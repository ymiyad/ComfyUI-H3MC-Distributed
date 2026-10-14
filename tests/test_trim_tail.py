"""Tail trimming of the H3 latent handed to Master.

The pure tests need only torch + safetensors. The equivalence test also runs
ComfyUI-H3-Motion-Context's own node; point it at checkouts with
    COMFYUI_SOURCE_ROOT=/path/to/ComfyUI H3_MOTION_CONTEXT_ROOT=/path/to/ComfyUI-H3-Motion-Context
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_package(name, root, module, stubs=()):
    for stub in stubs:
        sys.modules.setdefault(stub, types.ModuleType(stub))
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(root)]
    sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location(f"{name}.{module}", Path(root) / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")
h3mc = _load_package("h3mc_under_test", ROOT, "nodes")


def _clip(frames, seed=0, h=3, w=5):
    steps = (frames - 5) // 17 * 5 + 2
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 24, steps, h, w, generator=g),
            torch.randn(1, 32, 2, int(round(frames * 5 / 3)), generator=g))


@pytest.mark.parametrize("frames", [124, 141, 158, 243, 260, 277, 362, 481, 600])
@pytest.mark.parametrize("keep", [1, 56, 96, 240])
def test_trim_keeps_phase_overhang_and_tail(frames, keep):
    video, audio = _clip(frames, seed=frames)
    tv, ta, info = h3mc.trim_h3_latent_tail(video, audio, keep)
    if not info["trimmed"]:
        assert tv is video and ta is audio
        return
    total, kept = info["video_steps"]
    assert (total - kept) % 15 == 0 and kept % 5 == 2
    assert info["kept_frames"] >= keep
    assert torch.equal(tv, video[:, :, total - kept:])
    assert torch.equal(ta, audio[..., audio.shape[-1] - ta.shape[-1]:])
    before = audio.shape[-1] - 5 / 3 * h3mc._h3_pixel_frames(total)
    after = ta.shape[-1] - 5 / 3 * h3mc._h3_pixel_frames(kept)
    assert abs(before - after) < 1e-9


def test_trim_shrinks_a_15s_clip_about_six_times():
    video, audio = _clip(362)
    tv, ta, info = h3mc.trim_h3_latent_tail(video, audio, 56)
    assert info["video_steps"] == (107, 17) and info["audio_steps"] == (603, 93)


@pytest.mark.parametrize("keep", [0, -1, None])
def test_trim_disabled(keep):
    video, audio = _clip(362)
    tv, ta, info = h3mc.trim_h3_latent_tail(video, audio, keep)
    assert tv is video and ta is audio and not info["trimmed"]


def test_unexpected_layouts_are_left_whole():
    video, audio = _clip(362)
    for v, a in ((video[:, :, :-1], audio), (video, audio[..., :-3]), (video[0, 0], audio)):
        tv, ta, info = h3mc.trim_h3_latent_tail(v, a, 56)
        assert tv is v and ta is a and not info["trimmed"]


def test_unbatched_latent():
    video, audio = _clip(362)
    tv, ta, info = h3mc.trim_h3_latent_tail(video[0], audio[0], 56)
    assert info["trimmed"] and tv.shape[1] == 17 and ta.shape[-1] == 93


def test_matches_real_h3_motion_context():
    comfy_root = os.environ.get("COMFYUI_SOURCE_ROOT")
    h3_root = os.environ.get("H3_MOTION_CONTEXT_ROOT")
    if not (comfy_root and h3_root):
        pytest.skip("set COMFYUI_SOURCE_ROOT and H3_MOTION_CONTEXT_ROOT")
    sys.path.insert(0, comfy_root)
    sys.modules.pop("folder_paths", None)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    h3 = _load_package("h3_reference", h3_root, "nodes")
    node = h3.MiniMaxH3MotionContext()

    def apply(v, a, target, cl, acl):
        out, trim = node.apply([[torch.zeros(1, 4, 8), {}]], None, {"samples": [target, None]}, cl,
                               audio_context_length=acl, context_latent={"samples": [v, a]})
        return trim, out[0][1]["minimax_keyframes"]

    for frames in (124, 243, 362, 481):
        video, audio = _clip(frames, seed=frames)
        target = torch.zeros(1, 24, video.shape[2], 3, 5)
        tv, ta, _ = h3mc.trim_h3_latent_tail(video, audio, 56)
        for cl in (5, 22, 39, 56):
            for acl in (0, 3, 24, 56):
                (t1, k1), (t2, k2) = apply(video, audio, target, cl, acl), apply(tv, ta, target, cl, acl)
                assert t1 == t2 and len(k1) == len(k2)
                for x, y in zip(k1, k2):
                    assert x["resolved_frame_index"] == y["resolved_frame_index"]
                    for key in ("latent", "audio_latent"):
                        assert (key in x) == (key in y)
                        if key in x:
                            assert torch.equal(x[key], y[key])
