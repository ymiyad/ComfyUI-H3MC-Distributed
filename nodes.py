"""
ComfyUI-H3MC-Distributed

Three nodes that let ComfyUI-H3-Motion-Context chain generation run
across ComfyUI-Distributed workers, without any shared folder and
without modifying ComfyUI-H3-Motion-Context itself:

  H3 Motion Context Upload Latent (Distributed)
      Drop in place of the sampler -> ... link that used to go through
      H3 Motion Context Save Latent. Passes the latent through
      unchanged (so it stays on the dependency chain ComfyUI-Distributed
      ships to workers), and hands a copy to Master -- locally if this
      IS Master, over HTTP if this is a Worker -- keyed by this run's
      job id and this participant's batch position (matching Image
      Batch Divider's batch_N numbering exactly).

  H3 Motion Context Save Video With Latent (Distributed)
      Run on Master, once per candidate, after Distributed Collector +
      Image Batch Divider have split the round's results. Saves a video
      with its H3 AV latent embedded as container metadata (the same
      mechanism ComfyUI's own Save Video uses to embed workflow JSON),
      so the video file itself is the only thing you need to keep or
      hand off to continue the chain.

  H3 Motion Context Load Latent From Video (Distributed)
      Loads the latent embedded in a video saved by the node above.
      Uses a standard ComfyUI video-upload combo reading from
      ComfyUI/input, so ComfyUI-Distributed's own media-sync ships the
      chosen file to workers automatically for the next round -- no
      shared folder, no manual path-passing.

Why "batch position" and not a fixed "0 = Master" slot: whether Master
renders its own candidate or runs Orchestrator-only changes which
position its own DistributedCollector node reports, and a Worker has no
reliable way to detect Master's render/orchestrate mode from its own
copy of that node -- so master_participates on the Upload node is an
explicit switch you set to match your Distributed panel's "Include
Master" setting for this workflow.
"""

import os
import json
import base64
import urllib.request
import urllib.parse
import urllib.error

import folder_paths
from safetensors.torch import save as _st_save_bytes, load as _st_load_bytes

from . import _store

try:
    from comfy.cli_args import args as _comfy_args
except ImportError:
    _comfy_args = None

try:
    import av as _av
except ImportError:
    _av = None

try:
    from comfy_api.latest import Types as _ComfyVideoTypes
except ImportError:
    _ComfyVideoTypes = None


_H3_VIDEO_LATENT_METADATA_KEY = "h3_motion_context_latent_v1"


def _h3_streams_from_latent(latent):
    """Unpack an H3 AV latent the same way H3 Motion Context's own
    _streams_from_latent does: samples is usually a NestedTensor-like
    object (has .unbind()), not a plain list/tuple."""
    samples = latent["samples"] if isinstance(latent, dict) else latent
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context_video: expected a MiniMax H3 AV latent "
            "(a nested video/audio pair), got %r" % type(samples)
        )
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context_video: latent has no audio stream; wire "
            "the sampler output of an H3 AV graph."
        )
    return parts


_MASTER_STORE_IDX = -1  # sentinel: Master's own candidate (never collides
                        # with a worker_position, which is always >= 0)


def _h3_distributed_context(prompt):
    """Read what ComfyUI-Distributed already injected into this run's
    DistributedCollector node. Returns None on a plain (non-Distributed)
    run.

    worker_position is the 0-based index of this worker within
    enabled_worker_ids (config order) -- always unambiguous from a
    Worker's own perspective, unlike its eventual batch_N number.

    delegate_only is only meaningful (and only trustworthy) when
    is_worker is False: ComfyUI-Distributed always injects
    delegate_only=False into a Worker's own copy of this field
    regardless of Master's actual mode, since it's a Master-only
    concept. Nodes that need it must therefore only read it on Master --
    which H3 Motion Context Save Video With Latent can rely on, since
    ComfyUI-Distributed never ships nodes downstream of Distributed
    Collector to workers."""
    if not isinstance(prompt, dict):
        return None
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != "DistributedCollector":
            continue
        inputs = node.get("inputs", {}) or {}
        if "multi_job_id" not in inputs:
            continue
        try:
            enabled = json.loads(inputs.get("enabled_worker_ids") or "[]")
        except (TypeError, ValueError):
            enabled = []
        is_worker = bool(inputs.get("is_worker"))
        raw_worker_id = inputs.get("worker_id") or ""
        worker_position = (
            enabled.index(raw_worker_id)
            if (is_worker and raw_worker_id in enabled) else None
        )
        return {
            "is_worker": is_worker,
            "worker_position": worker_position,
            "delegate_only": (None if is_worker else bool(inputs.get("delegate_only"))),
            "multi_job_id": inputs.get("multi_job_id") or "",
            "master_url": inputs.get("master_url") or "",
        }
    return None


class H3MotionContextUploadLatent:
    """Replaces H3 Motion Context Save Latent in the chain -- put this
    directly after your sampler:

        Sampler -> H3 Motion Context Upload Latent -> VAE Decode -> ...

    Passes the latent through unchanged (so it stays on the dependency
    chain ComfyUI-Distributed ships to workers) and hands a copy to
    Master, keyed by this run's job id and this participant's own
    identity -- Master's own candidate, or this worker's 0-based
    position in enabled_worker_ids. Translating that into the batch_N
    numbering Image Batch Divider uses is left entirely to H3 Motion
    Context Save Video With Latent, which always runs on Master and can
    read Master's true Orchestrator-only status reliably (a Worker
    cannot). No shared folder required."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"latent": ("LATENT",)},
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Hand this participant's H3 latent to Master (over HTTP if "
        "running on a Distributed Worker), keyed by job id + this "
        "participant's identity. No shared folder needed."
    )

    def run(self, latent, prompt=None):
        ctx = _h3_distributed_context(prompt)
        video_t, audio_t = _h3_streams_from_latent(latent)[:2]
        payload = _st_save_bytes({
            "video": video_t.cpu().contiguous(),
            "audio": audio_t.cpu().contiguous(),
        })

        if ctx is None or not ctx["is_worker"]:
            # Master's own candidate (or a plain single-machine run,
            # where there is only ever one "participant"). If Master is
            # Orchestrator-only, this node simply never executes on
            # Master (nothing upstream renders there), so no incorrect
            # entry ever gets written for a candidate that doesn't exist.
            job_id = (ctx["multi_job_id"] if ctx else "") or "single"
            idx = _MASTER_STORE_IDX
            path = _store.write_latent(job_id, idx, payload)
            _store.cleanup_older_than()
            print(f"[H3MotionContextUploadLatent] stored locally (Master) "
                  f"job={job_id} -> {path}")
        else:
            job_id = ctx["multi_job_id"] or "single"
            idx = ctx["worker_position"] if ctx["worker_position"] is not None else 0
            self._upload(ctx["master_url"], job_id, idx, payload)
            print(f"[H3MotionContextUploadLatent] uploaded job={job_id} "
                  f"worker_position={idx} ({len(payload)} bytes) to "
                  f"{ctx['master_url']}")

        return (latent,)

    @staticmethod
    def _upload(master_url, job_id, idx, payload):
        if not master_url:
            raise RuntimeError(
                "h3_motion_context_video: no master_url available for "
                "this worker; cannot upload the latent."
            )
        url = (master_url.rstrip("/")
               + "/h3_distributed/latent?job=" + urllib.parse.quote(str(job_id))
               + "&idx=" + str(int(idx)))
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status >= 300:
                    raise RuntimeError(
                        "h3_motion_context_video: upload to %s failed: "
                        "HTTP %s" % (master_url, resp.status)
                    )
        except urllib.error.URLError as e:
            raise RuntimeError(
                "h3_motion_context_video: could not reach Master at %s "
                "(%s). Check that Master's ComfyUI is reachable from "
                "this worker." % (master_url, e)
            )


class H3MotionContextSaveVideoWithLatent:
    """Save a video AND the H3 Motion Context AV latent needed to
    continue it, as a single file. Run this on Master, once per
    candidate, after Distributed Collector + Image Batch Divider have
    split the round's results -- wire batch_N's decoded video in and
    set participant_index to that same N.

    The latent rides along as container metadata -- the same mechanism
    ComfyUI itself uses to embed workflow JSON into saved videos -- so
    there is no separate .safetensors file to keep in sync, overwrite
    by accident, or lose track of. To continue the chain, point
    H3 Motion Context Load Latent From Video at this exact file.

    The saved file also carries the same "prompt"/"workflow" metadata
    core ComfyUI Save Video writes, so it's draggable back into ComfyUI
    to restore the graph, and is skipped the same way under
    --disable-metadata.

    Re-encoding, transcoding, or re-muxing the saved file elsewhere will
    strip this metadata -- keep the original around for as long as you
    might want to continue the chain from it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {
                    "tooltip": "From Create Video, fed by the matching "
                               "batch_N from Image Batch Divider."}),
                "participant_index": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Must match the batch_N this video came "
                               "from (batch_1 -> 1, batch_2 -> 2, ...)."}),
                "filename_prefix": ("STRING", {"default": "h3_chain/clip"}),
                "format": (["mp4", "mkv", "webm"], {"default": "mp4"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Save a video with its H3 Motion Context AV latent embedded as "
        "container metadata: one file is both the video and the "
        "continuation state."
    )

    def save(self, video, participant_index, filename_prefix, format="mp4", prompt=None, extra_pnginfo=None):
        if _av is None:
            raise RuntimeError(
                "h3_motion_context_video: PyAV ('av') is not available."
            )

        ctx = _h3_distributed_context(prompt)
        job_id = (ctx["multi_job_id"] if ctx else "") or "single"

        # Translate Image Batch Divider's 1-based batch_N into the key
        # H3 Motion Context Upload Latent actually stored it under. Only
        # trustworthy here because this node always runs on Master (see
        # class docstring), so ctx["delegate_only"] reflects Master's
        # own real Orchestrator-only status, not a Worker's fixed
        # placeholder value.
        delegate_only = bool(ctx["delegate_only"]) if ctx and ctx["delegate_only"] is not None else False
        if ctx is None:
            store_idx = _MASTER_STORE_IDX  # plain single-machine run
        elif delegate_only:
            # No Master candidate: batch_1 = first worker.
            store_idx = participant_index - 1
        elif participant_index == 1:
            store_idx = _MASTER_STORE_IDX  # Master's own candidate
        else:
            store_idx = participant_index - 2

        try:
            payload_in = _store.read_latent(job_id, store_idx)
        except OSError as e:
            raise FileNotFoundError(
                "h3_motion_context_video: no latent stored for job=%r "
                "participant_index=%r (resolved store key %r) yet "
                "(%s). Make sure H3 Motion Context Upload Latent ran "
                "for that participant in this same run, and that "
                "participant_index matches the batch_N this video came "
                "from." % (job_id, participant_index, store_idx, e)
            )
        data_in = _st_load_bytes(payload_in)
        video_tensor = data_in["video"]
        audio_tensor = data_in["audio"]

        payload = _st_save_bytes({
            "video": video_tensor.cpu().contiguous(),
            "audio": audio_tensor.cpu().contiguous(),
        })
        b64 = base64.b64encode(payload).decode("ascii")

        width, height = video.get_dimensions()
        full_folder, filename, counter, subfolder, _prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(),
                width, height,
            )
        )
        file = f"{filename}_{counter:05}_.{format}"
        path = os.path.join(full_folder, file)

        # Same metadata shape ComfyUI's own Save Video node writes
        # (comfy_extras/nodes_video.py): "prompt" plus every extra_pnginfo
        # key (typically "workflow"), passed as RAW objects, not
        # pre-JSON-encoded. video.save_to() (VideoFromComponents.save_to,
        # specifically) always runs json.dumps() itself on every metadata
        # value it's given, even ones that are already strings -- so
        # pre-encoding here would double-encode prompt/workflow into a
        # JSON string containing a JSON string, which ComfyUI's own
        # drag-and-drop workflow loader can't parse back into a real
        # object. Passing raw dict values here (matching core exactly)
        # means save_to() JSON-encodes them exactly once. Skipped under
        # --disable-metadata, same as core.
        metadata = {}
        if _comfy_args is None or not getattr(_comfy_args, "disable_metadata", False):
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
        # Our own H3 latent key rides alongside prompt/workflow. It's
        # already a string (base64), but save_to() will run json.dumps()
        # on it too (wrapping it in quotes) -- H3MotionContextLoadLatent
        # FromVideo below undoes that with a json.loads() attempt before
        # base64-decoding, so this round-trips regardless of which
        # save_to() code path ends up handling it.
        metadata[_H3_VIDEO_LATENT_METADATA_KEY] = b64
        save_kwargs = {"metadata": metadata}
        if _ComfyVideoTypes is not None:
            save_kwargs["format"] = _ComfyVideoTypes.VideoContainer(format)
        else:
            save_kwargs["format"] = format
        video.save_to(path, **save_kwargs)

        # Relative to the output directory: this may be picked up on a
        # different machine than the one that saved it, and an absolute
        # path uses that machine's own OS path syntax (e.g. macOS
        # "/Volumes/..."), which won't resolve elsewhere.
        try:
            rel_path = os.path.relpath(path, folder_paths.get_output_directory())
        except ValueError:
            rel_path = path

        print(f"[H3MotionContextSaveVideoWithLatent] saved {path} "
              f"(embedded latent: {len(payload)} bytes)")

        # Same UI payload shape ComfyUI's own Save Video node returns
        # (comfy_extras/nodes_video.py, PreviewVideo.as_dict()), so this
        # node gets the same inline video-preview/playback widget.
        preview = {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}],
                   "animated": (True,)}
        return {"ui": preview, "result": (video, rel_path)}


class H3MotionContextLoadLatentFromVideo:
    """Load the H3 Motion Context AV latent embedded in a video saved by
    H3 Motion Context Save Video With Latent (Distributed), for the
    context_latent input.

    Uses a standard ComfyUI video-upload combo reading from
    ComfyUI/input: ComfyUI-Distributed's own media-sync ships whichever
    file you pick to every worker for the next round automatically, so
    there is no shared folder and no manual path-passing.

    Set enabled=False for the very first clip (no previous context)."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        file_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        video_extensions = ('.mp4', '.webm', '.mkv')
        file_list = [f for f in file_list if f.lower().endswith(video_extensions)]
        if not file_list:
            file_list = ["none"]
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Leave off for the first clip in a chain "
                               "(no previous context yet).",
                }),
                "video": (sorted(file_list), {
                    "video_upload": True,
                    "tooltip": "A video saved by H3 Motion Context Save "
                               "Video With Latent (Distributed).",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = "Load the H3 Motion Context AV latent embedded in a video's own metadata."

    @staticmethod
    def _resolve(video):
        p = folder_paths.get_annotated_filepath(video)
        if not p:
            raise ValueError("h3_motion_context_video: video is empty")
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(
            "h3_motion_context_video: video file not found: %r" % video
        )

    @classmethod
    def IS_CHANGED(cls, video, enabled=False):
        if not enabled:
            return False
        try:
            p = cls._resolve(video)
            return "%s:%d" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return float("nan")

    def load(self, enabled, video):
        if not enabled:
            print("[H3MotionContextLoadLatentFromVideo] disabled (no previous clip); returning None")
            return (None,)
        if _av is None:
            raise RuntimeError(
                "h3_motion_context_video: PyAV ('av') is not available; "
                "cannot read video metadata."
            )
        p = self._resolve(video)

        container = _av.open(p)
        try:
            raw_tag = container.metadata.get(_H3_VIDEO_LATENT_METADATA_KEY)
        finally:
            container.close()

        if not raw_tag:
            raise ValueError(
                "h3_motion_context_video: %s has no embedded H3 latent "
                "(missing '%s' metadata tag). Was it saved by H3 Motion "
                "Context Save Video With Latent (Distributed), or has "
                "it been re-encoded/re-muxed since?"
                % (p, _H3_VIDEO_LATENT_METADATA_KEY)
            )

        # video.save_to() always runs json.dumps() on every metadata
        # value on at least one of its code paths, so our base64 string
        # usually comes back JSON-quoted (e.g. '"<base64>..."'). Undo
        # that if present; fall back to the raw tag for files written
        # via a save_to() path that left plain strings alone.
        b64 = raw_tag
        try:
            unwrapped = json.loads(raw_tag)
            if isinstance(unwrapped, str):
                b64 = unwrapped
        except (TypeError, ValueError):
            pass

        payload = base64.b64decode(b64)
        data = _st_load_bytes(payload)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "h3_motion_context_video: %s's embedded latent is "
                "missing video/audio streams" % p
            )
        video_t = data["video"].contiguous().clone()
        audio_t = data["audio"].contiguous().clone()
        print(f"[H3MotionContextLoadLatentFromVideo] loaded latent from {p} "
              f"(video {tuple(video_t.shape)}, audio {tuple(audio_t.shape)}, "
              f"payload {len(payload)} bytes)")
        return ({"samples": [video_t, audio_t]},)


NODE_CLASS_MAPPINGS = {
    "H3MotionContextUploadLatent": H3MotionContextUploadLatent,
    "H3MotionContextSaveVideoWithLatent": H3MotionContextSaveVideoWithLatent,
    "H3MotionContextLoadLatentFromVideo": H3MotionContextLoadLatentFromVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MotionContextUploadLatent": "H3 Motion Context Upload Latent (Distributed)",
    "H3MotionContextSaveVideoWithLatent": "H3 Motion Context Save Video With Latent (Distributed)",
    "H3MotionContextLoadLatentFromVideo": "H3 Motion Context Load Latent From Video (Distributed)",
}
