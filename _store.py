"""Temp-file-backed store for H3 Motion Context latents uploaded from
Distributed workers to the Master, keyed by (job_id, participant_index).

Used by H3MotionContextUploadLatent (writer), the /h3_distributed/latent
HTTP route (writer, for cross-machine uploads), and
H3MotionContextFetchLatent (reader)."""

import os
import re
import time

import folder_paths

_STORE_DIR_NAME = "h3_distributed_latents"


def store_dir():
    d = os.path.join(folder_paths.get_temp_directory(), _STORE_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _safe_key(job_id, idx):
    job_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(job_id or "unknown"))[:128]
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = 0
    return job_id, idx


def store_path(job_id, idx):
    job_id, idx = _safe_key(job_id, idx)
    return os.path.join(store_dir(), f"{job_id}_{idx}.safetensors")


def write_latent(job_id, idx, payload: bytes) -> str:
    path = store_path(job_id, idx)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, path)
    return path


def read_latent(job_id, idx) -> bytes:
    path = store_path(job_id, idx)
    with open(path, "rb") as f:
        return f.read()


def cleanup_older_than(seconds: float = 6 * 3600):
    d = store_dir()
    now = time.time()
    try:
        for name in os.listdir(d):
            p = os.path.join(d, name)
            try:
                if now - os.path.getmtime(p) > seconds:
                    os.remove(p)
            except OSError:
                pass
    except OSError:
        pass
