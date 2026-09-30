"""Registers POST /h3_distributed/latent on Master's own ComfyUI server,
so a Distributed Worker can hand off an H3 latent without any shared
folder. Only meaningful on Master; harmless (and simply unused) on a
machine that only ever runs as a Worker."""

from aiohttp import web

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from . import _store


def _register():
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return

    routes = PromptServer.instance.routes

    @routes.post("/h3_distributed/latent")
    async def _h3_distributed_receive_latent(request):
        job_id = request.rel_url.query.get("job", "unknown")
        idx = request.rel_url.query.get("idx", "0")
        payload = await request.read()
        if not payload:
            return web.json_response({"ok": False, "error": "empty body"}, status=400)
        try:
            path = _store.write_latent(job_id, idx, payload)
        except Exception as e:  # noqa: BLE001 - report any storage failure to the caller
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        _store.cleanup_older_than()
        return web.json_response({"ok": True, "bytes": len(payload), "path": path})


_register()
