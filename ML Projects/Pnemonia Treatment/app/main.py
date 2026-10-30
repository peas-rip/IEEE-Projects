"""FastAPI application factory.

Owns two long-lived resources and nothing else:

* the :class:`~pneumonia.service.PipelineService` — the seven checkpoints,
  loaded once at startup
* the bundled prediction artifact — read once, so Playground slider moves
  recompute from arrays in memory instead of touching disk per request

Both are lazy. A missing artifact degrades the app to inference-only rather
than refusing to boot, which is the difference between "the Playground tab is
empty" and "the whole app is down".
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import router

log = logging.getLogger("pneumonia.app")

_SERVICE: Any = None
_TABLE: Any = None


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_service():
    """The process-wide PipelineService, created on first use."""
    global _SERVICE
    if _SERVICE is None:
        from pneumonia.service import PipelineService

        _SERVICE = PipelineService()
    return _SERVICE


def get_table():
    """The bundled prediction artifact, or None when it is absent."""
    global _TABLE
    if _TABLE is None:
        return None
    return _TABLE


def load_table() -> Any:
    """Load the artifact into the module-level cache, tolerating its absence."""
    global _TABLE
    from .predictions import DEFAULT_NPZ_PATH, load_prediction_table

    path = os.path.join(_project_root(), DEFAULT_NPZ_PATH)
    try:
        _TABLE = load_prediction_table(path)
    except (FileNotFoundError, ValueError) as exc:
        _TABLE = None
        log.warning("prediction artifact unavailable: %s", exc)
        return None

    log.info("loaded prediction artifact: %d images", len(_TABLE))
    return _TABLE


def reset_cache() -> None:
    """Drop cached resources. Used by tests to isolate app state."""
    global _SERVICE, _TABLE
    _SERVICE = None
    _TABLE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the artifact at startup. Checkpoints load on the first request.

    Loading 1.05 GiB of checkpoints during startup would delay the first HTTP
    response by several seconds on a laptop and delay any startup failure, so
    it is deferred until something actually asks for a prediction.
    """
    load_table()
    yield


def create_app(*, serve_static: bool = True) -> FastAPI:
    """Build the FastAPI app.

    ``serve_static`` is disabled by tests, which do not need the SPA.
    """
    app = FastAPI(
        title="Pneumonia Detection",
        description="Educational two-stage pneumonia pipeline. Not a medical device.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Dev servers bound to localhost, so a permissive CORS policy only ever
    # exposes the API to this machine. Kept explicit rather than implicit.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    if serve_static:
        _mount_static(app)

    return app


def _mount_static(app: FastAPI) -> None:
    """Serve the built SPA and the curated sample images."""
    root = _project_root()

    samples_dir = os.path.join(root, "samples")
    if os.path.isdir(samples_dir):
        app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")

    dist = os.path.join(root, "frontend", "dist")
    if not os.path.isdir(dist):
        log.warning("frontend/dist missing; run `npm run build` in frontend/")
        return

    assets = os.path.join(dist, "assets")
    if os.path.isdir(assets):
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = os.path.join(dist, "index.html")

    @app.get("/", include_in_schema=False)
    def spa_index() -> FileResponse:
        return FileResponse(index)


app = create_app()


def main() -> int:
    """Entry point for ``python -m app.main``."""
    import uvicorn

    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))

    print(f"Starting on http://{host}:{port}")
    print("Educational use only. Not a medical device.")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
