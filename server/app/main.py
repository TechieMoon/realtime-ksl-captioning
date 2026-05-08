from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Response, WebSocket

from app.config import Settings, get_settings
from app.models import KslModelAdapter, build_model_adapter
from app.ws import handle_caption_socket


def create_app(settings: Settings | None = None, model_adapter: KslModelAdapter | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    adapter = model_adapter or build_model_adapter(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = app_settings
        app.state.model_adapter = adapter
        await adapter.load()
        try:
            yield
        finally:
            await adapter.close()

    app = FastAPI(title=app_settings.app_name, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": app_settings.app_name}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict[str, object]:
        if not adapter.ready:
            response.status_code = 503
        return {
            "status": "ready" if adapter.ready else "not_ready",
            "model_backend": app_settings.model_backend,
            "model_id": app_settings.hf_model_id,
            "model_revision": app_settings.hf_model_revision,
            "device": app_settings.model_device,
            "adapter": adapter.name,
        }

    @app.websocket("/ws/captions")
    async def captions(websocket: WebSocket) -> None:
        session_id = websocket.query_params.get("session_id")
        if not session_id:
            await websocket.close(code=1008, reason="session_id query parameter is required")
            return
        await handle_caption_socket(
            websocket,
            session_id=session_id,
            settings=app_settings,
            model=adapter,
        )

    return app


app = create_app()

