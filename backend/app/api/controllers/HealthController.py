from backend.app.api.response import HealthResponse
from backend.app.api.routers.base import BaseController


class HealthController(BaseController):
    def __init__(self):
        pass

    async def get_health(self) -> HealthResponse:
        return HealthResponse(status="ok", models_loaded=True, version="1.0.0")

    async def get_info(self) -> dict:
        return {
            "service": "Price Tag Reader API",
            "version": "1.0.0",
            "endpoints": {
                "health": "GET /health",
                "process": "POST /process",
                "download": "GET /download/{video_id}",
                "delete": "DELETE /results/{video_id}",
                "docs": "/docs"
            }
        }