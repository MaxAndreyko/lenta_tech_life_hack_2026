from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.controllers.HealthController import HealthController
from backend.app.api.controllers.VideoController import VideoController
from backend.app.api.routers import health, video
from backend.app.processor.PriceTagProcessor import PriceTagProcessor


class Server:
    def __init__(
        self,
        upload_dir: str = "uploads",
        output_dir: str = "output",
        title: str = "Price Tag Reader API",
        version: str = "1.0.0"
    ):
        # todo change model
        self.processor = PriceTagProcessor("YOBA")
        self.health_controller = HealthController()
        self.video_controller = VideoController(
            processor=self.processor,
            upload_dir=upload_dir,
            output_dir=output_dir
        )
        self.app = self._create_app(title, version)

    def _create_app(self, title: str, version: str) -> FastAPI:
        app = FastAPI(title=title, version=version)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.state.health_controller = self.health_controller
        app.state.video_controller = self.video_controller

        app.include_router(health.router, tags=["Health"])
        app.include_router(video.router, tags=["Video"])

        return app

    def get_app(self) -> FastAPI:
        return self.app