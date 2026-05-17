from fastapi import UploadFile, HTTPException
from fastapi.responses import FileResponse

from backend.app.api.response import ProcessResponse, ProcessRequest, PriceTagResponse
from backend.app.api.routers.base import BaseController
from backend.app.processor.PriceTagProcessor import PriceTagProcessor
from backend.app.utils import files


class VideoController(BaseController):
    def __init__(
        self,
        processor: PriceTagProcessor,
        upload_dir: str = "uploads",
        output_dir: str = "output"
    ):
        self.processor = processor
        self.upload_dir = upload_dir
        self.output_dir = output_dir

    async def process_video(self, file: UploadFile, params: ProcessRequest) -> ProcessResponse:
        self._validate_format(file.filename)

        video_path, video_id = files.save_upload(file.file, file.filename, self.upload_dir)

        try:
            csv_path = files.get_csv_path(video_id, self.output_dir)
            results = self.processor.process_video(
                video_path=str(video_path),
                output_csv=str(csv_path),
                frame_interval=params.frame_interval,
                confidence_threshold=params.confidence_threshold
            )
            tags = self._results_to_tags(results)
        except Exception as e:
            files.delete_upload(video_path)
            raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")

        files.delete_upload(video_path)

        return ProcessResponse(
            video_id=video_id,
            total_tags=len(tags),
            tags=tags,
            csv_download_url=f"/download/{video_id}"
        )

    async def download_csv(self, video_id: str) -> FileResponse:
        csv_path = files.get_csv_path(video_id, self.output_dir)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Результаты не найдены")
        return FileResponse(
            path=str(csv_path),
            filename=f"price_tags_{video_id}.csv",
            media_type="text/csv"
        )

    async def delete_results(self, video_id: str) -> dict:
        deleted = files.delete_results(video_id, self.output_dir)
        if not deleted:
            raise HTTPException(status_code=404, detail="Результаты не найдены")
        return {"status": "deleted", "video_id": video_id}

    def _validate_format(self, filename: str) -> None:
        if not self.validate_video_format(filename):
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат. Разрешены: {self.ALLOWED_VIDEO_EXTENSIONS}"
            )

    def _results_to_tags(self, results) -> list[PriceTagResponse]:
        return []