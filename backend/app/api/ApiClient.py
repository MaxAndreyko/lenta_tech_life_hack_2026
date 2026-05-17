import uuid
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

from backend.app.api.response import PriceTagResponse


class ApiClient:
    """Клиент для управления файлами и вызова процессинга"""
    def __init__(
            self,
            upload_dir: str = "uploads",
            output_dir: str = "output"
    ):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_ready(self) -> bool:
        """Заглушка — всегда готов"""
        return True

    def save_uploaded_video(self, file_obj, filename: str) -> Tuple[Path, str]:
        """
        Сохраняет загруженное видео.
        Возвращает (путь, video_id).
        """
        video_id = str(uuid.uuid4())[:8]
        safe_name = f"{video_id}_{Path(filename).name}"
        file_path = self.upload_dir / safe_name

        with open(file_path, "wb") as f:
            f.write(file_obj.read())

        return file_path, video_id

    def process_video(
            self,
            video_path: Path,
            video_id: str,
            frame_interval: int = 5,
            confidence_threshold: float = 0.3
    ) -> Tuple[List[PriceTagResponse], Path]:
        """
        Обработка видео.
        Сейчас заглушка — возвращает пустой список.
        """
        # TODO: вызвать PriceTagProcessor.process_video()
        tags: List[PriceTagResponse] = []
        csv_path = self.output_dir / f"{video_id}_results.csv"

        # Заглушка: создаём пустой CSV с правильными колонками
        df = pd.DataFrame(columns=PriceTagResponse.model_fields.keys())
        df.to_csv(csv_path, index=False)

        return tags, csv_path

    def cleanup_video(self, video_path: Path):
        """Удаляет временное видео"""
        if video_path.exists():
            video_path.unlink()

    def get_csv_path(self, video_id: str) -> Optional[Path]:
        """Возвращает путь к CSV результатам"""
        csv_path = self.output_dir / f"{video_id}_results.csv"
        return csv_path if csv_path.exists() else None

    def delete_results(self, video_id: str) -> bool:
        """Удаляет CSV с результатами"""
        csv_path = self.get_csv_path(video_id)
        if csv_path:
            csv_path.unlink()
            return True
        return False