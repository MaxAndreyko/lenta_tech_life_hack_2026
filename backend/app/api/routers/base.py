from pathlib import Path
from typing import Set


class BaseController:
    ALLOWED_VIDEO_EXTENSIONS: Set[str] = {'.mp4', '.avi', '.mov', '.mkv'}

    def validate_video_format(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in self.ALLOWED_VIDEO_EXTENSIONS