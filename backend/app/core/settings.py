from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StorageSettings:
    """Filesystem paths used by the local video-processing pipeline."""

    data_dir: Path = BACKEND_ROOT / "data"
    videos_dir: Path = BACKEND_ROOT / "data" / "videos"
    frames_dir: Path = BACKEND_ROOT / "data" / "frames"
    crops_dir: Path = BACKEND_ROOT / "data" / "crops"
    outputs_dir: Path = BACKEND_ROOT / "data" / "outputs"
    temp_dir: Path = BACKEND_ROOT / "data" / "temp"
    logs_dir: Path = BACKEND_ROOT / "logs"

    def ensure_directories(self) -> None:
        """Create all configured local storage directories."""

        for path in (
            self.data_dir,
            self.videos_dir,
            self.frames_dir,
            self.crops_dir,
            self.outputs_dir,
            self.temp_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppSettings:
    """Top-level application settings container."""

    storage: StorageSettings = field(default_factory=StorageSettings)


settings = AppSettings()
