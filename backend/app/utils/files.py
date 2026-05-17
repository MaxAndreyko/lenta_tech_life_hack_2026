"""
Stateless функции для работы с файлами.
"""

import uuid
from pathlib import Path
from typing import Tuple


def get_upload_dir(base_dir: str = "uploads") -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(base_dir: str = "output") -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload(file_obj, original_filename: str, upload_dir: str = "uploads") -> Tuple[Path, str]:
    video_id = str(uuid.uuid4())[:8]
    safe_name = f"{video_id}_{Path(original_filename).name}"
    file_path = get_upload_dir(upload_dir) / safe_name

    with open(file_path, "wb") as f:
        f.write(file_obj.read())

    return file_path, video_id


def delete_upload(file_path: Path) -> None:
    if file_path.exists():
        file_path.unlink()


def get_csv_path(video_id: str, output_dir: str = "output") -> Path:
    return get_output_dir(output_dir) / f"{video_id}_results.csv"


def csv_exists(video_id: str, output_dir: str = "output") -> bool:
    return get_csv_path(video_id, output_dir).exists()


def delete_results(video_id: str, output_dir: str = "output") -> bool:
    csv_path = get_csv_path(video_id, output_dir)
    if csv_path.exists():
        csv_path.unlink()
        return True
    return False


def cleanup_old_uploads(upload_dir: str = "uploads", max_age_hours: int = 24) -> int:
    import time
    deleted = 0
    now = time.time()
    for file_path in get_upload_dir(upload_dir).iterdir():
        if file_path.is_file():
            if (now - file_path.stat().st_mtime) / 3600 > max_age_hours:
                file_path.unlink()
                deleted += 1
    return deleted


def cleanup_old_results(output_dir: str = "output", max_age_hours: int = 168) -> int:
    import time
    deleted = 0
    now = time.time()
    for file_path in get_output_dir(output_dir).iterdir():
        if file_path.is_file() and file_path.suffix == '.csv':
            if (now - file_path.stat().st_mtime) / 3600 > max_age_hours:
                file_path.unlink()
                deleted += 1
    return deleted