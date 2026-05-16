from pathlib import Path

import cv2


def load_css(css_path: str) -> str:
    """Reads .css from provided path"""
    css_file = Path(css_path)
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"⚠️ CSS file not found: {css_path}")
        return ""


def get_preview_video_frame(video_path):
    """Извлечение случайного кадра из видео для предпросмотра"""
    if video_path is None:
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Берем кадр из середины видео
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None
    except:
        return None