from pathlib import Path


def load_css(css_path: str) -> str:
    """Reads .css from provided path"""
    css_file = Path(css_path)
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"⚠️ CSS file not found: {css_path}")
        return ""