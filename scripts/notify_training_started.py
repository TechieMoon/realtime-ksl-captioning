from __future__ import annotations

import ctypes
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MARKER = LOG_DIR / "training_started.notify.txt"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def notify(message: str) -> None:
    MARKER.write_text(f"{stamp()} {message}\n", encoding="utf-8")
    try:
        ctypes.windll.user32.MessageBoxW(None, message, "Sign model training", 0x40)
    except Exception:
        pass


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if MARKER.exists():
        MARKER.unlink()

    seen = {path.name for path in LOG_DIR.glob("word_classifier_train_*.log")}
    while True:
        current = {path.name for path in LOG_DIR.glob("word_classifier_train_*.log")}
        new_logs = sorted(current - seen)
        if new_logs:
            notify(f"Keypoint extraction finished. Model training started: {new_logs[-1]}")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
