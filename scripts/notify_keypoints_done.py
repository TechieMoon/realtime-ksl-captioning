from __future__ import annotations

import ctypes
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
TRAIN_MANIFEST = ROOT / "manifests" / "train.json"
VAL_MANIFEST = ROOT / "manifests" / "val.json"
MARKER = LOG_DIR / "keypoints_done.notify.txt"
LOG_PATH = LOG_DIR / f"notify_keypoints_done_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp()} {message}\n")


def notify(message: str) -> None:
    MARKER.write_text(f"{stamp()} {message}\n", encoding="utf-8")
    try:
        ctypes.windll.user32.MessageBoxW(None, message, "Keypoint extraction", 0x40)
    except Exception:
        pass


def ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    if MARKER.exists():
        MARKER.unlink()
    log("Waiting for keypoint extraction to finish")
    while True:
        train_ready = ready(TRAIN_MANIFEST)
        val_ready = ready(VAL_MANIFEST)
        log(
            "status "
            f"train.json={'yes' if train_ready else 'no'} "
            f"val.json={'yes' if val_ready else 'no'}"
        )
        if train_ready and val_ready:
            notify("Keypoint extraction finished. Training has NOT been started.")
            log("notified keypoint extraction completion")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
