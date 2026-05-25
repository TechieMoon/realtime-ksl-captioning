from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
TRAIN_MANIFEST = ROOT / "manifests" / "train.json"
VAL_MANIFEST = ROOT / "manifests" / "val.json"
OUTPUT_DIR = ROOT / "checkpoints" / "word_classifier"
DONE_MARKER = LOG_DIR / "train_after_keypoints.done"
FAIL_MARKER = LOG_DIR / "train_after_keypoints.failed"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"train_after_keypoints_{RUN_ID}.log"


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp()} {message}\n")


def wait_for_manifests() -> None:
    log("Waiting for manifests/train.json and manifests/val.json")
    while True:
        train_ready = TRAIN_MANIFEST.exists() and TRAIN_MANIFEST.stat().st_size > 0
        val_ready = VAL_MANIFEST.exists() and VAL_MANIFEST.stat().st_size > 0
        if train_ready and val_ready:
            log("Both manifests are ready")
            return
        log(
            "Still waiting: "
            f"train.json={'yes' if train_ready else 'no'} "
            f"val.json={'yes' if val_ready else 'no'}"
        )
        time.sleep(60)


def run_training() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_log = LOG_DIR / f"word_classifier_train_{RUN_ID}.log"
    command = [
        sys.executable,
        "-m",
        "model.train_word_classifier",
        "--train-manifest",
        str(TRAIN_MANIFEST.relative_to(ROOT)),
        "--val-manifest",
        str(VAL_MANIFEST.relative_to(ROOT)),
        "--output-dir",
        str(OUTPUT_DIR.relative_to(ROOT)),
        "--epochs",
        "20",
        "--batch-size",
        "16",
        "--max-frames",
        "64",
    ]
    log(f"Starting training: {' '.join(command)}")
    log(f"Training output log: {train_log.relative_to(ROOT)}")
    with train_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()
    log(f"Training finished with exit={return_code}")
    return return_code


def main() -> None:
    log("train-after-keypoints watcher started")
    wait_for_manifests()
    return_code = run_training()
    if return_code == 0 and (OUTPUT_DIR / "best.pt").exists():
        DONE_MARKER.write_text(f"{stamp()} training complete\n", encoding="utf-8")
        log(f"Done: {OUTPUT_DIR / 'best.pt'}")
    else:
        FAIL_MARKER.write_text(f"{stamp()} training failed exit={return_code}\n", encoding="utf-8")
        raise SystemExit(return_code or 1)


if __name__ == "__main__":
    main()
