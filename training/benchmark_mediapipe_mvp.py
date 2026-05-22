from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
AI_MODEL_DIR = REPO_ROOT / "ai_model"
if str(AI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODEL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_model.mediapipe_mvp import MediaPipeMvpRecognizer  # noqa: E402
from training.train_mediapipe_mvp import _collect_samples, _default_cache_dir, _default_data_root, _load_label_infos  # noqa: E402


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_samples(
        data_root=data_root,
        labels={args.label},
        angles={args.angle},
        label_infos=_load_label_infos(data_root),
        max_per_label=1,
    )
    if not samples:
        raise SystemExit(f"No sample found for label={args.label!r} angle={args.angle!r}")

    sample = samples[0]
    recognizer = MediaPipeMvpRecognizer(args.artifact.resolve(), device=args.device)
    timings_ms: list[float] = []
    predictions = 0

    try:
        with ZipFile(sample.zip_path) as zip_file:
            video_bytes = zip_file.read(sample.member)

        scratch_dir = cache_dir / "_tmp"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=scratch_dir, delete=True) as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()

            capture = cv2.VideoCapture(temp_file.name)
            frame_id = 0
            while capture.isOpened() and frame_id < args.max_frames:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                started = time.perf_counter()
                result = recognizer.predict_one(frame_rgb, frame_id * 33)
                timings_ms.append((time.perf_counter() - started) * 1000)
                if result["text"]:
                    predictions += 1
                frame_id += 1
            capture.release()
    finally:
        recognizer.close()

    if not timings_ms:
        raise SystemExit("No frames were benchmarked.")

    mean_ms = statistics.fmean(timings_ms)
    sorted_timings = sorted(timings_ms)
    p95_ms = sorted_timings[min(len(sorted_timings) - 1, int(len(sorted_timings) * 0.95))]
    print(f"Sample: {sample.member}")
    print(f"Frames: {len(timings_ms)}")
    print(f"Predicted captions: {predictions}")
    print(f"Mean latency: {mean_ms:.2f} ms/frame")
    print(f"P95 latency: {p95_ms:.2f} ms/frame")
    print(f"Approx throughput: {1000.0 / mean_ms:.2f} fps")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the MediaPipe MVP recognizer on a real AIHub video clip.")
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    parser.add_argument("--artifact", type=Path, default=AI_MODEL_DIR / "mediapipe_mvp.joblib")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label", default="수어")
    parser.add_argument("--angle", default="F")
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    main()
