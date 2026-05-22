from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
AI_MODEL_DIR = REPO_ROOT / "ai_model"
if str(AI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODEL_DIR))

from mediapipe_mvp import FEATURE_DIM, _create_holistic, extract_mediapipe_features  # noqa: E402


VIDEO_RE = re.compile(r"NIA_SL_WORD(\d+)_REAL(\d+)_([FDLRU])\.mp4$")
LABEL_RE = re.compile(r"NIA_SL_WORD(\d+)_REAL(\d+)_([FDLRU])_morpheme\.json$")

DEFAULT_LABELS = ["수어", "좋다", "감사", "괜찮다", "싫다", "이해", "부탁", "모르다", "맞다", "힘"]


@dataclass(frozen=True)
class LabelInfo:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class VideoSample:
    zip_path: Path
    member: str
    word_id: int
    signer_id: int
    angle: str
    label: str
    start: float
    end: float


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    label_infos = _load_label_infos(data_root)
    samples = _collect_samples(
        data_root=data_root,
        labels=set(labels),
        angles=set(args.angles),
        label_infos=label_infos,
        max_per_label=args.max_per_label,
    )
    if not samples:
        raise SystemExit("No matching video samples found.")

    print(f"Selected {len(samples)} samples for labels: {labels}")
    for label in labels:
        print(f"  {label}: {sum(sample.label == label for sample in samples)} samples")

    holistic = _create_holistic()
    try:
        x, y, groups = _extract_dataset(samples, holistic, args.sequence_length)
    finally:
        holistic.close()

    label_to_index = {label: index for index, label in enumerate(labels)}
    y_index = np.asarray([label_to_index[label] for label in y], dtype=np.int64)

    train_mask = np.asarray([group != args.eval_signer for group in groups], dtype=bool)
    if train_mask.sum() == 0 or (~train_mask).sum() == 0:
        print("Eval signer split is not possible; falling back to stratified random split.")
        train_mask = _stratified_split(y_index, train_ratio=0.8, seed=args.seed)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
    )
    classifier.fit(x[train_mask], y_index[train_mask])

    predictions = classifier.predict(x[~train_mask])
    accuracy = accuracy_score(y_index[~train_mask], predictions)
    print(f"Validation accuracy: {accuracy:.3f}")
    print(classification_report(y_index[~train_mask], predictions, target_names=labels, zero_division=0))

    artifact = {
        "classifier": classifier,
        "labels": labels,
        "sequence_length": args.sequence_length,
        "feature_dim": FEATURE_DIM,
        "confidence_threshold": args.confidence_threshold,
        "training_summary": {
            "samples": len(samples),
            "labels": labels,
            "angles": args.angles,
            "eval_signer": args.eval_signer,
            "accuracy": accuracy,
        },
    }
    joblib.dump(artifact, output_path)
    print(f"Saved model artifact: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MediaPipe keypoint MVP classifier.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "수어 영상")
    parser.add_argument("--output", type=Path, default=AI_MODEL_DIR / "mediapipe_mvp.joblib")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--angles", nargs="+", default=["F", "D", "U", "L", "R"])
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--max-per-label", type=int, default=60)
    parser.add_argument("--eval-signer", type=int, default=18)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _load_label_infos(data_root: Path) -> dict[tuple[int, int, str], LabelInfo]:
    labels: dict[tuple[int, int, str], LabelInfo] = {}
    for zip_path in data_root.rglob("*real_word_morpheme.zip"):
        with ZipFile(zip_path) as zip_file:
            for info in zip_file.infolist():
                match = LABEL_RE.search(info.filename)
                if not match:
                    continue
                word_id = int(match.group(1))
                signer_id = int(match.group(2))
                angle = match.group(3)
                raw = json.loads(zip_file.read(info).decode("utf-8"))
                text, start, end = _parse_morpheme(raw)
                if text:
                    labels[(word_id, signer_id, angle)] = LabelInfo(text=text, start=start, end=end)
    return labels


def _parse_morpheme(raw: dict) -> tuple[str, float, float]:
    rows = raw.get("data", [])
    if not rows:
        return "", 0.0, float(raw.get("metaData", {}).get("duration", 0.0) or 0.0)

    first = rows[0]
    names = [attr.get("name") for attr in first.get("attributes", []) if attr.get("name")]
    text = names[0] if names else ""
    start = float(first.get("start", 0.0) or 0.0)
    end = float(first.get("end", raw.get("metaData", {}).get("duration", start + 1.0)) or start + 1.0)
    return text, start, max(start + 0.1, end)


def _collect_samples(
    *,
    data_root: Path,
    labels: set[str],
    angles: set[str],
    label_infos: dict[tuple[int, int, str], LabelInfo],
    max_per_label: int,
) -> list[VideoSample]:
    samples_by_label: dict[str, list[VideoSample]] = {label: [] for label in labels}
    for zip_path in data_root.rglob("*real_word_video.zip"):
        with ZipFile(zip_path) as zip_file:
            for info in zip_file.infolist():
                match = VIDEO_RE.search(info.filename)
                if not match:
                    continue
                word_id = int(match.group(1))
                signer_id = int(match.group(2))
                angle = match.group(3)
                if angle not in angles:
                    continue
                label_info = label_infos.get((word_id, signer_id, angle))
                if label_info is None or label_info.text not in labels:
                    continue
                samples_by_label[label_info.text].append(
                    VideoSample(
                        zip_path=zip_path,
                        member=info.filename,
                        word_id=word_id,
                        signer_id=signer_id,
                        angle=angle,
                        label=label_info.text,
                        start=label_info.start,
                        end=label_info.end,
                    )
                )

    samples: list[VideoSample] = []
    for label in sorted(samples_by_label):
        label_samples = sorted(
            samples_by_label[label],
            key=lambda sample: (sample.signer_id, sample.word_id, sample.angle, sample.member),
        )
        samples.extend(label_samples[:max_per_label])
    return samples


def _extract_dataset(samples: list[VideoSample], holistic: object, sequence_length: int) -> tuple[np.ndarray, list[str], list[int]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[int] = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{total}] {sample.label} signer={sample.signer_id} angle={sample.angle} {sample.member}")
        sequence = _extract_video_sequence(sample, holistic, sequence_length)
        if sequence is None:
            print("  skipped: no frames extracted")
            continue
        features.append(sequence.reshape(-1))
        labels.append(sample.label)
        groups.append(sample.signer_id)

    if not features:
        raise SystemExit("No features were extracted.")
    return np.asarray(features, dtype=np.float32), labels, groups


def _extract_video_sequence(sample: VideoSample, holistic: object, sequence_length: int) -> np.ndarray | None:
    with ZipFile(sample.zip_path) as zip_file:
        video_bytes = zip_file.read(sample.member)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
        temp_file.write(video_bytes)
        temp_file.flush()

        capture = cv2.VideoCapture(temp_file.name)
        if not capture.isOpened():
            return None

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_frame = max(0, int(sample.start * fps))
        end_frame = min(frame_count - 1, max(start_frame + 1, int(sample.end * fps)))
        frame_indices = np.linspace(start_frame, end_frame, num=sequence_length, dtype=int)

        sequence: list[np.ndarray] = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = capture.read()
            if not ok:
                sequence.append(np.zeros(FEATURE_DIM, dtype=np.float32))
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            sequence.append(extract_mediapipe_features(frame_rgb, holistic))
        capture.release()

    return np.asarray(sequence, dtype=np.float32)


def _stratified_split(y: np.ndarray, *, train_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(y), dtype=bool)
    for class_id in sorted(set(int(item) for item in y)):
        indices = np.where(y == class_id)[0]
        rng.shuffle(indices)
        train_count = max(1, int(round(len(indices) * train_ratio)))
        train_mask[indices[:train_count]] = True
    return train_mask


if __name__ == "__main__":
    main()
