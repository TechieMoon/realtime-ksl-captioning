from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
AI_MODEL_DIR = REPO_ROOT / "ai_model"
if str(AI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODEL_DIR))

from mediapipe_mvp import (  # noqa: E402
    FEATURE_DIM,
    MAX_MEDIAPIPE_WIDTH,
    _create_holistic,
    extract_mediapipe_features,
    sequence_to_model_vector,
)


ANGLE_RE = re.compile(r"_([FDLRU])(?:_morpheme)?$")
SIGNER_RE = re.compile(r"_(?:REAL|SYN|CROWD)?(\d+)_([FDLRU])(?:_morpheme)?$")
FEATURE_CACHE_VERSION = f"full_mediapipe_width_{MAX_MEDIAPIPE_WIDTH}_v1"
DEFAULT_CANDIDATES = "16:sgd,16:extra_trees,24:sgd,24:extra_trees,32:sgd"


@dataclass(frozen=True)
class LabelSegment:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class FullSample:
    split: str
    source: str
    zip_path: Path
    member: str
    key: str
    label: str
    start: float
    end: float
    signer_id: int
    angle: str


@dataclass(frozen=True)
class Candidate:
    sequence_length: int
    classifier: str
    include_deltas: bool


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    output_path = args.output.resolve()
    metrics_path = args.metrics_output.resolve()
    labels_output = args.labels_output.resolve()
    run_dir = args.run_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.time_budget_hours * 3600
    train_samples = discover_split(data_root / "1.Training", "train", args.sources)
    validation_samples = discover_split(data_root / "2.Validation", "validation", args.sources)

    labels = _build_vocabulary(
        train_samples=train_samples,
        validation_samples=validation_samples,
        min_train_per_label=args.min_train_per_label,
        min_validation_per_label=args.min_validation_per_label,
        max_labels=args.max_labels,
    )
    label_set = set(labels)
    train_samples = [sample for sample in train_samples if sample.label in label_set]
    validation_all = [sample for sample in validation_samples if sample.label in label_set]
    validation_unseen = sorted({sample.label for sample in validation_samples if sample.label not in label_set})

    if args.max_samples_per_label > 0:
        train_samples = _cap_samples_per_label(train_samples, args.max_samples_per_label)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "sources": args.sources,
        "labels": labels,
        "label_count": len(labels),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_all),
        "validation_unseen_labels": validation_unseen,
        "train_counts": Counter(sample.label for sample in train_samples),
        "validation_counts": Counter(sample.label for sample in validation_all),
    }
    _write_json(run_dir / "dataset_manifest.json", manifest)
    print(f"Discovered {len(train_samples)} train samples and {len(validation_all)} validation samples.")
    print(f"Vocabulary size: {len(labels)}")
    if validation_unseen:
        print(f"Validation labels without train samples: {len(validation_unseen)}")

    if args.dry_run:
        print(f"Dry run complete. Manifest: {run_dir / 'dataset_manifest.json'}")
        return
    if not train_samples or not validation_all:
        raise SystemExit("Need at least one train and one validation sample after filtering.")

    candidates = _parse_candidates(args.candidates)
    best: dict | None = None
    best_artifact: dict | None = None
    all_results: list[dict] = []

    for candidate in candidates:
        if _seconds_left(deadline) <= args.reserve_minutes * 60:
            print("Time budget is almost exhausted before the next candidate.")
            break

        print(
            "Candidate "
            f"sequence_length={candidate.sequence_length} "
            f"classifier={candidate.classifier} "
            f"include_deltas={candidate.include_deltas}"
        )
        train_result = _extract_matrix(
            samples=train_samples,
            sequence_length=candidate.sequence_length,
            include_deltas=candidate.include_deltas,
            cache_dir=cache_dir,
            deadline=deadline,
            reserve_seconds=args.reserve_minutes * 60,
            progress_every=args.progress_every,
        )
        validation_result = _extract_matrix(
            samples=validation_all,
            sequence_length=candidate.sequence_length,
            include_deltas=candidate.include_deltas,
            cache_dir=cache_dir,
            deadline=deadline,
            reserve_seconds=args.reserve_minutes * 60,
            progress_every=args.progress_every,
        )
        if train_result is None or validation_result is None:
            print("Stopped before completing feature extraction for this candidate.")
            break

        x_train, y_train_text = train_result
        x_validation, y_validation_text = validation_result
        label_to_index = {label: index for index, label in enumerate(labels)}
        y_train = np.asarray([label_to_index[label] for label in y_train_text], dtype=np.int64)
        y_validation = np.asarray([label_to_index[label] for label in y_validation_text], dtype=np.int64)

        classifier = _build_classifier(candidate.classifier, args.seed)
        started = time.perf_counter()
        classifier.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started

        predictions = classifier.predict(x_validation)
        accuracy = float(accuracy_score(y_validation, predictions))
        macro_f1 = float(f1_score(y_validation, predictions, average="macro", zero_division=0))
        report = classification_report(
            y_validation,
            predictions,
            labels=np.arange(len(labels)),
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )
        result = {
            "candidate": candidate.__dict__,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "train_samples": int(len(y_train)),
            "validation_samples": int(len(y_validation)),
            "classifier_train_seconds": train_seconds,
            "seconds_left": _seconds_left(deadline),
        }
        all_results.append(result)
        _write_json(run_dir / "candidate_results.json", all_results)
        print(f"accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} train_seconds={train_seconds:.1f}")

        if best is None or accuracy > best["accuracy"] or (
            accuracy == best["accuracy"] and macro_f1 > best["macro_f1"]
        ):
            best = result
            best_artifact = {
                "classifier": classifier,
                "labels": labels,
                "sequence_length": candidate.sequence_length,
                "feature_dim": FEATURE_DIM,
                "include_deltas": candidate.include_deltas,
                "confidence_threshold": args.confidence_threshold,
                "training_summary": {
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "classifier": candidate.classifier,
                    "samples_train": int(len(y_train)),
                    "samples_validation": int(len(y_validation)),
                    "sources": args.sources,
                    "max_mediapipe_width": MAX_MEDIAPIPE_WIDTH,
                    "feature_cache_version": FEATURE_CACHE_VERSION,
                },
            }
            joblib.dump(best_artifact, output_path)
            _write_labels(labels_output, labels)
            metrics = {
                **manifest,
                "best": {
                    **result,
                    "report": report,
                },
                "all_results": all_results,
                "output": str(output_path),
                "labels_output": str(labels_output),
                "metrics_output": str(metrics_path),
            }
            _write_json(metrics_path, metrics)
            print(f"Saved new best artifact: {output_path}")

    if best is None or best_artifact is None:
        raise SystemExit("No complete candidate finished within the time budget. Re-run to continue from cache.")

    print(f"Best validation accuracy: {best['accuracy']:.4f}")
    print(f"Best macro F1: {best['macro_f1']:.4f}")
    print(f"Saved metrics: {metrics_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full-dataset MediaPipe KSL word classifier.")
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "training" / "runs" / "full_mediapipe")
    parser.add_argument("--output", type=Path, default=AI_MODEL_DIR / "mediapipe_mvp.joblib")
    parser.add_argument("--metrics-output", type=Path, default=AI_MODEL_DIR / "metrics_full_mediapipe.json")
    parser.add_argument("--labels-output", type=Path, default=AI_MODEL_DIR / "labels.json")
    parser.add_argument("--time-budget-hours", type=float, default=12.0)
    parser.add_argument("--reserve-minutes", type=float, default=20.0)
    parser.add_argument("--sources", nargs="+", default=["real_word", "real_sen", "syn_word", "syn_sen", "crowd", "unknown"])
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--min-train-per-label", type=int, default=1)
    parser.add_argument("--min-validation-per-label", type=int, default=1)
    parser.add_argument("--max-labels", type=int, default=0, help="0 means use every label that passes filters.")
    parser.add_argument("--max-samples-per-label", type=int, default=0, help="0 means use every training sample.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _default_data_root() -> Path:
    env_root = os.environ.get("KSL_DATA_ROOT")
    if env_root:
        return Path(env_root)
    for candidate in (REPO_ROOT / "수어 영상", Path("D:/수어 영상")):
        if candidate.exists():
            return candidate
    return REPO_ROOT / "수어 영상"


def _default_cache_dir() -> Path:
    env_cache = os.environ.get("KSL_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    if Path("D:/수어 영상").exists():
        return Path("D:/ksl_cache/full_mediapipe_features")
    return REPO_ROOT / ".cache" / "full_mediapipe_features"


def discover_split(split_root: Path, split: str, allowed_sources: list[str]) -> list[FullSample]:
    if not split_root.exists():
        print(f"Missing split folder: {split_root}")
        return []

    allowed = set(allowed_sources)
    labels_by_key = _load_label_segments(split_root)
    samples: list[FullSample] = []
    video_zips = sorted(split_root.rglob("*video*.zip"))
    for zip_path in video_zips:
        try:
            with ZipFile(zip_path) as zip_file:
                for info in zip_file.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".mp4"):
                        continue
                    key = _sample_key(info.filename)
                    segments = labels_by_key.get(key)
                    if not segments:
                        continue
                    source = _source_kind(zip_path, info.filename)
                    if source not in allowed:
                        continue
                    signer_id, angle = _video_metadata(info.filename)
                    for segment in segments:
                        samples.append(
                            FullSample(
                                split=split,
                                source=source,
                                zip_path=zip_path,
                                member=info.filename,
                                key=key,
                                label=segment.text,
                                start=segment.start,
                                end=segment.end,
                                signer_id=signer_id,
                                angle=angle,
                            )
                        )
        except BadZipFile:
            print(f"Skipping bad video zip: {zip_path}")
    print(f"{split}: paired {len(samples)} samples from {len(video_zips)} video zips.")
    return samples


def _load_label_segments(split_root: Path) -> dict[str, list[LabelSegment]]:
    labels_by_key: dict[str, list[LabelSegment]] = {}
    label_zips = sorted(split_root.rglob("*morpheme*.zip"))
    for zip_path in label_zips:
        try:
            with ZipFile(zip_path) as zip_file:
                for info in zip_file.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".json"):
                        continue
                    key = _sample_key(info.filename)
                    raw = json.loads(zip_file.read(info).decode("utf-8"))
                    for segment in _parse_label_segments(raw):
                        labels_by_key.setdefault(key, [])
                        if segment not in labels_by_key[key]:
                            labels_by_key[key].append(segment)
        except BadZipFile:
            print(f"Skipping bad label zip: {zip_path}")
    print(f"Loaded labels for {len(labels_by_key)} videos from {len(label_zips)} label zips.")
    return labels_by_key


def _parse_label_segments(raw: dict) -> list[LabelSegment]:
    rows = raw.get("data", [])
    duration = float(raw.get("metaData", {}).get("duration", 0.0) or 0.0)
    segments: list[LabelSegment] = []
    for row in rows:
        text = _extract_label_text(row)
        if not text:
            continue
        start = float(row.get("start", 0.0) or 0.0)
        end = float(row.get("end", duration or start + 1.0) or start + 1.0)
        segments.append(LabelSegment(text=text, start=start, end=max(start + 0.1, end)))
    return segments


def _extract_label_text(row: dict) -> str:
    for attribute in row.get("attributes", []):
        name = str(attribute.get("name", "")).strip()
        if name:
            return name
    for key in ("text", "label", "name"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _sample_key(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_morpheme"):
        stem = stem[: -len("_morpheme")]
    return stem


def _source_kind(zip_path: Path, member: str) -> str:
    text = f"{zip_path.as_posix()}/{member}".lower()
    if "crowd" in text:
        return "crowd"
    if "syn_word" in text or "/syn/" in text and "word" in text:
        return "syn_word"
    if "syn_sen" in text or "/syn/" in text and "sen" in text:
        return "syn_sen"
    if "real_word" in text or "/real/word" in text:
        return "real_word"
    if "real_sen" in text or "/real/sen" in text:
        return "real_sen"
    return "unknown"


def _video_metadata(filename: str) -> tuple[int, str]:
    stem = Path(filename).stem
    signer_match = SIGNER_RE.search(stem)
    angle_match = ANGLE_RE.search(stem)
    signer_id = int(signer_match.group(1)) if signer_match else -1
    angle = signer_match.group(2) if signer_match else (angle_match.group(1) if angle_match else "")
    return signer_id, angle


def _build_vocabulary(
    *,
    train_samples: list[FullSample],
    validation_samples: list[FullSample],
    min_train_per_label: int,
    min_validation_per_label: int,
    max_labels: int,
) -> list[str]:
    train_counts = Counter(sample.label for sample in train_samples)
    validation_counts = Counter(sample.label for sample in validation_samples)
    labels = [
        label
        for label, count in train_counts.items()
        if count >= min_train_per_label and validation_counts.get(label, 0) >= min_validation_per_label
    ]
    labels.sort(key=lambda label: (-validation_counts[label], -train_counts[label], label))
    if max_labels > 0:
        labels = labels[:max_labels]
    return labels


def _cap_samples_per_label(samples: list[FullSample], max_samples_per_label: int) -> list[FullSample]:
    selected: list[FullSample] = []
    counts: Counter[str] = Counter()
    for sample in sorted(samples, key=lambda item: (item.label, item.source, item.signer_id, item.angle, item.key)):
        if counts[sample.label] >= max_samples_per_label:
            continue
        selected.append(sample)
        counts[sample.label] += 1
    return selected


def _parse_candidates(raw: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for chunk in raw.split(","):
        parts = [part.strip() for part in chunk.split(":") if part.strip()]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid candidate: {chunk}")
        sequence_length = int(parts[0])
        classifier = parts[1]
        include_deltas = True if len(parts) == 2 else parts[2].lower() not in {"0", "false", "no"}
        candidates.append(Candidate(sequence_length=sequence_length, classifier=classifier, include_deltas=include_deltas))
    return candidates


def _extract_matrix(
    *,
    samples: list[FullSample],
    sequence_length: int,
    include_deltas: bool,
    cache_dir: Path,
    deadline: float,
    reserve_seconds: float,
    progress_every: int,
) -> tuple[np.ndarray, list[str]] | None:
    sequences: list[np.ndarray] = []
    labels: list[str] = []
    holistic = _create_holistic()
    try:
        for index, sample in enumerate(samples, start=1):
            if _seconds_left(deadline) <= reserve_seconds:
                return None
            if index == 1 or index == len(samples) or index % progress_every == 0:
                print(f"[{sample.split} {index}/{len(samples)}] {sample.label} {sample.source} {sample.member}")
            sequence = _load_or_extract_video_sequence(sample, holistic, sequence_length, cache_dir)
            if sequence is None:
                continue
            sequences.append(sequence_to_model_vector(sequence, include_deltas)[0])
            labels.append(sample.label)
    finally:
        holistic.close()
    if not sequences:
        return None
    return np.asarray(sequences, dtype=np.float32), labels


def _load_or_extract_video_sequence(
    sample: FullSample,
    holistic: object,
    sequence_length: int,
    cache_dir: Path,
) -> np.ndarray | None:
    key = hashlib.sha1(
        f"{FEATURE_CACHE_VERSION}|{sequence_length}|{sample.zip_path}|{sample.member}|{sample.start:.4f}|{sample.end:.4f}".encode(
            "utf-8"
        )
    ).hexdigest()
    cache_path = cache_dir / f"{key}.npz"
    if cache_path.exists():
        return np.load(cache_path)["sequence"].astype(np.float32)

    sequence = _extract_video_sequence(sample, holistic, sequence_length, cache_dir)
    if sequence is not None:
        np.savez_compressed(cache_path, sequence=sequence)
    return sequence


def _extract_video_sequence(sample: FullSample, holistic: object, sequence_length: int, cache_dir: Path) -> np.ndarray | None:
    with ZipFile(sample.zip_path) as zip_file:
        video_bytes = zip_file.read(sample.member)

    scratch_dir = cache_dir / "_tmp"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=scratch_dir, delete=True) as temp_file:
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


def _build_classifier(name: str, seed: int) -> object:
    if name == "sgd":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=0.0001,
                max_iter=1000,
                early_stopping=True,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, n_jobs=-1),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(512, 256),
                alpha=0.001,
                batch_size=64,
                max_iter=800,
                early_stopping=True,
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=800,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classifier: {name}")


def _write_labels(path: Path, labels: list[str]) -> None:
    payload = {"labels": [{"id": index, "name": label, "text": label} for index, label in enumerate(labels)]}
    _write_json(path, payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seconds_left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


if __name__ == "__main__":
    main()
