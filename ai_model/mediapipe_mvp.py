from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import joblib
import numpy as np


POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FEATURE_DIM = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 * 2
MAX_MEDIAPIPE_WIDTH = 640
HOLISTIC_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)


@dataclass(frozen=True)
class MediaPipeMvpConfig:
    sequence_length: int
    confidence_threshold: float
    include_deltas: bool
    labels: list[str]


class MediaPipeMvpRecognizer:
    """Small keypoint-sequence classifier for the one-person MVP demo."""

    def __init__(self, artifact_path: Path, device: str = "cpu") -> None:
        artifact = joblib.load(artifact_path)
        self.classifier = artifact["classifier"]
        self.config = MediaPipeMvpConfig(
            sequence_length=int(artifact["sequence_length"]),
            confidence_threshold=float(artifact.get("confidence_threshold", 0.45)),
            include_deltas=bool(artifact.get("include_deltas", False)),
            labels=list(artifact["labels"]),
        )
        self.device = device
        self.buffer: deque[np.ndarray] = deque(maxlen=self.config.sequence_length)
        self.holistic = _create_holistic()

    def close(self) -> None:
        close = getattr(self.holistic, "close", None)
        if callable(close):
            close()

    def predict_one(self, frame: np.ndarray, timestamp_ms: int | None) -> dict[str, Any]:
        features = extract_mediapipe_features(frame, self.holistic)
        self.buffer.append(features)
        if len(self.buffer) < self.config.sequence_length:
            return {"text": "", "words": [], "is_final": False}

        sample = sequence_to_model_vector(np.asarray(self.buffer, dtype=np.float32), self.config.include_deltas)
        probabilities = _predict_probabilities(self.classifier, sample)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        if confidence < self.config.confidence_threshold:
            return {"text": "", "words": [], "is_final": False}

        text = self.config.labels[class_index]
        start_ms = int(timestamp_ms or 0)
        return {
            "text": text,
            "words": [
                {
                    "text": text,
                    "confidence": round(confidence, 4),
                    "start_ms": start_ms,
                    "end_ms": start_ms + 500,
                }
            ],
            "is_final": True,
        }


def has_mediapipe_artifact(model_dir: Path) -> bool:
    return (model_dir / "mediapipe_mvp.joblib").exists()


def load_mediapipe_mvp(model_dir: Path, device: str) -> MediaPipeMvpRecognizer:
    return MediaPipeMvpRecognizer(model_dir / "mediapipe_mvp.joblib", device=device)


def sequence_to_model_vector(sequence: np.ndarray, include_deltas: bool) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError("sequence must have shape (sequence_length, feature_dim)")
    if not include_deltas:
        return sequence.reshape(1, -1)

    deltas = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence, deltas], axis=1).reshape(1, -1)


def _predict_probabilities(classifier: Any, sample: np.ndarray) -> np.ndarray:
    predict_proba = getattr(classifier, "predict_proba", None)
    if callable(predict_proba):
        return predict_proba(sample)

    decision_function = getattr(classifier, "decision_function", None)
    if callable(decision_function):
        scores = np.asarray(decision_function(sample), dtype=np.float32)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    predictions = np.asarray(classifier.predict(sample), dtype=np.int64)
    class_count = max(int(predictions.max(initial=0)) + 1, 1)
    probabilities = np.zeros((len(predictions), class_count), dtype=np.float32)
    probabilities[np.arange(len(predictions)), predictions] = 1.0
    return probabilities


def extract_mediapipe_features(frame_rgb: np.ndarray, holistic: Any) -> np.ndarray:
    frame = _resize_for_mediapipe(_validate_rgb_frame(frame_rgb))
    results = holistic.process(frame)
    parts = [
        _flatten_landmarks(results.pose_landmarks, POSE_LANDMARKS, include_visibility=True),
        _flatten_landmarks(results.left_hand_landmarks, HAND_LANDMARKS, include_visibility=False),
        _flatten_landmarks(results.right_hand_landmarks, HAND_LANDMARKS, include_visibility=False),
    ]
    return np.concatenate(parts).astype(np.float32)


def _create_holistic() -> Any:
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        return mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    task_path = _ensure_holistic_task()
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return _TasksHolisticWrapper(mp.tasks.vision.HolisticLandmarker.create_from_options(options))


class _TasksHolisticWrapper:
    def __init__(self, landmarker: Any) -> None:
        self._landmarker = landmarker
        self._timestamp_ms = 0

    def process(self, frame_rgb: np.ndarray) -> Any:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        return _TasksHolisticResult(result)

    def close(self) -> None:
        self._landmarker.close()


class _TasksHolisticResult:
    def __init__(self, result: Any) -> None:
        self.pose_landmarks = _as_landmark_container(getattr(result, "pose_landmarks", None))
        self.left_hand_landmarks = _as_landmark_container(getattr(result, "left_hand_landmarks", None))
        self.right_hand_landmarks = _as_landmark_container(getattr(result, "right_hand_landmarks", None))


def _as_landmark_container(landmarks: Any) -> Any:
    if landmarks is None:
        return None
    if hasattr(landmarks, "landmark"):
        return landmarks
    if isinstance(landmarks, list):
        if not landmarks:
            return None
        if isinstance(landmarks[0], list):
            landmarks = landmarks[0]
        return _LandmarkContainer(landmarks)
    return None


class _LandmarkContainer:
    def __init__(self, landmarks: list[Any]) -> None:
        self.landmark = landmarks


def _ensure_holistic_task() -> Path:
    model_dir = Path(__file__).resolve().parent / "assets"
    model_dir.mkdir(parents=True, exist_ok=True)
    task_path = model_dir / "holistic_landmarker.task"
    if not task_path.exists():
        print(f"Downloading MediaPipe Holistic model to {task_path}")
        urlretrieve(HOLISTIC_TASK_URL, task_path)
    return task_path


def _flatten_landmarks(landmark_list: Any, count: int, *, include_visibility: bool) -> np.ndarray:
    width = 4 if include_visibility else 3
    if landmark_list is None:
        return np.zeros(count * width, dtype=np.float32)

    values: list[float] = []
    landmarks = list(landmark_list.landmark)[:count]
    for landmark in landmarks:
        values.extend([float(landmark.x), float(landmark.y), float(landmark.z)])
        if include_visibility:
            values.append(float(getattr(landmark, "visibility", 0.0)))

    missing = count - len(landmarks)
    if missing > 0:
        values.extend([0.0] * missing * width)
    return np.asarray(values, dtype=np.float32)


def _validate_rgb_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError("frame dtype must be uint8")
    return np.ascontiguousarray(frame)


def _resize_for_mediapipe(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= MAX_MEDIAPIPE_WIDTH:
        return frame

    import cv2

    scale = MAX_MEDIAPIPE_WIDTH / width
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (MAX_MEDIAPIPE_WIDTH, resized_height), interpolation=cv2.INTER_AREA)
