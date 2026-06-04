from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import SignKeypointLayout


# Face Mesh indices selected around lips, eyes, eyebrows, and nose.
# MediaPipe Holistic returns 468 face landmarks; sign recognition usually does
# not need the full mesh, so this list keeps the expression-heavy regions.
FACE_SELECTED_60 = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    33,
    133,
    159,
    145,
    263,
    362,
    386,
    374,
    70,
    63,
    105,
    66,
    336,
    296,
    334,
    293,
    1,
    2,
    98,
    327,
]


@dataclass
class HolisticExtractorConfig:
    target_fps: float = 15.0
    model_complexity: int = 1
    smooth_landmarks: bool = True
    refine_face_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    task_model_path: str | None = None
    max_frames: int | None = None


def _empty_keypoints(count: int) -> np.ndarray:
    return np.zeros((count, 4), dtype=np.float32)


def _landmark_to_xyzc(landmark, use_visibility: bool = False) -> np.ndarray:
    confidence = getattr(landmark, "visibility", 1.0) if use_visibility else 1.0
    return np.array([landmark.x, landmark.y, landmark.z, confidence], dtype=np.float32)


def _mean_keypoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = (a + b) * 0.5
    out[3] = min(a[3], b[3])
    return out


def _pose_upper_body(pose_landmarks) -> np.ndarray:
    lm = _landmark_sequence(pose_landmarks)
    if lm is None:
        return _empty_keypoints(13)

    left_shoulder = _landmark_to_xyzc(lm[11], use_visibility=True)
    right_shoulder = _landmark_to_xyzc(lm[12], use_visibility=True)
    left_hip = _landmark_to_xyzc(lm[23], use_visibility=True)
    right_hip = _landmark_to_xyzc(lm[24], use_visibility=True)

    return np.stack(
        [
            _mean_keypoint(left_shoulder, right_shoulder),  # synthetic neck
            left_shoulder,
            right_shoulder,
            _landmark_to_xyzc(lm[13], use_visibility=True),  # left elbow
            _landmark_to_xyzc(lm[14], use_visibility=True),  # right elbow
            _landmark_to_xyzc(lm[15], use_visibility=True),  # left wrist
            _landmark_to_xyzc(lm[16], use_visibility=True),  # right wrist
            _landmark_to_xyzc(lm[0], use_visibility=True),  # nose
            _landmark_to_xyzc(lm[2], use_visibility=True),  # left eye
            _landmark_to_xyzc(lm[5], use_visibility=True),  # right eye
            _landmark_to_xyzc(lm[7], use_visibility=True),  # left ear
            _landmark_to_xyzc(lm[8], use_visibility=True),  # right ear
            _mean_keypoint(left_hip, right_hip),  # synthetic mid hip
        ],
        axis=0,
    )


def _hand_keypoints(hand_landmarks) -> np.ndarray:
    lm = _landmark_sequence(hand_landmarks)
    if lm is None:
        return _empty_keypoints(21)
    return np.stack([_landmark_to_xyzc(landmark) for landmark in lm], axis=0)


def _face_keypoints(face_landmarks) -> np.ndarray:
    lm = _landmark_sequence(face_landmarks)
    if lm is None:
        return _empty_keypoints(len(FACE_SELECTED_60))
    return np.stack([_landmark_to_xyzc(lm[index]) for index in FACE_SELECTED_60], axis=0)


def _landmark_sequence(landmarks):
    if landmarks is None:
        return None
    if hasattr(landmarks, "landmark"):
        return landmarks.landmark
    if isinstance(landmarks, list) and landmarks:
        return landmarks
    return None


def _load_legacy_holistic_module():
    for module_name in (
        "mediapipe.solutions.holistic",
        "mediapipe.python.solutions.holistic",
    ):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    return None


def select_holistic_keypoints(
    results,
    layout: SignKeypointLayout | None = None,
) -> np.ndarray:
    """Convert MediaPipe Holistic results into the model layout [115, 4].

    Output order:
        0..12: upper body pose
        13..33: left hand
        34..54: right hand
        55..114: selected face points
    """

    layout = layout or SignKeypointLayout()
    keypoints = _empty_keypoints(layout.num_keypoints)
    keypoints[layout.pose_start : layout.pose_start + layout.pose_count] = (
        _pose_upper_body(results.pose_landmarks)
    )
    keypoints[
        layout.left_hand_start : layout.left_hand_start + layout.left_hand_count
    ] = _hand_keypoints(results.left_hand_landmarks)
    keypoints[
        layout.right_hand_start : layout.right_hand_start + layout.right_hand_count
    ] = _hand_keypoints(results.right_hand_landmarks)
    keypoints[layout.face_start : layout.face_start + layout.face_count] = (
        _face_keypoints(results.face_landmarks)
    )
    return keypoints


class HolisticKeypointExtractor:
    """Extract model-ready keypoints from mp4 files or individual frames."""

    def __init__(
        self,
        config: HolisticExtractorConfig | None = None,
        layout: SignKeypointLayout | None = None,
    ) -> None:
        self.config = config or HolisticExtractorConfig()
        self.layout = layout or SignKeypointLayout()
        self._legacy_holistic = _load_legacy_holistic_module()
        self._task_landmarker = None
        self._task_timestamp_offset_ms = 0

    def extract_frame(self, bgr_frame: np.ndarray, holistic) -> np.ndarray:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = holistic.process(rgb_frame)
        return select_holistic_keypoints(results, self.layout)

    def extract_video(self, video_path: str | Path) -> np.ndarray:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.config.target_fps
        frame_stride = max(1, round(source_fps / self.config.target_fps))
        frames: list[np.ndarray] = []
        frame_index = 0

        if self._legacy_holistic is not None:
            keypoints = self._extract_video_with_legacy(cap, frame_stride)
        elif self.config.task_model_path:
            keypoints = self._extract_video_with_tasks(cap, frame_stride, source_fps)
        else:
            cap.release()
            raise RuntimeError(
                "This mediapipe install does not expose mp.solutions.holistic. "
                "Install a legacy-compatible mediapipe version, or set "
                "HolisticExtractorConfig(task_model_path='holistic_landmarker.task') "
                "to use the current MediaPipe Tasks API."
            )

        cap.release()
        if not keypoints:
            return np.zeros((0, self.layout.num_keypoints, 4), dtype=np.float32)
        return np.stack(keypoints, axis=0).astype(np.float32)

    def _extract_video_with_legacy(
        self,
        cap: cv2.VideoCapture,
        frame_stride: int,
    ) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        frame_index = 0

        with self._legacy_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=self.config.smooth_landmarks,
            refine_face_landmarks=self.config.refine_face_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        ) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % frame_stride == 0:
                    frames.append(self.extract_frame(frame, holistic))
                    if self.config.max_frames and len(frames) >= self.config.max_frames:
                        break
                frame_index += 1

        return frames

    def _extract_video_with_tasks(
        self,
        cap: cv2.VideoCapture,
        frame_stride: int,
        source_fps: float,
    ) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        frame_index = 0
        landmarker = self._get_task_landmarker()
        last_timestamp_ms = self._task_timestamp_offset_ms
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = self._task_timestamp_offset_ms + int(
                    (frame_index / source_fps) * 1000
                )
                last_timestamp_ms = timestamp_ms
                results = landmarker.detect_for_video(mp_image, timestamp_ms)
                frames.append(select_holistic_keypoints(results, self.layout))
                if self.config.max_frames and len(frames) >= self.config.max_frames:
                    break
            frame_index += 1

        self._task_timestamp_offset_ms = last_timestamp_ms + 1
        return frames

    def _get_task_landmarker(self):
        if self._task_landmarker is not None:
            return self._task_landmarker

        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            HolisticLandmarker,
            HolisticLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.config.task_model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_landmarks_confidence=self.config.min_tracking_confidence,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_landmarks_confidence=self.config.min_tracking_confidence,
            min_hand_landmarks_confidence=self.config.min_tracking_confidence,
        )
        self._task_landmarker = HolisticLandmarker.create_from_options(options)
        return self._task_landmarker

    def close(self) -> None:
        if self._task_landmarker is not None:
            try:
                self._task_landmarker.close()
            finally:
                self._task_landmarker = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def extract_video_to_npy(
        self,
        video_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        keypoints = self.extract_video(video_path)
        temp_path = output_path.with_name(f"{output_path.name}.tmp")
        with temp_path.open("wb") as output_file:
            np.save(output_file, keypoints)
        temp_path.replace(output_path)
        return output_path
