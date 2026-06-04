from dataclasses import dataclass


@dataclass(frozen=True)
class SignKeypointLayout:
    """Default layout: upper-body pose + left hand + right hand + selected face."""

    pose_start: int = 0
    pose_count: int = 13
    left_hand_start: int = 13
    left_hand_count: int = 21
    right_hand_start: int = 34
    right_hand_count: int = 21
    face_start: int = 55
    face_count: int = 60
    neck_index: int = 0
    left_shoulder_index: int = 1
    right_shoulder_index: int = 2

    @property
    def num_keypoints(self) -> int:
        return self.face_start + self.face_count

    @property
    def left_wrist_index(self) -> int:
        return self.left_hand_start

    @property
    def right_wrist_index(self) -> int:
        return self.right_hand_start


@dataclass
class KeypointCTCConfig:
    num_keypoints: int = 115
    input_features: int = 16
    gloss_vocab_size: int = 1000
    blank_id: int = 0
    d_model: int = 256
    spatial_layers: int = 2
    temporal_layers: int = 4
    num_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_frames: int = 512
    use_causal_temporal_mask: bool = False


@dataclass
class Gloss2TextConfig:
    gloss_vocab_size: int = 1000
    text_vocab_size: int = 8000
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_gloss_len: int = 128
    max_text_len: int = 128
