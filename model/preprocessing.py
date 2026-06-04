from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import SignKeypointLayout


def build_part_ids(layout: SignKeypointLayout) -> Tensor:
    """Return body-part ids for pose, left hand, right hand, and face keypoints."""

    part_ids = torch.zeros(layout.num_keypoints, dtype=torch.long)
    part_ids[layout.pose_start : layout.pose_start + layout.pose_count] = 0
    part_ids[
        layout.left_hand_start : layout.left_hand_start + layout.left_hand_count
    ] = 1
    part_ids[
        layout.right_hand_start : layout.right_hand_start + layout.right_hand_count
    ] = 2
    part_ids[layout.face_start : layout.face_start + layout.face_count] = 3
    return part_ids


class KeypointPreprocessor(nn.Module):
    """Normalize raw keypoints and add motion/relative-position features.

    Input:
        raw_keypoints: [B, T, K, 4] containing x, y, z, confidence.

    Output:
        features: [B, T, K, 16]
    """

    def __init__(
        self,
        layout: SignKeypointLayout | None = None,
        eps: float = 1e-6,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.layout = layout or SignKeypointLayout()
        self.eps = eps
        self.min_scale = min_scale

    def forward(self, raw_keypoints: Tensor) -> Tensor:
        if raw_keypoints.ndim != 4 or raw_keypoints.size(-1) < 4:
            raise ValueError("raw_keypoints must have shape [B, T, K, 4+].")

        xyz = raw_keypoints[..., :3]
        confidence = raw_keypoints[..., 3:4].clamp(0.0, 1.0)

        left_shoulder = xyz[:, :, self.layout.left_shoulder_index]
        right_shoulder = xyz[:, :, self.layout.right_shoulder_index]
        neck = xyz[:, :, self.layout.neck_index]
        shoulder_center = 0.5 * (left_shoulder + right_shoulder)
        body_center = torch.where(
            torch.isfinite(neck).all(dim=-1, keepdim=True), neck, shoulder_center
        )

        shoulder_scale = (left_shoulder[..., :2] - right_shoulder[..., :2]).norm(
            dim=-1, keepdim=True
        )
        scale = shoulder_scale.clamp_min(self.min_scale).unsqueeze(-1)

        xyz_norm = (xyz - body_center.unsqueeze(2)) / (scale + self.eps)
        xyz_norm = torch.nan_to_num(xyz_norm)

        velocity = torch.zeros_like(xyz_norm)
        velocity[:, 1:] = xyz_norm[:, 1:] - xyz_norm[:, :-1]

        acceleration = torch.zeros_like(xyz_norm)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]

        left_wrist = xyz_norm[:, :, self.layout.left_wrist_index].unsqueeze(2)
        right_wrist = xyz_norm[:, :, self.layout.right_wrist_index].unsqueeze(2)
        rel_left_wrist = xyz_norm - left_wrist
        rel_right_wrist = xyz_norm - right_wrist

        return torch.cat(
            [
                xyz_norm,
                confidence,
                velocity,
                acceleration,
                rel_left_wrist,
                rel_right_wrist,
            ],
            dim=-1,
        )
