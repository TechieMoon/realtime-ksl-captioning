from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import KeypointCTCConfig, SignKeypointLayout
from .preprocessing import KeypointPreprocessor, build_part_ids


def _lengths_to_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


def _causal_mask(size: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.encoding[:, : x.size(1)]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, keypoint_mask: Tensor | None = None) -> Tensor:
        scores = self.score(x).squeeze(-1)
        if keypoint_mask is not None:
            scores = scores.masked_fill(keypoint_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=-2)


class KeypointToGlossCTC(nn.Module):
    """Spatial-temporal keypoint Transformer with a CTC gloss head.

    Use `forward_raw` when the input is raw [x, y, z, confidence] keypoints.
    Use `forward` when preprocessing was already done and the input is [B, T, K, F].
    """

    def __init__(
        self,
        config: KeypointCTCConfig | None = None,
        layout: SignKeypointLayout | None = None,
        part_ids: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or KeypointCTCConfig()
        self.layout = layout or SignKeypointLayout()
        self.preprocessor = KeypointPreprocessor(self.layout)

        self.input_projection = nn.Linear(self.config.input_features, self.config.d_model)
        self.keypoint_embedding = nn.Embedding(
            self.config.num_keypoints, self.config.d_model
        )
        self.part_embedding = nn.Embedding(4, self.config.d_model)
        if part_ids is None:
            part_ids = build_part_ids(self.layout)
        self.register_buffer("part_ids", part_ids.long(), persistent=False)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=self.config.spatial_layers
        )
        self.frame_pool = AttentionPooling(self.config.d_model)
        self.temporal_position = SinusoidalPositionalEncoding(
            self.config.d_model, self.config.max_frames, self.config.dropout
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=self.config.temporal_layers
        )
        self.ctc_head = nn.Linear(
            self.config.d_model, self.config.gloss_vocab_size + 1
        )

    def forward_raw(
        self,
        raw_keypoints: Tensor,
        lengths: Tensor | None = None,
        keypoint_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        features = self.preprocessor(raw_keypoints)
        return self.forward(features, lengths=lengths, keypoint_mask=keypoint_mask)

    def forward(
        self,
        features: Tensor,
        lengths: Tensor | None = None,
        keypoint_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, T, K, F].")

        batch_size, num_frames, num_keypoints, _ = features.shape
        if num_keypoints != self.config.num_keypoints:
            raise ValueError(
                f"expected {self.config.num_keypoints} keypoints, got {num_keypoints}."
            )
        if lengths is None:
            lengths = torch.full(
                (batch_size,), num_frames, dtype=torch.long, device=features.device
            )

        x = self.input_projection(features)
        keypoint_ids = torch.arange(num_keypoints, device=features.device)
        x = x + self.keypoint_embedding(keypoint_ids).view(1, 1, num_keypoints, -1)
        x = x + self.part_embedding(self.part_ids.to(features.device)).view(
            1, 1, num_keypoints, -1
        )

        x = x.reshape(batch_size * num_frames, num_keypoints, self.config.d_model)
        spatial_keypoint_mask = None
        if keypoint_mask is not None:
            spatial_keypoint_mask = keypoint_mask.reshape(batch_size * num_frames, num_keypoints)
        x = self.spatial_encoder(x, src_key_padding_mask=spatial_keypoint_mask)
        x = x.reshape(batch_size, num_frames, num_keypoints, self.config.d_model)

        x = self.frame_pool(x, keypoint_mask=keypoint_mask)
        x = self.temporal_position(x)

        temporal_padding_mask = _lengths_to_padding_mask(lengths, num_frames)
        temporal_mask = None
        if self.config.use_causal_temporal_mask:
            temporal_mask = _causal_mask(num_frames, features.device)
        x = self.temporal_encoder(
            x, mask=temporal_mask, src_key_padding_mask=temporal_padding_mask
        )

        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return {"logits": logits, "log_probs": log_probs, "lengths": lengths}

    def ctc_loss(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
        zero_infinity: bool = True,
    ) -> Tensor:
        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.config.blank_id,
            zero_infinity=zero_infinity,
        )
