from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import SignKeypointLayout, WordClassifierConfig
from .transformer_blocks import (
    AttentionPooling,
    SinusoidalPositionalEncoding,
    lengths_to_padding_mask,
)
from .preprocessing import KeypointPreprocessor, build_part_ids


class WordKeypointClassifier(nn.Module):
    """Single-word sign classifier for isolated Korean sign-language clips."""

    def __init__(
        self,
        num_classes: int,
        config: WordClassifierConfig | None = None,
        layout: SignKeypointLayout | None = None,
    ) -> None:
        super().__init__()
        self.config = config or WordClassifierConfig()
        self.layout = layout or SignKeypointLayout()
        self.preprocessor = KeypointPreprocessor(self.layout)

        self.input_projection = nn.Linear(self.config.input_features, self.config.d_model)
        self.keypoint_embedding = nn.Embedding(
            self.config.num_keypoints, self.config.d_model
        )
        self.part_embedding = nn.Embedding(4, self.config.d_model)
        self.register_buffer("part_ids", build_part_ids(self.layout), persistent=False)

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
        self.clip_pool = AttentionPooling(self.config.d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, num_classes),
        )

    def forward_raw(self, raw_keypoints: Tensor, lengths: Tensor | None = None) -> Tensor:
        return self.forward(self.preprocessor(raw_keypoints), lengths=lengths)

    def forward(self, features: Tensor, lengths: Tensor | None = None) -> Tensor:
        batch_size, num_frames, num_keypoints, _ = features.shape
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
        x = self.spatial_encoder(x)
        x = x.reshape(batch_size, num_frames, num_keypoints, self.config.d_model)

        x = self.frame_pool(x)
        x = self.temporal_position(x)

        padding_mask = lengths_to_padding_mask(lengths, num_frames)
        x = self.temporal_encoder(x, src_key_padding_mask=padding_mask)
        x = self.clip_pool(x, keypoint_mask=padding_mask)
        return self.classifier(x)
