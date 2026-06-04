from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def lengths_to_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


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
