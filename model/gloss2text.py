from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import Gloss2TextConfig


def _padding_mask(tokens: Tensor, pad_id: int) -> Tensor:
    return tokens.eq(pad_id)


def _causal_float_mask(size: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.full((size, size), float("-inf"), device=device), diagonal=1
    )


class TokenPositionalEncoding(nn.Module):
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
        return self.dropout(x + self.encoding[:, : x.size(1)])


class Gloss2TextTransformer(nn.Module):
    """Transformer encoder-decoder: gloss tokens -> Korean BPE/text tokens."""

    def __init__(self, config: Gloss2TextConfig | None = None) -> None:
        super().__init__()
        self.config = config or Gloss2TextConfig()
        self.gloss_embedding = nn.Embedding(
            self.config.gloss_vocab_size, self.config.d_model, padding_idx=self.config.pad_id
        )
        self.text_embedding = nn.Embedding(
            self.config.text_vocab_size, self.config.d_model, padding_idx=self.config.pad_id
        )
        self.gloss_position = TokenPositionalEncoding(
            self.config.d_model, self.config.max_gloss_len, self.config.dropout
        )
        self.text_position = TokenPositionalEncoding(
            self.config.d_model, self.config.max_text_len, self.config.dropout
        )
        self.transformer = nn.Transformer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            num_encoder_layers=self.config.num_layers,
            num_decoder_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_projection = nn.Linear(self.config.d_model, self.config.text_vocab_size)

    def forward(self, gloss_tokens: Tensor, text_input_tokens: Tensor) -> Tensor:
        gloss_padding = _padding_mask(gloss_tokens, self.config.pad_id)
        text_padding = _padding_mask(text_input_tokens, self.config.pad_id)
        target_mask = _causal_float_mask(text_input_tokens.size(1), text_input_tokens.device)

        source = self.gloss_position(
            self.gloss_embedding(gloss_tokens) * math.sqrt(self.config.d_model)
        )
        target = self.text_position(
            self.text_embedding(text_input_tokens) * math.sqrt(self.config.d_model)
        )
        hidden = self.transformer(
            src=source,
            tgt=target,
            tgt_mask=target_mask,
            src_key_padding_mask=gloss_padding,
            tgt_key_padding_mask=text_padding,
            memory_key_padding_mask=gloss_padding,
        )
        return self.output_projection(hidden)

    @torch.no_grad()
    def greedy_generate(self, gloss_tokens: Tensor, max_new_tokens: int = 64) -> Tensor:
        batch_size = gloss_tokens.size(0)
        generated = torch.full(
            (batch_size, 1),
            self.config.bos_id,
            dtype=torch.long,
            device=gloss_tokens.device,
        )

        for _ in range(max_new_tokens):
            logits = self.forward(gloss_tokens, generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.eq(self.config.eos_id).all():
                break
        return generated
