from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor


def greedy_ctc_decode(
    logits_or_ids: Tensor,
    blank_id: int = 0,
    lengths: Tensor | None = None,
) -> list[list[int]]:
    """Collapse repeated ids and remove CTC blanks.

    Accepts logits [B, T, V], log-probs [B, T, V], or already-decoded ids [B, T].
    """

    if logits_or_ids.ndim == 3:
        ids = logits_or_ids.argmax(dim=-1)
    elif logits_or_ids.ndim == 2:
        ids = logits_or_ids
    else:
        raise ValueError("input must have shape [B, T, V] or [B, T].")

    outputs: list[list[int]] = []
    for batch_index, sequence in enumerate(ids.detach().cpu().tolist()):
        limit = int(lengths[batch_index].item()) if lengths is not None else len(sequence)
        collapsed: list[int] = []
        prev = None
        for token_id in sequence[:limit]:
            if token_id != prev and token_id != blank_id:
                collapsed.append(int(token_id))
            prev = token_id
        outputs.append(collapsed)
    return outputs


def _common_prefix(sequences: Iterable[list[int]]) -> list[int]:
    seqs = list(sequences)
    if not seqs:
        return []
    prefix: list[int] = []
    for values in zip(*seqs):
        if len(set(values)) != 1:
            break
        prefix.append(values[0])
    return prefix


@dataclass
class StableGlossBuffer:
    """Small realtime stabilizer for streaming CTC gloss predictions."""

    history_size: int = 4
    min_repeats: int = 2
    committed: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._history: deque[list[int]] = deque(maxlen=self.history_size)

    def update(self, decoded_gloss_ids: list[int]) -> dict[str, list[int]]:
        self._history.append(decoded_gloss_ids)
        recent = list(self._history)[-self.min_repeats :]
        stable_prefix = _common_prefix(recent)

        new_commit = stable_prefix[len(self.committed) :]
        if new_commit:
            self.committed.extend(new_commit)

        partial = decoded_gloss_ids[len(self.committed) :]
        return {
            "committed": list(self.committed),
            "new_commit": new_commit,
            "partial": partial,
        }

    def reset(self) -> None:
        self._history.clear()
        self.committed.clear()


def majority_vote_beam(beam_results: list[list[int]]) -> list[int]:
    """Fallback helper when several chunk decodes are available."""

    if not beam_results:
        return []
    counter = Counter(tuple(result) for result in beam_results)
    return list(counter.most_common(1)[0][0])
