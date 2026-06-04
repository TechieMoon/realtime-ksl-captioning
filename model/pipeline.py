from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .ctc_decoder import StableGlossBuffer, greedy_ctc_decode
from .gloss2text import Gloss2TextTransformer
from .keypoint_ctc import KeypointToGlossCTC
from .keypoint_extractor import HolisticExtractorConfig, HolisticKeypointExtractor


class SignSubtitlePipeline:
    """Wrapper that connects mp4 -> Holistic -> CTC gloss -> Korean text tokens.

    The neural models still need trained checkpoints. This class wires the data
    flow so the same path can be used after training and during validation.
    """

    def __init__(
        self,
        keypoint_model: KeypointToGlossCTC,
        gloss2text_model: Gloss2TextTransformer | None = None,
        text_decoder: Callable[[list[int]], str] | None = None,
        extractor: HolisticKeypointExtractor | None = None,
        blank_id: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor = extractor or HolisticKeypointExtractor(HolisticExtractorConfig())
        self.keypoint_model = keypoint_model.to(self.device).eval()
        self.gloss2text_model = (
            gloss2text_model.to(self.device).eval() if gloss2text_model is not None else None
        )
        self.text_decoder = text_decoder
        self.blank_id = blank_id
        self.stable_buffer = StableGlossBuffer()

    def extract_keypoints(self, video_path: str | Path) -> np.ndarray:
        return self.extractor.extract_video(video_path)

    @torch.no_grad()
    def predict_gloss_ids(self, keypoints: np.ndarray) -> list[int]:
        if keypoints.ndim != 3:
            raise ValueError("keypoints must have shape [T, K, 4].")
        if keypoints.shape[0] == 0:
            return []

        raw = torch.from_numpy(keypoints).unsqueeze(0).float().to(self.device)
        lengths = torch.tensor([raw.size(1)], dtype=torch.long, device=self.device)
        out = self.keypoint_model.forward_raw(raw, lengths=lengths)
        return greedy_ctc_decode(out["logits"], blank_id=self.blank_id, lengths=lengths)[0]

    @torch.no_grad()
    def gloss_to_text_token_ids(self, gloss_ids: list[int], max_new_tokens: int = 64) -> list[int]:
        if self.gloss2text_model is None or not gloss_ids:
            return []

        config = self.gloss2text_model.config
        gloss_tokens = torch.tensor(
            [[config.bos_id, *gloss_ids, config.eos_id]],
            dtype=torch.long,
            device=self.device,
        )
        generated = self.gloss2text_model.greedy_generate(
            gloss_tokens, max_new_tokens=max_new_tokens
        )
        return generated.squeeze(0).detach().cpu().tolist()

    def predict_video(self, video_path: str | Path) -> dict[str, object]:
        keypoints = self.extract_keypoints(video_path)
        gloss_ids = self.predict_gloss_ids(keypoints)
        stable = self.stable_buffer.update(gloss_ids)
        text_token_ids = self.gloss_to_text_token_ids(stable["committed"])
        text = self.text_decoder(text_token_ids) if self.text_decoder else None

        return {
            "keypoints": keypoints,
            "gloss_ids": gloss_ids,
            "stable_gloss_ids": stable["committed"],
            "partial_gloss_ids": stable["partial"],
            "text_token_ids": text_token_ids,
            "text": text,
        }
