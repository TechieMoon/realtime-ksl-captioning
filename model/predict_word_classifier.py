from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import KeypointCTCConfig
from .keypoint_extractor import HolisticExtractorConfig, HolisticKeypointExtractor
from .word_classifier import WordKeypointClassifier
from .word_dataset import _fit_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a Korean sign-language word from an mp4 clip."
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to an mp4 file. Omit this when --keypoints is provided.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/word_classifier_epoch5_10_mixed/best.pt",
        help="Trained word-classifier checkpoint, usually best.pt or last.pt.",
    )
    parser.add_argument(
        "--keypoints",
        default=None,
        help="Optional pre-extracted .npy keypoint cache. Skips MediaPipe extraction.",
    )
    parser.add_argument(
        "--save-keypoints",
        default=None,
        help="Optional .npy path to save keypoints extracted from the mp4.",
    )
    parser.add_argument(
        "--task-model-path",
        default="model/assets/holistic_landmarker.task",
        help="MediaPipe HolisticLandmarker .task path.",
    )
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of a compact text summary.",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_keypoints(args: argparse.Namespace, max_frames: int) -> np.ndarray:
    if args.keypoints:
        return np.load(args.keypoints).astype(np.float32)

    if not args.video:
        raise ValueError("Provide an mp4 video path or --keypoints .npy path.")

    extractor = HolisticKeypointExtractor(
        HolisticExtractorConfig(
            target_fps=args.target_fps,
            task_model_path=args.task_model_path,
            max_frames=max_frames,
        )
    )
    try:
        keypoints = extractor.extract_video(args.video)
    finally:
        extractor.close()

    if args.save_keypoints:
        save_path = Path(args.save_keypoints)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, keypoints)

    return keypoints.astype(np.float32)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[WordKeypointClassifier, dict[str, int], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_to_id = checkpoint["label_to_id"]
    config = KeypointCTCConfig(**checkpoint["config"])
    model = WordKeypointClassifier(len(label_to_id), config=config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, label_to_id, checkpoint


@torch.no_grad()
def predict(
    model: WordKeypointClassifier,
    keypoints: np.ndarray,
    label_to_id: dict[str, int],
    max_frames: int,
    device: torch.device,
    top_k: int,
) -> dict[str, Any]:
    if keypoints.ndim != 3:
        raise ValueError("keypoints must have shape [frames, keypoints, features].")
    if keypoints.shape[0] == 0:
        raise ValueError("No keypoints were extracted from the video.")

    fitted, length = _fit_length(keypoints, max_frames)
    raw = torch.from_numpy(fitted).unsqueeze(0).float().to(device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)

    logits = model.forward_raw(raw, lengths)
    probabilities = torch.softmax(logits, dim=-1).squeeze(0)
    top_k = max(1, min(top_k, probabilities.numel()))
    values, indices = torch.topk(probabilities, k=top_k)

    id_to_label = {index: label for label, index in label_to_id.items()}
    predictions = [
        {
            "rank": rank + 1,
            "label": id_to_label[int(index)],
            "label_id": int(index),
            "probability": float(value),
        }
        for rank, (value, index) in enumerate(zip(values.cpu(), indices.cpu()))
    ]

    return {
        "prediction": predictions[0],
        "top_k": predictions,
        "input_frames": int(keypoints.shape[0]),
        "effective_frames": int(length),
        "max_frames": int(max_frames),
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, label_to_id, checkpoint = load_model(args.checkpoint, device)
    max_frames = int(checkpoint["config"].get("max_frames", model.config.max_frames))

    keypoints = load_keypoints(args, max_frames=max_frames)
    result = predict(
        model=model,
        keypoints=keypoints,
        label_to_id=label_to_id,
        max_frames=max_frames,
        device=device,
        top_k=args.top_k,
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "checkpoint_val_acc": checkpoint.get("val_acc"),
            "device": str(device),
        }
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    prediction = result["prediction"]
    print(
        f"prediction={prediction['label']} "
        f"label_id={prediction['label_id']} "
        f"probability={prediction['probability']:.4f}"
    )
    print(
        f"checkpoint_epoch={result['checkpoint_epoch']} "
        f"checkpoint_val_acc={result['checkpoint_val_acc']}"
    )
    print(
        f"frames={result['input_frames']} "
        f"effective_frames={result['effective_frames']} "
        f"device={result['device']}"
    )


if __name__ == "__main__":
    main()
