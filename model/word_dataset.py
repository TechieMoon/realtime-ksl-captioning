from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WordSample:
    video_path: str
    label_path: str
    cache_path: str
    label: str


def read_label(label_path: str | Path) -> tuple[str, str] | None:
    data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    video_name = data["metaData"]["name"]
    if not data.get("data"):
        return None
    if not data["data"][0].get("attributes"):
        return None
    label = data["data"][0]["attributes"][0]["name"]
    if not label:
        return None
    return video_name, label


def build_video_index(video_root: str | Path) -> dict[str, Path]:
    video_root = Path(video_root)
    video_index: dict[str, Path] = {}
    for video_path in video_root.rglob("*.mp4"):
        video_index.setdefault(video_path.name, video_path)
    return video_index


def _candidate_video_paths(split_root: Path, video_name: str) -> list[Path]:
    match = re.search(r"WORD(\d+)_REAL(\d+)", video_name)
    if not match:
        return []

    word_id = int(match.group(1))
    real_id = int(match.group(2))
    real_dir = f"{real_id:02d}" if word_id >= 1501 else f"{real_id:02d}-1"

    candidates: list[Path] = []
    if split_root.name.startswith("1."):
        source_id = (real_id * 2 - 1) if word_id >= 1501 else (real_id * 2)
        candidates.append(
            split_root / f"[원천]{source_id:02d}_real_word_video" / real_dir / video_name
        )
    else:
        candidates.append(
            split_root / "[원천]01_real_word_video" / "WORD" / real_dir / video_name
        )
    return candidates


def build_manifest(
    split_root: str | Path,
    cache_root: str | Path,
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[WordSample]:
    return list(
        iter_samples(
            split_root,
            cache_root,
            limit=limit,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    )


def iter_samples(
    split_root: str | Path,
    cache_root: str | Path,
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
):
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")

    split_root = Path(split_root)
    cache_root = Path(cache_root)
    label_root = split_root / "[라벨]01_real_word_morpheme" / "morpheme"
    fallback_video_index: dict[str, Path] | None = None

    matched_count = 0
    yielded_count = 0
    for label_path in sorted(label_root.rglob("*.json")):
        label_info = read_label(label_path)
        if label_info is None:
            continue
        video_name, label = label_info
        video_path = None
        for candidate in _candidate_video_paths(split_root, video_name):
            if candidate.exists():
                video_path = candidate
                break
        if video_path is None:
            if fallback_video_index is None:
                fallback_video_index = {}
                for video_root in [
                    path
                    for path in split_root.iterdir()
                    if path.is_dir()
                    and path.name.startswith("[원천]")
                    and path.name.endswith("video")
                ]:
                    fallback_video_index.update(build_video_index(video_root))
            video_path = fallback_video_index.get(video_name)
        if video_path is None:
            continue
        current_index = matched_count
        matched_count += 1
        if current_index % num_shards != shard_index:
            continue
        cache_path = cache_root / split_root.name / f"{Path(video_name).stem}.npy"
        yield WordSample(
            video_path=str(video_path),
            label_path=str(label_path),
            cache_path=str(cache_path),
            label=label,
        )
        yielded_count += 1
        if limit is not None and yielded_count >= limit:
            break


def save_manifest(samples: list[WordSample], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([sample.__dict__ for sample in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> list[WordSample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [WordSample(**item) for item in data]


def build_vocab(samples: list[WordSample]) -> dict[str, int]:
    labels = sorted({sample.label for sample in samples})
    return {label: index for index, label in enumerate(labels)}


def save_vocab(label_to_id: dict[str, int], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(label_to_id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fit_length(keypoints: np.ndarray, max_frames: int) -> tuple[np.ndarray, int]:
    length = min(len(keypoints), max_frames)
    if len(keypoints) > max_frames:
        indices = np.linspace(0, len(keypoints) - 1, max_frames).round().astype(np.int64)
        keypoints = keypoints[indices]
    output = np.zeros((max_frames, keypoints.shape[1], keypoints.shape[2]), dtype=np.float32)
    output[:length] = keypoints[:length]
    return output, length


class WordKeypointDataset(Dataset):
    def __init__(
        self,
        samples: list[WordSample],
        label_to_id: dict[str, int],
        max_frames: int = 64,
    ) -> None:
        self.samples = samples
        self.label_to_id = label_to_id
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        cache_path = Path(sample.cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing keypoint cache: {cache_path}. Run prepare_keypoints.py first."
            )

        keypoints = np.load(cache_path).astype(np.float32)
        keypoints, length = _fit_length(keypoints, self.max_frames)
        return {
            "keypoints": torch.from_numpy(keypoints),
            "length": torch.tensor(length, dtype=torch.long),
            "label_id": torch.tensor(self.label_to_id[sample.label], dtype=torch.long),
        }


def collate_word_batch(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "keypoints": torch.stack([item["keypoints"] for item in batch]),
        "lengths": torch.stack([item["length"] for item in batch]),
        "labels": torch.stack([item["label_id"] for item in batch]),
    }
