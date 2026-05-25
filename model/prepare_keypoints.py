from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .keypoint_extractor import HolisticExtractorConfig, HolisticKeypointExtractor
from .word_dataset import WordSample, iter_samples, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Holistic keypoints to .npy cache.")
    parser.add_argument("--split-root", required=True, help="ex: 1.Training or 2.Validation")
    parser.add_argument("--cache-root", default="keypoint_cache")
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--task-model-path", default="model/assets/holistic_landmarker.task")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first video extraction error instead of skipping bad files.",
    )
    return parser.parse_args()


def save_skipped(skipped: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    extractor = HolisticKeypointExtractor(
        HolisticExtractorConfig(
            task_model_path=args.task_model_path,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
        )
    )

    samples: list[WordSample] = []
    progress_path = Path(args.manifest_out).with_suffix(".progress.json")
    skipped: list[dict[str, Any]] = []
    skipped_path = Path(args.manifest_out).with_suffix(".skipped.json")
    try:
        for sample in tqdm(
            iter_samples(
                args.split_root,
                args.cache_root,
                limit=args.limit,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            ),
            desc=f"extract {Path(args.split_root).name} shard {args.shard_index}/{args.num_shards}",
        ):
            cache_path = Path(sample.cache_path)
            try:
                if not (args.skip_existing and cache_path.exists()):
                    extractor.extract_video_to_npy(sample.video_path, cache_path)
            except Exception as exc:
                if args.fail_fast:
                    raise
                skipped.append(
                    {
                        "video_path": sample.video_path,
                        "label_path": sample.label_path,
                        "cache_path": sample.cache_path,
                        "label": sample.label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                tqdm.write(f"skip failed video: {sample.video_path} ({type(exc).__name__}: {exc})")
                save_skipped(skipped, skipped_path)
                continue

            samples.append(sample)
            if len(samples) % 100 == 0:
                save_manifest(samples, progress_path)
    finally:
        extractor.close()

    save_manifest(samples, args.manifest_out)
    if skipped:
        save_skipped(skipped, skipped_path)
    elif skipped_path.exists():
        skipped_path.unlink()
    if progress_path.exists():
        progress_path.unlink()


if __name__ == "__main__":
    main()
