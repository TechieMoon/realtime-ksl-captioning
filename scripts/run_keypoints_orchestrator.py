from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MANIFEST_DIR = ROOT / "manifests"
KEYPOINT_DIR = ROOT / "keypoint_cache"
NUM_SHARDS = 8
MAX_FRAMES = 64


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


RUN_ID = timestamp()
MAIN_LOG = LOG_DIR / f"keypoints_orchestrator_{RUN_ID}.log"


def log(message: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with MAIN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def combine_manifests(prefix: str) -> None:
    samples: list[dict] = []
    for shard_index in range(NUM_SHARDS):
        path = MANIFEST_DIR / f"{prefix}_shard_{shard_index}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        samples.extend(json.loads(path.read_text(encoding="utf-8")))

    output = MANIFEST_DIR / f"{prefix}.json"
    output.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"combined {len(samples)} samples -> {output.relative_to(ROOT)}")


def run_split(split_root: str, prefix: str) -> None:
    log(f"START {split_root} keypoint extraction with {NUM_SHARDS} shards")
    processes: list[tuple[int, subprocess.Popen]] = []
    for shard_index in range(NUM_SHARDS):
        out_log = LOG_DIR / f"{prefix}_shard_{shard_index}_{RUN_ID}.out.log"
        err_log = LOG_DIR / f"{prefix}_shard_{shard_index}_{RUN_ID}.err.log"
        manifest = MANIFEST_DIR / f"{prefix}_shard_{shard_index}.json"
        args = [
            sys.executable,
            "-m",
            "model.prepare_keypoints",
            "--split-root",
            split_root,
            "--cache-root",
            str(KEYPOINT_DIR.relative_to(ROOT)),
            "--manifest-out",
            str(manifest.relative_to(ROOT)),
            "--task-model-path",
            "model/assets/holistic_landmarker.task",
            "--max-frames",
            str(MAX_FRAMES),
            "--num-shards",
            str(NUM_SHARDS),
            "--shard-index",
            str(shard_index),
            "--skip-existing",
        ]
        out_handle = out_log.open("w", encoding="utf-8")
        err_handle = err_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            args,
            cwd=ROOT,
            stdout=out_handle,
            stderr=err_handle,
        )
        out_handle.close()
        err_handle.close()
        processes.append((shard_index, process))
        skipped = MANIFEST_DIR / f"{prefix}_shard_{shard_index}.skipped.json"
        log(
            "started "
            f"{prefix} shard={shard_index} pid={process.pid} "
            f"manifest={manifest.relative_to(ROOT)} skipped={skipped.relative_to(ROOT)} "
            f"err={err_log.relative_to(ROOT)}"
        )

    failed = False
    for shard_index, process in processes:
        return_code = process.wait()
        log(f"finished {prefix} shard={shard_index} pid={process.pid} exit={return_code}")
        if return_code != 0:
            failed = True

    if failed:
        raise RuntimeError(f"{prefix} shard extraction failed; see logs for details.")

    combine_manifests(prefix)
    log(f"DONE {split_root} keypoint extraction")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log("Keypoint orchestrator started")
    run_split("1.Training", "train")
    run_split("2.Validation", "val")
    log("Keypoint orchestrator finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED {type(exc).__name__}: {exc}")
        raise
