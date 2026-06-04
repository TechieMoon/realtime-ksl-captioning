#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.models.huggingface import _import_repo_module, _select_torch_device  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate isolated KSL word videos in a folder.")
    parser.add_argument("input_dir", type=Path, help="Folder containing one video per word.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "test100")
    parser.add_argument("--repo-id", default="Seoyoung07/korean-sign-word-classifier-mediapipe")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    videos = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise ValueError(f"No videos found in {input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"test100_results_{timestamp}.csv"
    json_path = args.output_dir / f"test100_results_{timestamp}.json"
    md_path = args.output_dir / f"test100_report_{timestamp}.md"

    snapshot = Path(snapshot_download(repo_id=args.repo_id, revision=args.revision))
    module = _import_repo_module(snapshot, "model.predict_word_classifier")
    device = _select_torch_device(args.device)
    model, label_to_id, checkpoint = module.load_model(snapshot / "best.pt", device)
    max_frames = int(checkpoint["config"].get("max_frames", getattr(model.config, "max_frames", 64)))
    extractor = module.HolisticKeypointExtractor(
        module.HolisticExtractorConfig(
            target_fps=args.target_fps,
            task_model_path=str(snapshot / "model" / "assets" / "holistic_landmarker.task"),
            max_frames=max_frames,
        )
    )

    results: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    try:
        for index, video_path in enumerate(videos, start=1):
            expected = video_path.stem
            print(f"[{index}/{len(videos)}] {expected}", flush=True)
            item_start = time.perf_counter()
            row: dict[str, Any] = {
                "index": index,
                "file": str(video_path),
                "expected": expected,
                "expected_in_vocab": expected in label_to_id,
                "predicted": "",
                "confidence": None,
                "top_k": [],
                "top1_correct": False,
                "topk_contains_expected": False,
                "input_frames": None,
                "effective_frames": None,
                "elapsed_sec": None,
                "error": "",
            }
            try:
                keypoints = extractor.extract_video(video_path)
                prediction = module.predict(
                    model=model,
                    keypoints=keypoints,
                    label_to_id=label_to_id,
                    max_frames=max_frames,
                    device=device,
                    top_k=args.top_k,
                )
                top_k = prediction["top_k"]
                labels = [candidate["label"] for candidate in top_k]
                row.update(
                    {
                        "predicted": prediction["prediction"]["label"],
                        "confidence": prediction["prediction"]["probability"],
                        "top_k": top_k,
                        "top1_correct": prediction["prediction"]["label"] == expected,
                        "topk_contains_expected": expected in labels,
                        "input_frames": prediction["input_frames"],
                        "effective_frames": prediction["effective_frames"],
                    }
                )
            except Exception as exc:
                row["error"] = str(exc)
            finally:
                row["elapsed_sec"] = round(time.perf_counter() - item_start, 3)
                results.append(row)
    finally:
        extractor.close()

    total_elapsed = round(time.perf_counter() - started_at, 3)
    summary = build_summary(results, args, snapshot, total_elapsed)
    write_json(json_path, summary, results)
    write_csv(csv_path, results)
    write_markdown(md_path, summary, results)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(
        f"top1={summary['top1_correct']}/{summary['total']} "
        f"top5={summary['topk_contains_expected']}/{summary['total']} "
        f"errors={summary['errors']}"
    )


def build_summary(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    snapshot: Path,
    total_elapsed: float,
) -> dict[str, Any]:
    total = len(results)
    top1_correct = sum(1 for row in results if row["top1_correct"])
    topk_contains_expected = sum(1 for row in results if row["topk_contains_expected"])
    errors = sum(1 for row in results if row["error"])
    missing_vocab = sorted(row["expected"] for row in results if not row["expected_in_vocab"])
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir.expanduser().resolve()),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot": str(snapshot),
        "device": args.device,
        "target_fps": args.target_fps,
        "top_k": args.top_k,
        "total": total,
        "top1_correct": top1_correct,
        "top1_accuracy": round(top1_correct / total, 4) if total else 0,
        "topk_contains_expected": topk_contains_expected,
        "topk_accuracy": round(topk_contains_expected / total, 4) if total else 0,
        "errors": errors,
        "missing_vocab": missing_vocab,
        "elapsed_sec": total_elapsed,
    }


def write_json(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "index",
                "expected",
                "predicted",
                "confidence",
                "top1_correct",
                "topk_contains_expected",
                "expected_in_vocab",
                "input_frames",
                "effective_frames",
                "elapsed_sec",
                "error",
                "file",
                "top_k_labels",
            ],
        )
        writer.writeheader()
        for row in results:
            csv_row = dict(row)
            csv_row["top_k_labels"] = " | ".join(candidate["label"] for candidate in row["top_k"])
            csv_row.pop("top_k")
            writer.writerow(csv_row)


def write_markdown(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    correct = [row for row in results if row["top1_correct"]]
    failed = [row for row in results if not row["top1_correct"]]
    lines = [
        "# Test-100 KSL Word Classifier Evaluation",
        "",
        "## Summary",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Input directory: `{summary['input_dir']}`",
        f"- Model: `{summary['repo_id']}`",
        f"- Snapshot: `{Path(summary['snapshot']).name}`",
        f"- Device: `{summary['device']}`",
        f"- Target FPS: `{summary['target_fps']}`",
        f"- Total videos: **{summary['total']}**",
        f"- Top-1 correct: **{summary['top1_correct']} / {summary['total']}** ({summary['top1_accuracy'] * 100:.2f}%)",
        f"- Top-{summary['top_k']} contains expected: **{summary['topk_contains_expected']} / {summary['total']}** ({summary['topk_accuracy'] * 100:.2f}%)",
        f"- Inference errors: **{summary['errors']}**",
        f"- Total elapsed: **{summary['elapsed_sec']} sec**",
        "",
    ]
    if summary["missing_vocab"]:
        lines += [
            "## Labels Not Found In Model Vocabulary",
            "",
            ", ".join(f"`{label}`" for label in summary["missing_vocab"]),
            "",
        ]

    lines += [
        "## Top-1 Correct Words",
        "",
        ", ".join(f"`{row['expected']}`" for row in correct) if correct else "None",
        "",
        "## Top-1 Failures",
        "",
        "| # | Expected | Predicted | Confidence | Top-k Candidates | Error |",
        "|---:|---|---|---:|---|---|",
    ]
    for row in failed:
        confidence = "" if row["confidence"] is None else f"{row['confidence']:.4f}"
        top_k = ", ".join(
            f"{candidate['label']} ({candidate['probability']:.4f})" for candidate in row["top_k"]
        )
        lines.append(
            f"| {row['index']} | {row['expected']} | {row['predicted']} | {confidence} | {top_k} | {row['error']} |"
        )

    lines += [
        "",
        "## Full Results",
        "",
        "| # | Expected | Predicted | Top-1 | Top-k | Confidence | Frames | File |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in results:
        confidence = "" if row["confidence"] is None else f"{row['confidence']:.4f}"
        lines.append(
            f"| {row['index']} | {row['expected']} | {row['predicted']} | "
            f"{'OK' if row['top1_correct'] else 'FAIL'} | "
            f"{'OK' if row['topk_contains_expected'] else 'FAIL'} | "
            f"{confidence} | {row['input_frames'] or ''} | `{Path(row['file']).name}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
