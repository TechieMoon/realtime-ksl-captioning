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
DEFAULT_LABEL_ALIASES = {"꺠끗하다": "깨끗하다"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate isolated KSL word videos in a folder.")
    parser.add_argument("input_dir", type=Path, help="Folder containing one video per word.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "test100")
    parser.add_argument("--repo-id", default="Seoyoung07/korean-sign-word-classifier-mediapipe")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--label-alias",
        action="append",
        default=[],
        help="Expected-label alias in RAW=CANONICAL format. Can be repeated.",
    )
    parser.add_argument(
        "--disable-default-label-aliases",
        action="store_true",
        help="Do not apply built-in filename typo aliases such as 꺠끗하다=깨끗하다.",
    )
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
    ko_md_path = args.output_dir / f"test100_report_{timestamp}_ko.md"
    label_aliases = build_label_aliases(args)

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
            expected_raw = video_path.stem
            expected = normalize_label(expected_raw, label_aliases)
            print(f"[{index}/{len(videos)}] {expected_raw}", flush=True)
            item_start = time.perf_counter()
            row: dict[str, Any] = {
                "index": index,
                "file": str(video_path),
                "expected_raw": expected_raw,
                "expected": expected,
                "label_alias_applied": expected_raw != expected,
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
    summary = build_summary(results, args, snapshot, checkpoint, label_aliases, total_elapsed)
    write_json(json_path, summary, results)
    write_csv(csv_path, results)
    write_markdown(md_path, summary, results)
    write_markdown_ko(ko_md_path, summary, results)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {ko_md_path}")
    print(
        f"top1={summary['top1_correct']}/{summary['total']} "
        f"top5={summary['topk_contains_expected']}/{summary['total']} "
        f"errors={summary['errors']}"
    )


def build_summary(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    snapshot: Path,
    checkpoint: dict[str, Any],
    label_aliases: dict[str, str],
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
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_acc": checkpoint.get("val_acc"),
        "checkpoint_best_acc": checkpoint.get("best_acc"),
        "device": args.device,
        "target_fps": args.target_fps,
        "top_k": args.top_k,
        "label_aliases": label_aliases,
        "alias_applied": [
            {"raw": row["expected_raw"], "canonical": row["expected"]}
            for row in results
            if row["label_alias_applied"]
        ],
        "total": total,
        "top1_correct": top1_correct,
        "top1_accuracy": round(top1_correct / total, 4) if total else 0,
        "topk_contains_expected": topk_contains_expected,
        "topk_accuracy": round(topk_contains_expected / total, 4) if total else 0,
        "errors": errors,
        "missing_vocab": missing_vocab,
        "elapsed_sec": total_elapsed,
    }


def build_label_aliases(args: argparse.Namespace) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if not args.disable_default_label_aliases:
        aliases.update(DEFAULT_LABEL_ALIASES)
    for item in args.label_alias:
        if "=" not in item:
            raise ValueError(f"Invalid --label-alias value, expected RAW=CANONICAL: {item}")
        raw, canonical = item.split("=", 1)
        raw = raw.strip()
        canonical = canonical.strip()
        if not raw or not canonical:
            raise ValueError(f"Invalid --label-alias value, expected RAW=CANONICAL: {item}")
        aliases[raw] = canonical
    return aliases


def normalize_label(label: str, aliases: dict[str, str]) -> str:
    return aliases.get(label, label)


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
                "expected_raw",
                "expected",
                "label_alias_applied",
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
        f"- Checkpoint epoch: `{summary['checkpoint_epoch']}`",
        f"- Checkpoint validation accuracy: `{summary['checkpoint_val_acc']}`",
        f"- Device: `{summary['device']}`",
        f"- Target FPS: `{summary['target_fps']}`",
        f"- Total videos: **{summary['total']}**",
        f"- Top-1 correct: **{summary['top1_correct']} / {summary['total']}** ({summary['top1_accuracy'] * 100:.2f}%)",
        f"- Top-{summary['top_k']} contains expected: **{summary['topk_contains_expected']} / {summary['total']}** ({summary['topk_accuracy'] * 100:.2f}%)",
        f"- Inference errors: **{summary['errors']}**",
        f"- Total elapsed: **{summary['elapsed_sec']} sec**",
        "",
    ]
    if summary["alias_applied"]:
        alias_text = ", ".join(
            f"`{item['raw']}` -> `{item['canonical']}`" for item in summary["alias_applied"]
        )
        lines += [
            "## Expected Label Normalization",
            "",
            f"Applied aliases: {alias_text}",
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
        ", ".join(format_expected_label(row) for row in correct) if correct else "None",
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
            f"| {row['index']} | {format_expected_label(row)} | {row['predicted']} | {confidence} | {top_k} | {row['error']} |"
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
            f"| {row['index']} | {format_expected_label(row)} | {row['predicted']} | "
            f"{'OK' if row['top1_correct'] else 'FAIL'} | "
            f"{'OK' if row['topk_contains_expected'] else 'FAIL'} | "
            f"{confidence} | {row['input_frames'] or ''} | `{Path(row['file']).name}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_expected_label(row: dict[str, Any]) -> str:
    if row.get("label_alias_applied"):
        return f"`{row['expected_raw']}` -> `{row['expected']}`"
    return f"`{row['expected']}`"


def write_markdown_ko(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    correct = [row for row in results if row["top1_correct"]]
    failed = [row for row in results if not row["top1_correct"]]
    topk_only = [row for row in failed if row["topk_contains_expected"]]
    lines = [
        "# Test-100 수어 단어 모델 평가 보고서",
        "",
        "## 요약",
        "",
        f"- 평가 일시: `{summary['created_at']}`",
        f"- 테스트 폴더: `{summary['input_dir']}`",
        f"- 모델: `{summary['repo_id']}`",
        f"- 모델 스냅샷: `{Path(summary['snapshot']).name}`",
        f"- 체크포인트 epoch: `{summary['checkpoint_epoch']}`",
        f"- 체크포인트 validation accuracy: `{summary['checkpoint_val_acc']}`",
        f"- 실행 장치: `{summary['device']}`",
        f"- 입력 영상 수: **{summary['total']}개**",
        f"- Top-1 정답: **{summary['top1_correct']} / {summary['total']}개** ({summary['top1_accuracy'] * 100:.2f}%)",
        f"- Top-{summary['top_k']} 안에 정답 포함: **{summary['topk_contains_expected']} / {summary['total']}개** ({summary['topk_accuracy'] * 100:.2f}%)",
        f"- Top-1 오답: **{len(failed)}개**",
        f"- 추론 에러: **{summary['errors']}개**",
        f"- 총 평가 시간: **{summary['elapsed_sec']}초**",
        "",
    ]
    if summary["alias_applied"]:
        alias_text = ", ".join(
            f"`{item['raw']}` -> `{item['canonical']}`" for item in summary["alias_applied"]
        )
        lines += [
            "## 정답 라벨 정규화",
            "",
            f"- 적용한 alias: {alias_text}",
            "- 파일명 오타 또는 표기 차이만 있는 경우 canonical 라벨 기준으로 정답 여부를 계산했습니다.",
            "",
        ]
    if summary["missing_vocab"]:
        lines += [
            "## 모델 vocabulary에 없는 정답 라벨",
            "",
            ", ".join(f"`{label}`" for label in summary["missing_vocab"]),
            "",
        ]
    lines += [
        "## 맞은 단어 Top-1 기준",
        "",
        f"총 {len(correct)}개: "
        + (", ".join(format_expected_label(row) for row in correct) if correct else "없음"),
        "",
        "## 틀린 단어 Top-1 기준",
        "",
        f"총 {len(failed)}개입니다. 아래 표에서 `Top-{summary['top_k']} 후보`에 정답이 들어간 경우는 모델의 1순위는 틀렸지만 후보 안에는 정답이 있었던 항목입니다.",
        "",
        f"| # | 정답 | 예측 1순위 | confidence | Top-{summary['top_k']} 후보 |",
        "|---:|---|---|---:|---|",
    ]
    for row in failed:
        confidence = "" if row["confidence"] is None else f"{row['confidence']:.4f}"
        top_k = ", ".join(
            f"{candidate['label']} ({candidate['probability']:.4f})" for candidate in row["top_k"]
        )
        lines.append(
            f"| {row['index']} | {format_expected_label(row)} | {row['predicted']} | {confidence} | {top_k} |"
        )
    lines += [
        "",
        f"## Top-1은 틀렸지만 Top-{summary['top_k']} 안에 정답이 있었던 단어",
        "",
        f"총 {len(topk_only)}개: "
        + (", ".join(format_expected_label(row) for row in topk_only) if topk_only else "없음"),
        "",
        "## 전체 결과",
        "",
        f"| # | 정답 | 예측 1순위 | Top-1 | Top-{summary['top_k']} | confidence | 추출 프레임 | 파일 |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in results:
        confidence = "" if row["confidence"] is None else f"{row['confidence']:.4f}"
        lines.append(
            f"| {row['index']} | {format_expected_label(row)} | {row['predicted']} | "
            f"{'정답' if row['top1_correct'] else '오답'} | "
            f"{'포함' if row['topk_contains_expected'] else '미포함'} | "
            f"{confidence} | {row['input_frames'] or ''} | `{Path(row['file']).name}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
