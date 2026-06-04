from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import KeypointCTCConfig
from .word_classifier import WordKeypointClassifier
from .word_dataset import (
    WordKeypointDataset,
    build_vocab,
    collate_word_batch,
    load_manifest,
    save_vocab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train isolated-word sign classifier.")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", default="checkpoints/word_classifier")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from, usually output_dir/last.pt.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Train epochs and save checkpoints without running validation.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only from --resume checkpoint.",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    return parser.parse_args()


def run_epoch(
    model: WordKeypointClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in tqdm(loader, leave=False):
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model.forward_raw(keypoints, lengths)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += logits.argmax(dim=-1).eq(labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def load_best_acc(output_dir: Path, fallback: float = 0.0) -> float:
    best_path = output_dir / "best.pt"
    if not best_path.exists():
        return fallback
    try:
        checkpoint = torch.load(best_path, map_location="cpu")
    except Exception:
        return fallback
    return max(fallback, float(checkpoint.get("val_acc", 0.0)))


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.validate_only and args.resume is None:
        raise ValueError("--validate-only requires --resume.")
    if not args.validate_only and args.train_manifest is None:
        raise ValueError("--train-manifest is required unless --validate-only is used.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    resume_checkpoint = load_checkpoint(args.resume, device) if args.resume else None

    train_samples = load_manifest(args.train_manifest) if args.train_manifest else []
    val_samples = load_manifest(args.val_manifest)
    label_to_id = (
        resume_checkpoint["label_to_id"] if resume_checkpoint is not None else build_vocab(train_samples)
    )
    save_vocab(label_to_id, output_dir / "label_to_id.json")

    if resume_checkpoint is not None:
        config = KeypointCTCConfig(**resume_checkpoint["config"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
    else:
        config = KeypointCTCConfig(
            gloss_vocab_size=len(label_to_id),
            d_model=args.d_model,
            spatial_layers=args.spatial_layers,
            temporal_layers=args.temporal_layers,
            num_heads=args.heads,
            max_frames=args.max_frames,
        )
        start_epoch = 1

    model = WordKeypointClassifier(len(label_to_id), config=config)
    model.to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])

    train_loader = None
    if not args.validate_only:
        train_loader = DataLoader(
            WordKeypointDataset(train_samples, label_to_id, max_frames=config.max_frames),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_word_batch,
        )
    val_loader = DataLoader(
        WordKeypointDataset(val_samples, label_to_id, max_frames=config.max_frames),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_word_batch,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    if resume_checkpoint is not None and "optimizer_state" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
    criterion = nn.CrossEntropyLoss()
    best_acc = (
        float(resume_checkpoint.get("best_acc", resume_checkpoint.get("val_acc", 0.0)))
        if resume_checkpoint is not None
        else 0.0
    )
    best_acc = load_best_acc(output_dir, best_acc)

    (output_dir / "config.json").write_text(
        json.dumps(
            vars(args)
            | {
                "num_classes": len(label_to_id),
                "effective_max_frames": config.max_frames,
                "start_epoch": start_epoch,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if resume_checkpoint is not None:
        if args.validate_only:
            print(
                f"validating checkpoint {args.resume} "
                f"from epoch {resume_checkpoint['epoch']:03d}",
                flush=True,
            )
        else:
            print(
                f"resumed checkpoint {args.resume} "
                f"from epoch {resume_checkpoint['epoch']:03d}; "
                f"training through epoch {args.epochs:03d}",
                flush=True,
            )
        if not (output_dir / "best.pt").exists():
            torch.save(resume_checkpoint, output_dir / "best.pt")

    if args.validate_only:
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        validation_checkpoint = dict(resume_checkpoint)
        validation_checkpoint.update(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": max(best_acc, val_acc),
                "validation_pending": False,
            }
        )
        print(
            f"validation epoch {validation_checkpoint['epoch']:03d} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        torch.save(validation_checkpoint, output_dir / "last.pt")
        if val_acc >= best_acc:
            torch.save(validation_checkpoint, output_dir / "best.pt")
        return

    if start_epoch > args.epochs:
        print(
            f"checkpoint is already at epoch {start_epoch - 1:03d}; "
            f"--epochs {args.epochs:03d} leaves nothing to train",
            flush=True,
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        assert train_loader is not None
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        train_checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
            "label_to_id": label_to_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": None,
            "val_acc": None,
            "best_acc": best_acc,
            "validation_pending": True,
        }
        train_checkpoint_path = output_dir / f"train_epoch_{epoch:03d}.pt"
        torch.save(train_checkpoint, train_checkpoint_path)
        torch.save(train_checkpoint, output_dir / "last_train.pt")
        print(
            f"epoch {epoch:03d} train_complete "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"checkpoint={train_checkpoint_path}",
            flush=True,
        )
        if args.skip_validation:
            torch.save(train_checkpoint, output_dir / "last.pt")
            continue

        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
            "label_to_id": label_to_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": max(best_acc, val_acc),
            "validation_pending": False,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
