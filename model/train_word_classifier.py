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
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", default="checkpoints/word_classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_manifest(args.train_manifest)
    val_samples = load_manifest(args.val_manifest)
    label_to_id = build_vocab(train_samples)
    save_vocab(label_to_id, output_dir / "label_to_id.json")

    config = KeypointCTCConfig(
        gloss_vocab_size=len(label_to_id),
        d_model=args.d_model,
        spatial_layers=args.spatial_layers,
        temporal_layers=args.temporal_layers,
        num_heads=args.heads,
        max_frames=args.max_frames,
    )
    model = WordKeypointClassifier(len(label_to_id), config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        WordKeypointDataset(train_samples, label_to_id, max_frames=args.max_frames),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_word_batch,
    )
    val_loader = DataLoader(
        WordKeypointDataset(val_samples, label_to_id, max_frames=args.max_frames),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_word_batch,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    (output_dir / "config.json").write_text(
        json.dumps(vars(args) | {"num_classes": len(label_to_id)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "label_to_id": label_to_id,
            "epoch": epoch,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
