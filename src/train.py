from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models import FROM_SCRATCH_MODELS, TRANSFER_MODELS, build_model, count_parameters
from utils import (
    append_summary_row,
    build_dataloaders,
    epochs_to_accuracy,
    evaluate,
    get_device,
    plot_bn_comparison,
    plot_history,
    save_classification_report,
    save_confusion_matrix,
    save_history,
    save_json,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood-cell classification experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--mode", choices=["single", "scratch", "bn", "transfer"], default="single")
    parser.add_argument("--model", default="lenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--high-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional small subset for smoke tests")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_experiment(
    args: argparse.Namespace,
    model_name: str,
    *,
    lr: float | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    device = get_device()
    output_name = run_name or model_name
    transfer = model_name.startswith("resnet18") or model_name.startswith("vgg16")
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        transfer=transfer,
        limit_samples=args.limit_samples,
    )
    model = build_model(model_name, num_classes=len(classes), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    effective_lr = lr if lr is not None else args.lr
    if model_name.endswith("_full"):
        effective_lr = min(effective_lr, 1e-5)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=effective_lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_path = args.results_dir / "checkpoints" / f"{output_name}.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_time_sec": train_metrics["epoch_time_sec"],
            "lr": effective_lr,
        }
        history.append(row)
        print(
            f"{output_name} epoch {epoch:02d}/{args.epochs} "
            f"train_acc={row['train_accuracy']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"time={row['epoch_time_sec']:.1f}s"
        )
        if row["val_accuracy"] > best_val_acc:
            best_val_acc = row["val_accuracy"]
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)

    history_path = args.results_dir / "tables" / f"{output_name}_history.csv"
    save_history(history, history_path)
    plot_history(history, output_name, args.results_dir / "figures" / f"{output_name}_curves.png")
    save_confusion_matrix(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        classes,
        args.results_dir / "figures" / f"{output_name}_confusion_matrix.png",
    )
    save_classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        classes,
        args.results_dir / "tables" / f"{output_name}_classification_report.json",
    )

    summary = {
        "model": output_name,
        "base_model": model_name,
        "parameters": count_parameters(model),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": args.epochs,
        "mean_epoch_time_sec": sum(row["epoch_time_sec"] for row in history) / len(history),
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "epochs_to_80_val_accuracy": epochs_to_accuracy(history),
        "lr": effective_lr,
        "limit_samples": args.limit_samples or "",
    }
    save_json(summary, args.results_dir / "tables" / f"{output_name}_summary.json")
    append_summary_row(args.results_dir / "tables" / "summary.csv", summary)
    return {"summary": summary, "history": history}


def run_scratch(args: argparse.Namespace) -> None:
    for spec in FROM_SCRATCH_MODELS:
        run_experiment(args, spec.name)


def run_bn(args: argparse.Namespace) -> None:
    base = run_experiment(args, "vgg11", lr=args.lr)
    bn = run_experiment(args, "vgg11_bn", lr=args.lr)
    plot_bn_comparison(
        {"VGG-11": base["history"], "VGG-11+BN": bn["history"]},
        args.results_dir / "figures" / "bn_loss_comparison.png",
    )

    high_base = run_experiment(args, "vgg11", lr=args.high_lr, run_name="vgg11_high_lr")
    high_bn = run_experiment(args, "vgg11_bn", lr=args.high_lr, run_name="vgg11_bn_high_lr")
    plot_bn_comparison(
        {"VGG-11 LR alta": high_base["history"], "VGG-11+BN LR alta": high_bn["history"]},
        args.results_dir / "figures" / "bn_high_lr_loss_comparison.png",
    )

    high_lr_results = {
        "vgg11_high_lr": high_base["summary"],
        "vgg11_bn_high_lr": high_bn["summary"],
    }
    save_json(high_lr_results, args.results_dir / "tables" / "bn_high_lr_experiment.json")


def run_transfer(args: argparse.Namespace) -> None:
    for spec in TRANSFER_MODELS:
        run_experiment(args, spec.name)


def main() -> None:
    args = parse_args()
    if args.mode == "scratch":
        run_scratch(args)
    elif args.mode == "bn":
        run_bn(args)
    elif args.mode == "transfer":
        run_transfer(args)
    else:
        run_experiment(args, args.model)


if __name__ == "__main__":
    main()
