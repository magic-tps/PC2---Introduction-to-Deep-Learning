from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def has_dataset_layout(data_dir: Path) -> bool:
    train_dir = data_dir / "images" / "TRAIN"
    test_dir = data_dir / "images" / "TEST"
    return (
        train_dir.is_dir()
        and test_dir.is_dir()
        and any(train_dir.glob("*/*.jpeg"))
        and any(test_dir.glob("*/*.jpeg"))
    )


def resolve_data_dir(data_dir: Path) -> Path:
    candidates: list[Path] = []
    data_dir = data_dir.expanduser()
    if data_dir.is_absolute():
        candidates.append(data_dir)
    else:
        candidates.extend([Path.cwd() / data_dir, PROJECT_ROOT / data_dir])
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "dataset",
            PROJECT_ROOT.parent / "dataset2-master",
            PROJECT_ROOT.parent / "dataset2-master" / "dataset2-master",
            PROJECT_ROOT.parent / "dataset-master",
            PROJECT_ROOT.parent / "dataset-master" / "dataset-master",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if has_dataset_layout(resolved):
            return resolved

    checked = "\n  - ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        "Dataset not found with ImageFolder layout. Expected images/TRAIN and images/TEST in one of:\n"
        f"  - {checked}\n"
        "Use data/download_data.sh or pass --data-dir to dataset2-master."
    )


def build_transforms(image_size: int = 64, transfer: bool = False) -> tuple[transforms.Compose, transforms.Compose]:
    size = 224 if transfer else image_size
    train_tfms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def build_dataloaders(
    data_dir: Path,
    *,
    batch_size: int,
    image_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
    transfer: bool,
    limit_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    data_dir = resolve_data_dir(data_dir)
    train_tfms, eval_tfms = build_transforms(image_size=image_size, transfer=transfer)
    train_root = data_dir / "images" / "TRAIN"
    test_root = data_dir / "images" / "TEST"

    train_full = datasets.ImageFolder(train_root, transform=train_tfms)
    val_full = datasets.ImageFolder(train_root, transform=eval_tfms)
    test_dataset = datasets.ImageFolder(test_root, transform=eval_tfms)
    classes = train_full.classes

    if limit_samples:
        rng = random.Random(seed)
        train_indices = list(range(len(train_full)))
        test_indices = list(range(len(test_dataset)))
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
        train_indices = train_indices[: min(limit_samples, len(train_indices))]
        test_indices = test_indices[: min(max(1, limit_samples // 4), len(test_indices))]
        train_full = Subset(train_full, train_indices)
        val_full = Subset(val_full, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    val_size = max(1, int(len(train_full) * val_split))
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset_for_indices = random_split(train_full, [train_size, val_size], generator=generator)
    _, val_subset = random_split(val_full, [train_size, val_size], generator=generator)
    val_subset.indices = val_subset_for_indices.indices

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_subset, shuffle=True, **kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    start = time.perf_counter()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "epoch_time_sec": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (preds == targets).sum().item()
        total += batch_size
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_history(history: list[dict[str, Any]], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validacion")
    axes[0].set_title("Perdida")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="validacion")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bn_comparison(histories: dict[str, list[dict[str, Any]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=f"{label} train")
        axes[1].plot(epochs, [row["val_loss"] for row in history], label=f"{label} val")
    axes[0].set_title("Perdida de entrenamiento")
    axes[1].set_title("Perdida de validacion")
    for axis in axes:
        axis.set_xlabel("Epoca")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(y_true: list[int], y_pred: list[int], classes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_classification_report(y_true: list[int], y_pred: list[int], classes: list[str], path: Path) -> None:
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    save_json(report, path)


def epochs_to_accuracy(history: list[dict[str, Any]], threshold: float = 0.80, metric: str = "val_accuracy") -> int | None:
    for row in history:
        if row[metric] >= threshold:
            return int(row["epoch"])
    return None


def _same_summary_key(existing: dict[str, Any], row: dict[str, Any]) -> bool:
    return (
        str(existing.get("model", "")) == str(row.get("model", ""))
        and str(existing.get("lr", "")) == str(row.get("lr", ""))
        and str(existing.get("limit_samples", "")) == str(row.get("limit_samples", ""))
    )


def append_summary_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))
            rows = [existing for existing in reader if not _same_summary_key(existing, row)]
    rows.append({key: row.get(key, "") for key in fieldnames})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: existing.get(key, "") for key in fieldnames} for existing in rows)
