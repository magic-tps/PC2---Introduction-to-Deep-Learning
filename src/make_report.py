from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "informe.md"
SUMMARY_CSV = ROOT / "results" / "tables" / "summary.csv"
FIGURES_DIR = ROOT / "results" / "figures"
OUTPUT_PDF = ROOT / "informe.pdf"


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold", va="top")
    wrapped = []
    for paragraph in body.splitlines():
        if not paragraph.strip():
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(paragraph, width=92))
    fig.text(0.08, 0.89, "\n".join(wrapped[:54]), fontsize=9.5, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    if not SUMMARY_CSV.exists():
        ax.text(0.02, 0.95, "Tabla resumen pendiente: ejecutar src/train.py para generar summary.csv.", fontsize=12)
    else:
        df = pd.read_csv(SUMMARY_CSV)
        keep = [
            "model",
            "parameters",
            "mean_epoch_time_sec",
            "best_val_accuracy",
            "test_accuracy",
            "epochs_to_80_val_accuracy",
            "lr",
        ]
        df = df[[column for column in keep if column in df.columns]].round(4)
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        ax.set_title("Resultados comparativos", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def get_best_confusion_matrix() -> Path | None:
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        if {"model", "test_accuracy"}.issubset(df.columns) and not df.empty:
            best_model = df.sort_values("test_accuracy", ascending=False).iloc[0]["model"]
            candidate = FIGURES_DIR / f"{best_model}_confusion_matrix.png"
            if candidate.exists():
                return candidate
    confusion = sorted(FIGURES_DIR.glob("*confusion_matrix.png"))
    return confusion[0] if confusion else None


def get_best_model_name() -> str | None:
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        if {"model", "test_accuracy"}.issubset(df.columns) and not df.empty:
            return str(df.sort_values("test_accuracy", ascending=False).iloc[0]["model"])
    return None


def add_figure_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.imshow(plt.imread(image_path))
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def add_figure_grid_page(pdf: PdfPages, title: str, figures: list[tuple[Path, str]]) -> None:
    figures = [(path, label) for path, label in figures if path.exists()]
    if not figures:
        return
    if len(figures) == 1:
        add_figure_page(pdf, figures[0][0], title)
        return

    cols = 2
    rows = math.ceil(len(figures) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))
    flat_axes = list(axes.flat)
    for ax, (image_path, label) in zip(flat_axes, figures):
        ax.imshow(plt.imread(image_path))
        ax.axis("off")
        ax.set_title(label, fontsize=10, weight="bold")
    for ax in flat_axes[len(figures) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    text = REPORT_MD.read_text(encoding="utf-8") if REPORT_MD.exists() else ""
    with PdfPages(OUTPUT_PDF) as pdf:
        add_text_page(pdf, "Informe tecnico", text)
        add_table_page(pdf)
        add_figure_grid_page(
            pdf,
            "EDA",
            [
                (FIGURES_DIR / "eda_class_counts.png", "Distribucion de clases"),
                (FIGURES_DIR / "eda_samples.png", "Muestras por clase"),
            ],
        )
        add_figure_grid_page(
            pdf,
            "Tarea 1: modelos desde cero",
            [
                (FIGURES_DIR / "lenet_curves.png", "LeNet"),
                (FIGURES_DIR / "lenet_bn_curves.png", "LeNet+BN"),
                (FIGURES_DIR / "vgg11_curves.png", "VGG-11"),
                (FIGURES_DIR / "vgg11_bn_curves.png", "VGG-11+BN"),
            ],
        )
        add_figure_grid_page(
            pdf,
            "Tarea 2: Batch Normalization",
            [
                (FIGURES_DIR / "bn_loss_comparison.png", "BN controlado"),
                (FIGURES_DIR / "bn_high_lr_loss_comparison.png", "BN con LR alta"),
            ],
        )
        add_figure_grid_page(
            pdf,
            "Tarea 3: imagenes del notebook 03_transfer",
            [(FIGURES_DIR / f"{get_best_model_name()}_curves.png", "Mejor modelo")],
        )
        best_confusion = get_best_confusion_matrix()
        if best_confusion:
            add_figure_page(pdf, best_confusion, "Matriz de confusion del mejor modelo")
    print(f"Report written to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
