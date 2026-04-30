import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["Epoch"]),
                    "loss": float(row["Loss"]),
                    "accuracy": float(row["Accuracy"]),
                    "precision": float(row["Precision"]),
                    "recall": float(row["Recall"]),
                    "f1": float(row["F1 Score"]),
                    "auc": float(row["AUC"]),
                    "specificity": float(row["Specificity"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def best_point(rows, key):
    best = max(rows, key=lambda row: row[key])
    return best["epoch"], best[key]


def plot_validation_curves(rows, output_path, title):
    epochs = [row["epoch"] for row in rows]
    loss = [row["loss"] for row in rows]
    acc = [row["accuracy"] for row in rows]
    precision = [row["precision"] for row in rows]
    recall = [row["recall"] for row in rows]
    f1 = [row["f1"] for row in rows]
    auc = [row["auc"] for row in rows]
    specificity = [row["specificity"] for row in rows]

    best_acc_epoch, best_acc = best_point(rows, "accuracy")
    best_auc_epoch, best_auc = best_point(rows, "auc")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    axes[0, 0].plot(epochs, loss, marker="o", color="#b84a3a", linewidth=2)
    axes[0, 0].set_title("Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, acc, marker="o", label="Accuracy", color="#1f77b4", linewidth=2)
    axes[0, 1].plot(epochs, f1, marker="s", label="F1", color="#2ca02c", linewidth=2)
    axes[0, 1].scatter(best_acc_epoch, best_acc, color="#1f77b4", s=55, zorder=3)
    axes[0, 1].annotate(
        f"Best Acc\nE{best_acc_epoch}: {best_acc:.4f}",
        xy=(best_acc_epoch, best_acc),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0, 1].set_title("Validation Accuracy / F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, auc, marker="o", label="AUC", color="#9467bd", linewidth=2)
    axes[1, 0].plot(epochs, specificity, marker="s", label="Specificity", color="#ff7f0e", linewidth=2)
    axes[1, 0].scatter(best_auc_epoch, best_auc, color="#9467bd", s=55, zorder=3)
    axes[1, 0].annotate(
        f"Best AUC\nE{best_auc_epoch}: {best_auc:.4f}",
        xy=(best_auc_epoch, best_auc),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )
    axes[1, 0].set_title("Validation AUC / Specificity")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, precision, marker="o", label="Precision", color="#8c564b", linewidth=2)
    axes[1, 1].plot(epochs, recall, marker="s", label="Recall", color="#17becf", linewidth=2)
    axes[1, 1].set_title("Validation Precision / Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Validation CSV log path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--title", default="Validation Curves", help="Figure title")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = load_rows(args.csv)
    plot_validation_curves(rows, args.output, args.title)


if __name__ == "__main__":
    main()
