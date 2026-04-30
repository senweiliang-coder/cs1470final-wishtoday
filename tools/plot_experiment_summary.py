import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_series(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["Epoch"]),
                    "loss": float(row["Loss"]),
                    "accuracy": float(row["Accuracy"]),
                    "auc": float(row["AUC"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def load_test_row(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    row = rows[-1]
    return {
        "checkpoint": row["Checkpoint"],
        "epoch": int(row["Epoch"]),
        "loss": float(row["Loss"]),
        "accuracy": float(row["Accuracy"]),
        "auc": float(row["AUC"]),
    }


def plot_summary(train_rows, val_rows, test_row, output_path, title):
    train_epochs = [row["epoch"] for row in train_rows]
    val_epochs = [row["epoch"] for row in val_rows]

    train_loss = [row["loss"] for row in train_rows]
    val_loss = [row["loss"] for row in val_rows]
    train_acc = [row["accuracy"] for row in train_rows]
    val_acc = [row["accuracy"] for row in val_rows]
    train_auc = [row["auc"] for row in train_rows]
    val_auc = [row["auc"] for row in val_rows]

    best_val_acc = max(val_rows, key=lambda row: row["accuracy"])
    best_val_auc = max(val_rows, key=lambda row: row["auc"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title, fontsize=14)

    axes[0, 0].plot(train_epochs, train_loss, marker="o", label="Train Loss", color="#b84a3a", linewidth=2)
    axes[0, 0].plot(val_epochs, val_loss, marker="s", label="Val Loss", color="#1f77b4", linewidth=2)
    axes[0, 0].axhline(test_row["loss"], linestyle="--", color="#2ca02c", label=f'Test Loss {test_row["loss"]:.4f}')
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(train_epochs, train_acc, marker="o", label="Train ACC", color="#8c564b", linewidth=2)
    axes[0, 1].plot(val_epochs, val_acc, marker="s", label="Val ACC", color="#17becf", linewidth=2)
    axes[0, 1].axhline(test_row["accuracy"], linestyle="--", color="#9467bd", label=f'Test ACC {test_row["accuracy"]:.4f}')
    axes[0, 1].scatter(best_val_acc["epoch"], best_val_acc["accuracy"], color="#17becf", s=55, zorder=3)
    axes[0, 1].annotate(
        f'Best Val ACC\nE{best_val_acc["epoch"]}: {best_val_acc["accuracy"]:.4f}',
        xy=(best_val_acc["epoch"], best_val_acc["accuracy"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("ACC")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(train_epochs, train_auc, marker="o", label="Train AUC", color="#ff7f0e", linewidth=2)
    axes[1, 0].plot(val_epochs, val_auc, marker="s", label="Val AUC", color="#1f77b4", linewidth=2)
    axes[1, 0].axhline(test_row["auc"], linestyle="--", color="#2ca02c", label=f'Test AUC {test_row["auc"]:.4f}')
    axes[1, 0].scatter(best_val_auc["epoch"], best_val_auc["auc"], color="#1f77b4", s=55, zorder=3)
    axes[1, 0].annotate(
        f'Best Val AUC\nE{best_val_auc["epoch"]}: {best_val_auc["auc"]:.4f}',
        xy=(best_val_auc["epoch"], best_val_auc["auc"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )
    axes[1, 0].set_title("AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    summary_text = "\n".join(
        [
            f"Checkpoint: {test_row['checkpoint']}",
            f"Test epoch: {test_row['epoch']}",
            "",
            f"Best val ACC: epoch {best_val_acc['epoch']} = {best_val_acc['accuracy']:.4f}",
            f"Best val AUC: epoch {best_val_auc['epoch']} = {best_val_auc['auc']:.4f}",
            "",
            f"Test loss: {test_row['loss']:.4f}",
            f"Test ACC: {test_row['accuracy']:.4f}",
            f"Test AUC: {test_row['auc']:.4f}",
        ]
    )
    axes[1, 1].text(0.02, 0.98, summary_text, va="top", ha="left", fontsize=11, family="monospace")

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Experiment Summary")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    train_rows = load_series(args.train_csv)
    val_rows = load_series(args.val_csv)
    test_row = load_test_row(args.test_csv)
    plot_summary(train_rows, val_rows, test_row, args.output, args.title)


if __name__ == "__main__":
    main()
