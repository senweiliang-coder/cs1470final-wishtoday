import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_fairvision import FairVisionDataset
from fusion_net import MedFusion


DATASET_CONFIG = {
    "fairvision_dr": {"disease": "DR", "num_classes": 2},
    "fairvision_glaucoma": {"disease": "Glaucoma", "num_classes": 2},
    "fairvision_amd": {"disease": "AMD", "num_classes": 4},
}


def compute_auc_and_specificity(all_targets, all_predictions, all_probabilities):
    all_probabilities = np.array(all_probabilities)
    unique_targets = sorted(set(all_targets))
    specificity = None

    if len(unique_targets) < 2:
        return float("nan"), None

    if len(unique_targets) == 2:
        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
        conf_matrix = confusion_matrix(all_targets, all_predictions, labels=unique_targets)
        if conf_matrix.shape == (2, 2):
            tn = conf_matrix[0, 0]
            fp = conf_matrix[0, 1]
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    else:
        classes = list(range(all_probabilities.shape[1]))
        all_targets_one_hot = label_binarize(all_targets, classes=classes)
        auc = roc_auc_score(all_targets_one_hot, all_probabilities, multi_class="ovr")

    return auc, specificity


def build_model(args, num_classes, device):
    model_args = SimpleNamespace(
        mode="test",
        batch_size=args.train_batch_size,
        fundus_encoder=args.fundus_encoder,
        oct_encoder=args.oct_encoder,
    )
    model = MedFusion(num_classes, 2, [(96, 96, 96), (384, 384)], model_args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("epoch")


def write_metrics(output_path, row):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "Checkpoint",
        "Epoch",
        "Loss",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC",
        "Specificity",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIG))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fundus_encoder", default="convnext")
    parser.add_argument("--oct_encoder", default="resnet3d")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FairVisionDataset(
        dataset_root=args.data_root,
        disease=cfg["disease"],
        mode="test",
        model_base="transformer",
        condition="normal",
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.train_batch_size // 2),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model, checkpoint_epoch = build_model(args, cfg["num_classes"], device)

    total_loss = 0.0
    total_count = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in tqdm(loader):
            low_view = data[0]
            inputs = {k: v.float().to(device) for k, v in low_view.items()}
            target = target.long().to(device)

            pred, loss, _ = model(inputs, target, checkpoint_epoch or 0)
            batch_size = target.size(0)
            total_loss += float(loss.mean().item()) * batch_size
            total_count += batch_size

            predicted = pred.argmax(dim=-1)
            probabilities = F.softmax(pred, dim=1)

            all_targets.extend(target.cpu().numpy().tolist())
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_count, 1)
    accuracy = float(np.mean(np.array(all_targets) == np.array(all_predictions)))
    precision = precision_score(all_targets, all_predictions, average="weighted")
    recall = recall_score(all_targets, all_predictions, average="weighted")
    f1 = f1_score(all_targets, all_predictions, average="weighted")
    auc, specificity = compute_auc_and_specificity(all_targets, all_predictions, all_probabilities)

    row = {
        "Checkpoint": os.path.basename(args.checkpoint),
        "Epoch": checkpoint_epoch,
        "Loss": f"{avg_loss:.6f}",
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1 Score": f"{f1:.4f}",
        "AUC": f"{auc:.4f}",
        "Specificity": "" if specificity is None else f"{specificity:.4f}",
    }
    write_metrics(args.output, row)

    print(row)


if __name__ == "__main__":
    main()
