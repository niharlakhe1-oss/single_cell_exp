"""
train_5fold_cv.py
=================
5-fold stratified cross-validation training for the attention pooling
model (Normal vs Crohn disease classification from scGPT embeddings).

Key differences from the original discovery/validation split:
  - All donors go into a single pool; no held-out set at dataset level
  - StratifiedKFold ensures each fold has the same Crohn:Normal ratio
  - Splits are done at DONOR level — no single donor appears in both
    train and val within any fold
  - Per-fold metrics are averaged for a robust estimate of generalisation
  - Best model per fold is saved as best_model_fold{k}.pt
  - Aggregate report printed at the end: mean ± std across folds

Usage
-----
    python train_5fold_cv.py

Requires
--------
    attention_model.py   (in same directory)
    data/all/            folder with ALL donor .npy files
    data/all_labels.npy  matching labels array

    pip install torch scikit-learn numpy pandas matplotlib
"""

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
)
import matplotlib.pyplot as plt

from attention_model import (
    AttentionPoolingModel,
    PatientDataset,
    collate_fn,
    load_all_patients_flat,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────
#  Hyperparameters — edit here only
# ─────────────────────────────────────────────────────────────────

D_H           = 512          # scGPT embedding dimension
NUM_CLASSES   = 2
BATCH_SIZE    = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 20
N_FOLDS       = 5
RANDOM_SEED   = 42

# Data: all patients in a single flat folder (no pre-split)
ALL_DATA_DIR    = "data/all/"
ALL_LABELS_PATH = "data/all_labels.npy"

LABEL_NAMES   = {0: "Normal", 1: "Crohn disease"}
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model() -> nn.Module:
    return AttentionPoolingModel(
        d_h=D_H, num_classes=NUM_CLASSES
    ).to(DEVICE)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weighting for imbalanced classes."""
    n_total = len(labels)
    weights = []
    for c in range(NUM_CLASSES):
        n_c = (labels == c).sum()
        weights.append(n_total / (NUM_CLASSES * n_c))
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for Z, mask, labels in loader:
        Z, mask, labels = Z.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        Z = F.normalize(Z, dim=-1)

        logits, _, _ = model(Z, mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for Z, mask, labels in loader:
        Z, mask, labels = Z.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        Z = F.normalize(Z, dim=-1)

        logits, _, _ = model(Z, mask)
        probs = torch.softmax(logits, dim=1)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    preds_np  = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs_np  = torch.cat(all_probs).numpy()

    return (
        total_loss / len(loader),
        correct / total * 100,
        preds_np,
        labels_np,
        probs_np,
    )


def compute_fold_metrics(labels_np, preds_np, probs_np) -> dict:
    """Compute full suite of metrics for one fold."""
    disease_probs = probs_np[:, 1]

    cm = confusion_matrix(labels_np, preds_np)
    TN, FP, FN, TP = cm.ravel()

    accuracy    = (TP + TN) / len(labels_np)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)

    # Guard AUROC — needs both classes present in fold
    try:
        auroc = roc_auc_score(labels_np, disease_probs)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(labels_np, disease_probs)
    except ValueError:
        auprc = float("nan")

    mcc   = matthews_corrcoef(labels_np, preds_np)
    brier = brier_score_loss(labels_np, disease_probs)

    return dict(
        accuracy=accuracy, sensitivity=sensitivity, specificity=specificity,
        precision=precision, f1=f1, auroc=auroc, auprc=auprc,
        mcc=mcc, brier=brier,
        TP=TP, TN=TN, FP=FP, FN=FN,
    )


# ─────────────────────────────────────────────────────────────────
#  Main 5-fold CV loop
# ─────────────────────────────────────────────────────────────────

def run_cv():
    set_seed(RANDOM_SEED)
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_TYPE}")
    print(f"Folds  : {N_FOLDS}")
    print("=" * 65)

    # ── Load all patients ─────────────────────────────────────────
    patient_list, donor_ids, all_labels = load_all_patients_flat(
        ALL_DATA_DIR, ALL_LABELS_PATH
    )
    print(f"Total donors  : {len(patient_list)}")
    print(f"  Normal (0)  : {(all_labels == 0).sum()}")
    print(f"  Crohn  (1)  : {(all_labels == 1).sum()}")
    print("=" * 65)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics  = []
    all_fold_results = []

    # ── Per-fold training ────────────────────────────────────────
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(len(patient_list)), all_labels), start=1
    ):
        print(f"\n{'─'*65}")
        print(f"  FOLD {fold}/{N_FOLDS}   "
              f"train={len(train_idx)} donors  val={len(val_idx)} donors")
        print(f"{'─'*65}")

        # ── Verify no donor overlap ───────────────────────────────
        assert len(set(train_idx) & set(val_idx)) == 0

        train_data = [patient_list[i] for i in train_idx]
        val_data   = [patient_list[i] for i in val_idx]

        train_labels_fold = all_labels[train_idx]
        print(f"  Train class balance: "
              f"Normal={( train_labels_fold==0).sum()}  "
              f"Crohn={(train_labels_fold==1).sum()}")

        # ── DataLoaders ──────────────────────────────────────────
        g = torch.Generator()
        g.manual_seed(RANDOM_SEED + fold)

        train_loader = DataLoader(
            PatientDataset(train_data),
            batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(
            PatientDataset(val_data),
            batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g,
        )

        # ── Model, optimiser, loss ───────────────────────────────
        set_seed(RANDOM_SEED + fold)
        model     = build_model()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS
        )
        criterion = nn.CrossEntropyLoss(
            weight=compute_class_weights(train_labels_fold)
        )

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params : {total_params:,}")

        best_val_acc  = 0.0
        best_epoch    = 1
        best_preds    = None
        best_labels   = None
        best_probs    = None

        train_losses, val_losses, val_accs = [], [], []

        for epoch in range(NUM_EPOCHS):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            vl_loss, vl_acc, preds_np, labels_np, probs_np = evaluate(
                model, val_loader, criterion
            )
            scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            val_accs.append(vl_acc)

            if vl_acc > best_val_acc:
                best_val_acc  = vl_acc
                best_epoch    = epoch + 1
                best_preds    = preds_np
                best_labels   = labels_np
                best_probs    = probs_np
                torch.save(model.state_dict(), f"best_model_fold{fold}.pt")
                print(f"    ✓ Saved fold {fold} best model  "
                      f"(epoch {epoch+1}, val_acc={vl_acc:.1f}%)")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Ep [{epoch+1:02d}/{NUM_EPOCHS}]  "
                      f"TrainLoss={tr_loss:.4f}  TrainAcc={tr_acc:.1f}%  "
                      f"ValLoss={vl_loss:.4f}  ValAcc={vl_acc:.1f}%")

        # ── Compute metrics at best epoch ─────────────────────────
        metrics = compute_fold_metrics(best_labels, best_preds, best_probs)
        metrics["fold"]       = fold
        metrics["best_epoch"] = best_epoch
        metrics["best_val_acc"] = best_val_acc
        fold_metrics.append(metrics)

        # ── Per-donor result table for this fold ──────────────────
        def outcome(t, p):
            if t == 0 and p == 0: return "TN — Correct Normal"
            if t == 1 and p == 1: return "TP — Correct Crohn"
            if t == 0 and p == 1: return "FP — Normal→Crohn"
            return                       "FN — Crohn→Normal"

        fold_rows = [
            {
                "fold":      fold,
                "donor_id":  donor_ids[val_idx[i]],
                "true":      LABEL_NAMES[int(best_labels[i])],
                "pred":      LABEL_NAMES[int(best_preds[i])],
                "prob_normal": round(float(best_probs[i, 0]), 3),
                "prob_crohn":  round(float(best_probs[i, 1]), 3),
                "correct":   bool(best_preds[i] == best_labels[i]),
                "outcome":   outcome(int(best_labels[i]), int(best_preds[i])),
            }
            for i in range(len(best_preds))
        ]
        all_fold_results.extend(fold_rows)

        print(f"\n  Fold {fold} summary → "
              f"AUROC={metrics['auroc']:.3f}  "
              f"MCC={metrics['mcc']:.3f}  "
              f"F1={metrics['f1']:.3f}  "
              f"Acc={metrics['accuracy']*100:.1f}%  "
              f"Sen={metrics['sensitivity']*100:.1f}%  "
              f"Spe={metrics['specificity']*100:.1f}%")

    # ── Aggregate report ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  5-FOLD CV — AGGREGATE METRICS")
    print("=" * 65)

    metric_keys = [
        "accuracy", "sensitivity", "specificity",
        "precision", "f1", "auroc", "auprc", "mcc", "brier",
    ]
    summary_rows = []

    for key in metric_keys:
        values = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        mean_v = np.mean(values)
        std_v  = np.std(values)
        label  = key.upper().replace("_", " ")
        print(f"  {label:<20} : {mean_v:.3f} ± {std_v:.3f}   "
              f"[{min(values):.3f} – {max(values):.3f}]")
        summary_rows.append({"metric": key, "mean": mean_v, "std": std_v,
                              "min": min(values), "max": max(values)})

    print("=" * 65)
    print("\nPer-fold breakdown:")
    for m in fold_metrics:
        print(f"  Fold {m['fold']}  "
              f"Acc={m['accuracy']*100:.1f}%  "
              f"AUROC={m['auroc']:.3f}  "
              f"MCC={m['mcc']:.3f}  "
              f"TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}  "
              f"best_epoch={m['best_epoch']}")

    # ── Save CSVs ─────────────────────────────────────────────────
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv("cv_donor_results.csv", index=False)
    print("\nSaved → cv_donor_results.csv ✓")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("cv_metric_summary.csv", index=False)
    print("Saved → cv_metric_summary.csv ✓")

    # ── Plot per-fold metrics ─────────────────────────────────────
    plot_cv_results(fold_metrics)

    return fold_metrics, results_df, summary_df


# ─────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────

def plot_cv_results(fold_metrics: list):
    folds     = [m["fold"] for m in fold_metrics]
    metrics_to_plot = ["accuracy", "auroc", "sensitivity", "specificity", "mcc"]
    colors    = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 4))

    for ax, key, color in zip(axes, metrics_to_plot, colors):
        values = [m[key] for m in fold_metrics]
        mean_v = np.mean(values)

        ax.bar(folds, values, color=color, alpha=0.75, edgecolor="white",
               linewidth=1.5)
        ax.axhline(mean_v, color="black", linestyle="--", alpha=0.6,
                   label=f"Mean={mean_v:.3f}")
        ax.set_xlabel("Fold", fontsize=11)
        ax.set_ylabel(key.upper(), fontsize=11)
        ax.set_title(key.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xticks(folds)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        f"5-Fold CV — Normal vs Crohn Disease  (Softmax Attention Pooling)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("cv_fold_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → cv_fold_metrics.png ✓")


# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fold_metrics, results_df, summary_df = run_cv()
