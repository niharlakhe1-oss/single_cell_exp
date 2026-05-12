"""
cell_type_importance.py  (adapted for SingleCellMetricModel / train.py pipeline)
==================================================================================
Finds which cell types drive Normal vs Crohn disease classification using the
attention weights from your trained SingleCellMetricModel.

What changed vs the original:
  - Uses SingleCellMetricModel (scGPT + LoRA + AggregatorPlusClassifier)
    instead of AttentionPoolingModel
  - Uses DataModule for data loading — no pre-computed _embeddings.npy needed
  - Attention weights come from AggregatorPlusClassifier.AttentionPooling,
    which already returns (1, N_cells) softmax weights as the 3rd return value
    of model.forward()
  - Fold splits reproduced with the same StratifiedKFold + seed as train.py

Prerequisites:
  1. Run save_cell_type_metadata.py first
  2. Have best_model_fold{0..4}.pt in CHECKPOINT_DIR

Produces:
    fig_A_cell_type_importance.png   — ranked boxplot of attention per cell type
    fig_B_normal_vs_crohn.png        — Normal vs Crohn attention for top cell types
    fig_C_fold_heatmap.png           — cell type attention heatmap across folds
    fig_D_patient_umap.png           — patient UMAP coloured by top cell type
    cell_type_attention_summary.csv  — full numeric results
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold
from umap import UMAP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_loader import DataModule
from src.framework import SingleCellMetricModel, build_aggregator
from scgpt.tokenizer import GeneVocab


# ── CONFIG — copy these values exactly from your train.py run ────────────────
H5AD_PATH             = "data/SCP1884_subset_50donors.h5ad"
MODEL_DIR             = "scgpt/continual_pretrained"    # args.json, vocab.json
CHECKPOINT_DIR        = "checkpoints"                   # best_model_fold{N}.pt
CELL_TYPE_META_DIR    = "data/cell_type_metadata/"      # from save_cell_type_metadata.py
OUTPUT_DIR            = "cell_type_importance_figures"

GENE_COL              = "index"
DONOR_COL             = "donor_id"
LABEL_COL             = "disease__ontology_label"
DISEASE_LABEL         = "Crohn's disease"
MAX_CELLS_PER_PATIENT = 5000
N_HVG                 = 5000
TEST_SIZE             = 0.15
NUM_CLASSES           = 2
LORA_R                = 8
DROPOUT               = 0.2
MAX_SEQ_LENGTH        = 1200
CHUNK_SIZE            = 256
N_FOLDS               = 5
SEED                  = 42

MIN_DONORS_PER_CT     = 3       # drop cell types present in fewer donors
TOP_N_CELL_TYPES      = 15      # how many to show in figures A/B
DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES     = {0: "Normal", 1: "Crohn disease"}
DISEASE_PALETTE = {"Normal": "#1f77b4", "Crohn disease": "#e74c3c"}
# ─────────────────────────────────────────────────────────────────────────────


def safe_donor_id(donor_id: str) -> str:
    return donor_id.replace("/", "_")


def load_model(fold_idx: int, model_config: dict, vocab: GeneVocab) -> SingleCellMetricModel:
    """Build and load a trained SingleCellMetricModel for one fold."""
    aggregator = build_aggregator(
        emb_size    = model_config["embsize"],
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
        normalize   = False,       # normalisation is done inside model.forward()
    )
    model = SingleCellMetricModel(
        model_config               = model_config,
        checkpoint_path            = str(Path(MODEL_DIR) / "best_model.pt"),
        vocab                      = vocab,
        aggregator_plus_classifier = aggregator,
        lora_r                     = LORA_R,
        max_seq_length             = MAX_SEQ_LENGTH,
        device                     = str(DEVICE),
    )
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold{fold_idx}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"    Loaded checkpoint: {ckpt_path} ✓")
    return model


def load_cell_types(donor_id: str) -> np.ndarray:
    """Load per-cell type labels saved by save_cell_type_metadata.py."""
    safe_did = safe_donor_id(donor_id)
    path = os.path.join(CELL_TYPE_META_DIR, f"{safe_did}_celltypes.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cell type file not found: {path}\n"
            "Run save_cell_type_metadata.py first."
        )
    return np.load(path, allow_pickle=True).astype(str)


@torch.no_grad()
def extract_attention_weights(
    model: SingleCellMetricModel,
    batch: dict,
    donor_id: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run one patient through the model and return per-cell attention weights.

    Returns:
        weights    : (n_cells,)  softmax attention weight per cell
        cell_types : (n_cells,)  cell type label per cell
        label      : int         0=Normal, 1=Crohn
    """
    # model.forward() returns (preds, patient_emb, weights)
    # weights shape: (1, N_cells)  — from AggregatorPlusClassifier.AttentionPooling
    _, _, weights_t = model(batch, chunk_size=CHUNK_SIZE)
    weights = weights_t.squeeze(0).cpu().numpy()    # (N_cells,)

    cell_types = load_cell_types(donor_id)
    label      = int(batch["label"].item())

    if len(weights) != len(cell_types):
        raise AssertionError(
            f"Donor {donor_id}: weight length {len(weights)} "
            f"!= cell type length {len(cell_types)}.\n"
            "Re-run save_cell_type_metadata.py with the same MAX_CELLS_PER_PATIENT "
            "and SEED to fix cell count mismatch."
        )

    return weights, cell_types, label


def aggregate_by_cell_type(weights: np.ndarray, cell_types: np.ndarray) -> dict:
    """Mean attention weight per cell type for one patient."""
    return {
        ct: float(weights[cell_types == ct].mean())
        for ct in np.unique(cell_types)
    }


# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device : {DEVICE}\n")

# ── Load model config and vocab (shared across folds) ────────────────────────
model_dir   = Path(MODEL_DIR)
with open(model_dir / "args.json") as f:
    model_config = json.load(f)

vocab = GeneVocab.from_file(str(model_dir / "vocab.json"))
for tok in ["<pad>", "<cls>", "<eoc>"]:
    if tok not in vocab:
        vocab.append_token(tok)

# ── Build DataModule with same preprocessing as train.py ─────────────────────
import scanpy as sc
print("Loading and preprocessing data ...")
adata = sc.read_h5ad(H5AD_PATH)

dm = DataModule(
    adata                 = adata,
    gene_vocab_file       = str(model_dir / "vocab.json"),
    gene_col              = GENE_COL,
    patient_col           = DONOR_COL,
    bag_col               = None,
    label_col             = LABEL_COL,
    disease_label         = DISEASE_LABEL,
    max_cells_per_patient = MAX_CELLS_PER_PATIENT,
    batch_size            = 1,
)
dm.preprocess_for_scgpt(n_hvg=N_HVG)
dm.perform_initial_split(test_size=TEST_SIZE, seed=SEED)
dm.prepare_folds(n_splits=N_FOLDS, seed=SEED)

# ── Collect donor IDs and labels in the same order DataModule uses ────────────
# NOTE: adjust the attribute names below to match your DataModule implementation.
# Common patterns: dm.patient_ids, dm.donor_ids, dm.all_donor_ids
# If unavailable, derive from the preprocessed adata directly:
preprocessed_adata = dm.adata    # adjust if stored under a different name
all_donors         = sorted(preprocessed_adata.obs[DONOR_COL].unique())
all_labels_arr     = np.array([
    1 if preprocessed_adata.obs[preprocessed_adata.obs[DONOR_COL] == d][LABEL_COL]
         .iloc[0] == DISEASE_LABEL else 0
    for d in all_donors
])

print(f"\nTotal donors : {len(all_donors)}")
print(f"  Normal (0) : {(all_labels_arr == 0).sum()}")
print(f"  Crohn  (1) : {(all_labels_arr == 1).sum()}")


# ── Cross-validation attention extraction ─────────────────────────────────────
# Reproduce the exact same fold splits used in train.py
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_records = []     # one dict per validation donor, accumulated across folds
all_embs    = []     # patient embeddings for UMAP
emb_labels  = []
emb_donors  = []

for fold_idx, (train_idx, val_idx) in enumerate(
    skf.split(np.arange(len(all_donors)), all_labels_arr)
):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx + 1} / {N_FOLDS}  —  {len(val_idx)} validation donors")
    print(f"{'='*60}")

    model = load_model(fold_idx, model_config, vocab)

    # Get the DataModule's validation loader for this fold
    _, val_loader = dm.get_fold_loaders(fold_idx)

    # val_loader yields one patient at a time (batch_size=1)
    # We pair each batch with its donor id using val_idx + all_donors
    for batch_i, batch in enumerate(val_loader):
        # Map batch position back to donor id.
        # Assumes val_loader iterates in the same order as val_idx.
        # If your DataModule shuffles val, set shuffle=False or add donor_id to batch.
        donor_id = all_donors[val_idx[batch_i]]
        label    = int(batch["label"].item())

        try:
            weights, cell_types, label = extract_attention_weights(
                model, batch, donor_id
            )
        except (AssertionError, FileNotFoundError) as e:
            print(f"  WARNING: skipping {donor_id} — {e}")
            continue

        ct_weights = aggregate_by_cell_type(weights, cell_types)

        record = {
            "donor_id" : donor_id,
            "label"    : label,
            "disease"  : LABEL_NAMES[label],
            "fold"     : fold_idx + 1,
        }
        record.update(ct_weights)
        all_records.append(record)

        # Collect patient embedding for UMAP (2nd return value of model.forward)
        with torch.no_grad():
            _, patient_emb, _ = model(batch, chunk_size=CHUNK_SIZE)
        all_embs.append(patient_emb.squeeze(0).cpu())
        emb_labels.append(label)
        emb_donors.append(donor_id)

    print(f"  Collected {len([r for r in all_records if r['fold']==fold_idx+1])} donor records ✓")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Build results DataFrame ───────────────────────────────────────────────────
results_df = pd.DataFrame(all_records)

cell_type_cols = [
    c for c in results_df.columns
    if c not in {"donor_id", "label", "disease", "fold"}
]

# Filter to cell types seen in enough donors
donors_per_ct = {
    ct: results_df[ct].notna().sum()
    for ct in cell_type_cols
}
common_cts = [ct for ct, n in donors_per_ct.items() if n >= MIN_DONORS_PER_CT]
print(f"\nCell types with >= {MIN_DONORS_PER_CT} donors: {len(common_cts)} / {len(cell_type_cols)}")

# Long-form DataFrame for seaborn
long_df = results_df[["donor_id", "label", "disease", "fold"] + common_cts].melt(
    id_vars    = ["donor_id", "label", "disease", "fold"],
    value_vars = common_cts,
    var_name   = "cell_type",
    value_name = "mean_attention",
).dropna()

# Rank cell types by overall median attention
ct_order = (
    long_df.groupby("cell_type")["mean_attention"]
    .median()
    .sort_values()
    .index.tolist()
)
top_cts = ct_order[-TOP_N_CELL_TYPES:]    # highest median attention

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, "cell_type_attention_summary.csv")
results_df.to_csv(csv_path, index=False)
print(f"Saved → {csv_path}")


# ── FIGURE A — Ranked boxplot: overall attention per cell type ────────────────
print("\nPlotting Figure A ...")

long_top = long_df[long_df["cell_type"].isin(top_cts)].copy()
long_top["cell_type"] = pd.Categorical(
    long_top["cell_type"], categories=top_cts, ordered=True
)

fig, ax = plt.subplots(figsize=(max(10, len(top_cts) * 1.2), 6))
sns.boxplot(
    data=long_top, x="cell_type", y="mean_attention",
    order=top_cts, palette="Blues_r", width=0.6, ax=ax,
)
sns.stripplot(
    data=long_top, x="cell_type", y="mean_attention",
    order=top_cts, color="black", alpha=0.35, size=3.5, jitter=True, ax=ax,
)
ax.set_xlabel("Cell Type", fontsize=12)
ax.set_ylabel("Mean Attention Weight", fontsize=12)
ax.set_title(
    "Cell Type Importance — Ranked by Median Attention Weight\n"
    "(higher = more influence on patient-level classification)",
    fontsize=13, fontweight="bold",
)
ax.tick_params(axis="x", rotation=35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
path_a = os.path.join(OUTPUT_DIR, "fig_A_cell_type_importance.png")
plt.savefig(path_a, dpi=150, bbox_inches="tight")
print(f"Saved → {path_a}")
plt.close()


# ── FIGURE B — Normal vs Crohn attention for top cell types ──────────────────
print("Plotting Figure B ...")

fig, ax = plt.subplots(figsize=(max(10, len(top_cts) * 1.5), 7))
sns.boxplot(
    data=long_top, x="cell_type", y="mean_attention",
    hue="disease", palette=DISEASE_PALETTE,
    dodge=True, width=0.6, ax=ax,
    hue_order=["Normal", "Crohn disease"],
)
sns.stripplot(
    data=long_top, x="cell_type", y="mean_attention",
    hue="disease", palette=DISEASE_PALETTE,
    dodge=True, ax=ax, alpha=0.4, size=4, jitter=True,
    hue_order=["Normal", "Crohn disease"],
    legend=False,
)

# Mann-Whitney U p-value stars
y_max = long_top["mean_attention"].max()
for i, ct in enumerate(top_cts):
    g_norm  = long_top[(long_top["cell_type"] == ct) & (long_top["disease"] == "Normal")]["mean_attention"]
    g_crohn = long_top[(long_top["cell_type"] == ct) & (long_top["disease"] == "Crohn disease")]["mean_attention"]
    if len(g_norm) < 2 or len(g_crohn) < 2:
        continue
    _, pval = mannwhitneyu(g_norm, g_crohn, alternative="two-sided")
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ax.text(i, y_max * 1.06, stars, ha="center", fontsize=13,
            color="#2c3e50", fontweight="bold")

ax.set_xlabel("Cell Type", fontsize=12)
ax.set_ylabel("Mean Attention Weight", fontsize=12)
ax.set_title(
    "Attention Weight by Cell Type: Normal vs Crohn Disease\n"
    "(Mann-Whitney U, * p<0.05  ** p<0.01  *** p<0.001)",
    fontsize=13, fontweight="bold",
)
ax.tick_params(axis="x", rotation=30)
handles, labels_leg = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels_leg):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), title="Disease", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
path_b = os.path.join(OUTPUT_DIR, "fig_B_normal_vs_crohn.png")
plt.savefig(path_b, dpi=150, bbox_inches="tight")
print(f"Saved → {path_b}")
plt.close()


# ── FIGURE C — Heatmap: mean attention per cell type × fold ──────────────────
print("Plotting Figure C ...")

heatmap_data = (
    long_df[long_df["cell_type"].isin(top_cts)]
    .groupby(["fold", "cell_type"])["mean_attention"]
    .mean()
    .unstack("cell_type")
    .reindex(columns=top_cts)
)

fig, ax = plt.subplots(figsize=(max(10, len(top_cts) * 1.2), 4))
sns.heatmap(
    heatmap_data, annot=True, fmt=".3f",
    cmap="Blues", linewidths=0.5, ax=ax,
    cbar_kws={"label": "Mean attention weight"},
)
ax.set_xlabel("Cell Type", fontsize=12)
ax.set_ylabel("Fold", fontsize=12)
ax.set_title(
    "Mean Attention Weight per Cell Type Across Folds\n"
    "(consistent patterns across folds = robust biological signal)",
    fontsize=12, fontweight="bold",
)
ax.tick_params(axis="x", rotation=35)
plt.tight_layout()
path_c = os.path.join(OUTPUT_DIR, "fig_C_fold_heatmap.png")
plt.savefig(path_c, dpi=150, bbox_inches="tight")
print(f"Saved → {path_c}")
plt.close()


# ── FIGURE D — Patient UMAP coloured by top cell type attention ───────────────
print("\nComputing patient UMAP for Figure D ...")

top1_ct       = ct_order[-1]
all_embs_np   = torch.stack(all_embs).numpy()
all_labels_np = np.array(emb_labels)
print(f"  Colouring UMAP by top cell type: '{top1_ct}'")

# Gather top-cell-type attention per donor from results_df
top_ct_attn = []
for did in emb_donors:
    row = results_df[results_df["donor_id"] == did]
    if top1_ct in row.columns and not row[top1_ct].isna().all():
        top_ct_attn.append(float(row[top1_ct].iloc[0]))
    else:
        top_ct_attn.append(np.nan)
top_ct_attn = np.array(top_ct_attn)

reducer = UMAP(n_components=2, n_neighbors=5, min_dist=0.3, random_state=SEED)
umap_2d = reducer.fit_transform(all_embs_np)
print("  UMAP fitted ✓")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: disease label
for lv, name in LABEL_NAMES.items():
    idx = all_labels_np == lv
    axes[0].scatter(
        umap_2d[idx, 0], umap_2d[idx, 1],
        c=list(DISEASE_PALETTE.values())[lv],
        label=name, s=100, alpha=0.85,
        edgecolors="white", linewidths=0.8,
    )
axes[0].set_title("Patient UMAP — Disease Label", fontsize=12, fontweight="bold")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")
axes[0].legend(fontsize=10)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Right panel: top cell type attention
valid_mask = ~np.isnan(top_ct_attn)
vmin = np.nanpercentile(top_ct_attn, 5)
vmax = np.nanpercentile(top_ct_attn, 95)
sc_plot = axes[1].scatter(
    umap_2d[valid_mask, 0], umap_2d[valid_mask, 1],
    c=top_ct_attn[valid_mask],
    cmap="YlOrRd", vmin=vmin, vmax=vmax,
    s=100, alpha=0.85, edgecolors="white", linewidths=0.8,
)
plt.colorbar(sc_plot, ax=axes[1], label=f"Mean attention\n({top1_ct})")
axes[1].set_title(
    f"Patient UMAP — '{top1_ct}' Attention",
    fontsize=12, fontweight="bold",
)
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle(
    "Patient-Level UMAP with Cell Type Attention Colouring\n"
    "(validation donors, all 5 folds combined)",
    fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout()
path_d = os.path.join(OUTPUT_DIR, "fig_D_patient_umap.png")
plt.savefig(path_d, dpi=150, bbox_inches="tight")
print(f"Saved → {path_d}")
plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  CELL TYPE IMPORTANCE SUMMARY")
print("="*60)
summary = (
    long_df.groupby(["cell_type", "disease"])["mean_attention"]
    .agg(["median", "mean", "std", "count"])
    .reset_index()
    .sort_values("median", ascending=False)
)
print(summary.to_string(index=False))
print(f"\nAll figures saved to: {OUTPUT_DIR}/")
print("Done ✓")
