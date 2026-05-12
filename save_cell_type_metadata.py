"""
save_cell_type_metadata.py  (adapted for SingleCellMetricModel / train.py pipeline)
=====================================================================================
Run this ONCE before cell_type_importance.py.

What changed vs the original:
  - No longer depends on pre-existing *_embeddings.npy files
  - Runs the SAME DataModule preprocessing as train.py (HVG selection,
    normalisation, subsampling) so cell ordering is guaranteed to match
    what SingleCellMetricModel sees at inference time
  - Reads cell type labels from adata.obs AFTER preprocessing so the
    donor/cell correspondence is exact

After running, OUTPUT_DIR will contain:
    {safe_donor_id}_celltypes.npy   (n_cells,) str  — one per donor

Usage:
    python save_cell_type_metadata.py
    (edit the CONFIG block below to match your train.py run)
"""

import os
import sys
import json
import numpy as np
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_loader import DataModule          # same import as train.py


# ── CONFIG — copy these values exactly from your train.py run ────────────────
H5AD_PATH             = "data/SCP1884_subset_50donors.h5ad"
MODEL_DIR             = "scgpt/continual_pretrained"   # contains vocab.json
OUTPUT_DIR            = "data/cell_type_metadata/"

GENE_COL              = "index"
DONOR_COL             = "donor_id"
LABEL_COL             = "disease__ontology_label"
DISEASE_LABEL         = "Crohn's disease"
MAX_CELLS_PER_PATIENT = 5000
N_HVG                 = 5000
SEED                  = 42

# The obs column that holds cell type labels.
# Check adata.obs.columns — common names: "cell_type", "cell_type_fine",
# "leiden", "celltype_l2", etc.
CELL_TYPE_COL         = "cell_type"
# ─────────────────────────────────────────────────────────────────────────────


def safe_donor_id(donor_id: str) -> str:
    """Mirrors DataModule's safe_did logic (replace / with _)."""
    return donor_id.replace("/", "_")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vocab_path = str(Path(MODEL_DIR) / "vocab.json")

    # ── Load raw AnnData ──────────────────────────────────────────────────────
    print("Loading AnnData ...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"  Total cells   : {adata.shape[0]:,}")
    print(f"  obs columns   : {adata.obs.columns.tolist()}")
    print(f"  Unique donors : {adata.obs[DONOR_COL].nunique()}")
    print(f"  Cell types    : {sorted(adata.obs[CELL_TYPE_COL].unique().tolist())}")

    # ── Run the SAME preprocessing as train.py ────────────────────────────────
    # This ensures gene filtering (HVG), normalisation, and any obs filtering
    # match exactly what the model saw during training.
    print("\nRunning DataModule preprocessing (same as train.py) ...")
    dm = DataModule(
        adata=adata,
        gene_vocab_file=vocab_path,
        gene_col=GENE_COL,
        patient_col=DONOR_COL,
        bag_col=None,
        label_col=LABEL_COL,
        disease_label=DISEASE_LABEL,
        max_cells_per_patient=MAX_CELLS_PER_PATIENT,
        batch_size=1,
    )
    dm.preprocess_for_scgpt(n_hvg=N_HVG)

    # Access the preprocessed AnnData.
    # NOTE: if your DataModule stores it under a different attribute name
    # (e.g. dm.adata_, dm.processed_adata), adjust the line below.
    preprocessed_adata = dm.adata
    print(f"\nPreprocessed shape : {preprocessed_adata.shape}")
    print(f"HVGs kept          : {preprocessed_adata.shape[1]}")

    # Verify the cell type column survived preprocessing
    if CELL_TYPE_COL not in preprocessed_adata.obs.columns:
        raise ValueError(
            f"Column '{CELL_TYPE_COL}' not found in preprocessed adata.obs.\n"
            f"Available columns: {preprocessed_adata.obs.columns.tolist()}"
        )

    # ── Save per-donor cell type arrays ──────────────────────────────────────
    donors = sorted(preprocessed_adata.obs[DONOR_COL].unique())
    print(f"\nSaving cell type arrays for {len(donors)} donors → {OUTPUT_DIR}\n")

    saved, errors = 0, []

    for donor_id in donors:
        mask        = preprocessed_adata.obs[DONOR_COL] == donor_id
        donor_cells = preprocessed_adata[mask]

        cell_types = donor_cells.obs[CELL_TYPE_COL].values.astype(str)
        n_total    = len(cell_types)

        # ── Subsample if needed — mirrors DataModule's behaviour ──────────────
        # DataModule limits each patient to MAX_CELLS_PER_PATIENT.
        # If your DataModule subsamples randomly (not just truncates), set the
        # same rng seed so the cell ordering matches exactly.
        #
        # Scenario A — DataModule takes the FIRST N cells (deterministic):
        #   cell_types = cell_types[:MAX_CELLS_PER_PATIENT]
        #
        # Scenario B — DataModule samples randomly with a fixed seed (default):
        if n_total > MAX_CELLS_PER_PATIENT:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(n_total, MAX_CELLS_PER_PATIENT, replace=False)
            idx.sort()                             # preserve original cell order
            cell_types = cell_types[idx]

        safe_did  = safe_donor_id(donor_id)
        save_path = os.path.join(OUTPUT_DIR, f"{safe_did}_celltypes.npy")
        np.save(save_path, cell_types)
        saved += 1

        print(f"  {safe_did:<40} {len(cell_types):>5}/{n_total:<5} cells  "
              f"types={sorted(set(cell_types))[:3]}...")

    print(f"\nDone. Saved {saved} cell type files.")
    if errors:
        print(f"Errors ({len(errors)}): {errors}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    sample_donor = safe_donor_id(donors[0])
    ct_check = np.load(
        os.path.join(OUTPUT_DIR, f"{sample_donor}_celltypes.npy"), allow_pickle=True
    )
    print(f"\nVerification — {sample_donor}:")
    print(f"  Cells saved : {ct_check.shape[0]}")
    print(f"  Unique types: {np.unique(ct_check).tolist()}")
    print("✓ Cell type metadata ready for cell_type_importance.py")


if __name__ == "__main__":
    main()
