#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

CURATED_CSV  = OUTDIR / "chembl_xo_curated.csv"
MORGAN_NPY   = OUTDIR / "morgan_matrix.npy"
MACCS_NPY    = OUTDIR / "maccs_matrix.npy"
LABELS_NPY   = OUTDIR / "labels.npy"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S04  –  Molecular Fingerprint Computation")
    print("=" * 60)

    if not CURATED_CSV.exists():
        raise FileNotFoundError(
            f"Curated data not found: {CURATED_CSV}\n"
            "  Run S03_data_curation.py first."
        )

    # ── Load and restore mol objects ──────────────────────────────────────
    print("\nLoading curated dataset …")
    df = pd.read_csv(CURATED_CSV)
    df["mol"] = df["canon_smi"].apply(sanitise_smiles)
    df.dropna(subset=["mol"], inplace=True)
    print(f"  {len(df):,} molecules loaded")

    # ── Morgan fingerprints ───────────────────────────────────────────────
    if MORGAN_NPY.exists():
        print(f"\nCached Morgan matrix found: {MORGAN_NPY.name}")
        morgan_matrix = np.load(MORGAN_NPY)
    else:
        print(f"\nComputing Morgan fingerprints (radius={MORGAN_RADIUS}, {MORGAN_BITS} bits) …")
        t0 = time.time()
        morgan_matrix = np.vstack(df["mol"].apply(mol_to_morgan).tolist())
        print(f"  Done in {time.time()-t0:.1f} s  |  shape: {morgan_matrix.shape}")
        np.save(MORGAN_NPY, morgan_matrix)
        print(f"  Saved → {MORGAN_NPY}")

    # ── MACCS structural keys ─────────────────────────────────────────────
    if MACCS_NPY.exists():
        print(f"\nCached MACCS matrix found: {MACCS_NPY.name}")
        maccs_matrix = np.load(MACCS_NPY)
    else:
        print(f"\nComputing MACCS structural keys ({MACCS_BITS} bits) …")
        t0 = time.time()
        maccs_matrix = np.vstack(df["mol"].apply(mol_to_maccs).tolist())
        print(f"  Done in {time.time()-t0:.1f} s  |  shape: {maccs_matrix.shape}")
        np.save(MACCS_NPY, maccs_matrix)
        print(f"  Saved → {MACCS_NPY}")

    # ── Label array ───────────────────────────────────────────────────────
    labels = df["label"].values
    np.save(LABELS_NPY, labels)
    print(f"\nLabel array: Active={labels.sum()}  Inactive={(1-labels).sum()}")
    print(f"  Saved → {LABELS_NPY}")

    print(f"\nNext → run S05_dataset_split.py")
