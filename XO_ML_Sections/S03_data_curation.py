#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

RAW_CSV     = OUTDIR / "chembl_xo_ic50_raw.csv"
CURATED_CSV = OUTDIR / "chembl_xo_curated.csv"


# =============================================================================
# CURATION FUNCTION
# =============================================================================

def curate_bioactivity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full ChEMBL bioactivity curation pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw records from S02 (output of fetch_chembl_ic50).

    Returns
    -------
    pd.DataFrame
        Clean, labelled dataset with columns:
        canon_smi | ic50_um | pIC50 | label | mol
    """
    df = df.copy()

    # ── Step 1: drop missing values ───────────────────────────────────────
    df.dropna(subset=["canonical_smiles", "standard_value"], inplace=True)
    df = df[df["canonical_smiles"].str.strip().ne("")]

    # ── Step 2: numeric IC50 and unit normalisation to µM ─────────────────
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df.dropna(subset=["standard_value"], inplace=True)
    df = df[df["standard_value"] > 0]

    unit_to_um = {"NM": 1e-3, "UM": 1.0, "MM": 1e3, "PM": 1e-6}
    df["unit_norm"] = (
        df["standard_units"].str.strip().str.upper()
        .str.replace("µ", "U", regex=False)
        .str.replace("'", "", regex=False)
    )
    df = df[df["unit_norm"].isin(unit_to_um)]
    df["ic50_um"] = df["standard_value"] * df["unit_norm"].map(unit_to_um)
    df = df[(df["ic50_um"] > 0) & (df["ic50_um"] < 1e6)]

    # ── Step 3: sanitise SMILES ───────────────────────────────────────────
    df["mol"] = df["canonical_smiles"].apply(sanitise_smiles)
    df.dropna(subset=["mol"], inplace=True)

    # ── Step 4: Lipinski Ro5 (≤ 1 violation) quality filter ───────────────
    df = df[df["mol"].apply(lambda m: lipinski_ro5(m, max_violations=1))].copy()

    # ── Step 5: canonical SMILES + geometric mean for duplicates ──────────
    df["canon_smi"] = df["mol"].apply(Chem.MolToSmiles)
    df["ln_ic50"]   = np.log(df["ic50_um"])

    agg = (
        df.groupby("canon_smi")["ln_ic50"]
          .mean()
          .reset_index()
          .rename(columns={"ln_ic50": "mean_ln_ic50"})
    )
    agg["ic50_um"] = np.exp(agg["mean_ln_ic50"])
    agg["pIC50"]   = -np.log10(agg["ic50_um"] * 1e-6)

    # ── Step 6: binary activity labels ────────────────────────────────────
    agg["label"] = (agg["ic50_um"] <= IC50_CUTOFF_UM).astype(int)

    # Re-attach RDKit mol objects (first occurrence per canonical SMILES)
    smi_mol = df.groupby("canon_smi")["mol"].first().to_dict()
    agg["mol"] = agg["canon_smi"].map(smi_mol)

    # ── Summary report ────────────────────────────────────────────────────
    n_a = (agg["label"] == 1).sum()
    n_i = (agg["label"] == 0).sum()
    print(f"\n  Curated dataset: {len(agg):,} unique compounds")
    print(f"    Active   (IC50 ≤ {IC50_CUTOFF_UM} µM)  : {n_a:,}  ({100*n_a/len(agg):.1f}%)")
    print(f"    Inactive (IC50 >  {IC50_CUTOFF_UM} µM)  : {n_i:,}  ({100*n_i/len(agg):.1f}%)")

    return agg[["canon_smi", "ic50_um", "pIC50", "label", "mol"]].reset_index(drop=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S03  –  Data Curation and Preprocessing")
    print("=" * 60)

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_CSV}\n"
            "  Run S02_chembl_retrieval.py first."
        )

    if CURATED_CSV.exists():
        print(f"\nCached curated file found: {CURATED_CSV.name}")
        df_curated = pd.read_csv(CURATED_CSV)
        df_curated["mol"] = df_curated["canon_smi"].apply(sanitise_smiles)
        n_a = (df_curated["label"] == 1).sum()
        n_i = (df_curated["label"] == 0).sum()
        print(f"  {len(df_curated):,} compounds  |  Active: {n_a}  Inactive: {n_i}")
    else:
        print("\nLoading raw ChEMBL data …")
        df_raw = pd.read_csv(RAW_CSV, low_memory=False)
        print(f"  {len(df_raw):,} raw records")

        print("\nRunning curation pipeline …")
        df_curated = curate_bioactivity_data(df_raw)

        # Save without mol column (not directly serialisable)
        df_curated.drop(columns=["mol"]).to_csv(CURATED_CSV, index=False)
        print(f"\n  Saved → {CURATED_CSV}")

    print(f"\nNext → run S04_fingerprints.py")
