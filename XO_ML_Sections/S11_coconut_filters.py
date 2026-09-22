#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

COCONUT_CSV      = Path("coconut_complete.csv")   # ← set path here
FILTERED_CSV     = OUTDIR / "coconut_filtered.csv"


# =============================================================================
# DATABASE LOADER
# =============================================================================

def load_coconut(csv_path: Path) -> pd.DataFrame:
    """
    Load COCONUT CSV and normalise SMILES column name.

    Accepted column names for SMILES:
        'smiles', 'canonical_smiles', 'smiles_string', 'smi'

    Parameters
    ----------
    csv_path : Path

    Returns
    -------
    pd.DataFrame  with columns ['coconut_id', 'smiles']
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"COCONUT file not found: {csv_path}\n"
            "  Download from https://coconut.naturalproducts.net/"
        )

    print(f"Loading COCONUT database from {csv_path.name} …", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    for candidate in ["smiles", "canonical_smiles", "smiles_string", "smi"]:
        if candidate in df.columns:
            df.rename(columns={candidate: "smiles"}, inplace=True)
            break

    if "smiles" not in df.columns:
        raise KeyError(
            "No SMILES column found. "
            "Expected one of: smiles, canonical_smiles, smiles_string, smi"
        )

    if "coconut_id" not in df.columns:
        df["coconut_id"] = [f"CNP{i:07d}" for i in range(len(df))]

    df = df[["coconut_id", "smiles"]].dropna().reset_index(drop=True)
    print(f"  Total entries loaded: {len(df):,}")
    return df


# =============================================================================
# FILTER PIPELINE
# =============================================================================

def apply_structural_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply PAINS → Brenk → Lipinski (strict) filters sequentially.

    Parameters
    ----------
    df : pd.DataFrame  with columns ['coconut_id', 'smiles']

    Returns
    -------
    pd.DataFrame  of compounds passing all three filters,
                  with an additional 'mol' column.
    """
    counts = {}
    n_init = len(df)
    df = df.copy()

    # ── Parse SMILES ──────────────────────────────────────────────────────
    print("  Parsing SMILES … ", end="", flush=True)
    df["mol"] = df["smiles"].apply(sanitise_smiles)
    df.dropna(subset=["mol"], inplace=True)
    print(f"{len(df):,} valid structures")
    counts["initial"] = n_init

    # ── (1) PAINS filter ──────────────────────────────────────────────────
    print("  Applying PAINS filter … ", end="", flush=True)
    df = df[df["mol"].apply(passes_pains)].copy()
    counts["after_pains"] = len(df)
    pct = 100 * (n_init - len(df)) / n_init
    print(f"{len(df):,}  ({pct:.2f} % removed from initial)")

    # ── (2) Brenk filter ─────────────────────────────────────────────────
    print("  Applying Brenk filter … ", end="", flush=True)
    df = df[df["mol"].apply(passes_brenk)].copy()
    counts["after_brenk"] = len(df)
    pct = 100 * (counts["after_pains"] - len(df)) / counts["after_pains"]
    print(f"{len(df):,}  ({pct:.2f} % removed from PAINS-filtered pool)")

    # ── (3) Lipinski Ro5 – strict, zero violations ────────────────────────
    print("  Applying Lipinski Ro5 (strict, 0 violations) … ", end="", flush=True)
    df = df[df["mol"].apply(lambda m: lipinski_ro5(m, max_violations=0))].copy()
    counts["after_lipinski"] = len(df)
    pct = 100 * (counts["after_brenk"] - len(df)) / counts["after_brenk"]
    print(f"{len(df):,}  ({pct:.2f} % removed from Brenk-filtered pool)")

    # ── Summary table (Table 2) ───────────────────────────────────────────
    print("\n  Screening cascade summary (Table 2):")
    print(f"  {'Stage':<30}  {'Compounds':>10}  {'Reduction %':>12}")
    print("  " + "-" * 57)
    print(f"  {'Initial COCONUT':<30}  {counts['initial']:>10,}  {'—':>12}")
    print(f"  {'After PAINS':<30}  {counts['after_pains']:>10,}  "
          f"{100*(counts['initial']-counts['after_pains'])/counts['initial']:>11.2f}%")
    print(f"  {'After Brenk':<30}  {counts['after_brenk']:>10,}  "
          f"{100*(counts['after_pains']-counts['after_brenk'])/counts['after_pains']:>11.2f}%")
    print(f"  {'After Lipinski (strict)':<30}  {counts['after_lipinski']:>10,}  "
          f"{100*(counts['after_brenk']-counts['after_lipinski'])/counts['after_brenk']:>11.2f}%")

    return df.reset_index(drop=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S11  –  COCONUT Database Structural Quality Filters")
    print("=" * 60)

    if FILTERED_CSV.exists():
        print(f"\nCached filtered file found: {FILTERED_CSV.name}")
        df_filtered = pd.read_csv(FILTERED_CSV)
        df_filtered["mol"] = df_filtered["smiles"].apply(sanitise_smiles)
        df_filtered.dropna(subset=["mol"], inplace=True)
        print(f"  {len(df_filtered):,} compounds ready for ML screening")
    else:
        if not COCONUT_CSV.exists():
            print(f"\n[ERROR] COCONUT file not found at: {COCONUT_CSV}")
            print("  Download from https://coconut.naturalproducts.net/")
            print("  Then set COCONUT_CSV path at the top of this file.")
            sys.exit(1)

        df_coconut  = load_coconut(COCONUT_CSV)
        print("\nApplying structural filters …")
        df_filtered = apply_structural_filters(df_coconut)

        # Save without mol column
        df_filtered.drop(columns=["mol"]).to_csv(FILTERED_CSV, index=False)
        print(f"\n  Filtered compounds saved → {FILTERED_CSV}")

    print(f"\nCompounds entering ML screening: {len(df_filtered):,}")
    print(f"\nNext → run S12_virtual_screening.py")
