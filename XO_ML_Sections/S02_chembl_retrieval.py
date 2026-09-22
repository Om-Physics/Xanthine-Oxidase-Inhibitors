#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

# =============================================================================
# TARGET DEFINITIONS
# =============================================================================

XO_TARGETS = {
    "CHEMBL4878": "Homo sapiens",
    "CHEMBL2094": "Bos taurus",
}

RAW_CSV = OUTDIR / "chembl_xo_ic50_raw.csv"


# =============================================================================
# RETRIEVAL FUNCTION
# =============================================================================

def fetch_chembl_ic50(target_dict: dict) -> pd.DataFrame:
    """
    Query ChEMBL v33 REST API for IC50 records against XO.

    Filters applied:
        standard_type      = IC50
        standard_relation  ∈ {=, <}
        assay_type         ∈ {B, F}  (binding + functional)

    Parameters
    ----------
    target_dict : dict  {ChEMBL_ID : organism_label}

    Returns
    -------
    pd.DataFrame  with columns:
        molecule_chembl_id, canonical_smiles, standard_value,
        standard_units, standard_relation, assay_type,
        pchembl_value, target_chembl_id, organism
    """
    if not _CHEMBL:
        raise ImportError(
            "Install the ChEMBL client:\n"
            "    pip install chembl-webresource-client"
        )

    act_api  = _chembl_api.activity
    all_rows = []

    for tid, organism in target_dict.items():
        print(f"  Querying {organism} ({tid}) … ", end="", flush=True)
        batch = list(
            act_api.filter(
                target_chembl_id      = tid,
                standard_type         = "IC50",
                standard_relation__in = ["=", "<"],
                assay_type__in        = ["B", "F"],
            ).only([
                "molecule_chembl_id", "canonical_smiles",
                "standard_value",     "standard_units",
                "standard_relation",  "assay_type",
                "pchembl_value",      "target_chembl_id",
            ])
        )
        for rec in batch:
            rec["organism"] = organism
        all_rows.extend(batch)
        print(f"{len(batch):,} records")

    df = pd.DataFrame(all_rows)
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S02  –  ChEMBL Bioactivity Data Retrieval")
    print("=" * 60)

    if RAW_CSV.exists():
        print(f"\nCached file found: {RAW_CSV.name}")
        df_raw = pd.read_csv(RAW_CSV, low_memory=False)
    else:
        print("\nDownloading from ChEMBL API …")
        df_raw = fetch_chembl_ic50(XO_TARGETS)
        df_raw.to_csv(RAW_CSV, index=False)
        print(f"  Saved → {RAW_CSV}")

    print(f"\nTotal raw records : {len(df_raw):,}")
    print("\nRecord counts by target and assay type:")
    print(df_raw[["target_chembl_id", "assay_type"]].value_counts().to_string())
    print(f"\nNext → run S03_data_curation.py")
