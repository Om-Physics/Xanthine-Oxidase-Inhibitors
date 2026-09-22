#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

RFC_PKL    = OUTDIR / "best_rfc_model.pkl"
TOP20_CSV  = OUTDIR / "feature_importance_top20.csv"
ALL_IMP    = OUTDIR / "feature_importance_all.npy"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S10  –  Morgan Fingerprint Feature Importance Analysis")
    print("=" * 60)

    if not RFC_PKL.exists():
        raise FileNotFoundError(
            f"Missing: {RFC_PKL}\n"
            "  Run S07_hyperparameter_optimisation.py first."
        )

    with open(RFC_PKL, "rb") as fh:
        best_rfc = pickle.load(fh)

    imp         = best_rfc.feature_importances_
    ranked_bits = np.argsort(imp)[::-1]

    # ── Top-20 table ──────────────────────────────────────────────────────
    top20_df = pd.DataFrame({
        "Rank"         : range(1, 21),
        "Bit_Position" : ranked_bits[:20],
        "Importance"   : np.round(imp[ranked_bits[:20]], 6),
    })
    top20_df.to_csv(TOP20_CSV, index=False)
    np.save(ALL_IMP, imp)

    print("\nTop-20 Morgan bit positions (mean decrease in impurity):")
    print(f"  {'Rank':>4}  {'Bit':>6}  {'Importance':>12}")
    print("  " + "-" * 27)
    for _, row in top20_df.iterrows():
        flag = " ◄ KEY BIT" if int(row["Bit_Position"]) in KEY_BITS else ""
        print(f"  {int(row['Rank']):>4}  {int(row['Bit_Position']):>6}  "
              f"{row['Importance']:>12.6f}{flag}")

    # ── Manuscript key bits ───────────────────────────────────────────────
    print("\nManuscript-highlighted key bits (XO pharmacophore):")
    print(f"  {'Bit':>5}  {'Rank':>5}  {'Importance':>12}  Description")
    print("  " + "-" * 65)
    for bit, desc in KEY_BITS.items():
        rank_arr = np.where(ranked_bits == bit)[0]
        rank_str = str(rank_arr[0] + 1) if len(rank_arr) else "N/A"
        print(f"  {bit:>5}  {rank_str:>5}  {imp[bit]:>12.6f}  {desc}")

    print(f"\n  Top-20 CSV saved → {TOP20_CSV}")
    print(f"  Full importance array saved → {ALL_IMP}")
    print(f"\nNext → run S11_coconut_filters.py")
