#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

FILTERED_CSV  = OUTDIR / "coconut_filtered.csv"
RFC_PKL       = OUTDIR / "best_rfc_model.pkl"
HITS_CSV      = OUTDIR / "ml_vs_hits_for_docking.csv"
SUMMARY_TXT   = OUTDIR / "ml_vs_hits_summary.txt"


# =============================================================================
# VIRTUAL SCREENING FUNCTION
# =============================================================================

def virtual_screen(df: pd.DataFrame,
                   rfc_model,
                   threshold: float = ML_PROB_CUTOFF) -> pd.DataFrame:
    """
    Score every compound in df with the trained RFC and retain hits.

    Parameters
    ----------
    df        : DataFrame with columns ['coconut_id', 'smiles', 'mol']
    rfc_model : Trained RandomForestClassifier (from S07)
    threshold : Minimum activity probability (default 0.60)

    Returns
    -------
    pd.DataFrame
        Hit list sorted by activity_prob (descending) with 1-based rank
        and computed physicochemical descriptors.
    """
    n_screened = len(df)

    # ── Compute Morgan fingerprints ────────────────────────────────────────
    print(f"  Computing Morgan fingerprints for {n_screened:,} compounds …",
          flush=True)
    t0  = time.time()
    fps = np.vstack(df["mol"].apply(mol_to_morgan).tolist())
    print(f"  Done in {time.time()-t0:.1f} s")

    # ── RFC activity probability ───────────────────────────────────────────
    print("  Running RFC prediction …", flush=True)
    proba = rfc_model.predict_proba(fps)[:, 1]

    df = df.copy()
    df["activity_prob"] = proba

    hits = (
        df[df["activity_prob"] >= threshold]
        .copy()
        .sort_values("activity_prob", ascending=False)
        .reset_index(drop=True)
    )
    hits.index = hits.index + 1    # 1-based docking priority rank

    # ── Physicochemical descriptors for the hit list ───────────────────────
    print("  Computing physicochemical descriptors …")
    hits["mol_weight"]   = hits["mol"].apply(Descriptors.MolWt).round(3)
    hits["clogp"]        = hits["mol"].apply(Descriptors.MolLogP).round(3)
    hits["hba"]          = hits["mol"].apply(rdMolDescriptors.CalcNumHBA)
    hits["hbd"]          = hits["mol"].apply(rdMolDescriptors.CalcNumHBD)
    hits["tpsa"]         = hits["mol"].apply(rdMolDescriptors.CalcTPSA).round(2)
    hits["n_rot_bonds"]  = hits["mol"].apply(rdMolDescriptors.CalcNumRotatableBonds)
    hits["n_rings"]      = hits["mol"].apply(rdMolDescriptors.CalcNumRings)
    hits["n_arom_rings"] = hits["mol"].apply(rdMolDescriptors.CalcNumAromaticRings)

    # Enrichment factor vs structurally-filtered pool
    ef = n_screened / max(len(hits), 1)

    print(f"\n  Virtual screening results  (p >= {threshold})")
    print(f"  {'Metric':<35}  {'Value':>12}")
    print("  " + "-" * 50)
    print(f"  {'Input compounds':<35}  {n_screened:>12,}")
    print(f"  {'Hits retained':<35}  {len(hits):>12,}")
    print(f"  {'Enrichment factor':<35}  {ef:>12,.0f}x")
    print(f"  {'Top-hit activity prob':<35}  {hits['activity_prob'].iloc[0]:>12.3f}")
    print(f"  {'Mean hit activity prob':<35}  {hits['activity_prob'].mean():>12.3f}")

    return hits


# =============================================================================
# MANN-WHITNEY U ENRICHMENT TEST  (Section 3.3)
# =============================================================================

def mann_whitney_enrichment(ml_be: np.ndarray,
                             random_be: np.ndarray,
                             n_random: int = 200) -> dict:
    """
    One-sided Mann-Whitney U test confirming ML-selected compounds
    have significantly more negative docking energies than random draws.

    Parameters
    ----------
    ml_be     : binding energies of ML-selected compounds (kcal/mol)
    random_be : binding energies of randomly drawn compounds (kcal/mol)
    n_random  : size of random sample to draw

    Returns
    -------
    dict  {U, p_value, ml_mean, random_mean}
    """
    rng = np.random.default_rng(SEED)
    sample = rng.choice(random_be, size=min(n_random, len(random_be)),
                        replace=False)
    u_stat, p_val = stats.mannwhitneyu(ml_be, sample, alternative="less")

    print(f"\n  Mann-Whitney U test (ML-selected vs random draw):")
    print(f"    ML-selected mean BE  : {np.mean(ml_be):.2f} kcal/mol")
    print(f"    Random draw mean BE  : {np.mean(sample):.2f} kcal/mol")
    print(f"    U statistic          : {u_stat:.1f}")
    print(f"    p-value              : {p_val:.3e}")
    return {"U": u_stat, "p_value": p_val,
            "ml_mean": round(float(np.mean(ml_be)), 3),
            "random_mean": round(float(np.mean(sample)), 3)}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S12  –  ML-Based Virtual Screening  (p >= 0.60)")
    print("=" * 60)

    for req in [FILTERED_CSV, RFC_PKL]:
        if not req.exists():
            raise FileNotFoundError(
                f"Missing: {req}\n"
                "  Run S11_coconut_filters.py and S07 first."
            )

    # ── Load filtered COCONUT compounds ───────────────────────────────────
    print("\nLoading structurally filtered COCONUT compounds …")
    df_filtered = pd.read_csv(FILTERED_CSV)
    df_filtered["mol"] = df_filtered["smiles"].apply(sanitise_smiles)
    df_filtered.dropna(subset=["mol"], inplace=True)
    print(f"  {len(df_filtered):,} compounds loaded")

    # ── Load best RFC ─────────────────────────────────────────────────────
    with open(RFC_PKL, "rb") as fh:
        best_rfc = pickle.load(fh)
    print(f"  RFC model loaded from {RFC_PKL.name}")

    # ── Virtual screening ─────────────────────────────────────────────────
    if HITS_CSV.exists():
        print(f"\nCached hit list found: {HITS_CSV.name}")
        df_hits = pd.read_csv(HITS_CSV, index_col=0)
        print(f"  {len(df_hits):,} hits loaded")
    else:
        print("\nRunning virtual screen …")
        df_hits = virtual_screen(df_filtered, best_rfc, ML_PROB_CUTOFF)

        # Save without mol column
        df_hits.drop(columns=["mol"], errors="ignore").to_csv(HITS_CSV)
        print(f"\n  Hit list saved → {HITS_CSV}")

    # ── Summary file for docking workflow ────────────────────────────────
    lines = [
        "ML Virtual Screening Summary",
        "=" * 50,
        f"RFC model            : best_rfc_model.pkl",
        f"Probability cutoff   : {ML_PROB_CUTOFF}",
        f"Input compounds      : {len(df_filtered):,}",
        f"Hits retained        : {len(df_hits):,}",
        f"Enrichment factor    : {len(df_filtered)//max(len(df_hits),1):,}x",
        "",
        "Top-10 candidates for AutoDock Vina (PDB: 3NVW):",
        f"  Grid box: 25 x 25 x 25 A, centred on catalytic Mo site",
        "",
        f"{'Rank':<6}  {'COCONUT_ID':<15}  {'Prob':>6}  {'MW':>8}  {'cLogP':>7}",
        "-" * 50,
    ]
    for idx, row in df_hits.head(10).iterrows():
        lines.append(
            f"  {idx:<4}  {row['coconut_id']:<15}  "
            f"{row['activity_prob']:>6.3f}  "
            f"{row.get('mol_weight', 0):>8.2f}  "
            f"{row.get('clogp', 0):>7.3f}"
        )

    with open(SUMMARY_TXT, "w") as fh:
        fh.write("\n".join(lines))

    print(f"\n  Docking summary saved → {SUMMARY_TXT}")
    print("\nTop-10 hits for AutoDock Vina:")
    display_cols = [c for c in ["coconut_id","smiles","activity_prob",
                                "mol_weight","clogp","hba","hbd"]
                    if c in df_hits.columns]
    print(df_hits[display_cols].head(10).to_string())
    print(f"\nNext → run S13_visualisation.py")
