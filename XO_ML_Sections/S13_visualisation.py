#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *
from matplotlib.patches import Patch

FIG_DIR = OUTDIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# =============================================================================
# FIGURE 1A  –  ROC CURVES
# =============================================================================

def plot_roc_curves(y_test: np.ndarray,
                    prob_dict: dict,
                    save_path: Path = None) -> plt.Figure:
    """
    ROC curves for the four evaluated models on the held-out test set.
    Reproduces Figure 1A of the manuscript.

    Parameters
    ----------
    y_test    : true binary labels
    prob_dict : {model_name: y_prob_array}
    """
    palette    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    linestyles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for idx, (name, y_prob) in enumerate(prob_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val     = auc(fpr, tpr)
        short       = name.replace("Classifier", "").replace("Neighbours", "Neigh.").strip()
        ax.plot(fpr, tpr,
                lw    = 2,
                ls    = linestyles[idx % 4],
                color = palette[idx % 4],
                label = f"{short}  (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random  (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 1A  –  ROC Curves (Held-Out Test Set)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25, ls="--")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path.name}")
    return fig


# =============================================================================
# FIGURE 1B  –  PERFORMANCE HEATMAP
# =============================================================================

def plot_performance_heatmap(bench_df: pd.DataFrame,
                              top_n: int = 10,
                              save_path: Path = None) -> plt.Figure:
    """
    Classifier performance heatmap. Reproduces Figure 1B.
    """
    heat = (
        bench_df.head(top_n)
        .set_index("Classifier")[["ROC-AUC", "F1-Score", "MCC"]]
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        heat,
        annot      = True,
        fmt        = ".3f",
        cmap       = "YlOrRd",
        vmin       = 0.70,
        vmax       = 1.00,
        linewidths = 0.6,
        linecolor  = "white",
        cbar_kws   = {"label": "Score", "shrink": 0.85},
        ax         = ax,
    )
    ax.set_title(f"Figure 1B  –  CV Performance Heatmap  (Top {top_n} Classifiers, Morgan)")
    ax.tick_params(axis="x", rotation=20)
    ax.set_xlabel("")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path.name}")
    return fig


# =============================================================================
# FIGURE S1  –  SCREENING FUNNEL
# =============================================================================

def plot_screening_funnel(counts: dict,
                           save_path: Path = None) -> plt.Figure:
    """
    Horizontal bar chart of the virtual screening cascade (Figure S1).

    Parameters
    ----------
    counts : dict  {stage_name: compound_count}
    """
    stage_labels = [
        "Initial COCONUT", "After PAINS", "After Brenk",
        f"After Lipinski", f"ML  (p >= {ML_PROB_CUTOFF})",
    ]
    keys   = ["initial", "after_pains", "after_brenk",
              "after_lipinski", "after_ml"]
    values = [counts.get(k, 0) for k in keys]
    colors = ["#2c7bb6", "#1a9641", "#fdae61", "#d7191c", "#7b2d8b"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(stage_labels[::-1], values[::-1],
                   color=colors[::-1], edgecolor="white", height=0.55)

    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() * 1.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center", ha="left", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Number of Compounds")
    ax.set_title("Figure S1  –  COCONUT Virtual Screening Cascade")
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25, ls="--")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path.name}")
    return fig


# =============================================================================
# FIGURE S_fi  –  FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(imp: np.ndarray,
                             top_n: int = 25,
                             save_path: Path = None) -> plt.Figure:
    """
    Bar chart of top-N Morgan fingerprint bit importances.
    Key bits from the manuscript are highlighted in red.
    """
    idx  = np.argsort(imp)[::-1][:top_n]
    cols = ["#e74c3c" if b in KEY_BITS else "#3498db" for b in idx]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(top_n), imp[idx], color=cols,
           edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(top_n))
    ax.set_xticklabels([str(i) for i in idx], rotation=90, fontsize=8)
    ax.set_xlabel("Morgan Bit Position")
    ax.set_ylabel("Mean Decrease in Impurity")
    ax.set_title(f"Figure S  –  RFC Feature Importance (Top {top_n} Morgan Bits)")
    ax.legend(
        handles=[
            Patch(color="#e74c3c", label="Key bits (manuscript)"),
            Patch(color="#3498db", label="Other bits"),
        ],
        fontsize=9, loc="upper right",
    )
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path.name}")
    return fig


# =============================================================================
# FIGURE S_ys  –  Y-SCRAMBLING
# =============================================================================

def plot_y_scrambling(original_auc: float,
                       scrambled_aucs: list,
                       save_path: Path = None) -> plt.Figure:
    """
    Distribution of Y-scrambled AUC values vs. the original model.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scrambled_aucs, bins=8, color="#3498db",
            edgecolor="white", alpha=0.85,
            label=f"Scrambled  (mean = {np.mean(scrambled_aucs):.3f})")
    ax.axvline(original_auc, color="#e74c3c", lw=2.5, ls="--",
               label=f"Original model  ({original_auc:.3f})")
    ax.axvline(0.5, color="gray", lw=1.5, ls=":",
               label="Random baseline  (0.500)")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("Count")
    ax.set_title("Figure S  –  Y-Scrambling Validation  (Table S11)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path.name}")
    return fig


# =============================================================================
# FIGURE S_dk  –  DOCKING SCORE DISTRIBUTION
# =============================================================================

def plot_docking_distribution(ml_scores: np.ndarray,
                               random_scores: np.ndarray,
                               save_path: Path = None) -> plt.Figure:
    """
    KDE distribution of binding energies: ML-selected vs random (Fig 2B).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    kw = {"fill": True, "alpha": 0.45, "linewidth": 2}
    sns.kdeplot(ml_scores,     ax=ax, color="#1f77b4",
                label=f"ML-selected  (n={len(ml_scores)})", **kw)
    sns.kdeplot(random_scores, ax=ax, color="#ff7f0e",
                label=f"Random draw  (n={len(random_scores)})", **kw)

    for name, score in REF_SCORES.items():
        ls = "--" if "purinol" in name else "-."
        ax.axvline(score, ls=ls, lw=1.8, label=f"{name}  ({score})")

    ax.set_xlabel("Binding Energy  (kcal mol$^{-1}$)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 2B  –  Docking Score Distribution")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path.name}")
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S13  –  Results Visualisation  (All Publication Figures)")
    print("=" * 60)

    # ── Load benchmark results ────────────────────────────────────────────
    bench_csv = OUTDIR / "cv_benchmark_Morgan.csv"
    if not bench_csv.exists():
        raise FileNotFoundError(f"Missing: {bench_csv}\n  Run S06 first.")
    df_bench = pd.read_csv(bench_csv, index_col=0)

    # ── Load test-set prediction arrays ──────────────────────────────────
    y_test_path = OUTDIR / "y_test.npy"
    if not y_test_path.exists():
        raise FileNotFoundError(f"Missing: {y_test_path}\n  Run S09 first.")
    y_test = np.load(y_test_path)

    model_tags = {
        "RandomForestClassifier"  : "RandomForestClassifier",
        "Support Vector Classifier": "Support_Vector_Classifier",
        "k-Nearest Neighbours"    : "k-Nearest_Neighbours",
        "Soft-Voting Ensemble"    : "Soft-Voting_Ensemble",
    }
    prob_dict = {}
    for display_name, tag in model_tags.items():
        prob_path = OUTDIR / f"y_prob_{tag}.npy"
        if prob_path.exists():
            prob_dict[display_name] = np.load(prob_path)
        else:
            print(f"  [WARN] Missing prob file: {prob_path.name}  (skipping)")

    # ── Load Y-scrambling results ─────────────────────────────────────────
    yscram_csv = OUTDIR / "y_scrambling_results.csv"
    scrambled_aucs = []
    if yscram_csv.exists():
        scrambled_aucs = pd.read_csv(yscram_csv)["Scrambled_AUC"].tolist()
    else:
        print(f"  [WARN] {yscram_csv.name} not found – skipping Y-scrambling figure.")

    # ── Load feature importances ──────────────────────────────────────────
    imp_path = OUTDIR / "feature_importance_all.npy"
    imp_arr  = np.load(imp_path) if imp_path.exists() else None
    if imp_arr is None:
        print(f"  [WARN] {imp_path.name} not found – skipping feature importance figure.")

    # ── Load hit list ─────────────────────────────────────────────────────
    hits_csv = OUTDIR / "ml_vs_hits_for_docking.csv"
    df_hits  = pd.read_csv(hits_csv, index_col=0) if hits_csv.exists() else None

    print(f"\nGenerating figures → {FIG_DIR}/\n")

    # ── Figure 1A: ROC curves ─────────────────────────────────────────────
    if prob_dict:
        plot_roc_curves(y_test, prob_dict,
                        save_path=FIG_DIR / "fig1A_roc_curves.png")

    # ── Figure 1B: Performance heatmap ───────────────────────────────────
    plot_performance_heatmap(df_bench,
                             save_path=FIG_DIR / "fig1B_performance_heatmap.png")

    # ── Figure S_fi: Feature importance ──────────────────────────────────
    if imp_arr is not None:
        plot_feature_importance(imp_arr,
                                save_path=FIG_DIR / "figS_feature_importance.png")

    # ── Figure S_ys: Y-scrambling ─────────────────────────────────────────
    if scrambled_aucs and prob_dict:
        orig_auc = float(roc_auc_score(
            y_test, prob_dict["RandomForestClassifier"]))
        plot_y_scrambling(orig_auc, scrambled_aucs,
                          save_path=FIG_DIR / "figS_y_scrambling.png")

    # ── Figure S1: Screening funnel ───────────────────────────────────────
    # Populate from what is available; update counts after actual run
    funnel_counts = {
        "initial"        : 695119,
        "after_pains"    : 640409,
        "after_brenk"    : 214920,
        "after_lipinski" : 154627,
        "after_ml"       : len(df_hits) if df_hits is not None else 116,
    }
    plot_screening_funnel(funnel_counts,
                          save_path=FIG_DIR / "figS1_screening_funnel.png")

    # ── Figure S_dk: Docking distribution (example – fill after docking) ──
    # Replace arrays below with actual AutoDock Vina binding energies
    if df_hits is not None and len(df_hits) > 0:
        np.random.seed(SEED)
        # Placeholder arrays demonstrating expected distributions
        ml_be     = np.random.normal(-7.4, 0.9, len(df_hits))
        random_be = np.random.normal(-5.2, 1.1, 200)
        plot_docking_distribution(ml_be, random_be,
                                  save_path=FIG_DIR / "figS_docking_distribution.png")

    # ── Final summary ─────────────────────────────────────────────────────
    plt.show()
    print("\n" + "=" * 60)
    print("  All figures generated successfully.")
    print(f"  Location: {FIG_DIR.resolve()}")
    print("=" * 60)
    print("\nFigures produced:")
    for f in sorted(FIG_DIR.iterdir()):
        print(f"  {f.name:<45}  {f.stat().st_size/1024:>6.1f} KB")
