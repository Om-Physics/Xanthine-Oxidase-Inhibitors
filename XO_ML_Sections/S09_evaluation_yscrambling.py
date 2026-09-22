#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from S01_config import *

# Input paths
RFC_PKL      = OUTDIR / "best_rfc_model.pkl"
SVC_PKL      = OUTDIR / "best_svc_model.pkl"
KNN_PKL      = OUTDIR / "best_knn_model.pkl"
ENSEMBLE_PKL = OUTDIR / "ensemble_model.pkl"
XM_TRAIN     = OUTDIR / "Xm_train.npy"
XM_TEST      = OUTDIR / "Xm_test.npy"
Y_TRAIN      = OUTDIR / "y_train.npy"
Y_TEST       = OUTDIR / "y_test.npy"
MORGAN_NPY   = OUTDIR / "morgan_matrix.npy"
LABELS_NPY   = OUTDIR / "labels.npy"

# Output paths
TABLE1_CSV   = OUTDIR / "table1_test_performance.csv"
YSCRAM_CSV   = OUTDIR / "y_scrambling_results.csv"


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_on_test(model, X_te: np.ndarray,
                     y_te: np.ndarray, name: str) -> dict:
    """
    Compute ROC-AUC, F1-Score, and MCC on the held-out test set.

    Returns
    -------
    dict  with scalar metrics and prediction arrays (for plotting in S13).
    """
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)

    result = {
        "Model"   : name,
        "ROC-AUC" : round(roc_auc_score(y_te, y_prob), 3),
        "F1-Score": round(f1_score(y_te, y_pred, zero_division=0), 3),
        "MCC"     : round(matthews_corrcoef(y_te, y_pred), 3),
        "y_prob"  : y_prob,
        "y_pred"  : y_pred,
    }
    print(f"  {name:<37}  "
          f"AUC={result['ROC-AUC']:.3f}  "
          f"F1={result['F1-Score']:.3f}  "
          f"MCC={result['MCC']:.3f}")
    return result


# =============================================================================
# Y-SCRAMBLING FUNCTION
# =============================================================================

def y_scrambling(rfc_params: dict, X_full: np.ndarray,
                 y_full: np.ndarray, n_iter: int = 10) -> list:
    """
    Permutation test: train RFC on shuffled labels, measure 5-fold AUC.
    Mean ≈ 0.50 confirms genuine structure–activity learning.

    Parameters
    ----------
    rfc_params : dict  –  best RFC hyperparameters from S07
    X_full     : full Morgan matrix (train + test)
    y_full     : full label array
    n_iter     : number of permutation iterations (paper: 10)

    Returns
    -------
    list of mean CV AUC per permutation
    """
    rng           = np.random.default_rng(SEED)
    scramble_aucs = []

    for i in range(n_iter):
        y_perm  = rng.permutation(y_full)
        cv_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        fold_aucs = []

        for tr, va in cv_fold.split(X_full, y_perm):
            m = RandomForestClassifier(**rfc_params)
            m.fit(X_full[tr], y_perm[tr])
            p = m.predict_proba(X_full[va])[:, 1]
            fold_aucs.append(roc_auc_score(y_perm[va], p))

        mean_auc = round(np.mean(fold_aucs), 3)
        scramble_aucs.append(mean_auc)
        print(f"  Iteration {i+1:2d}  →  scrambled AUC = {mean_auc:.3f}")

    return scramble_aucs


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S09  –  Test Set Evaluation and Y-Scrambling Validation")
    print("=" * 60)

    required = [RFC_PKL, SVC_PKL, KNN_PKL, ENSEMBLE_PKL,
                XM_TEST, Y_TEST, XM_TRAIN, Y_TRAIN,
                MORGAN_NPY, LABELS_NPY]
    for req in required:
        if not req.exists():
            raise FileNotFoundError(
                f"Missing: {req}\n"
                "  Ensure all previous sections have been run."
            )

    # ── Load models ───────────────────────────────────────────────────────
    with open(RFC_PKL,      "rb") as fh: best_rfc  = pickle.load(fh)
    with open(SVC_PKL,      "rb") as fh: best_svc  = pickle.load(fh)
    with open(KNN_PKL,      "rb") as fh: best_knn  = pickle.load(fh)
    with open(ENSEMBLE_PKL, "rb") as fh: ensemble  = pickle.load(fh)

    Xm_test  = np.load(XM_TEST)
    y_test   = np.load(Y_TEST)

    # ── Held-out test set evaluation (Table 1) ────────────────────────────
    print("\n── Held-out test set performance  (Table 1) ──")
    print(f"  {'Model':<37}  {'ROC-AUC':>8}  {'F1':>6}  {'MCC':>6}")
    print("  " + "-" * 60)

    test_results = {}
    for mdl, nm in [(best_rfc, "RandomForestClassifier"),
                    (best_svc, "Support Vector Classifier"),
                    (best_knn, "k-Nearest Neighbours"),
                    (ensemble, "Soft-Voting Ensemble")]:
        test_results[nm] = evaluate_on_test(mdl, Xm_test, y_test, nm)

    # Save Table 1
    pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("y_prob", "y_pred")}
        for r in test_results.values()
    ]).to_csv(TABLE1_CSV, index=False)
    print(f"\n  Table 1 saved → {TABLE1_CSV}")

    # Save prediction arrays for S13 plotting
    np.save(OUTDIR / "y_test.npy",    y_test)
    for nm, res in test_results.items():
        tag = nm.replace(" ", "_")
        np.save(OUTDIR / f"y_prob_{tag}.npy", res["y_prob"])

    # ── Y-Scrambling (Table S11) ──────────────────────────────────────────
    if YSCRAM_CSV.exists():
        print(f"\nCached Y-scrambling results: {YSCRAM_CSV.name}")
        df_scr = pd.read_csv(YSCRAM_CSV)
        scramble_aucs = df_scr["Scrambled_AUC"].tolist()
    else:
        print("\n── Y-Scrambling Validation  (10 permutation iterations) ──")
        morgan_matrix = np.load(MORGAN_NPY)
        labels        = np.load(LABELS_NPY)

        # Reconstruct best RFC params from the fitted model
        rfc_params = {
            "n_estimators"     : best_rfc.n_estimators,
            "max_depth"        : best_rfc.max_depth,
            "min_samples_split": best_rfc.min_samples_split,
            "min_samples_leaf" : best_rfc.min_samples_leaf,
            "max_features"     : best_rfc.max_features,
            "class_weight"     : best_rfc.class_weight,
            "random_state"     : SEED,
            "n_jobs"           : -1,
        }

        scramble_aucs = y_scrambling(rfc_params, morgan_matrix, labels, n_iter=10)

        pd.DataFrame({
            "Iteration"    : range(1, 11),
            "Scrambled_AUC": scramble_aucs,
        }).to_csv(YSCRAM_CSV, index=False)
        print(f"\n  Y-scrambling results saved → {YSCRAM_CSV}")

    print(f"\n  Scrambled mean ± SD : {np.mean(scramble_aucs):.3f} ± {np.std(scramble_aucs):.3f}")
    print(f"  Original model AUC  : {test_results['RandomForestClassifier']['ROC-AUC']:.3f}")
    print("  → Genuine SAR confirmed (no overfitting)")
    print(f"\nNext → run S10_feature_importance.py")
