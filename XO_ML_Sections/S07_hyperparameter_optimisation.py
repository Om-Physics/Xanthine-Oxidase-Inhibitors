#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

XM_TRAIN    = OUTDIR / "Xm_train.npy"
Y_TRAIN     = OUTDIR / "y_train.npy"
RFC_PKL     = OUTDIR / "best_rfc_model.pkl"
SVC_PKL     = OUTDIR / "best_svc_model.pkl"
KNN_PKL     = OUTDIR / "best_knn_model.pkl"
GS_CSV      = OUTDIR / "hyperparameter_results.csv"

inner_cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)


# =============================================================================
# GRID DEFINITIONS
# =============================================================================

RFC_GRID = {
    "n_estimators"     : [100, 200, 300, 500],
    "max_depth"        : [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"     : ["sqrt", "log2", 0.5],
    "class_weight"     : [None, "balanced"],
}

SVC_GRID = {
    "C"     : [0.1, 1, 10, 100],
    "kernel": ["rbf", "poly"],
    "gamma" : ["scale", "auto"],
}

KNN_GRID = {
    "n_neighbors": [3, 5, 7, 11, 15, 21],
    "weights"    : ["uniform", "distance"],
    "metric"     : ["minkowski", "jaccard"],
}


# =============================================================================
# GRID SEARCH HELPER
# =============================================================================

def grid_search(estimator, param_grid: dict, X, y,
                pkl_path: Path, label: str):
    """
    Run GridSearchCV and return the best estimator.
    Loads from cache if pkl_path already exists.
    """
    if pkl_path.exists():
        print(f"  Cached model found: {pkl_path.name}")
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    print(f"\n  [GridSearchCV]  {label} …")
    gs = GridSearchCV(
        estimator  = estimator,
        param_grid = param_grid,
        cv         = inner_cv,
        scoring    = "roc_auc",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    gs.fit(X, y)
    best = gs.best_estimator_

    with open(pkl_path, "wb") as fh:
        pickle.dump(best, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV AUC : {gs.best_score_:.4f}")
    print(f"  Saved       → {pkl_path}")

    return best, gs.best_params_, round(gs.best_score_, 4)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S07  –  Hyperparameter Optimisation  (Tables S2–S3)")
    print("=" * 60)

    for req in [XM_TRAIN, Y_TRAIN]:
        if not req.exists():
            raise FileNotFoundError(f"Missing: {req}\n  Run S05_dataset_split.py first.")

    Xm_train = np.load(XM_TRAIN)
    y_train  = np.load(Y_TRAIN)

    summary_rows = []

    # ── RandomForestClassifier (Table S2) ────────────────────────────────
    result = grid_search(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        RFC_GRID, Xm_train, y_train,
        RFC_PKL, "RandomForestClassifier"
    )
    if isinstance(result, tuple):
        best_rfc, rfc_params, rfc_auc = result
        summary_rows.append({"Classifier": "RandomForestClassifier",
                              "Best_Params": str(rfc_params), "Best_CV_AUC": rfc_auc})
    else:
        best_rfc = result

    # ── Support Vector Classifier (Table S3) ─────────────────────────────
    result = grid_search(
        SVC(probability=True, random_state=SEED),
        SVC_GRID, Xm_train, y_train,
        SVC_PKL, "Support Vector Classifier"
    )
    if isinstance(result, tuple):
        best_svc, svc_params, svc_auc = result
        summary_rows.append({"Classifier": "Support Vector Classifier",
                              "Best_Params": str(svc_params), "Best_CV_AUC": svc_auc})
    else:
        best_svc = result

    # ── k-Nearest Neighbours ─────────────────────────────────────────────
    result = grid_search(
        KNeighborsClassifier(n_jobs=-1),
        KNN_GRID, Xm_train, y_train,
        KNN_PKL, "k-Nearest Neighbours"
    )
    if isinstance(result, tuple):
        best_knn, knn_params, knn_auc = result
        summary_rows.append({"Classifier": "k-Nearest Neighbours",
                              "Best_Params": str(knn_params), "Best_CV_AUC": knn_auc})
    else:
        best_knn = result

    # ── Save summary CSV ─────────────────────────────────────────────────
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(GS_CSV, index=False)
        print(f"\nGrid search summary saved → {GS_CSV}")

    print(f"\nNext → run S08_ensemble.py")
