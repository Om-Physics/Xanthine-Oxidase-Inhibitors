#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

XM_TRAIN         = OUTDIR / "Xm_train.npy"
XC_TRAIN         = OUTDIR / "Xc_train.npy"
Y_TRAIN          = OUTDIR / "y_train.npy"
BENCH_MORGAN_CSV = OUTDIR / "cv_benchmark_Morgan.csv"
BENCH_MACCS_CSV  = OUTDIR / "cv_benchmark_MACCS.csv"


# =============================================================================
# CLASSIFIER SUITE  (Table S1)
# =============================================================================

def build_classifier_suite() -> dict:
    """
    All 20 classifiers evaluated in the benchmark.
    Ordering follows Table S1 sorted by expected ROC-AUC.
    """
    suite = {
        # Tree-based ensembles
        "RandomForestClassifier"       : RandomForestClassifier(n_jobs=-1, random_state=SEED),
        "Extra Trees Classifier"       : ExtraTreesClassifier(n_jobs=-1, random_state=SEED),
        "Gradient Boosting Classifier" : GradientBoostingClassifier(random_state=SEED),
        "Bagging Classifier"           : BaggingClassifier(n_jobs=-1, random_state=SEED),
        "AdaBoost Classifier"          : AdaBoostClassifier(random_state=SEED),
        # Kernel / distance
        "Support Vector Classifier"    : SVC(probability=True, random_state=SEED),
        "k-Nearest Neighbours"         : KNeighborsClassifier(n_jobs=-1),
        # Linear methods – calibrated wrappers for predict_proba
        "Logistic Regression"          : LogisticRegression(max_iter=2000, n_jobs=-1, random_state=SEED),
        "Linear SVC"                   : CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=SEED), cv=3),
        "SGD Classifier"               : CalibratedClassifierCV(SGDClassifier(max_iter=1000, random_state=SEED), cv=3),
        "Ridge Classifier"             : CalibratedClassifierCV(RidgeClassifier(), cv=3),
        "Passive Aggressive"           : CalibratedClassifierCV(PassiveAggressiveClassifier(max_iter=1000, random_state=SEED), cv=3),
        # Neural network
        "MLP Classifier"               : MLPClassifier(max_iter=1000, random_state=SEED),
        # Probabilistic / discriminant
        "Bernoulli Naive Bayes"        : BernoulliNB(),
        "Gaussian Naive Bayes"         : GaussianNB(),
        "LDA"                          : LDA(),
        "Quadratic Discriminant"       : QDA(),
    }
    # Optional gradient-boosting libraries
    if _LGBM:
        suite["LightGBM"] = LGBMClassifier(n_jobs=-1, random_state=SEED, verbose=-1)
    if _XGB:
        suite["XGBoost"]  = XGBClassifier(n_jobs=-1, random_state=SEED,
                                           eval_metric="logloss", verbosity=0)
    if _CAT:
        suite["CatBoost"] = CatBoostClassifier(random_seed=SEED, verbose=0, thread_count=-1)

    return suite


# =============================================================================
# BENCHMARK FUNCTION
# =============================================================================

def run_cv_benchmark(X: np.ndarray, y: np.ndarray,
                     fp_label: str) -> pd.DataFrame:
    """
    Run the 25 × 5-fold CV benchmark for all classifiers.

    Parameters
    ----------
    X        : fingerprint matrix (train set)
    y        : label array (train set)
    fp_label : "Morgan" or "MACCS" – used in console output and filename

    Returns
    -------
    pd.DataFrame sorted by ROC-AUC (descending)
    """
    cv_scheme   = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED
    )
    classifiers = build_classifier_suite()
    records     = []

    print(f"\n  Descriptor : {fp_label}  |  "
          f"{CV_REPEATS} × {CV_SPLITS}-fold CV  |  {len(classifiers)} classifiers")
    print(f"  {'Classifier':<37}  {'ROC-AUC':>8}  {'F1':>6}  {'MCC':>6}  {'Time':>7}")
    print("  " + "-" * 65)

    for name, clf in classifiers.items():
        auc_vals, f1_vals, mcc_vals = [], [], []
        t_start = time.time()

        for tr_idx, va_idx in cv_scheme.split(X, y):
            clf_ = clone(clf)
            clf_.fit(X[tr_idx], y[tr_idx])
            yp   = clf_.predict_proba(X[va_idx])[:, 1]
            ypred= clf_.predict(X[va_idx])
            auc_vals.append(roc_auc_score(y[va_idx], yp))
            f1_vals.append(f1_score(y[va_idx], ypred, zero_division=0))
            mcc_vals.append(matthews_corrcoef(y[va_idx], ypred))

        elapsed = time.time() - t_start
        row = {
            "Classifier"  : name,
            "ROC-AUC"     : round(np.mean(auc_vals), 3),
            "ROC-AUC SD"  : round(np.std(auc_vals),  3),
            "F1-Score"    : round(np.mean(f1_vals),  3),
            "F1-Score SD" : round(np.std(f1_vals),   3),
            "MCC"         : round(np.mean(mcc_vals), 3),
            "MCC SD"      : round(np.std(mcc_vals),  3),
        }
        records.append(row)
        print(f"  {name:<37}  {row['ROC-AUC']:>8.3f}  "
              f"{row['F1-Score']:>6.3f}  {row['MCC']:>6.3f}  {elapsed:>6.0f}s")

    df_bench = (
        pd.DataFrame(records)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )
    df_bench.index = df_bench.index + 1    # 1-based rank

    out_path = OUTDIR / f"cv_benchmark_{fp_label}.csv"
    df_bench.to_csv(out_path)
    print(f"\n  Benchmark results saved → {out_path}")
    return df_bench


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S06  –  Twenty-Classifier Benchmark  (25 × 5-Fold CV)")
    print("=" * 60)

    for req in [XM_TRAIN, XC_TRAIN, Y_TRAIN]:
        if not req.exists():
            raise FileNotFoundError(f"Missing: {req}\n  Run S05_dataset_split.py first.")

    Xm_train = np.load(XM_TRAIN)
    Xc_train = np.load(XC_TRAIN)
    y_train  = np.load(Y_TRAIN)

    # Morgan fingerprints (primary descriptor)
    if BENCH_MORGAN_CSV.exists():
        print(f"\nCached Morgan benchmark found: {BENCH_MORGAN_CSV.name}")
        df_bench_morgan = pd.read_csv(BENCH_MORGAN_CSV, index_col=0)
    else:
        print("\n[Morgan fingerprints]")
        df_bench_morgan = run_cv_benchmark(Xm_train, y_train, "Morgan")

    # MACCS keys (comparison)
    if BENCH_MACCS_CSV.exists():
        print(f"Cached MACCS benchmark found: {BENCH_MACCS_CSV.name}")
        df_bench_maccs = pd.read_csv(BENCH_MACCS_CSV, index_col=0)
    else:
        print("\n[MACCS structural keys]")
        df_bench_maccs  = run_cv_benchmark(Xc_train, y_train, "MACCS")

    print("\n── Top-5 Morgan results (Table S1) ──")
    print(df_bench_morgan[["Classifier","ROC-AUC","F1-Score","MCC"]].head(5).to_string())
    print("\n── Top-5 MACCS results (Table S1) ──")
    print(df_bench_maccs[["Classifier","ROC-AUC","F1-Score","MCC"]].head(5).to_string())
    print(f"\nNext → run S07_hyperparameter_optimisation.py")
