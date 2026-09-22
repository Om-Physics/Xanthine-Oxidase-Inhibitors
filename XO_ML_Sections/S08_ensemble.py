#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

RFC_PKL      = OUTDIR / "best_rfc_model.pkl"
SVC_PKL      = OUTDIR / "best_svc_model.pkl"
KNN_PKL      = OUTDIR / "best_knn_model.pkl"
XM_TRAIN     = OUTDIR / "Xm_train.npy"
Y_TRAIN      = OUTDIR / "y_train.npy"
ENSEMBLE_PKL = OUTDIR / "ensemble_model.pkl"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S08  –  Soft-Voting Ensemble Construction")
    print("=" * 60)

    for req in [RFC_PKL, SVC_PKL, KNN_PKL, XM_TRAIN, Y_TRAIN]:
        if not req.exists():
            raise FileNotFoundError(
                f"Missing: {req}\n"
                "  Run S07_hyperparameter_optimisation.py first."
            )

    print("\nLoading tuned base classifiers …")
    with open(RFC_PKL, "rb") as fh:
        best_rfc = pickle.load(fh)
    with open(SVC_PKL, "rb") as fh:
        best_svc = pickle.load(fh)
    with open(KNN_PKL, "rb") as fh:
        best_knn = pickle.load(fh)

    Xm_train = np.load(XM_TRAIN)
    y_train  = np.load(Y_TRAIN)

    if ENSEMBLE_PKL.exists():
        print(f"Cached ensemble found: {ENSEMBLE_PKL.name}")
        with open(ENSEMBLE_PKL, "rb") as fh:
            ensemble = pickle.load(fh)
    else:
        print("\nFitting soft-voting ensemble (RFC + SVC + kNN) …")
        ensemble = VotingClassifier(
            estimators=[
                ("RFC", best_rfc),
                ("SVC", best_svc),
                ("kNN", best_knn),
            ],
            voting = "soft",
            n_jobs = -1,
        )
        ensemble.fit(Xm_train, y_train)

        with open(ENSEMBLE_PKL, "wb") as fh:
            pickle.dump(ensemble, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Ensemble fitted and saved → {ENSEMBLE_PKL}")

    print(f"\nEnsemble members: {[name for name, _ in ensemble.estimators]}")
    print(f"Voting strategy : {ensemble.voting}")
    print(f"\nNext → run S09_evaluation_yscrambling.py")
