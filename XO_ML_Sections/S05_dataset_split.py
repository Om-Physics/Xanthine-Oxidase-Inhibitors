#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from S01_config import *

MORGAN_NPY  = OUTDIR / "morgan_matrix.npy"
MACCS_NPY   = OUTDIR / "maccs_matrix.npy"
LABELS_NPY  = OUTDIR / "labels.npy"

XM_TRAIN    = OUTDIR / "Xm_train.npy"
XM_TEST     = OUTDIR / "Xm_test.npy"
XC_TRAIN    = OUTDIR / "Xc_train.npy"
XC_TEST     = OUTDIR / "Xc_test.npy"
Y_TRAIN     = OUTDIR / "y_train.npy"
Y_TEST      = OUTDIR / "y_test.npy"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S05  –  Dataset Preparation  (80 / 20 Stratified Split)")
    print("=" * 60)

    for req in [MORGAN_NPY, MACCS_NPY, LABELS_NPY]:
        if not req.exists():
            raise FileNotFoundError(
                f"Required file missing: {req}\n"
                "  Run S04_fingerprints.py first."
            )

    if all(p.exists() for p in [XM_TRAIN, XM_TEST, XC_TRAIN, XC_TEST, Y_TRAIN, Y_TEST]):
        print("\nCached split arrays found – loading …")
        Xm_train = np.load(XM_TRAIN)
        Xm_test  = np.load(XM_TEST)
        Xc_train = np.load(XC_TRAIN)
        Xc_test  = np.load(XC_TEST)
        y_train  = np.load(Y_TRAIN)
        y_test   = np.load(Y_TEST)
    else:
        print("\nLoading fingerprint matrices …")
        morgan_matrix = np.load(MORGAN_NPY)
        maccs_matrix  = np.load(MACCS_NPY)
        labels        = np.load(LABELS_NPY)

        print(f"  Morgan shape : {morgan_matrix.shape}")
        print(f"  MACCS  shape : {maccs_matrix.shape}")
        print(f"  Labels       : Active={labels.sum()}  Inactive={(1-labels).sum()}")

        print(f"\nSplitting  (stratified, test={TEST_FRAC}, seed={SEED}) …")
        (Xm_train, Xm_test,
         Xc_train, Xc_test,
         y_train,  y_test) = train_test_split(
            morgan_matrix, maccs_matrix, labels,
            test_size    = TEST_FRAC,
            stratify     = labels,
            random_state = SEED,
        )

        for arr, path in [(Xm_train, XM_TRAIN), (Xm_test,  XM_TEST),
                          (Xc_train, XC_TRAIN), (Xc_test,  XC_TEST),
                          (y_train,  Y_TRAIN),  (y_test,   Y_TEST)]:
            np.save(path, arr)

    print(f"\n  Train: {Xm_train.shape[0]:,} molecules"
          f"  (Active={y_train.sum()}  Inactive={(1-y_train).sum()})")
    print(f"  Test : {Xm_test.shape[0]:,}  molecules"
          f"  (Active={y_test.sum()}   Inactive={(1-y_test).sum()})")
    print("\nAll split arrays saved to XO_ML_Results/")
    print(f"\nNext → run S06_classifier_benchmark.py")
