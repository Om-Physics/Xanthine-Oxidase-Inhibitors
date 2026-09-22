#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import time
import pickle
import warnings
from pathlib import Path

# ── Scientific stack ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

# ── scikit-learn core ─────────────────────────────────────────────────────────
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RepeatedStratifiedKFold, GridSearchCV,
)
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef,
    roc_curve, auc, classification_report,
)
from sklearn.calibration import CalibratedClassifierCV

# ── scikit-learn classifiers ──────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, BaggingClassifier,
    AdaBoostClassifier, VotingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier,
    RidgeClassifier, PassiveAggressiveClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Optional gradient-boosting libraries ─────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    _LGBM = True
except ImportError:
    _LGBM = False

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False

try:
    from catboost import CatBoostClassifier
    _CAT = True
except ImportError:
    _CAT = False

try:
    from chembl_webresource_client.new_client import new_client as _chembl_api
    _CHEMBL = True
except ImportError:
    _CHEMBL = False

warnings.filterwarnings("ignore")

# =============================================================================
# PROJECT-WIDE CONSTANTS  (match manuscript Section 2.1 exactly)
# =============================================================================
SEED             = 57       # random state throughout
TEST_FRAC        = 0.20     # 80/20 stratified train-test split
CV_SPLITS        = 5        # k in stratified k-fold
CV_REPEATS       = 25       # repetitions for repeated CV
IC50_CUTOFF_UM   = 1.0      # µM  – IC50 ≤ 1.0 → Active label
ML_PROB_CUTOFF   = 0.60     # minimum RFC probability for VS hit
MORGAN_RADIUS    = 3        # Morgan circular fingerprint radius
MORGAN_BITS      = 2048     # Morgan fingerprint bit-length
MACCS_BITS       = 166      # MACCS structural keys (bits 1-166)

# Key Morgan bit positions identified in manuscript feature analysis
KEY_BITS = {
    378: "aza-arene ring N (pyridine/pyrimidine)",
    285: "pyrimidine-imine linkage",
    507: "aryl-aryl connectivity",
    323: "conjugated pyrido/pyrimidine bridge",
    420: "oxadiazole substructure",
}

# Reference XO inhibitor docking scores (kcal/mol) for comparison
REF_SCORES = {
    "Topiroxostat": -9.2,
    "Febuxostat":   -9.8,
    "Allopurinol":  -6.4,
    "Xanthine":     -5.8,
}

# Output directory – all section files write here
OUTDIR = Path("XO_ML_Results")
OUTDIR.mkdir(exist_ok=True)

# Matplotlib styling
matplotlib.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.labelsize" : 12,
    "axes.titlesize" : 13,
    "legend.fontsize": 9,
    "figure.dpi"     : 150,
})
np.random.seed(SEED)

# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================

def sanitise_smiles(smi: str):
    """Return a sanitised RDKit Mol or None if the SMILES is invalid."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def lipinski_ro5(mol, max_violations: int = 1) -> bool:
    """
    Lipinski Rule-of-Five filter.

    max_violations=1  during curation  (Section 2.1)
    max_violations=0  for VS screening (Section 2.2 / Section 11)
    """
    n = sum([
        Descriptors.MolWt(mol)          > 500,
        Descriptors.MolLogP(mol)         >   5,
        rdMolDescriptors.CalcNumHBA(mol) >  10,
        rdMolDescriptors.CalcNumHBD(mol) >   5,
    ])
    return n <= max_violations


def mol_to_morgan(mol, radius: int = MORGAN_RADIUS,
                  n_bits: int = MORGAN_BITS) -> np.ndarray:
    """Morgan circular fingerprint → 1-D uint8 array."""
    fp  = AllChem.GetMorganFingerprintAsBitVect(
              mol, radius=radius, nBits=n_bits, useFeatures=False)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def mol_to_maccs(mol) -> np.ndarray:
    """MACCS structural keys → 166-bit uint8 array."""
    fp  = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr[1:]      # drop unused index-0 bit


def _make_filter_catalog(flag) -> FilterCatalog:
    """Build an RDKit FilterCatalog for PAINS or Brenk."""
    p = FilterCatalogParams()
    p.AddCatalog(flag)
    return FilterCatalog(p)


# Build once at import time (thread-safe, reused across sections)
PAINS_CATALOG = _make_filter_catalog(FilterCatalogParams.FilterCatalogs.PAINS)
BRENK_CATALOG = _make_filter_catalog(FilterCatalogParams.FilterCatalogs.BRENK)


def passes_pains(mol) -> bool:
    """True if the molecule has NO PAINS alert."""
    return PAINS_CATALOG.GetFirstMatch(mol) is None


def passes_brenk(mol) -> bool:
    """True if the molecule has NO Brenk alert."""
    return BRENK_CATALOG.GetFirstMatch(mol) is None


if __name__ == "__main__":
    print("S01_config.py loaded successfully.")
    print(f"  Output directory : {OUTDIR.resolve()}")
    print(f"  SEED={SEED}  |  MORGAN_BITS={MORGAN_BITS}  |  ML_PROB_CUTOFF={ML_PROB_CUTOFF}")
    print(f"  LightGBM={'yes' if _LGBM else 'no'}  "
          f"XGBoost={'yes' if _XGB else 'no'}  "
          f"CatBoost={'yes' if _CAT else 'no'}  "
          f"ChEMBL={'yes' if _CHEMBL else 'no'}")
