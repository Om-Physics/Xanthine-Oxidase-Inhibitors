# XO ML Pipeline  –  Section Files
## Machine Learning-Driven Discovery of Natural Xanthine Oxidase Inhibitors

**Authors:** Om Jha, Dhirendra Prasad Upadhyay, Arjun Acharya  
**Correspondence:** arjun.acharya@trc.tu.edu.np  
**Repository:** https://github.com/Om-Physics/Xanthine-Oxidase-Inhibitors

---

## File Map

| File | Section | Purpose | Input | Output |
|---|---|---|---|---|
| `S01_config.py` | 1 | Imports, constants, shared utilities | — | Imported by all |
| `S02_chembl_retrieval.py` | 2 | ChEMBL v33 IC50 data fetch | ChEMBL API | `chembl_xo_ic50_raw.csv` |
| `S03_data_curation.py` | 3 | Curation, Lipinski, geometric mean | S02 CSV | `chembl_xo_curated.csv` |
| `S04_fingerprints.py` | 4 | Morgan (2048-bit) + MACCS (166-bit) | S03 CSV | `morgan_matrix.npy`, `maccs_matrix.npy`, `labels.npy` |
| `S05_dataset_split.py` | 5 | 80/20 stratified split (seed=57) | S04 npy | `Xm_train.npy`, `Xm_test.npy`, `y_train.npy`, `y_test.npy` |
| `S06_classifier_benchmark.py` | 6 | 20 classifiers × 25×5-fold CV | S05 npy | `cv_benchmark_Morgan.csv`, `cv_benchmark_MACCS.csv` |
| `S07_hyperparameter_optimisation.py` | 7 | GridSearchCV for RFC, SVC, kNN | S05 npy | `best_rfc_model.pkl`, `best_svc_model.pkl`, `best_knn_model.pkl` |
| `S08_ensemble.py` | 8 | Soft-voting ensemble (RFC+SVC+kNN) | S07 pkl | `ensemble_model.pkl` |
| `S09_evaluation_yscrambling.py` | 9 | Test-set metrics + Y-scrambling | S07/S08 | `table1_test_performance.csv`, `y_scrambling_results.csv` |
| `S10_feature_importance.py` | 10 | Morgan bit importance analysis | S07 pkl | `feature_importance_top20.csv`, `feature_importance_all.npy` |
| `S11_coconut_filters.py` | 11 | PAINS → Brenk → Lipinski filters | COCONUT CSV | `coconut_filtered.csv` |
| `S12_virtual_screening.py` | 12 | RFC screening, p≥0.60, Mann-Whitney | S11 CSV + S07 pkl | `ml_vs_hits_for_docking.csv` |
| `S13_visualisation.py` | 13 | All publication figures (300 DPI) | S06–S12 outputs | `figures/*.png` |

---

## Run Order

```bash
# Step 0 – verify dependencies
pip install rdkit chembl-webresource-client scikit-learn numpy pandas \
            matplotlib seaborn scipy lightgbm xgboost catboost

# Step 1 – run sections in order from the XO_Sections/ folder
cd XO_Sections/

python S02_chembl_retrieval.py          # downloads ChEMBL data (internet needed once)
python S03_data_curation.py             # curate and label compounds
python S04_fingerprints.py              # compute Morgan + MACCS fingerprints
python S05_dataset_split.py             # 80/20 stratified split
python S06_classifier_benchmark.py      # 20-classifier CV benchmark  [slow ~hours]
python S07_hyperparameter_optimisation.py  # GridSearchCV top-3  [slow ~hours]
python S08_ensemble.py                  # soft-voting ensemble
python S09_evaluation_yscrambling.py    # test-set metrics + Y-scrambling
python S10_feature_importance.py        # Morgan bit importances
python S11_coconut_filters.py           # COCONUT structural filters
python S12_virtual_screening.py         # ML virtual screening
python S13_visualisation.py             # generate all figures
```

> **Note:** S01_config.py is never run directly — it is imported by all other files.

---

## Key Parameters (S01_config.py)

| Parameter | Value | Source |
|---|---|---|
| `SEED` | 57 | Section 2.1 |
| `TEST_FRAC` | 0.20 | Section 2.1 |
| `CV_SPLITS` | 5 | Section 2.1 |
| `CV_REPEATS` | 25 | Section 2.1 |
| `IC50_CUTOFF_UM` | 1.0 µM | Section 2.1 |
| `ML_PROB_CUTOFF` | 0.60 | Section 2.2 |
| `MORGAN_RADIUS` | 3 | Section 2.1 |
| `MORGAN_BITS` | 2048 | Section 2.1 |
| `MACCS_BITS` | 166 | Section 2.1 |

---

## COCONUT Database

Download the March 2025 snapshot from:  
https://coconut.naturalproducts.net/

Place the CSV as `coconut_complete.csv` in your working directory,
then update `COCONUT_CSV` at the top of `S11_coconut_filters.py`.

---

## Expected Key Results

| Metric | Value |
|---|---|
| RFC ROC-AUC (test) | 0.960 |
| RFC F1-Score (test) | 0.943 |
| RFC MCC (test) | 0.876 |
| Y-scrambling AUC | 0.52 ± 0.04 |
| COCONUT → hits | 695,119 → 116 |
| Enrichment factor | ~5,992× |
| Lead compound | CNP0149788.0 |
| Lead binding energy | −10.6 kcal/mol |
