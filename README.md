# Architectural Complexity and Miscalibration in Proteogenomic Survival Prediction

Code accompanying the paper *"Architectural Complexity and Miscalibration in
Proteogenomic Survival Prediction: A Failure Analysis on Real Clinical
Missingness,"* currently under review at
[ICBINB-BIO](https://icbinb-bio.github.io/) (NeurIPS 2026 Workshop).

## What this is

We test two common but rarely-verified assumptions in multi-omic clinical
machine learning: (1) that purpose-built missing-modality architectures
outperform simple imputation, and (2) that discrimination metrics (C-index,
AUC) alone certify a model's outputs are trustworthy. On a real, adequately
powered TCGA-BRCA proteogenomic cohort (n=1,080, 151 deaths, real RPPA/mutation
missingness), neither assumption survives controlled comparison. This
repository contains the full pipeline: data acquisition, the leakage
correction we had to apply to our own initial analysis, every baseline and
ablation, the calibration bug we found and fixed, and every sensitivity
analysis reported in the paper.

## Repository structure

```
proteogenomic-failure-analysis/
├── README.md
├── requirements.txt
├── pipeline/
│   ├── utils.py
│   ├── 01_data_acquisition.py
│   ├── 02_feature_screening_cv.py
│   ├── 03_baselines.py
│   ├── 04_architecture_ablation.py
│   ├── 05_calibration.py
│   ├── 06_ph_testing_dca.py
│   ├── 07_missingness_stress_test.py
│   └── 08_sensitivity_ablations.py
└── results/
```

## Data source

All data is public and requires no special access: [TCGA-BRCA PanCancer
Atlas](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018),
accessed programmatically via the [cBioPortal REST
API](https://www.cbioportal.org/api). No data files are bundled in this repo
&mdash; [`pipeline/01_data_acquisition.py`](pipeline/01_data_acquisition.py)
downloads everything needed.

## Pipeline

Run in order from the **repository root**; each stage depends on outputs from
earlier stages, all saved to [`results/`](results/).

| Script | What it does | Key output | Paper section |
|---|---|---|---|
| [`pipeline/01_data_acquisition.py`](pipeline/01_data_acquisition.py) | cBioPortal download, cohort construction, missingness handling | `results/target_df.csv`, `results/X_prot_final.csv`, `results/X_gen_tcga.csv` | §1 Problem |
| [`pipeline/02_feature_screening_cv.py`](pipeline/02_feature_screening_cv.py) | Leakage-corrected 5-fold CV, primary reported result | `results/oof_risk_v2.npy` (OOF C-index 0.590) | §3 Observed outcome, Appendix A |
| [`pipeline/03_baselines.py`](pipeline/03_baselines.py) | Elastic-net Cox, random survival forest, gradient-boosted survival | `results/baseline_oof.pkl` | §3, Table 1 |
| [`pipeline/04_architecture_ablation.py`](pipeline/04_architecture_ablation.py) | Impute+indicator baseline, training-time augmentation, paired significance tests | `results/oof_risk_notoken.npy`, `results/oof_risk_augmented.npy` | §3, Table 1 |
| [`pipeline/05_calibration.py`](pipeline/05_calibration.py) | Calibration slope, Brier score (corrected), recalibration | `results/oof_risk_recalibrated.npy` | §3, Table 2 |
| [`pipeline/06_ph_testing_dca.py`](pipeline/06_ph_testing_dca.py) | Proportional hazards testing, stratified Cox, decision curve analysis | `results/dca_curves.pkl` | §3, Figure 1, Appendix B |
| [`pipeline/07_missingness_stress_test.py`](pipeline/07_missingness_stress_test.py) | 5-seed averaged synthetic missingness stress test | `results/stress_test_results.pkl` | §4 Reason for failure, Figure 2 |
| [`pipeline/08_sensitivity_ablations.py`](pipeline/08_sensitivity_ablations.py) | Site-level (TSS) sensitivity, feature-count ablation | printed results | §5 Limitations |

All shared classes (dataset, model architecture, loss function) and metric
helpers (C-index, time-dependent AUC, censoring-aware net benefit) live in
[`pipeline/utils.py`](pipeline/utils.py) and are imported by every numbered
script.

## Setup

```bash
git clone https://github.com/vaibhavalakshmiravideshik/tcga-brca-missing-modality-failure.git
cd tcga-brca-missing-modality-failure
pip install -r requirements.txt
```

## Reproducing the paper's headline numbers

Run all commands from the repository root:

```bash
python pipeline/01_data_acquisition.py         # ~2-5 min, network-bound
python pipeline/02_feature_screening_cv.py     # ~5-10 min, retrains 5 models
python pipeline/03_baselines.py                # ~2-3 min
python pipeline/04_architecture_ablation.py    # ~10-15 min, retrains 10 models
python pipeline/05_calibration.py              # <1 min
python pipeline/06_ph_testing_dca.py           # ~1-2 min
python pipeline/07_missingness_stress_test.py  # ~15-20 min, retrains 5 models
python pipeline/08_sensitivity_ablations.py    # ~20-30 min, retrains 15 models
```

Each script prints its result alongside the reference value reported in the
paper. Exact figures may drift slightly (sub-0.01 on C-index) between runs due
to non-deterministic GPU operations and cBioPortal API ordering &mdash; this
run-to-run variance is itself discussed in the paper's Appendix B as a
reproducibility check, not a discrepancy to resolve.

## Key findings

- **Architecture doesn't win.** A learnable missing-modality token, including
  a variant with training-time missingness augmentation, does not
  significantly outperform simple median-imputation-plus-indicator, nor
  classical elastic-net Cox / random survival forest / gradient-boosted
  survival, on identical features and evaluation (paired bootstrap
  $p = 0.13$&ndash;$0.70$ across all comparisons).
- **Discrimination isn't calibration.** The model has real, statistically
  confirmed discrimination (C-index $0.59$) but its raw risk outputs are
  substantially miscalibrated (slope $0.42$) &mdash; badly enough that an
  uncalibrated Brier score loses to a trivial constant-risk prediction.
- **We report our own mistakes.** A feature-selection leakage and a Brier
  score sign-inversion bug, both found during this project, are documented
  and corrected in the paper and reproduced in this code, rather than
  silently fixed and hidden.

## A note on how this was built

Portions of this codebase were developed interactively with a large language
model (Claude, Anthropic) under direct human review at every step, including
identification of the feature-selection leakage and Brier-score sign error
that the paper itself reports and corrects. All results were generated by
code executed by the author; nothing was fabricated by the model. Full
disclosure is in the paper's LLM Usage Disclosure section.

## Status

This paper is currently under review at ICBINB-BIO (NeurIPS 2026 Workshop).
A citation will be added here once the review outcome and any camera-ready
version are finalized.
