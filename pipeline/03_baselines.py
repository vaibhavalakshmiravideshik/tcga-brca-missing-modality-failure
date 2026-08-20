"""
Stage 3: Classical survival baselines.

Compares the missing-token architecture against elastic-net penalized Cox
regression, random survival forest, and gradient-boosted survival analysis,
trained on identical per-fold-screened features under identical 5-fold OOF
evaluation. Result reported in the paper: all four methods statistically
indistinguishable (C-index 0.590-0.607, fully overlapping 95% CIs).

Requires: results from 01_data_acquisition.py and 02_feature_screening_cv.py
Output: results/baseline_oof.pkl
"""
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from utils import bootstrap_c_index_ci

OUT_DIR = "results"
SEED = 42


def make_sksurv_y(times, statuses):
    return Surv.from_arrays(event=statuses.astype(bool), time=times)


def main():
    X_prot_final = pd.read_csv(os.path.join(OUT_DIR, "X_prot_final.csv"))
    X_gen_tcga = pd.read_csv(os.path.join(OUT_DIR, "X_gen_tcga.csv"))
    target_df = pd.read_csv(os.path.join(OUT_DIR, "target_df.csv"))
    with open(os.path.join(OUT_DIR, "fold_feature_log.pkl"), "rb") as f:
        fold_feature_log = pickle.load(f)
    oof_risk_v2 = np.load(os.path.join(OUT_DIR, "oof_risk_v2.npy"))

    os_time_arr = target_df['OS_Time'].values
    os_status_arr = target_df['OS_Status'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_indices = np.arange(len(target_df))

    baseline_oof = {
        'elastic_net_cox': np.full(len(target_df), np.nan),
        'rsf': np.full(len(target_df), np.nan),
        'gbm_survival': np.full(len(target_df), np.nan),
    }

    fold_num = 0
    for train_idx, val_idx in skf.split(all_indices, target_df['OS_Status'].values):
        fold_num += 1
        print(f"--- Baseline fold {fold_num}/5 ---")
        sel_prot = fold_feature_log[fold_num - 1]['prot']
        sel_gen = fold_feature_log[fold_num - 1]['gen']

        X_prot_fold = X_prot_final[sel_prot]
        X_gen_fold_raw = X_gen_tcga[sel_gen]
        gen_medians = X_gen_fold_raw.iloc[train_idx].median()
        X_gen_fold_filled = X_gen_fold_raw.fillna(gen_medians).fillna(0.0)
        X_prot_fold_filled = X_prot_fold.fillna(X_prot_fold.iloc[train_idx].median())

        X_combined = pd.concat([X_prot_fold_filled, X_gen_fold_filled], axis=1)
        X_train_b, X_val_b = X_combined.iloc[train_idx].values, X_combined.iloc[val_idx].values
        y_train_b = make_sksurv_y(os_time_arr[train_idx], os_status_arr[train_idx])

        # Elastic-net Cox
        cox_df_train = X_combined.iloc[train_idx].copy()
        cox_df_train['OS_Time'] = os_time_arr[train_idx]
        cox_df_train['OS_Status'] = os_status_arr[train_idx]
        cph_en = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
        try:
            cph_en.fit(cox_df_train, duration_col='OS_Time', event_col='OS_Status', show_progress=False)
            risk_en = cph_en.predict_partial_hazard(X_combined.iloc[val_idx]).values
            baseline_oof['elastic_net_cox'][val_idx] = risk_en
        except Exception as e:
            print(f"  Elastic-net Cox failed on fold {fold_num}: {e}")

        # Random Survival Forest
        rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10, max_depth=4,
                                    random_state=SEED, n_jobs=-1)
        rsf.fit(X_train_b, y_train_b)
        baseline_oof['rsf'][val_idx] = rsf.predict(X_val_b)

        # Gradient Boosted Survival
        gbm = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.05,
                                                max_depth=2, random_state=SEED)
        gbm.fit(X_train_b, y_train_b)
        baseline_oof['gbm_survival'][val_idx] = gbm.predict(X_val_b)

        print(f"  Fold {fold_num} baselines fit.")

    print("\n=== Baseline OOF C-index comparison ===")
    our_ci = bootstrap_c_index_ci(oof_risk_v2, os_time_arr, os_status_arr)
    our_c = concordance_index(os_time_arr, -oof_risk_v2, os_status_arr)
    print(f"{'Missing-token model':25s} {our_c:.4f}  95% CI ({our_ci[0]:.4f}, {our_ci[1]:.4f})")

    for name, risk in baseline_oof.items():
        valid = ~np.isnan(risk)
        c_idx = concordance_index(os_time_arr[valid], -risk[valid], os_status_arr[valid])
        ci_lo, ci_hi = bootstrap_c_index_ci(risk[valid], os_time_arr[valid], os_status_arr[valid])
        print(f"{name:25s} {c_idx:.4f}  95% CI ({ci_lo:.4f}, {ci_hi:.4f})")

    with open(os.path.join(OUT_DIR, "baseline_oof.pkl"), "wb") as f:
        pickle.dump(baseline_oof, f)
    print(f"\n✓ Saved baseline_oof.pkl to {OUT_DIR}/")


if __name__ == "__main__":
    main()
