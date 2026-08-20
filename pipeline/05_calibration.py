"""
Stage 5: Calibration assessment and recalibration.

Assesses whether raw model risk estimates are calibrated (not just
rank-discriminative), computes IPCW Brier score correctly (this script's
horizon_brier_score function is the CORRECTED version -- our first
implementation passed risk where the metric expected survival probability,
producing a spuriously bad score of 0.54; that bug and its fix are
documented in the paper's Reason for Failure section as an example of why
calibration metrics need the same implementation scrutiny as discrimination
metrics), and applies Cox-based recalibration.

Requires: results from 01_data_acquisition.py, 02_feature_screening_cv.py
Output: results/oof_risk_recalibrated.npy
"""
import numpy as np
import pandas as pd
import os
from lifelines import CoxPHFitter
from sksurv.metrics import brier_score
from sksurv.util import Surv

from utils import km_event_prob_by_horizon, HORIZON_DCA

OUT_DIR = "results"


def make_sksurv_y(times, statuses):
    return Surv.from_arrays(event=statuses.astype(bool), time=times)


def horizon_brier_score(risk_scores, times, statuses, horizon):
    """CORRECT version: sksurv's brier_score expects predicted SURVIVAL
    probability S(t), not event risk 1-S(t). Passing risk directly
    (the bug in our first implementation) inverts the score and produces
    an implausibly bad, meaningless number."""
    y_struct = make_sksurv_y(times, statuses)
    survival_prob = 1.0 - np.clip(risk_scores, 0.001, 0.999)
    try:
        _, bs = brier_score(y_struct, y_struct, survival_prob.reshape(-1, 1).repeat(1, axis=1), [horizon])
        return bs[0]
    except Exception as e:
        print(f"Brier score computation failed: {e}")
        return np.nan


def main():
    target_df = pd.read_csv(os.path.join(OUT_DIR, "target_df.csv"))
    oof_risk_v2 = np.load(os.path.join(OUT_DIR, "oof_risk_v2.npy"))
    os_time_arr = target_df['OS_Time'].values
    os_status_arr = target_df['OS_Status'].values

    # --- Decile calibration table ---
    n_bins = 10
    risk_deciles = pd.qcut(oof_risk_v2, n_bins, labels=False, duplicates='drop')
    print("=== Calibration table (raw), predicted vs. KM-observed 5-year risk ===")
    for b in sorted(np.unique(risk_deciles)):
        mask = risk_deciles == b
        mean_pred = oof_risk_v2[mask].mean()
        observed = km_event_prob_by_horizon(os_time_arr[mask], os_status_arr[mask], HORIZON_DCA)
        print(f"  Decile {b}: n={mask.sum():3d}  predicted={mean_pred:.3f}  KM-observed={observed:.3f}")

    # --- Calibration slope via Cox regression on complementary log-log of risk ---
    eps = 1e-6
    cll_pred = np.log(-np.log(1 - np.clip(oof_risk_v2, eps, 1 - eps)))
    calib_cox_df = pd.DataFrame({'cll_pred': cll_pred, 'OS_Time': os_time_arr, 'OS_Status': os_status_arr})
    cph_calib = CoxPHFitter()
    cph_calib.fit(calib_cox_df, duration_col='OS_Time', event_col='OS_Status', show_progress=False)
    raw_slope = cph_calib.summary['coef'].values[0]
    print(f"\nRaw calibration slope: {raw_slope:.4f} (ideal = 1.0; reference value in paper: 0.42)")

    # --- Brier scores: raw vs. trivial baseline ---
    bs_raw = horizon_brier_score(oof_risk_v2, os_time_arr, os_status_arr, HORIZON_DCA)
    base_rate = km_event_prob_by_horizon(os_time_arr, os_status_arr, HORIZON_DCA)
    trivial_risk = np.full_like(oof_risk_v2, base_rate)
    bs_trivial = horizon_brier_score(trivial_risk, os_time_arr, os_status_arr, HORIZON_DCA)
    print(f"\nRaw Brier score: {bs_raw:.4f}")
    print(f"Trivial (constant base-rate) Brier score: {bs_trivial:.4f}")
    print(f"Raw model beats trivial baseline: {bs_raw < bs_trivial} "
          f"(reference in paper: FALSE, raw 0.1567 > trivial 0.1500)")

    # --- Recalibration ---
    recal_input_df = pd.DataFrame({'cll_pred': cll_pred})
    recal_surv = cph_calib.predict_survival_function(recal_input_df, times=[HORIZON_DCA])
    oof_risk_recalibrated = 1.0 - recal_surv.loc[HORIZON_DCA].values

    cll_recal = np.log(-np.log(1 - np.clip(oof_risk_recalibrated, eps, 1 - eps)))
    calib_check_df = pd.DataFrame({'cll_pred': cll_recal, 'OS_Time': os_time_arr, 'OS_Status': os_status_arr})
    cph_check = CoxPHFitter()
    cph_check.fit(calib_check_df, duration_col='OS_Time', event_col='OS_Status', show_progress=False)
    recal_slope = cph_check.summary['coef'].values[0]
    bs_recal = horizon_brier_score(oof_risk_recalibrated, os_time_arr, os_status_arr, HORIZON_DCA)

    print(f"\nRecalibrated slope: {recal_slope:.4f} (should be ~1.0 by construction)")
    print(f"Recalibrated Brier score: {bs_recal:.4f}  "
          f"(vs. trivial {bs_trivial:.4f} -- reference in paper: 0.1482 vs. 0.1500, marginal improvement)")

    np.save(os.path.join(OUT_DIR, "oof_risk_recalibrated.npy"), oof_risk_recalibrated)
    print(f"\n✓ Saved oof_risk_recalibrated.npy to {OUT_DIR}/")


if __name__ == "__main__":
    main()
