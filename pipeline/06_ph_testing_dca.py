"""
Stage 6: Proportional hazards testing, stratified Cox model, and decision
curve analysis (DCA).

Tests whether AJCC stage, PAM50 subtype, and the proteogenomic risk score
satisfy the Cox proportional hazards assumption (paper finding: stage and
subtype violate it under ordinal coding, the risk score does not), refits
a PH-respecting stratified model, and computes censoring-aware DCA using
recalibrated risk estimates.

Requires: results from stages 01, 02, 05
Output: results/dca_curves.pkl, prints all Section 3/Appendix B numbers
"""
import numpy as np
import pandas as pd
import os
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import pickle

from utils import (STAGE_MAP, SUBTYPE_RISK_MAP, HORIZON_DCA,
                    km_event_prob_by_horizon, net_benefit_with_ci)

OUT_DIR = "results"


def main():
    target_df = pd.read_csv(os.path.join(OUT_DIR, "target_df.csv"))
    oof_risk_recalibrated = np.load(os.path.join(OUT_DIR, "oof_risk_recalibrated.npy"))
    os_time_arr = target_df['OS_Time'].values
    os_status_arr = target_df['OS_Status'].values

    stage_numeric_full = target_df['AJCC_Stage'].map(STAGE_MAP)
    subtype_numeric_full = target_df['Subtype'].map(SUBTYPE_RISK_MAP)
    valid_full = stage_numeric_full.notna() & subtype_numeric_full.notna()

    combo_df = pd.DataFrame({
        'model_risk': oof_risk_recalibrated[valid_full.values],
        'stage': stage_numeric_full[valid_full].values,
        'subtype_risk': subtype_numeric_full[valid_full].values,
        'OS_Time': os_time_arr[valid_full.values],
        'OS_Status': os_status_arr[valid_full.values],
    })
    print(f"Combined-model evaluation cohort: n={valid_full.sum()}, events={int(combo_df['OS_Status'].sum())}")

    # --- Non-stratified Cox (used for DCA's absolute risk estimates) ---
    cph = CoxPHFitter()
    cph.fit(combo_df, duration_col='OS_Time', event_col='OS_Status', show_progress=False)
    print("\n=== Non-stratified Cox model ===")
    print(cph.summary[['coef', 'p']])
    print(f"C-index: {cph.concordance_index_:.4f}  (reference: 0.7273)")

    # --- Proportional hazards testing ---
    print("\n=== Proportional hazards test (Schoenfeld residuals) ===")
    ph_results = proportional_hazard_test(cph, combo_df, time_transform='rank')
    print(ph_results.summary)
    print("Reference: stage p<0.005, subtype p=0.0021 (both violate PH); "
          "model_risk p=0.95-0.98 (does not violate PH)")

    # --- Stratified Cox (PH-respecting specification) ---
    combo_df_strat = combo_df.copy()
    combo_df_strat['stage'] = combo_df_strat['stage'].astype(int).astype('category')
    combo_df_strat['subtype_risk'] = combo_df_strat['subtype_risk'].astype(int).astype('category')
    cph_strat = CoxPHFitter()
    cph_strat.fit(combo_df_strat, duration_col='OS_Time', event_col='OS_Status',
                  strata=['stage', 'subtype_risk'], show_progress=False)
    print("\n=== Stratified Cox model (PH-respecting) ===")
    print(cph_strat.summary[['coef', 'p']])
    print(f"Stratified C-index: {cph_strat.concordance_index_:.4f}  "
          f"(reference: model_risk p=0.0118, C-index=0.6170)")

    # --- Stage-only baseline for DCA comparator ---
    stage_only_df = pd.DataFrame({'stage': combo_df['stage'].values,
                                   'OS_Time': combo_df['OS_Time'].values,
                                   'OS_Status': combo_df['OS_Status'].values})
    cph_stage_only = CoxPHFitter()
    cph_stage_only.fit(stage_only_df, duration_col='OS_Time', event_col='OS_Status', show_progress=False)
    print(f"\nStage-only C-index (same n): {cph_stage_only.concordance_index_:.4f}  (reference: 0.6892)")

    # --- Decision curve analysis, recalibrated risk, bootstrap CIs ---
    print("\n=== Decision curve analysis (recalibrated risk, bootstrap 200 resamples) ===")
    stage_surv = cph_stage_only.predict_survival_function(
        pd.DataFrame({'stage': combo_df['stage'].values}), times=[HORIZON_DCA])
    stage_risk = 1.0 - stage_surv.loc[HORIZON_DCA].values

    combo_surv = cph.predict_survival_function(
        combo_df[['model_risk', 'stage', 'subtype_risk']], times=[HORIZON_DCA])
    combo_risk = 1.0 - combo_surv.loc[HORIZON_DCA].values

    thresholds = np.linspace(0.03, 0.40, 30)
    eval_time = combo_df['OS_Time'].values
    eval_status = combo_df['OS_Status'].values

    nb_combo_pt, nb_combo_lo, nb_combo_hi = net_benefit_with_ci(
        combo_risk, eval_time, eval_status, HORIZON_DCA, thresholds, n_boot=200)
    nb_stage_pt, nb_stage_lo, nb_stage_hi = net_benefit_with_ci(
        stage_risk, eval_time, eval_status, HORIZON_DCA, thresholds, n_boot=200)

    advantage = nb_combo_pt - nb_stage_pt
    if (advantage > 0).any():
        print(f"Combined beats Stage-alone across thresholds: "
              f"{thresholds[advantage > 0].min():.3f} to {thresholds[advantage > 0].max():.3f}  "
              f"(reference: 8.1%-40%)")
    print(f"Combined stays above Treat-None throughout: {(nb_combo_pt > 0).all()}")

    with open(os.path.join(OUT_DIR, "dca_curves.pkl"), "wb") as f:
        pickle.dump({
            'thresholds': thresholds,
            'nb_combo_pt': nb_combo_pt, 'nb_combo_lo': nb_combo_lo, 'nb_combo_hi': nb_combo_hi,
            'nb_stage_pt': nb_stage_pt, 'nb_stage_lo': nb_stage_lo, 'nb_stage_hi': nb_stage_hi,
        }, f)
    print(f"\n✓ Saved dca_curves.pkl to {OUT_DIR}/")


if __name__ == "__main__":
    main()
