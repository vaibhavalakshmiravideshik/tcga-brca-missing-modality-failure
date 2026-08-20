"""
Stage 8: Site-level sensitivity analysis and feature-count ablation.

Part A: Uses TCGA Tissue Source Site (TSS, encoded in the patient barcode)
as a proxy for batch effects -- cBioPortal does not expose literal RPPA
plate/batch metadata. Tests whether the independent-value finding survives
site stratification. Paper finding: it does, but on a thinner margin
(p=0.0125 unstratified vs. p=0.0440 site-stratified), reported as a
genuine limitation, not resolved.

Part B: Reruns the full 5-fold OOF pipeline at three feature-count
settings (10/5, 25/15, 50/30 proteins/genes) to confirm the 25/15 setting
used throughout was not an arbitrary choice. Paper finding: flat 10->25,
declining at 50 (overfitting once feature count grows relative to events).

Requires: results from 01_data_acquisition.py, 02_feature_screening_cv.py
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import copy
import os
import warnings
warnings.filterwarnings('ignore')

from utils import (TCGADataset, TCGASurvivalModel, DiscreteSurvivalLoss,
                    univariate_cox_screen_fold, cumulative_risk_at_horizon, TIME_BINS_TCGA)

OUT_DIR = "results"
SEED = 42


def part_a_site_sensitivity(target_df, oof_risk_v2, os_time_arr, os_status_arr):
    print("=" * 70)
    print("PART A: Site-level (TSS) sensitivity analysis")
    print("=" * 70)

    target_df = target_df.copy()
    target_df['TSS_Site'] = target_df['patientId'].str.split('-').str[1]

    site_summary = target_df.groupby('TSS_Site').agg(
        n=('patientId', 'count'), rppa_missing_rate=('Has_RPPA', lambda x: 1 - x.mean()),
        event_rate=('OS_Status', 'mean')).sort_values('n', ascending=False)
    print("\nTop sites by n:")
    print(site_summary.head(15))

    MIN_N, MIN_EVENTS = 30, 5
    eligible_sites = site_summary[site_summary['n'] >= MIN_N].index.tolist()
    print(f"\nPer-site OOF C-index (n>={MIN_N}):")
    for site in eligible_sites:
        mask = (target_df['TSS_Site'] == site).values
        n_events = int(os_status_arr[mask].sum())
        if n_events < MIN_EVENTS:
            print(f"  {site}: n={mask.sum()}, events={n_events} -- excluded")
            continue
        c = concordance_index(os_time_arr[mask], -oof_risk_v2[mask], os_status_arr[mask])
        print(f"  {site}: n={mask.sum():3d}, events={n_events:2d}, C-index={c:.4f}")

    # Site-stratified Cox vs. non-stratified, on model_risk alone
    site_counts = target_df['TSS_Site'].value_counts()
    sites_ok = site_counts[site_counts >= MIN_N].index
    strat_mask = target_df['TSS_Site'].isin(sites_ok).values
    strat_df = pd.DataFrame({
        'model_risk': oof_risk_v2[strat_mask], 'site': target_df['TSS_Site'].values[strat_mask],
        'OS_Time': os_time_arr[strat_mask], 'OS_Status': os_status_arr[strat_mask],
    })
    strat_df['site'] = strat_df['site'].astype('category')

    cph_no_site = CoxPHFitter().fit(strat_df[['model_risk', 'OS_Time', 'OS_Status']],
                                      'OS_Time', 'OS_Status', show_progress=False)
    cph_site = CoxPHFitter().fit(strat_df, duration_col='OS_Time', event_col='OS_Status',
                                   strata=['site'], show_progress=False)
    print(f"\nmodel_risk p-value without site stratification: {cph_no_site.summary.loc['model_risk', 'p']:.4f}")
    print(f"model_risk p-value WITH site stratification:      {cph_site.summary.loc['model_risk', 'p']:.4f}")
    print("Reference: 0.0125 unstratified vs. 0.0440 site-stratified -- "
          "finding survives but on a thinner margin, reported as a limitation.")


def train_one_setting(n_prot, n_gen, X_prot_final, X_gen_tcga, target_df,
                       has_rppa_mask, has_mut_mask, os_time_arr, os_status_arr, device):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_risk = np.full(len(target_df), np.nan)
    for train_idx, val_idx in skf.split(np.arange(len(target_df)), target_df['OS_Status'].values):
        sel_prot = univariate_cox_screen_fold(X_prot_final, os_time_arr, os_status_arr, train_idx, has_rppa_mask, n_prot)
        sel_gen = univariate_cox_screen_fold(X_gen_tcga, os_time_arr, os_status_arr, train_idx, has_mut_mask, n_gen)
        X_prot_fold = X_prot_final[sel_prot]
        X_gen_fold_raw = X_gen_tcga[sel_gen]
        X_gen_fold_filled = X_gen_fold_raw.fillna(X_gen_fold_raw.iloc[train_idx].median()).fillna(0.0)

        train_ds = TCGADataset(X_prot_fold, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                                os_time_arr, os_status_arr, train_idx)
        val_ds = TCGADataset(X_prot_fold, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                              os_time_arr, os_status_arr, val_idx)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = TCGASurvivalModel(n_prot, n_gen, 20, len(TIME_BINS_TCGA), 0.35).to(device)
        criterion = DiscreteSurvivalLoss(TIME_BINS_TCGA)
        optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        best_loss, best_state, no_improve = float('inf'), None, 0
        for epoch in range(1, 101):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                hazards = model(batch['prot'].to(device), batch['gen'].to(device),
                                 batch['prot_missing'].to(device), batch['gen_missing'].to(device))
                loss = criterion(hazards, batch['os_time'].to(device), batch['os_status'].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_running = sum(
                    criterion(model(b['prot'].to(device), b['gen'].to(device),
                                     b['prot_missing'].to(device), b['gen_missing'].to(device)),
                              b['os_time'].to(device), b['os_status'].to(device)).item()
                    for b in val_loader) / len(val_loader)
            if val_running < best_loss:
                best_loss, best_state, no_improve = val_running, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
            if no_improve >= 20:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            full_val = DataLoader(val_ds, batch_size=len(val_idx), shuffle=False)
            batch = next(iter(full_val))
            hazards_val = model(batch['prot'].to(device), batch['gen'].to(device),
                                 batch['prot_missing'].to(device), batch['gen_missing'].to(device)).cpu().numpy()
        oof_risk[val_idx] = cumulative_risk_at_horizon(hazards_val, len(TIME_BINS_TCGA) - 1)
    return concordance_index(os_time_arr, -oof_risk, os_status_arr)


def part_b_feature_count(X_prot_final, X_gen_tcga, target_df, has_rppa_mask, has_mut_mask,
                          os_time_arr, os_status_arr, device):
    print("\n" + "=" * 70)
    print("PART B: Feature-count ablation")
    print("=" * 70)
    settings = [(10, 5), (25, 15), (50, 30)]
    results = {}
    for n_prot, n_gen in settings:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        print(f"\n--- {n_prot} proteins / {n_gen} genes ---")
        c = train_one_setting(n_prot, n_gen, X_prot_final, X_gen_tcga, target_df,
                               has_rppa_mask, has_mut_mask, os_time_arr, os_status_arr, device)
        results[f"{n_prot}p_{n_gen}g"] = c
        print(f"  OOF C-index: {c:.4f}")
    print("\nSummary:", results)
    print("Reference values from paper: 10p_5g=0.594, 25p_15g=0.593, 50p_30g=0.560")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_prot_final = pd.read_csv(os.path.join(OUT_DIR, "X_prot_final.csv"))
    X_gen_tcga = pd.read_csv(os.path.join(OUT_DIR, "X_gen_tcga.csv"))
    target_df = pd.read_csv(os.path.join(OUT_DIR, "target_df.csv"))
    oof_risk_v2 = np.load(os.path.join(OUT_DIR, "oof_risk_v2.npy"))

    has_rppa_mask = target_df['Has_RPPA'].values.astype(bool)
    has_mut_mask = target_df['Has_Mutation_Data'].values.astype(bool)
    os_time_arr = target_df['OS_Time'].values
    os_status_arr = target_df['OS_Status'].values

    part_a_site_sensitivity(target_df, oof_risk_v2, os_time_arr, os_status_arr)
    part_b_feature_count(X_prot_final, X_gen_tcga, target_df, has_rppa_mask, has_mut_mask,
                          os_time_arr, os_status_arr, device)


if __name__ == "__main__":
    main()
