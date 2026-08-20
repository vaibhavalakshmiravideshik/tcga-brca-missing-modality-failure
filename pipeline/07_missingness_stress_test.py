"""
Stage 7: Synthetic missingness stress test.

Forces increasing levels of synthetic RPPA missingness (0-100%) onto
complete-data patients at inference time and evaluates discrimination,
averaged across 5 independently-seeded models trained on an identical
split, to isolate a genuine trend from single-run training noise.

Paper finding: discrimination remains well above chance through 50%
missingness (mean C-index 0.62-0.68), with a non-monotonic early dip and
mid-range plateau, declining toward chance only under severe (75-100%)
missingness.

Requires: results from 01_data_acquisition.py
Output: results/stress_test_results.pkl
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import copy
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from utils import (TCGADataset, TCGASurvivalModel, DiscreteSurvivalLoss,
                    univariate_cox_screen_fold, cumulative_risk_at_horizon, TIME_BINS_TCGA)

OUT_DIR = "results"
N_SEEDS = 5
STRESS_PROBS = [0.0, 0.15, 0.30, 0.50, 0.75, 1.0]


def evaluate_with_forced_missingness(model, X_prot, X_gen, indices, force_prot_missing_prob, seed, device):
    rng = np.random.RandomState(seed)
    n_idx = len(indices)
    prot_force = rng.rand(n_idx) < force_prot_missing_prob
    X_p = torch.tensor(X_prot.iloc[indices].fillna(0.0).values, dtype=torch.float32).to(device)
    X_g = torch.tensor(X_gen.iloc[indices].values, dtype=torch.float32).to(device)
    p_missing = torch.tensor(prot_force.astype(np.float32)).to(device)
    g_missing = torch.zeros(n_idx, dtype=torch.float32).to(device)
    with torch.no_grad():
        hazards = model(X_p, X_g, p_missing, g_missing).cpu().numpy()
    return cumulative_risk_at_horizon(hazards, len(TIME_BINS_TCGA) - 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_prot_final = pd.read_csv(os.path.join(OUT_DIR, "X_prot_final.csv"))
    X_gen_tcga = pd.read_csv(os.path.join(OUT_DIR, "X_gen_tcga.csv"))
    target_df = pd.read_csv(os.path.join(OUT_DIR, "target_df.csv"))
    has_rppa_mask = target_df['Has_RPPA'].values.astype(bool)
    has_mut_mask = target_df['Has_Mutation_Data'].values.astype(bool)
    os_time_arr = target_df['OS_Time'].values
    os_status_arr = target_df['OS_Status'].values

    n = len(target_df)
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=0.25, random_state=42, stratify=target_df['OS_Status'].values)
    sel_prot = univariate_cox_screen_fold(X_prot_final, os_time_arr, os_status_arr, train_idx, has_rppa_mask, 25)
    sel_gen = univariate_cox_screen_fold(X_gen_tcga, os_time_arr, os_status_arr, train_idx, has_mut_mask, 15)
    X_prot_s = X_prot_final[sel_prot]
    X_gen_s_raw = X_gen_tcga[sel_gen]
    X_gen_s_filled = X_gen_s_raw.fillna(X_gen_s_raw.iloc[train_idx].median()).fillna(0.0)

    complete_data_mask = has_rppa_mask & has_mut_mask
    complete_idx = np.where(complete_data_mask)[0]
    complete_time = os_time_arr[complete_idx]
    complete_status = os_status_arr[complete_idx]

    all_seed_results = np.full((N_SEEDS, len(STRESS_PROBS)), np.nan)

    for seed_i in range(N_SEEDS):
        print(f"=== Training model {seed_i + 1}/{N_SEEDS} (seed={seed_i}) ===")
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)

        train_ds = TCGADataset(X_prot_s, X_gen_s_filled, has_rppa_mask, has_mut_mask,
                                os_time_arr, os_status_arr, train_idx)
        test_ds = TCGADataset(X_prot_s, X_gen_s_filled, has_rppa_mask, has_mut_mask,
                               os_time_arr, os_status_arr, test_idx)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        model = TCGASurvivalModel(len(sel_prot), len(sel_gen), 20, len(TIME_BINS_TCGA), 0.35).to(device)
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
                    for b in test_loader
                ) / len(test_loader)
            if val_running < best_loss:
                best_loss, best_state, no_improve = val_running, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
            if no_improve >= 20:
                break
        model.load_state_dict(best_state)
        model.eval()

        for j, p in enumerate(STRESS_PROBS):
            risk = evaluate_with_forced_missingness(model, X_prot_s, X_gen_s_filled, complete_idx, p, 42, device)
            all_seed_results[seed_i, j] = concordance_index(complete_time, -risk, complete_status)

        print(f"  Results: {[f'{v:.3f}' for v in all_seed_results[seed_i]]}")

    mean_c = np.nanmean(all_seed_results, axis=0)
    min_c = np.nanmin(all_seed_results, axis=0)
    max_c = np.nanmax(all_seed_results, axis=0)

    print(f"\n=== Averaged across {N_SEEDS} seeds ===")
    for j, p in enumerate(STRESS_PROBS):
        print(f"  missingness={p:.2f}: mean C-index={mean_c[j]:.4f}  range=[{min_c[j]:.4f}, {max_c[j]:.4f}]")
    print("\nReference values from paper: [0.681, 0.620, 0.641, 0.645, 0.613, 0.517]")

    with open(os.path.join(OUT_DIR, "stress_test_results.pkl"), "wb") as f:
        pickle.dump({'probs': STRESS_PROBS, 'mean_c': mean_c, 'min_c': min_c, 'max_c': max_c,
                     'all_seed_results': all_seed_results}, f)
    print(f"\n✓ Saved stress_test_results.pkl to {OUT_DIR}/")


if __name__ == "__main__":
    main()
