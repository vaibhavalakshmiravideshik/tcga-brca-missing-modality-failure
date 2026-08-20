"""
Stage 2: Leakage-corrected 5-fold cross-validation.

Trains the primary missing-token model with feature screening refit
independently within each fold's own training partition (correcting the
single-split-screening leakage documented in Appendix A of the paper: OOF
C-index moved from 0.6015 -> 0.5900 after this fix).

Requires: results/X_prot_final.csv, results/X_gen_tcga.csv, results/target_df.csv
  (run 01_data_acquisition.py first)
Output: results/oof_risk_v2.npy, results/fold_feature_log.pkl
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
import copy
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from utils import (TCGADataset, TCGASurvivalModel, DiscreteSurvivalLoss,
                    univariate_cox_screen_fold, cumulative_risk_at_horizon,
                    bootstrap_c_index_ci, TIME_BINS_TCGA)

OUT_DIR = "results"
SEED = 42


def train_one_fold(X_prot_fold, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                    os_time_arr, os_status_arr, train_idx, val_idx, device,
                    n_prot, n_gen, hidden_dim=20, dropout=0.35, patience=20, max_epochs=100):
    train_ds = TCGADataset(X_prot_fold, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                            os_time_arr, os_status_arr, train_idx)
    val_ds = TCGADataset(X_prot_fold, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                          os_time_arr, os_status_arr, val_idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = TCGASurvivalModel(n_prot, n_gen, hidden_dim, len(TIME_BINS_TCGA), dropout).to(device)
    criterion = DiscreteSurvivalLoss(TIME_BINS_TCGA)
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_loss, best_state, no_improve = float('inf'), None, 0
    for epoch in range(1, max_epochs + 1):
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
                for b in val_loader
            ) / len(val_loader)
        if val_running < best_loss:
            best_loss, best_state, no_improve = val_running, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full_val_loader = DataLoader(val_ds, batch_size=len(val_idx), shuffle=False)
        batch = next(iter(full_val_loader))
        hazards_val = model(batch['prot'].to(device), batch['gen'].to(device),
                             batch['prot_missing'].to(device), batch['gen_missing'].to(device)).cpu().numpy()
    return hazards_val, best_loss


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_prot_final = pd.read_csv(os.path.join(OUT_DIR, "X_prot_final.csv"))
    X_gen_tcga = pd.read_csv(os.path.join(OUT_DIR, "X_gen_tcga.csv"))
    target_df = pd.read_csv(os.path.join(OUT_DIR, "target_df.csv"))

    has_rppa_mask = target_df['Has_RPPA'].values.astype(bool)
    has_mut_mask = target_df['Has_Mutation_Data'].values.astype(bool)
    os_time_arr = target_df['OS_Time'].values
    os_status_arr = target_df['OS_Status'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_indices = np.arange(len(target_df))
    oof_risk_v2 = np.full(len(target_df), np.nan)
    fold_feature_log = []

    print("Running 5-fold CV with per-fold feature screening (leakage-corrected)...")
    fold_num = 0
    for train_idx, val_idx in skf.split(all_indices, target_df['OS_Status'].values):
        fold_num += 1
        sel_prot = univariate_cox_screen_fold(X_prot_final, os_time_arr, os_status_arr, train_idx, has_rppa_mask, 25)
        sel_gen = univariate_cox_screen_fold(X_gen_tcga, os_time_arr, os_status_arr, train_idx, has_mut_mask, 15)
        fold_feature_log.append({'fold': fold_num, 'prot': sel_prot, 'gen': sel_gen})

        X_prot_fold = X_prot_final[sel_prot]
        X_gen_fold_raw = X_gen_tcga[sel_gen]
        X_gen_fold_filled = X_gen_fold_raw.fillna(X_gen_fold_raw.iloc[train_idx].median()).fillna(0.0)

        hazards_val, best_loss = train_one_fold(
            X_prot_fold, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
            os_time_arr, os_status_arr, train_idx, val_idx, device,
            n_prot=len(sel_prot), n_gen=len(sel_gen))

        oof_risk_v2[val_idx] = cumulative_risk_at_horizon(hazards_val, len(TIME_BINS_TCGA) - 1)
        print(f"  Fold {fold_num}: best val loss={best_loss:.4f}")

    oof_c_index = concordance_index(os_time_arr, -oof_risk_v2, os_status_arr)
    ci_lo, ci_hi = bootstrap_c_index_ci(oof_risk_v2, os_time_arr, os_status_arr)
    print(f"\n=== Leakage-corrected OOF C-index: {oof_c_index:.4f} (95% CI: {ci_lo:.4f}-{ci_hi:.4f}) ===")
    print("Reference value reported in paper: 0.5900 (95% CI: 0.5376-0.6458)")

    stable = pd.Series([f for log in fold_feature_log for f in log['prot']]).value_counts()
    print(f"\nProteins selected in all 5 folds: {(stable == 5).sum()}/{len(stable.index.unique())} unique proteins seen")

    np.save(os.path.join(OUT_DIR, "oof_risk_v2.npy"), oof_risk_v2)
    with open(os.path.join(OUT_DIR, "fold_feature_log.pkl"), "wb") as f:
        pickle.dump(fold_feature_log, f)
    print(f"\n✓ Saved oof_risk_v2.npy and fold_feature_log.pkl to {OUT_DIR}/")


if __name__ == "__main__":
    main()
