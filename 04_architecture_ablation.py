"""
Stage 4: Missing-token architecture ablation.

Tests whether the learnable missing-token mechanism outperforms (a) simple
median-imputation-plus-indicator, and (b) a variant strengthened with
training-time synthetic missingness augmentation. All comparisons use
paired bootstrap testing (identical resampling across compared models).

Paper result: no comparison reaches significance (p=0.13-0.70), the
architecture's added complexity is not empirically justified at this
sample size.

Requires: results from 01_data_acquisition.py, 02_feature_screening_cv.py
Output: results/oof_risk_notoken.npy, results/oof_risk_augmented.npy
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
import copy
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from utils import (TCGADataset, TCGASurvivalModel, TCGASurvivalModelNoToken,
                    DiscreteSurvivalLoss, cumulative_risk_at_horizon,
                    bootstrap_c_index_ci, paired_bootstrap_c_index_diff, TIME_BINS_TCGA)

OUT_DIR = "results"
SEED = 42


class TCGADatasetAugmented(torch.utils.data.Dataset):
    """As TCGADataset, but additionally injects SYNTHETIC missingness
    during training only, at the given probabilities, on top of any
    already-present real missingness. Gives the missing-token mechanism
    more exposure to the missing condition than the real 19%/7% rate
    alone provides."""

    def __init__(self, X_prot, X_gen, prot_present_mask, gen_present_mask,
                 os_time, os_status, indices,
                 synthetic_prot_missing_prob=0.0, synthetic_gen_missing_prob=0.0, is_train=True):
        idx = indices
        self.X_prot = torch.tensor(X_prot.iloc[idx].fillna(0.0).values, dtype=torch.float32)
        self.X_gen = torch.tensor(X_gen.iloc[idx].values, dtype=torch.float32)
        self.real_prot_missing = torch.tensor((~prot_present_mask[idx]).astype(np.float32))
        self.real_gen_missing = torch.tensor((~gen_present_mask[idx]).astype(np.float32))
        self.os_time = torch.tensor(os_time[idx], dtype=torch.float32)
        self.os_status = torch.tensor(os_status[idx], dtype=torch.float32)
        self.synthetic_prot_missing_prob = synthetic_prot_missing_prob
        self.synthetic_gen_missing_prob = synthetic_gen_missing_prob
        self.is_train = is_train

    def __len__(self):
        return len(self.os_time)

    def __getitem__(self, i):
        prot_missing = self.real_prot_missing[i].clone()
        gen_missing = self.real_gen_missing[i].clone()
        if self.is_train:
            if prot_missing == 0 and np.random.rand() < self.synthetic_prot_missing_prob:
                prot_missing = torch.tensor(1.0)
            if gen_missing == 0 and np.random.rand() < self.synthetic_gen_missing_prob:
                gen_missing = torch.tensor(1.0)
        return {'prot': self.X_prot[i], 'gen': self.X_gen[i],
                'prot_missing': prot_missing, 'gen_missing': gen_missing,
                'os_time': self.os_time[i], 'os_status': self.os_status[i]}


def train_generic(model, train_loader, val_loader, criterion, device, max_epochs=100, patience=20):
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
    return model, best_loss


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_prot_final = pd.read_csv(os.path.join(OUT_DIR, "X_prot_final.csv"))
    X_gen_tcga = pd.read_csv(os.path.join(OUT_DIR, "X_gen_tcga.csv"))
    target_df = pd.read_csv(os.path.join(OUT_DIR, "target_df.csv"))
    with open(os.path.join(OUT_DIR, "fold_feature_log.pkl"), "rb") as f:
        fold_feature_log = pickle.load(f)
    oof_risk_v2 = np.load(os.path.join(OUT_DIR, "oof_risk_v2.npy"))

    has_rppa_mask = target_df['Has_RPPA'].values.astype(bool)
    has_mut_mask = target_df['Has_Mutation_Data'].values.astype(bool)
    os_time_arr = target_df['OS_Time'].values
    os_status_arr = target_df['OS_Status'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_indices = np.arange(len(target_df))

    oof_risk_notoken = np.full(len(target_df), np.nan)
    oof_risk_augmented = np.full(len(target_df), np.nan)

    fold_num = 0
    for train_idx, val_idx in skf.split(all_indices, target_df['OS_Status'].values):
        fold_num += 1
        sel_prot = fold_feature_log[fold_num - 1]['prot']
        sel_gen = fold_feature_log[fold_num - 1]['gen']
        X_prot_fold = X_prot_final[sel_prot]
        X_gen_fold_raw = X_gen_tcga[sel_gen]
        X_gen_fold_filled = X_gen_fold_raw.fillna(X_gen_fold_raw.iloc[train_idx].median()).fillna(0.0)
        X_prot_fold_filled = X_prot_fold.fillna(X_prot_fold.iloc[train_idx].median())

        # --- Ablation A: impute + indicator (no token) ---
        train_ds_nt = TCGADataset(X_prot_fold_filled, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                                   os_time_arr, os_status_arr, train_idx)
        val_ds_nt = TCGADataset(X_prot_fold_filled, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                                 os_time_arr, os_status_arr, val_idx)
        model_nt = TCGASurvivalModelNoToken(len(sel_prot), len(sel_gen), 20, len(TIME_BINS_TCGA), 0.35).to(device)
        model_nt, loss_nt = train_generic(
            model_nt, DataLoader(train_ds_nt, batch_size=32, shuffle=True),
            DataLoader(val_ds_nt, batch_size=64, shuffle=False), DiscreteSurvivalLoss(TIME_BINS_TCGA), device)
        with torch.no_grad():
            full_val = DataLoader(val_ds_nt, batch_size=len(val_idx), shuffle=False)
            batch = next(iter(full_val))
            hazards_val = model_nt(batch['prot'].to(device), batch['gen'].to(device),
                                    batch['prot_missing'].to(device), batch['gen_missing'].to(device)).cpu().numpy()
        oof_risk_notoken[val_idx] = cumulative_risk_at_horizon(hazards_val, len(TIME_BINS_TCGA) - 1)

        # --- Ablation B: token + training-time synthetic missingness augmentation ---
        train_ds_aug = TCGADatasetAugmented(X_prot_fold, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                                             os_time_arr, os_status_arr, train_idx,
                                             synthetic_prot_missing_prob=0.20, synthetic_gen_missing_prob=0.10,
                                             is_train=True)
        val_ds_aug = TCGADatasetAugmented(X_prot_fold, X_gen_fold_filled, has_rppa_mask, has_mut_mask,
                                           os_time_arr, os_status_arr, val_idx, is_train=False)
        model_aug = TCGASurvivalModel(len(sel_prot), len(sel_gen), 20, len(TIME_BINS_TCGA), 0.35).to(device)
        model_aug, loss_aug = train_generic(
            model_aug, DataLoader(train_ds_aug, batch_size=32, shuffle=True),
            DataLoader(val_ds_aug, batch_size=64, shuffle=False), DiscreteSurvivalLoss(TIME_BINS_TCGA), device)
        with torch.no_grad():
            full_val = DataLoader(val_ds_aug, batch_size=len(val_idx), shuffle=False)
            batch = next(iter(full_val))
            hazards_val = model_aug(batch['prot'].to(device), batch['gen'].to(device),
                                     batch['prot_missing'].to(device), batch['gen_missing'].to(device)).cpu().numpy()
        oof_risk_augmented[val_idx] = cumulative_risk_at_horizon(hazards_val, len(TIME_BINS_TCGA) - 1)

        print(f"Fold {fold_num}: no-token loss={loss_nt:.4f}, augmented loss={loss_aug:.4f}")

    print("\n=== Architecture ablation summary ===")
    for name, risk in [("Impute+indicator (no token)", oof_risk_notoken),
                        ("Token + augmentation", oof_risk_augmented)]:
        c = concordance_index(os_time_arr, -risk, os_status_arr)
        ci = bootstrap_c_index_ci(risk, os_time_arr, os_status_arr)
        print(f"  {name:35s}: C-index={c:.4f}  95% CI=({ci[0]:.4f}, {ci[1]:.4f})")

    print("\n=== Paired bootstrap significance tests ===")
    diff, lo, hi, p = paired_bootstrap_c_index_diff(oof_risk_augmented, oof_risk_notoken, os_time_arr, os_status_arr)
    print(f"  Augmented token - Impute+indicator: diff={diff:+.4f}  95% CI=({lo:.4f},{hi:.4f})  p={p:.4f}")
    diff2, lo2, hi2, p2 = paired_bootstrap_c_index_diff(oof_risk_augmented, oof_risk_v2, os_time_arr, os_status_arr)
    print(f"  Augmented token - Non-augmented token: diff={diff2:+.4f}  95% CI=({lo2:.4f},{hi2:.4f})  p={p2:.4f}")
    print("\nNone of these should reach significance (p<0.05) -- consistent with the paper's finding "
          "that architectural complexity is not empirically justified at this sample size.")

    np.save(os.path.join(OUT_DIR, "oof_risk_notoken.npy"), oof_risk_notoken)
    np.save(os.path.join(OUT_DIR, "oof_risk_augmented.npy"), oof_risk_augmented)
    print(f"\n✓ Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
