"""
Shared utilities for the proteogenomic missing-modality failure analysis pipeline.

Contains: PyTorch dataset/model/loss classes, discrete-time survival helpers,
Cox-based feature screening, and evaluation metrics (C-index, time-dependent
AUC, censoring-aware net benefit) used consistently across all pipeline stages.

Corresponding paper: "Architectural Complexity and Miscalibration in
Proteogenomic Survival Prediction: A Failure Analysis on Real Clinical
Missingness" (ICBINB-BIO, NeurIPS 2026 Workshop).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

TIME_BINS_TCGA = [365.0, 1095.0, 1825.0, 2555.0]  # 1y, 3y, 5y, 7y
HORIZON_LABELS_TCGA = ["1y", "3y", "5y", "7y"]
HORIZON_DCA = 1825.0  # 5-year, matches primary reported results

# TCGA-BRCA-relevant fixed 44-gene driver panel (avoids per-cohort top-N
# selection instability across cross-validation folds).
BRCA_DRIVER_PANEL = sorted(set([
    'TP53', 'PIK3CA', 'GATA3', 'MAP3K1', 'CDH1', 'PTEN', 'AKT1', 'ERBB2', 'ESR1',
    'MYC', 'CCND1', 'RB1', 'BRCA1', 'BRCA2', 'NF1', 'ARID1A', 'KMT2C', 'FOXA1',
    'RUNX1', 'TBX3', 'CBFB', 'PIK3R1', 'NCOR1', 'SF3B1', 'CTCF', 'MAP2K4',
    'CDKN1B', 'SMAD4', 'ATR', 'STK11', 'ERBB3', 'KRAS', 'NRAS', 'BRAF', 'EGFR',
    'MTOR', 'TSC1', 'TSC2', 'CDKN2A', 'MDM2', 'MDM4', 'ATM', 'CHEK2', 'PALB2',
]))

STAGE_MAP = {
    'STAGE I': 1, 'STAGE IA': 1, 'STAGE IB': 1,
    'STAGE II': 2, 'STAGE IIA': 2, 'STAGE IIB': 2,
    'STAGE III': 3, 'STAGE IIIA': 3, 'STAGE IIIB': 3, 'STAGE IIIC': 3,
    'STAGE IV': 4,
}
SUBTYPE_RISK_MAP = {
    'BRCA_LumA': 1, 'BRCA_Normal': 1, 'BRCA_LumB': 2, 'BRCA_Her2': 3, 'BRCA_Basal': 3,
}


# ==============================================================================
# PyTorch dataset and model
# ==============================================================================
class TCGADataset(Dataset):
    """Wraps proteomic/genomic features, modality-missingness masks, and
    survival targets. Missing-modality rows are zero-filled in the tensor;
    the boolean mask (not the zero value) is what tells the model a
    modality is absent, consumed by the missing-token substitution in
    TCGASurvivalModel.forward()."""

    def __init__(self, X_prot, X_gen, prot_present_mask, gen_present_mask,
                 os_time, os_status, indices):
        idx = indices
        self.X_prot = torch.tensor(X_prot.iloc[idx].fillna(0.0).values, dtype=torch.float32)
        self.X_gen = torch.tensor(X_gen.iloc[idx].values, dtype=torch.float32)
        self.prot_missing = torch.tensor((~prot_present_mask[idx]).astype(np.float32))
        self.gen_missing = torch.tensor((~gen_present_mask[idx]).astype(np.float32))
        self.os_time = torch.tensor(os_time[idx], dtype=torch.float32)
        self.os_status = torch.tensor(os_status[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.os_time)

    def __getitem__(self, i):
        return {
            'prot': self.X_prot[i], 'gen': self.X_gen[i],
            'prot_missing': self.prot_missing[i], 'gen_missing': self.gen_missing[i],
            'os_time': self.os_time[i], 'os_status': self.os_status[i],
        }


class TCGASurvivalModel(nn.Module):
    """Compact discrete-time survival network with a learnable missing-
    modality token. A trainable embedding vector is substituted for a
    modality's raw input whenever that modality is flagged absent for a
    given patient. Outputs conditional hazard probabilities across
    len(TIME_BINS_TCGA) discrete horizons."""

    def __init__(self, num_prot, num_gen, hidden_dim=20, n_time_bins=4, dropout=0.35):
        super().__init__()
        self.missing_prot_token = nn.Parameter(torch.randn(num_prot) * 0.02)
        self.missing_gen_token = nn.Parameter(torch.randn(num_gen) * 0.02)
        in_dim = num_prot + num_gen
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_time_bins), nn.Sigmoid(),
        )

    def forward(self, x_prot, x_gen, prot_missing, gen_missing):
        B = x_prot.size(0)
        prot_tok = self.missing_prot_token.unsqueeze(0).repeat(B, 1)
        gen_tok = self.missing_gen_token.unsqueeze(0).repeat(B, 1)
        x_prot = torch.where(prot_missing.unsqueeze(1).bool(), prot_tok, x_prot)
        x_gen = torch.where(gen_missing.unsqueeze(1).bool(), gen_tok, x_gen)
        return self.net(torch.cat([x_prot, x_gen], dim=1))


class TCGASurvivalModelNoToken(nn.Module):
    """Ablation baseline: identical to TCGASurvivalModel but without the
    learnable missing token. Missing modalities are median-imputed with a
    binary missingness-indicator feature appended per modality — the
    classical imputation+indicator approach compared against the token
    mechanism in 04_architecture_ablation.py."""

    def __init__(self, num_prot, num_gen, hidden_dim=20, n_time_bins=4, dropout=0.35):
        super().__init__()
        in_dim = num_prot + num_gen + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_time_bins), nn.Sigmoid(),
        )

    def forward(self, x_prot, x_gen, prot_missing, gen_missing):
        x = torch.cat([x_prot, x_gen, prot_missing.unsqueeze(1), gen_missing.unsqueeze(1)], dim=1)
        return self.net(x)


class DiscreteSurvivalLoss(nn.Module):
    """Censoring-aware discrete-time negative log-likelihood. Each patient
    contributes to exactly the intervals in which they were observed to be
    at risk: -log(hazard) if the event occurred in that interval, -log(1 -
    hazard) if they survived through it (event/censoring later, or
    ongoing follow-up), and no contribution to intervals after their
    follow-up ended."""

    def __init__(self, time_bins):
        super().__init__()
        self.time_bins = time_bins

    def forward(self, hazards, os_time, os_status):
        hazards = torch.clamp(hazards, 1e-4, 1.0 - 1e-4)
        n_bins = hazards.shape[1]
        device_ = hazards.device
        bin_starts = torch.tensor([0.0] + self.time_bins[:-1], device=device_)
        bin_ends = torch.tensor(self.time_bins, device=device_)
        total_loss = 0.0
        for b in range(n_bins):
            at_risk = (os_time >= bin_starts[b]).float()
            event_in_bin = ((os_time >= bin_starts[b]) & (os_time < bin_ends[b]) & (os_status == 1)).float()
            survived_bin = (at_risk - event_in_bin) * (((os_time >= bin_ends[b]) | (os_status == 0)).float())
            p = hazards[:, b]
            loss_b = (-event_in_bin * torch.log(p) - survived_bin * torch.log(1.0 - p)).sum() / (at_risk.sum() + 1e-6)
            total_loss = total_loss + loss_b
        return total_loss / n_bins


# ==============================================================================
# Feature screening
# ==============================================================================
def univariate_cox_screen_fold(features_df, os_time, os_status, fold_train_idx,
                                 valid_mask, top_k):
    """Univariate Cox screening restricted to a single fold's own training
    partition (never validation-fold data), avoiding the feature-selection
    leakage documented in Appendix A of the paper. Returns the top_k
    feature names by univariate p-value."""
    eligible_idx = fold_train_idx[valid_mask[fold_train_idx]]
    train_df = features_df.iloc[eligible_idx].copy()
    train_df['OS_Time'] = os_time[eligible_idx]
    train_df['OS_Status'] = os_status[eligible_idx]
    results = []
    for col in features_df.columns:
        sub = train_df[[col, 'OS_Time', 'OS_Status']].dropna()
        if sub[col].nunique() < 2 or len(sub) < 30:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col='OS_Time', event_col='OS_Status', show_progress=False)
            results.append((col, cph.summary['p'].values[0]))
        except Exception:
            continue
    results_df = pd.DataFrame(results, columns=['feature', 'p_value']).sort_values('p_value')
    return results_df.head(top_k)['feature'].tolist()


# ==============================================================================
# Evaluation metrics
# ==============================================================================
def cumulative_risk_at_horizon(hazards, horizon_idx):
    """Converts per-interval conditional hazards into cumulative event risk
    by the end of interval horizon_idx: 1 - prod(1 - hazard_b)."""
    return 1.0 - np.prod(1.0 - hazards[:, :horizon_idx + 1], axis=1)


def km_event_prob_by_horizon(times, statuses, horizon):
    """Kaplan-Meier-estimated P(event by horizon), robust to censoring."""
    if len(times) < 5 or statuses.sum() == 0:
        return np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=statuses)
    try:
        return 1.0 - kmf.survival_function_at_times(horizon).values[0]
    except Exception:
        return np.nan


def time_dependent_auc(risk_scores, os_time, os_status, horizon_days):
    """AUC for predicting event-by-horizon; patients censored before the
    horizon with no observed event are excluded (standard practice, since
    their true status at the horizon is unknown)."""
    event_by_horizon = ((os_time <= horizon_days) & (os_status == 1)).astype(int)
    ambiguous = (os_time < horizon_days) & (os_status == 0)
    keep = ~ambiguous
    if len(np.unique(event_by_horizon[keep])) < 2:
        return np.nan, keep.sum()
    return roc_auc_score(event_by_horizon[keep], risk_scores[keep]), keep.sum()


def bootstrap_ci(risk_scores, os_time, os_status, horizon_days, n_boot=1000, seed=42):
    """Bootstrap 95% CI for time-dependent AUC."""
    rng = np.random.RandomState(seed)
    n = len(risk_scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        auc, _ = time_dependent_auc(risk_scores[idx], os_time[idx], os_status[idx], horizon_days)
        if not np.isnan(auc):
            boots.append(auc)
    if len(boots) < 100:
        return (np.nan, np.nan)
    return (np.percentile(boots, 2.5), np.percentile(boots, 97.5))


def bootstrap_c_index_ci(risk_scores, os_time, os_status, n_boot=1000, seed=42):
    """Bootstrap 95% CI for concordance index."""
    rng = np.random.RandomState(seed)
    n = len(risk_scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(os_status[idx])) < 2:
            continue
        try:
            boots.append(concordance_index(os_time[idx], -risk_scores[idx], os_status[idx]))
        except Exception:
            continue
    return np.percentile(boots, [2.5, 97.5])


def paired_bootstrap_c_index_diff(risk_a, risk_b, times, statuses, n_boot=2000, seed=42):
    """Paired bootstrap test for a C-index difference between two risk
    scores computed on the SAME patients (e.g. two competing models).
    Uses identical resampled indices for both scores each iteration so the
    comparison is properly paired, not just two independent CIs eyeballed
    for overlap. Returns (point_diff, ci_lo, ci_hi, two_sided_p_value)."""
    rng = np.random.RandomState(seed)
    n = len(risk_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(statuses[idx])) < 2:
            continue
        try:
            c_a = concordance_index(times[idx], -risk_a[idx], statuses[idx])
            c_b = concordance_index(times[idx], -risk_b[idx], statuses[idx])
            diffs.append(c_a - c_b)
        except Exception:
            continue
    diffs = np.array(diffs)
    point_diff = concordance_index(times, -risk_a, statuses) - concordance_index(times, -risk_b, statuses)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return point_diff, ci_lo, ci_hi, p_value


def net_benefit_curve(risk_scores, times, statuses, horizon, thresholds):
    """Censoring-aware net benefit (Vickers-style decision curve analysis),
    using Kaplan-Meier-estimated event probability within each threshold-
    defined high-risk subgroup rather than naive binary classification
    counting, which would be biased by patients censored before the
    horizon."""
    n = len(risk_scores)
    net_benefits = []
    for pt in thresholds:
        high_risk_mask = risk_scores >= pt
        n_high = high_risk_mask.sum()
        if n_high == 0:
            net_benefits.append(0.0)
            continue
        p_event = km_event_prob_by_horizon(times[high_risk_mask], statuses[high_risk_mask], horizon)
        if np.isnan(p_event):
            net_benefits.append(np.nan)
            continue
        frac_high = n_high / n
        nb = frac_high * p_event - frac_high * (1.0 - p_event) * (pt / (1.0 - pt))
        net_benefits.append(nb)
    return np.array(net_benefits)


def net_benefit_with_ci(risk_scores, times, statuses, horizon, thresholds, n_boot=200, seed=42):
    """Net benefit curve with bootstrap 95% confidence bands."""
    point = net_benefit_curve(risk_scores, times, statuses, horizon, thresholds)
    rng = np.random.RandomState(seed)
    n = len(risk_scores)
    boot_curves = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_curves.append(net_benefit_curve(risk_scores[idx], times[idx], statuses[idx], horizon, thresholds))
    boot_curves = np.array(boot_curves)
    ci_lo = np.nanpercentile(boot_curves, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_curves, 97.5, axis=0)
    return point, ci_lo, ci_hi
