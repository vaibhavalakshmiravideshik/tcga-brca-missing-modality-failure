"""
Stage 1: Data acquisition and cohort construction.

Pulls TCGA-BRCA PanCancer Atlas clinical, RPPA proteomic, and somatic
mutation data from the public cBioPortal REST API, verifies real survival
labels (no fabricated/imputed outcomes), and constructs the aligned
proteomic + genomic + clinical cohort used by every downstream script.

Output: results/X_prot_final.csv, results/X_gen_tcga.csv, results/target_df.csv
"""
import requests
import pandas as pd
import numpy as np
import os

from utils import BRCA_DRIVER_PANEL

CBIOPORTAL_API = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga_pan_can_atlas_2018"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def cbio_get(endpoint, params=None):
    r = requests.get(f"{CBIOPORTAL_API}{endpoint}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def cbio_get_all_clinical(study_id, page_size=1000):
    records, page = [], 0
    while True:
        resp = requests.get(
            f"{CBIOPORTAL_API}/studies/{study_id}/clinical-data",
            params={"clinicalDataType": "PATIENT", "pageSize": page_size,
                    "pageNumber": page, "projection": "DETAILED"}, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        records.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return records


def cbio_get_molecular_data(molecular_profile_id, sample_ids, chunk_size=500, projection="DETAILED"):
    all_data = []
    for i in range(0, len(sample_ids), chunk_size):
        chunk = sample_ids[i:i + chunk_size]
        resp = requests.post(
            f"{CBIOPORTAL_API}/molecular-profiles/{molecular_profile_id}/molecular-data/fetch",
            json={"sampleIds": chunk}, params={"projection": projection}, timeout=120)
        resp.raise_for_status()
        all_data.extend(resp.json())
    return all_data


def cbio_get_mutations(profile_id, sample_ids, chunk_size=500):
    all_data = []
    for i in range(0, len(sample_ids), chunk_size):
        chunk = sample_ids[i:i + chunk_size]
        resp = requests.post(
            f"{CBIOPORTAL_API}/molecular-profiles/{profile_id}/mutations/fetch",
            json={"sampleIds": chunk}, params={"projection": "DETAILED"}, timeout=120)
        resp.raise_for_status()
        all_data.extend(resp.json())
    return all_data


def main():
    print("=== Step 1: Clinical data + survival verification ===")
    clinical_records = cbio_get_all_clinical(STUDY_ID)
    clin_df_raw = pd.DataFrame(clinical_records)
    clin_wide = clin_df_raw.pivot_table(
        index='patientId', columns='clinicalAttributeId', values='value', aggfunc='first')

    # Real, verified survival columns only -- no fallback synthetic labels.
    valid_os_mask = clin_wide['OS_STATUS'].notna() & pd.to_numeric(clin_wide['OS_MONTHS'], errors='coerce').notna()
    clin_valid = clin_wide[valid_os_mask].copy()
    clin_valid['OS_Status'] = clin_valid['OS_STATUS'].str.startswith('1').astype(int)
    clin_valid['OS_Time_Days'] = pd.to_numeric(clin_valid['OS_MONTHS'], errors='coerce') * 30.4375
    valid_patient_ids = clin_valid.index.tolist()
    print(f"Valid-survival patients: {len(clin_valid)}, events: {int(clin_valid['OS_Status'].sum())}")

    print("\n=== Step 2: RPPA proteomics ===")
    all_samples = cbio_get(f"/studies/{STUDY_ID}/samples", params={"pageSize": 10000})
    sample_df = pd.DataFrame(all_samples)[['sampleId', 'patientId']]
    sample_df = sample_df[sample_df['patientId'].isin(valid_patient_ids)]
    sample_ids = sample_df['sampleId'].tolist()

    rppa_raw = cbio_get_molecular_data(f"{STUDY_ID}_rppa_Zscores", sample_ids, projection="DETAILED")
    rppa_df = pd.DataFrame(rppa_raw)
    rppa_df['hugoGeneSymbol'] = rppa_df['gene'].apply(lambda g: g.get('hugoGeneSymbol'))
    rppa_wide = rppa_df.pivot_table(index='sampleId', columns='hugoGeneSymbol', values='value', aggfunc='first')
    print(f"RPPA: {rppa_wide.shape}, samples with RPPA: {rppa_wide.shape[0]}")

    print("\n=== Step 3: Mutations + fixed driver panel ===")
    mut_raw = cbio_get_mutations(f"{STUDY_ID}_mutations", sample_ids)
    mut_df = pd.DataFrame(mut_raw)
    mut_df['hugoGeneSymbol'] = mut_df['gene'].apply(lambda g: g.get('hugoGeneSymbol'))
    samples_with_mutations = set(mut_df['sampleId'].unique())

    mut_filtered = mut_df[mut_df['hugoGeneSymbol'].isin(BRCA_DRIVER_PANEL)]
    gen_wide = pd.crosstab(mut_filtered['sampleId'], mut_filtered['hugoGeneSymbol'])
    gen_wide = (gen_wide > 0).astype(int)

    print("\n=== Step 4: Align proteomics/genomics/clinical, handle missingness ===")
    master_index = sample_df.set_index('sampleId')
    X_prot_raw = rppa_wide.reindex(master_index.index)
    has_rppa = X_prot_raw.notna().any(axis=1)

    gen_wide_reindexed = gen_wide.reindex(columns=BRCA_DRIVER_PANEL, fill_value=0).reindex(master_index.index)
    assayed_mask = master_index.index.isin(samples_with_mutations)
    # Modality-level missingness (no mutation calling at all) stays NaN --
    # genuinely different from "assayed, wildtype" (0).
    gen_wide_reindexed.loc[~assayed_mask] = np.nan
    gen_wide_reindexed.loc[assayed_mask] = gen_wide_reindexed.loc[assayed_mask].fillna(0)
    has_mutation_data = pd.Series(assayed_mask, index=master_index.index)
    X_gen_tcga = gen_wide_reindexed

    target_df = master_index.copy()
    target_df['OS_Time'] = clin_valid.loc[target_df['patientId'], 'OS_Time_Days'].values
    target_df['OS_Status'] = clin_valid.loc[target_df['patientId'], 'OS_Status'].values
    target_df['Has_RPPA'] = has_rppa.values
    target_df['Has_Mutation_Data'] = has_mutation_data.values
    target_df['AJCC_Stage'] = clin_valid.loc[target_df['patientId'], 'AJCC_PATHOLOGIC_TUMOR_STAGE'].values
    target_df['Subtype'] = clin_valid.loc[target_df['patientId'], 'SUBTYPE'].values
    target_df = target_df.reset_index(drop=True)

    # Within-panel (individual antibody probe) missingness for patients who
    # DO have RPPA: train-partition-agnostic median impute here since this
    # script has no train/test split yet; downstream scripts re-impute
    # using train-only statistics per their own split/fold.
    X_prot_raw = X_prot_raw.reset_index(drop=True)
    has_rppa_mask = target_df['Has_RPPA'].values.astype(bool)
    protein_medians = X_prot_raw.loc[has_rppa_mask].median()
    X_prot_final = X_prot_raw.copy()
    X_prot_final.loc[has_rppa_mask] = X_prot_final.loc[has_rppa_mask].fillna(protein_medians)
    X_gen_tcga = X_gen_tcga.reset_index(drop=True)

    print(f"\nFinal cohort: {len(target_df)} patients, {int(target_df['OS_Status'].sum())} events")
    print(f"RPPA coverage: {target_df['Has_RPPA'].mean():.1%}, "
          f"Mutation coverage: {target_df['Has_Mutation_Data'].mean():.1%}")

    X_prot_final.to_csv(os.path.join(OUT_DIR, "X_prot_final.csv"), index=False)
    X_gen_tcga.to_csv(os.path.join(OUT_DIR, "X_gen_tcga.csv"), index=False)
    target_df.to_csv(os.path.join(OUT_DIR, "target_df.csv"), index=False)
    print(f"\n✓ Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
