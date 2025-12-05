import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

# ==========================
# Configuration (EDIT HERE)
# ==========================
K = 50  # Top-k genes for the panel (change this to 16, 24, 32, 50, etc.)

DATA_DIR = Path("data/raw")
TRAIN_CSV = DATA_DIR / "data_set_ALL_AML_train.csv"
LABELS_CSV = DATA_DIR / "actual.csv"

OUTPUT_DIR = Path("./results_golub_panel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# Helper functions (from notebook)
# ==========================

def _safe_sd(x, eps: float = 1e-9) -> float:
    """Unbiased sample std (ddof=1) with epsilon to avoid division by zero."""
    return max(np.std(x, ddof=1), eps)


def load_golub_train(train_csv: str | Path, labels_csv: str | Path):
    """
    Load Golub training data and labels.

    Returns
    -------
    X_train : pd.DataFrame
        Gene expression matrix with shape (n_genes, n_samples).
    y_train : np.ndarray
        Array of labels 'ALL' / 'AML' of length n_samples.
    patient_ids : np.ndarray
        Array of patient IDs.
    gene_accessions : dict
        Mapping from gene description to accession number.
    """
    train_csv = Path(train_csv)
    labels_csv = Path(labels_csv)

    # Load expression matrix - need both gene description and accession
    df_raw = pd.read_csv(train_csv)
    gene_descriptions = df_raw['Gene Description'].tolist()
    gene_accessions_list = df_raw['Gene Accession Number'].tolist()
    
    # Create mapping from description to accession
    gene_accessions = dict(zip(gene_descriptions, gene_accessions_list))
    
    # Set gene description as index
    X_train_raw = df_raw.set_index('Gene Description').drop(columns=['Gene Accession Number'])
    
    # Remove call columns
    call_cols = [c for c in X_train_raw.columns if c.startswith('call')]
    X_train_raw = X_train_raw.drop(columns=call_cols, errors='ignore')
    
    print(f"Raw X_train shape: {X_train_raw.shape}")

    # Remove duplicate gene names (keep first occurrence)
    if X_train_raw.index.duplicated().any():
        n_duplicates = X_train_raw.index.duplicated().sum()
        print(f"Warning: Found {n_duplicates} duplicate gene names. Keeping first occurrence only.")
        X_train_raw = X_train_raw[~X_train_raw.index.duplicated(keep="first")]
        print(f"After removing duplicates: {X_train_raw.shape}")

    # Load labels
    labels = pd.read_csv(labels_csv)
    print(f"Labels shape: {labels.shape}")
    print(f"Label columns: {labels.columns.tolist()}")

    # Identify ID and diagnosis columns heuristically
    id_col_candidates = [c for c in labels.columns if "id" in c.lower() or "sample" in c.lower() or "patient" in c.lower()]
    cls_col_candidates = [
        c
        for c in labels.columns
        if "diag" in c.lower()
        or "class" in c.lower()
        or "type" in c.lower()
        or "cancer" in c.lower()
    ]

    id_col = id_col_candidates[0] if id_col_candidates else labels.columns[0]
    cls_col = cls_col_candidates[0] if cls_col_candidates else labels.columns[1]

    labels = labels[[id_col, cls_col]].rename(columns={id_col: "sample_id", cls_col: "diagnosis"})
    labels["diagnosis"] = labels["diagnosis"].str.upper().str.strip()
    labels["sample_id"] = labels["sample_id"].astype(str)

    assert set(labels["diagnosis"]).issubset({"ALL", "AML"}), "Labels must be ALL or AML"

    # Determine orientation: are samples in columns or rows?
    sample_ids = set(labels["sample_id"])
    cols_overlap = len(sample_ids & set(map(str, X_train_raw.columns)))
    rows_overlap = len(sample_ids & set(map(str, X_train_raw.index)))

    print(f"Overlap with columns: {cols_overlap}, with rows: {rows_overlap}")

    if rows_overlap > cols_overlap:
        X_train_raw = X_train_raw.T
        print("Transposed X_train (samples were rows, now columns)")

    # Align and reorder columns to match label order
    common_samples = [s for s in labels["sample_id"] if s in X_train_raw.columns]
    X_train = X_train_raw.loc[:, common_samples]
    labels = labels[labels["sample_id"].isin(common_samples)].reset_index(drop=True)

    # Create aligned label and patient ID arrays
    y_train = labels.set_index("sample_id").loc[X_train.columns, "diagnosis"].values
    patient_ids = np.array([int(s) for s in X_train.columns])

    print(f"\nFinal X_train shape (genes x samples): {X_train.shape}")
    print("Class distribution:")
    print(pd.Series(y_train).value_counts())
    print(f"\nFirst 5 samples: {X_train.columns[:5].tolist()}")

    return X_train, y_train, patient_ids, gene_accessions


def balance_patient_samples_snr(X: pd.DataFrame, y: np.ndarray, patient_ids: np.ndarray):
    """Balance samples by taking equal numbers of ALL and AML patients.

    NOTE: This balances PATIENTS (samples/data points), NOT genes (features).
    X is (genes x samples), so we select columns.
    """
    all_indices = np.where(y == 'ALL')[0]
    aml_indices = np.where(y == 'AML')[0]
    
    n_minority = min(len(all_indices), len(aml_indices))
    
    # Take all from minority, first n from majority
    if len(aml_indices) <= len(all_indices):
        selected_aml = aml_indices
        selected_all = all_indices[:n_minority]
    else:
        selected_all = all_indices
        selected_aml = aml_indices[:n_minority]
    
    selected_indices = np.concatenate([selected_all, selected_aml])
    
    X_balanced = X.iloc[:, selected_indices]
    y_balanced = y[selected_indices]
    patient_ids_balanced = patient_ids[selected_indices]
    
    print(f"[INFO] Balanced samples: {len(selected_all)} ALL + {len(selected_aml)} AML = {len(selected_indices)} total")
    
    return X_balanced, y_balanced, patient_ids_balanced


def compute_P_scores(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """
    Compute Golub P-scores:

        P(g,c) = (μ_ALL - μ_AML) / (σ_ALL + σ_AML)

    for each gene g, where μ_c and σ_c are the mean and std within class c.
    X is assumed to have shape (n_genes, n_samples), index = gene names.
    """
    mask_ALL = (y == "ALL")
    mask_AML = (y == "AML")

    P = {}
    for gene, row in X.iterrows():
        x_all = row.values[mask_ALL]
        x_aml = row.values[mask_AML]

        mu_all = np.mean(x_all)
        mu_aml = np.mean(x_aml)
        sd_all = _safe_sd(x_all)
        sd_aml = _safe_sd(x_aml)

        P[gene] = (mu_all - mu_aml) / (sd_all + sd_aml)

    # Sort by |P| descending (as in the notebook)
    return pd.Series(P).sort_values(key=lambda s: s.abs(), ascending=False)


def fit_weighted_voting(X: pd.DataFrame, y: np.ndarray, genes: list[str]):
    """
    Fit Golub's weighted voting classifier on a selected set of genes.

    Returns
    -------
    a : pd.Series
        P(g,c) weights with sign, indexed by gene.
    b : pd.Series
        Thresholds b_g = 0.5 * (μ_ALL + μ_AML), indexed by gene.
    """
    mask_ALL = (y == "ALL")
    mask_AML = (y == "AML")

    a = {}
    b = {}

    for g in genes:
        x_all = X.loc[g, mask_ALL].astype(float)
        x_aml = X.loc[g, mask_AML].astype(float)

        mu_all = x_all.mean()
        mu_aml = x_aml.mean()
        sd_all = _safe_sd(x_all.values)
        sd_aml = _safe_sd(x_aml.values)

        a[g] = (mu_all - mu_aml) / (sd_all + sd_aml)   # P(g,c)
        b[g] = 0.5 * (mu_all + mu_aml)                # midpoint threshold

    return pd.Series(a), pd.Series(b)


def predict_weighted_voting(X: pd.DataFrame, a: pd.Series, b: pd.Series):
    """
    Predict labels for samples in X using the weighted voting model (a, b).

    X is (n_genes, n_samples), index = gene names.
    """
    assert len(a) == len(b), f"Mismatch: a has {len(a)} genes, b has {len(b)} genes"
    assert all(a.index == b.index), "a and b must have the same gene order"

    model_genes = set(a.index)
    X_genes = set(X.index)
    common_genes = list(model_genes & X_genes)

    # Keep order from a.index
    common_genes = [g for g in a.index if g in common_genes]

    a_sub = a.loc[common_genes]
    b_sub = b.loc[common_genes]
    X_sub = X.loc[common_genes].astype(float)

    assert len(a_sub) == len(b_sub) == len(X_sub), (
        f"Shape mismatch: a_sub={len(a_sub)}, b_sub={len(b_sub)}, X_sub={len(X_sub)}"
    )

    votes_all = []
    votes_aml = []

    for col in X_sub.columns:
        x_sample = X_sub[col].values
        v_g = a_sub.values * (x_sample - b_sub.values)

        V_all = np.sum(np.abs(v_g[v_g >= 0]))
        V_aml = np.sum(np.abs(v_g[v_g < 0]))

        votes_all.append(V_all)
        votes_aml.append(V_aml)

    votes_all = np.array(votes_all)
    votes_aml = np.array(votes_aml)

    y_pred = np.where(votes_all >= votes_aml, "ALL", "AML")
    ps = np.abs(votes_all - votes_aml) / (votes_all + votes_aml + 1e-12)

    return y_pred, ps


# ==========================
# Main panel generation
# ==========================

def load_independent_set(ind_csv: str | Path, labels_csv: str | Path):
    """Load independent test set."""
    ind_csv = Path(ind_csv)
    labels_csv = Path(labels_csv)

    df_raw = pd.read_csv(ind_csv)
    
    # Get gene accession mapping
    gene_descriptions = df_raw['Gene Description'].tolist()
    gene_accessions_list = df_raw['Gene Accession Number'].tolist()
    gene_accessions = dict(zip(gene_descriptions, gene_accessions_list))
    
    X_ind_raw = df_raw.set_index('Gene Description').drop(columns=['Gene Accession Number'])
    
    # Remove call columns
    call_cols = [c for c in X_ind_raw.columns if c.startswith('call')]
    X_ind_raw = X_ind_raw.drop(columns=call_cols, errors='ignore')

    if X_ind_raw.index.duplicated().any():
        X_ind_raw = X_ind_raw[~X_ind_raw.index.duplicated(keep="first")]

    labels = pd.read_csv(labels_csv)
    id_col = [c for c in labels.columns if "id" in c.lower() or "sample" in c.lower() or "patient" in c.lower()]
    cls_col = [c for c in labels.columns if "diag" in c.lower() or "class" in c.lower() or "type" in c.lower() or "cancer" in c.lower()]
    id_col = id_col[0] if id_col else labels.columns[0]
    cls_col = cls_col[0] if cls_col else labels.columns[1]

    labels = labels[[id_col, cls_col]].rename(columns={id_col: "sample_id", cls_col: "diagnosis"})
    labels["diagnosis"] = labels["diagnosis"].str.upper().str.strip()
    labels["sample_id"] = labels["sample_id"].astype(str)

    # Independent set samples are 39-72
    ind_sample_ids = [str(i) for i in range(39, 73)]
    common_samples = [s for s in ind_sample_ids if s in X_ind_raw.columns]
    X_ind = X_ind_raw.loc[:, common_samples]

    labels_ind = labels[labels["sample_id"].isin(common_samples)].reset_index(drop=True)
    y_ind = labels_ind.set_index("sample_id").loc[X_ind.columns, "diagnosis"].values
    patient_ids = np.array([int(s) for s in X_ind.columns])

    return X_ind, y_ind, patient_ids, gene_accessions


def split_train_test(
    X: pd.DataFrame,
    y: np.ndarray,
    patient_ids: np.ndarray,
    test_ratio: float
):
    """Split data into internal train and test sets (stratified).

    Note: X is (samples x genes) for this function (transposed from main SNR format).
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=y,
        random_state=42
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    patient_ids_train = patient_ids[train_idx]

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]
    patient_ids_test = patient_ids[test_idx]

    return X_train, y_train, patient_ids_train, X_test, y_test, patient_ids_test


def split_cross_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    patient_ids: np.ndarray,
    n_folds: int
):
    """Split data into k-fold cross-validation sets (stratified).

    Note: X is (samples x genes) for this function (transposed from main SNR format).
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        patient_ids_train = patient_ids[train_idx]

        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y[test_idx]
        patient_ids_test = patient_ids[test_idx]

        folds.append((X_train, y_train, patient_ids_train, X_test, y_test, patient_ids_test))

    return folds


def run_snr_selection(
    k: int,
    out_dir: str = "results",
    train_csv: str = None,
    ind_csv: str = None,
    labels_csv: str = None,
    use_all_data: bool = False,
    balanced_genes: bool = False,  # NEW - gene balance only
    use_balanced_patients: bool = False,  # NEW - sample balance only
    validation_strategy: dict | None = None,  # NEW - {"method": "split", "ratio": 0.7} or {"method": "cv", "folds": 5}
):
    """Run SNR feature selection and output train/test CSVs.

    Args:
        k: Number of top genes to select
        out_dir: Output directory
        train_csv: Path to training CSV
        ind_csv: Path to independent test CSV
        labels_csv: Path to labels CSV
        use_all_data: If True, combine train+independent for selection and output single file
        balanced_genes: Select k/2 ALL-favoring + k/2 AML-favoring genes
        use_balanced_patients: Balance patient samples (11 ALL + 11 AML)
        validation_strategy: Internal validation config (train/test split or CV)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = train_csv or DATA_DIR / "data_set_ALL_AML_train.csv"
    ind_csv = ind_csv or DATA_DIR / "data_set_ALL_AML_independent.csv"
    labels_csv = labels_csv or LABELS_CSV

    print(f"\n=== SNR Feature Selection (K={k}) ===\n")

    # Load training data
    X_train, y_train, patient_ids, gene_accessions = load_golub_train(train_csv, labels_csv)
    
    # If using all data, also load and combine independent set
    X_ind = None
    y_ind = None
    patient_ids_ind = None
    if use_all_data and ind_csv and Path(ind_csv).exists():
        print(f"[INFO] Loading independent data for combined selection...")
        X_ind, y_ind, patient_ids_ind, gene_accessions_ind = load_independent_set(ind_csv, labels_csv)
        # Update gene_accessions with any from independent set
        gene_accessions.update(gene_accessions_ind)
        # Combine train and independent (both are genes x samples)
        X_train = pd.concat([X_train, X_ind], axis=1)
        y_train = np.concatenate([y_train, y_ind])
        patient_ids = np.concatenate([patient_ids, patient_ids_ind])
        print(f"[INFO] Combined data shape: {X_train.shape}, Labels: ALL={sum(y_train=='ALL')}, AML={sum(y_train=='AML')}")
    
    # Balance patient samples if requested (equal ALL and AML patients)
    if use_balanced_patients:
        X_train, y_train, patient_ids = balance_patient_samples_snr(X_train, y_train, patient_ids)

    # Compute P-scores
    P_all = compute_P_scores(X_train, y_train)

    # Select genes based on gene balance flag
    if balanced_genes:
        # Balanced gene selection: k/2 ALL-favoring (positive P) + k/2 AML-favoring (negative P)
        if k % 2 != 0:
            k += 1
            print(f"[INFO] k must be even for balanced selection. Adjusted to k={k}.")
        
        # Get ALL-favoring genes (positive P, sorted by highest)
        all_genes = P_all[P_all > 0].sort_values(ascending=False)
        all_biomarkers = all_genes.index[:k//2].tolist()
        
        # Get AML-favoring genes (negative P, sorted by most negative)
        aml_genes = P_all[P_all < 0].sort_values(ascending=True)
        aml_biomarkers = aml_genes.index[:k//2].tolist()
        
        topk_genes = all_biomarkers + aml_biomarkers
        print(f"[INFO] Balanced gene selection: {len(all_biomarkers)} ALL + {len(aml_biomarkers)} AML biomarkers")
    else:
        # Unbalanced: just top-k by |P|
        topk_genes = P_all.index[:k].tolist()
    
    # Get accession numbers for selected genes
    topk_accessions = [gene_accessions.get(g, g) for g in topk_genes]

    print(f"Top-{k} genes selected:")
    for i, gene in enumerate(topk_genes[:10]):
        print(f"  {i+1}. {gene[:50]}... P={P_all[gene]:.3f}")

    # Output data files based on validation strategy
    # Note: X_train is (genes x samples), need to transpose for splitting
    if validation_strategy is None or validation_strategy["method"] == "split":
        # Train/test split mode
        if validation_strategy:
            test_ratio = 1.0 - validation_strategy["ratio"]
            # Transpose to (samples x genes) for splitting
            X_train_T = X_train.loc[topk_genes].T.copy()
            X_train_T.columns = topk_accessions

            X_train_int, y_train_int, pids_train_int, X_test_int, y_test_int, pids_test_int = split_train_test(
                X_train_T, y_train, patient_ids, test_ratio
            )

            # Internal training set
            train_int_df = X_train_int.copy()
            train_int_df["cancer"] = y_train_int
            train_int_df["patient"] = pids_train_int
            train_int_df.to_csv(out_path / f"train_internal_top_{k}_snr.csv", index=False)
            print(f"[INFO] Internal train CSV saved: {out_path / f'train_internal_top_{k}_snr.csv'}")

            # Internal test set
            test_int_df = X_test_int.copy()
            test_int_df["cancer"] = y_test_int
            test_int_df["patient"] = pids_test_int
            test_int_df.to_csv(out_path / f"test_internal_top_{k}_snr.csv", index=False)
            print(f"[INFO] Internal test CSV saved: {out_path / f'test_internal_top_{k}_snr.csv'}")
        else:
            # No internal split - output full training set (backward compatibility)
            train_data = X_train.loc[topk_genes].T.copy()
            train_data.columns = topk_accessions
            train_data["cancer"] = y_train
            train_data["patient"] = patient_ids
            train_data.to_csv(out_path / f"train_top_{k}_snr.csv", index=False)
            print(f"[INFO] Train CSV saved: {out_path / f'train_top_{k}_snr.csv'}")

    elif validation_strategy["method"] == "cv":
        # Cross-validation mode
        n_folds = validation_strategy["folds"]
        # Transpose to (samples x genes) for splitting
        X_train_T = X_train.loc[topk_genes].T.copy()
        X_train_T.columns = topk_accessions

        folds = split_cross_validation(X_train_T, y_train, patient_ids, n_folds)

        for fold_idx, (X_tr, y_tr, pids_tr, X_te, y_te, pids_te) in enumerate(folds, start=1):
            # Fold training set
            fold_train_df = X_tr.copy()
            fold_train_df["cancer"] = y_tr
            fold_train_df["patient"] = pids_tr
            fold_train_df.to_csv(out_path / f"fold_{fold_idx}_train_top_{k}_snr.csv", index=False)

            # Fold test set
            fold_test_df = X_te.copy()
            fold_test_df["cancer"] = y_te
            fold_test_df["patient"] = pids_te
            fold_test_df.to_csv(out_path / f"fold_{fold_idx}_test_top_{k}_snr.csv", index=False)

        print(f"[INFO] {n_folds}-fold CV sets saved (fold_1 through fold_{n_folds})")

    # Independent test set (always generated if available)
    if ind_csv and Path(ind_csv).exists():
        if X_ind is None:
            X_ind, y_ind, patient_ids_ind, _ = load_independent_set(ind_csv, labels_csv)

        # Filter to selected genes
        common_genes = [g for g in topk_genes if g in X_ind.index]
        if len(common_genes) < k:
            print(f"Warning: Only {len(common_genes)}/{k} genes found in independent set")

        common_accessions = [gene_accessions.get(g, g) for g in common_genes]

        ind_data = X_ind.loc[common_genes].T.copy()
        ind_data.columns = common_accessions
        ind_data["cancer"] = y_ind
        ind_data["patient"] = patient_ids_ind
        ind_data.to_csv(out_path / f"independent_top_{k}_snr.csv", index=False)
        print(f"[INFO] Independent CSV saved: {out_path / f'independent_top_{k}_snr.csv'}")

    # Build scores dataframe
    scores_df = pd.DataFrame({
        "gene_description": topk_genes,
        "gene_accession": topk_accessions,
        "snr_score": [P_all[g] for g in topk_genes],
        "favors_class": ["ALL" if P_all[g] > 0 else "AML" for g in topk_genes],
    })

    # Split by class for ranking
    all_genes_df = scores_df[scores_df['favors_class'] == 'ALL'].copy()
    aml_genes_df = scores_df[scores_df['favors_class'] == 'AML'].copy()

    # Assign rank within each class (based on absolute SNR score)
    all_genes_df = all_genes_df.sort_values('snr_score', ascending=False)
    aml_genes_df = aml_genes_df.sort_values('snr_score', ascending=True)  # More negative = higher rank

    all_genes_df['rank_within_class'] = range(1, len(all_genes_df) + 1)
    aml_genes_df['rank_within_class'] = range(1, len(aml_genes_df) + 1)

    # Combine and sort by absolute SNR score for overall ranking
    scores_df = pd.concat([all_genes_df, aml_genes_df])
    scores_df = scores_df.assign(abs_snr=lambda x: x['snr_score'].abs()).sort_values('abs_snr', ascending=False)
    scores_df['overall_rank'] = range(1, len(scores_df) + 1)
    scores_df = scores_df.drop(columns=['abs_snr'])

    print(f"[INFO] SNR-based selection rankings:")
    print(f"  - ALL genes: {len(all_genes_df)} genes with positive SNR")
    print(f"  - AML genes: {len(aml_genes_df)} genes with negative SNR")

    # FILE 1: Full details with rankings and scores
    topk_output = scores_df[['overall_rank', 'rank_within_class', 'gene_description', 'gene_accession', 'snr_score', 'favors_class']]
    topk_output.to_csv(out_path / f"topk_snr_{k}genes.csv", index=False)
    print(f"[INFO] Top-k details saved to {out_path / f'topk_snr_{k}genes.csv'}")

    # FILE 2: Selected genes only (just accession numbers, one per line, no header)
    with open(out_path / f"selected_genes_snr_{k}genes.csv", 'w') as f:
        for acc in topk_accessions:
            f.write(f"{acc}\n")
    print(f"[INFO] Selected genes saved to {out_path / f'selected_genes_snr_{k}genes.csv'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SNR feature selection for Golub dataset")
    parser.add_argument("--k", type=int, default=K, help="Number of top features to select")
    parser.add_argument("--out_dir", default="results", help="Output directory")
    parser.add_argument("--train_csv", default=None, help="Training CSV path")
    parser.add_argument("--ind_csv", default=None, help="Independent test CSV path")
    parser.add_argument("--labels_csv", default=None, help="Labels CSV path")
    parser.add_argument(
        "--use_all_data",
        action="store_true",
        help="Combine train and independent data for feature selection (outputs single file)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Balance samples (equal ALL/AML patients)",
    )
    args = parser.parse_args()

    run_snr_selection(
        k=args.k,
        out_dir=args.out_dir,
        train_csv=args.train_csv,
        ind_csv=args.ind_csv,
        labels_csv=args.labels_csv,
        use_all_data=args.use_all_data,
        balanced=args.balanced,
    )


if __name__ == "__main__":
    main()