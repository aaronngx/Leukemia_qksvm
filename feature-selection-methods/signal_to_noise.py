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
    """
    train_csv = Path(train_csv)
    labels_csv = Path(labels_csv)

    # Load expression matrix with genes as index
    X_train_raw = pd.read_csv(train_csv, index_col=0)
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
    id_col_candidates = [c for c in labels.columns if "id" in c.lower() or "sample" in c.lower()]
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

    # Create aligned label array
    y_train = labels.set_index("sample_id").loc[X_train.columns, "diagnosis"].values

    print(f"\nFinal X_train shape (genes x samples): {X_train.shape}")
    print("Class distribution:")
    print(pd.Series(y_train).value_counts())
    print(f"\nFirst 5 samples: {X_train.columns[:5].tolist()}")

    return X_train, y_train


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
    X_ind_raw = df_raw.set_index('Gene Description').drop(columns=['Gene Accession Number'])

    if X_ind_raw.index.duplicated().any():
        X_ind_raw = X_ind_raw[~X_ind_raw.index.duplicated(keep="first")]

    labels = pd.read_csv(labels_csv)
    id_col = [c for c in labels.columns if "id" in c.lower() or "sample" in c.lower()]
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

    return X_ind, y_ind


def run_snr_selection(
    k: int,
    out_dir: str = "results",
    train_csv: str = None,
    ind_csv: str = None,
    labels_csv: str = None,
):
    """Run SNR feature selection and output train/test CSVs."""
    import argparse

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = train_csv or DATA_DIR / "data_set_ALL_AML_train.csv"
    ind_csv = ind_csv or DATA_DIR / "data_set_ALL_AML_independent.csv"
    labels_csv = labels_csv or LABELS_CSV

    print(f"\n=== SNR Feature Selection (K={k}) ===\n")

    # Load training data
    X_train, y_train = load_golub_train(train_csv, labels_csv)

    # Compute P-scores
    P_all = compute_P_scores(X_train, y_train)
    topk_genes = P_all.index[:k].tolist()

    print(f"Top-{k} genes selected by |P(g,c)|:")
    for i, gene in enumerate(topk_genes[:10]):
        print(f"  {i+1}. {gene[:50]}... P={P_all[gene]:.3f}")

    # Build training CSV with selected features
    train_data = X_train.loc[topk_genes].T.copy()
    train_data.columns = topk_genes  # Use actual gene names
    train_data["cancer"] = y_train  # Keep as ALL/AML strings
    train_data["patient"] = [f"{i+1}" for i in range(len(train_data))]
    train_data.to_csv(out_path / f"train_top_{k}_snr.csv", index=False)
    print(f"\nTrain CSV saved: {out_path / f'train_top_{k}_snr.csv'}")

    # Load and process independent set
    if Path(ind_csv).exists():
        X_ind, y_ind = load_independent_set(ind_csv, labels_csv)

        # Filter to selected genes
        common_genes = [g for g in topk_genes if g in X_ind.index]
        if len(common_genes) < k:
            print(f"Warning: Only {len(common_genes)}/{k} genes found in independent set")

        ind_data = X_ind.loc[common_genes].T.copy()
        ind_data.columns = common_genes  # Use actual gene names
        ind_data["cancer"] = y_ind  # Keep as ALL/AML strings
        ind_data["patient"] = [f"{i+39}" for i in range(len(ind_data))]  # Independent set is patients 39-72
        ind_data.to_csv(out_path / f"independent_top_{k}_snr.csv", index=False)
        print(f"Independent CSV saved: {out_path / f'independent_top_{k}_snr.csv'}")

    # Save gene scores
    scores_df = pd.DataFrame({
        "gene_description": topk_genes,
        "snr_score": [P_all[g] for g in topk_genes],
        "favors_class": ["ALL" if P_all[g] > 0 else "AML" for g in topk_genes],
    })
    scores_df.to_csv(out_path / f"top_{k}_snr_scores.csv", index=False)
    print(f"Scores CSV saved: {out_path / f'top_{k}_snr_scores.csv'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SNR feature selection for Golub dataset")
    parser.add_argument("--k", type=int, default=K, help="Number of top features to select")
    parser.add_argument("--out_dir", default="results", help="Output directory")
    parser.add_argument("--train_csv", default=None, help="Training CSV path")
    parser.add_argument("--ind_csv", default=None, help="Independent test CSV path")
    parser.add_argument("--labels_csv", default=None, help="Labels CSV path")
    args = parser.parse_args()

    run_snr_selection(
        k=args.k,
        out_dir=args.out_dir,
        train_csv=args.train_csv,
        ind_csv=args.ind_csv,
        labels_csv=args.labels_csv,
    )


if __name__ == "__main__":
    main()