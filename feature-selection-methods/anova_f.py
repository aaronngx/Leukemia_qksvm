import argparse
from pathlib import Path

import pandas as pd


def calculate_anova_f(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calculate ANOVA F-statistic = between-class variance / within-class variance."""
    # Convert labels to numeric for computation if they are strings
    if y.dtype == 'object':
        label_map = {'ALL': 0, 'AML': 1}
        y_numeric = y.map(label_map)
    else:
        y_numeric = y

    class_0 = X[y_numeric == 0]
    class_1 = X[y_numeric == 1]

    n_0 = len(class_0)
    n_1 = len(class_1)
    n_total = n_0 + n_1

    overall_mean = X.mean()
    mean_0 = class_0.mean()
    mean_1 = class_1.mean()

    between_class_var = (
        n_0 * (mean_0 - overall_mean) ** 2 +
        n_1 * (mean_1 - overall_mean) ** 2
    ) / (2 - 1)

    var_0 = class_0.var(ddof=1)
    var_1 = class_1.var(ddof=1)
    within_class_var = ((n_0 - 1) * var_0 + (n_1 - 1) * var_1) / (n_total - 2)

    within_class_var[within_class_var == 0] = 1e-10
    f_scores = between_class_var / within_class_var

    return f_scores


def select_features_anova_f(X_train: pd.DataFrame, y_train: pd.Series, k: int) -> list[str]:
    """Select top-k features using ANOVA F-test."""
    f_scores = calculate_anova_f(X_train, y_train)
    top_features = f_scores.nlargest(k).index.tolist()
    return top_features


def select_features_anova_f_balanced(X_train: pd.DataFrame, y_train: pd.Series, k: int) -> list[str]:
    """Select balanced features: k/2 higher in class 0, k/2 higher in class 1 (based on mean expression)."""
    if k % 2 != 0:
        k += 1
        print(f"[INFO] k must be even for balanced selection. Adjusted to k={k}.")

    # Convert labels to numeric for computation if they are strings
    if y_train.dtype == 'object':
        label_map = {'ALL': 0, 'AML': 1}
        y_numeric = y_train.map(label_map)
    else:
        y_numeric = y_train

    X_all = X_train[y_numeric == 0]
    X_aml = X_train[y_numeric == 1]

    all_means = X_all.mean()
    aml_means = X_aml.mean()

    all_biomarkers = (all_means - aml_means).nlargest(k // 2).index.tolist()
    aml_biomarkers = (aml_means - all_means).nlargest(k // 2).index.tolist()

    balanced_features = all_biomarkers + aml_biomarkers
    print(f"[INFO] Selected {len(all_biomarkers)} ALL biomarkers and {len(aml_biomarkers)} AML biomarkers.")
    return balanced_features


def preprocess_raw_data(raw_csv: str, labels_csv: str, start_patient_id: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Preprocess raw Golub dataset: transpose to (samples x genes) and add labels.
    
    Returns:
        X_df: DataFrame with samples as rows, gene accession numbers as columns
        y: Series of labels (ALL/AML)
        metadata_df: DataFrame mapping gene_accession to gene_description
        patient_ids: Series of patient IDs
    """
    df_raw = pd.read_csv(raw_csv)

    gene_descriptions = df_raw['Gene Description'].tolist()
    gene_accessions = df_raw['Gene Accession Number'].tolist()

    numeric_cols = []
    for col in df_raw.columns:
        if col not in ['Gene Description', 'Gene Accession Number']:
            if not col.startswith('call'):
                numeric_cols.append(col)

    print(f"[INFO] Found {len(numeric_cols)} sample columns in {raw_csv}")

    X = df_raw[numeric_cols].values.T
    # Use gene accession numbers as column names (they are unique)
    X_df = pd.DataFrame(X, columns=gene_accessions)

    metadata_df = pd.DataFrame({
        'gene_accession': gene_accessions,
        'gene_description': gene_descriptions
    })

    # Load labels and create mapping from patient ID to label
    df_labels = pd.read_csv(labels_csv)
    patient_to_label = dict(zip(df_labels['patient'].astype(str), df_labels['cancer']))
    
    # The column names in raw data are sample IDs (as strings)
    # Map each sample to its label
    sample_ids = [str(col) for col in numeric_cols]
    labels_list = [patient_to_label.get(sid, 'UNKNOWN') for sid in sample_ids]
    patient_ids_list = [int(sid) for sid in sample_ids]
    
    y = pd.Series(labels_list, name='cancer')
    patient_ids = pd.Series(patient_ids_list, name='patient')

    print(f"[INFO] Preprocessed shape: {X_df.shape}, Labels: ALL={sum(y=='ALL')}, AML={sum(y=='AML')}")

    return X_df, y, metadata_df, patient_ids


def balance_patient_samples(X: pd.DataFrame, y: pd.Series, patient_ids: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Balance samples by taking equal numbers of ALL and AML patients.

    NOTE: This balances PATIENTS (samples/data points), NOT genes (features).
    Takes all samples from minority class and matches with same number from majority class.
    """
    # Find indices for each class
    all_indices = y[y == 'ALL'].index.tolist()
    aml_indices = y[y == 'AML'].index.tolist()
    
    n_minority = min(len(all_indices), len(aml_indices))
    
    # Take all from minority, first n from majority
    if len(aml_indices) <= len(all_indices):
        # AML is minority - take all AML and first n ALL
        selected_aml = aml_indices
        selected_all = all_indices[:n_minority]
    else:
        # ALL is minority - take all ALL and first n AML
        selected_all = all_indices
        selected_aml = aml_indices[:n_minority]
    
    selected_indices = selected_all + selected_aml
    
    X_balanced = X.iloc[selected_indices].reset_index(drop=True)
    y_balanced = y.iloc[selected_indices].reset_index(drop=True)
    patient_ids_balanced = patient_ids.iloc[selected_indices].reset_index(drop=True)
    
    print(f"[INFO] Balanced samples: {len(selected_all)} ALL + {len(selected_aml)} AML = {len(selected_indices)} total")
    
    return X_balanced, y_balanced, patient_ids_balanced


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    patient_ids: pd.Series,
    test_ratio: float
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into internal train and test sets (stratified)."""
    from sklearn.model_selection import train_test_split
    import numpy as np

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=y,
        random_state=42
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    patient_ids_train = patient_ids.iloc[train_idx].reset_index(drop=True)

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    patient_ids_test = patient_ids.iloc[test_idx].reset_index(drop=True)

    return X_train, y_train, patient_ids_train, X_test, y_test, patient_ids_test


def split_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    patient_ids: pd.Series,
    n_folds: int
) -> list[tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]]:
    """Split data into k-fold cross-validation sets (stratified)."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        patient_ids_train = patient_ids.iloc[train_idx].reset_index(drop=True)

        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        patient_ids_test = patient_ids.iloc[test_idx].reset_index(drop=True)

        folds.append((X_train, y_train, patient_ids_train, X_test, y_test, patient_ids_test))

    return folds


def split_loocv(
    X: pd.DataFrame,
    y: pd.Series,
    patient_ids: pd.Series
) -> list[tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]]:
    """Split data into Leave-One-Out cross-validation sets.

    For each data point, train on all other points and test on that single point.

    Returns:
        List of (X_train, y_train, patient_ids_train, X_test, y_test, patient_ids_test) tuples,
        one per sample in the dataset.
    """
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    folds = []

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        patient_ids_train = patient_ids.iloc[train_idx].reset_index(drop=True)

        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        patient_ids_test = patient_ids.iloc[test_idx].reset_index(drop=True)

        folds.append((X_train, y_train, patient_ids_train, X_test, y_test, patient_ids_test))

    return folds


def run_feature_selection(
    input_train: str,
    input_ind: str | None,
    input_actual: str | None,
    k: int,
    out_dir: str,
    balanced_genes: bool = False,  # NEW - gene balance only
    use_balanced_patients: bool = False,  # NEW - sample balance only
    validation_strategy: dict | None = None,  # NEW - {"method": "split", "ratio": 0.7} or {"method": "cv", "folds": 5}
    labels_csv: str | None = None,
    use_all_data: bool = False,
) -> None:
    """Run ANOVA F-test feature selection on train set and apply internal validation splits.

    Args:
        input_train: Path to training CSV
        input_ind: Path to independent test CSV (optional)
        input_actual: Path to actual test CSV (optional)
        k: Number of genes/features to select
        out_dir: Output directory
        balanced_genes: Select k/2 ALL-favoring + k/2 AML-favoring genes
        use_balanced_patients: Balance patient samples (11 ALL + 11 AML)
        validation_strategy: Internal validation config (train/test split or CV)
        labels_csv: Path to labels CSV (required for raw data)
        use_all_data: If True, combine train+independent for selection
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_peek = pd.read_csv(input_train, nrows=1)
    needs_preprocessing = 'label' not in df_peek.columns and 'cancer' not in df_peek.columns

    metadata = None
    patient_ids = None
    
    if needs_preprocessing:
        if labels_csv is None:
            raise ValueError("Raw data detected but no labels_csv provided!")
        print(f"[INFO] Preprocessing raw data from {input_train}")
        X_train, y_train, metadata, patient_ids = preprocess_raw_data(input_train, labels_csv)
        
        # If using all data, also load and combine independent set
        if use_all_data and input_ind is not None:
            print(f"[INFO] Preprocessing independent data from {input_ind}")
            X_ind, y_ind, _, patient_ids_ind = preprocess_raw_data(input_ind, labels_csv, start_patient_id=39)
            # Combine train and independent
            X_train = pd.concat([X_train, X_ind], ignore_index=True)
            y_train = pd.concat([y_train, y_ind], ignore_index=True)
            patient_ids = pd.concat([patient_ids, patient_ids_ind], ignore_index=True)
            print(f"[INFO] Combined data shape: {X_train.shape}, Labels: ALL={sum(y_train=='ALL')}, AML={sum(y_train=='AML')}")
    else:
        df_train = pd.read_csv(input_train)
        y_train = df_train["cancer"] if "cancer" in df_train.columns else df_train["label"]
        patient_ids = df_train["patient"] if "patient" in df_train.columns else pd.Series(range(1, len(df_train)+1))
        cols_to_drop = [c for c in ["cancer", "patient", "label"] if c in df_train.columns]
        X_train = df_train.drop(columns=cols_to_drop)
        
        # If using all data, also load and combine independent set
        if use_all_data and input_ind is not None:
            df_ind = pd.read_csv(input_ind)
            y_ind = df_ind["cancer"] if "cancer" in df_ind.columns else df_ind["label"]
            patient_ids_ind = df_ind["patient"] if "patient" in df_ind.columns else pd.Series(range(39, 39+len(df_ind)))
            cols_to_drop_ind = [c for c in ["cancer", "patient", "label"] if c in df_ind.columns]
            X_ind = df_ind.drop(columns=cols_to_drop_ind)
            X_train = pd.concat([X_train, X_ind], ignore_index=True)
            y_train = pd.concat([y_train, y_ind], ignore_index=True)
            patient_ids = pd.concat([patient_ids, patient_ids_ind], ignore_index=True)
            print(f"[INFO] Combined data shape: {X_train.shape}")
    
    # Balance patient samples if requested (equal ALL and AML patients)
    if use_balanced_patients:
        X_train, y_train, patient_ids = balance_patient_samples(X_train, y_train, patient_ids)

    # Select features based on gene balance flag
    if balanced_genes:
        feature_names = select_features_anova_f_balanced(X_train, y_train, k)
    else:
        feature_names = select_features_anova_f(X_train, y_train, k)

    print(f"[INFO] Selected {len(feature_names)} features using ANOVA F-test.")
    print("[INFO] First 10 features:", feature_names[:10])

    f_scores = calculate_anova_f(X_train, y_train)

    # Convert labels to numeric for comparison if they are strings
    if y_train.dtype == 'object':
        label_map = {'ALL': 0, 'AML': 1}
        y_numeric = y_train.map(label_map)
    else:
        y_numeric = y_train

    mean_all = X_train[y_numeric == 0].mean()
    mean_aml = X_train[y_numeric == 1].mean()
    favors_class = (mean_all > mean_aml).map({True: 'ALL', False: 'AML'})

    # Filter to only the selected features (now accession numbers)
    f_scores_selected = f_scores.loc[feature_names]
    favors_class_selected = favors_class.loc[feature_names]

    # Create mapping from accession to description for output
    if metadata is not None:
        acc_to_desc = dict(zip(metadata['gene_accession'], metadata['gene_description']))
        gene_descriptions = [acc_to_desc.get(f, f) for f in feature_names]
    else:
        gene_descriptions = feature_names

    f_scores_df = pd.DataFrame({
        'gene_accession': f_scores_selected.index,
        'gene_description': gene_descriptions,
        'f_score': f_scores_selected.values,
        'favors_class': favors_class_selected.values
    }).sort_values('f_score', ascending=False)

    # Add rankings based on mode
    if balanced_genes:
        # For balanced: rank within each class
        all_genes = f_scores_df[f_scores_df['favors_class'] == 'ALL'].copy()
        aml_genes = f_scores_df[f_scores_df['favors_class'] == 'AML'].copy()

        all_genes['rank_within_class'] = range(1, len(all_genes) + 1)
        aml_genes['rank_within_class'] = range(1, len(aml_genes) + 1)

        # Combine and add overall rank
        f_scores_df = pd.concat([all_genes, aml_genes]).sort_values('f_score', ascending=False)
        f_scores_df['overall_rank'] = range(1, len(f_scores_df) + 1)

        print(f"[INFO] Balanced selection rankings:")
        print(f"  - ALL genes: ranks 1-{len(all_genes)} within ALL class")
        print(f"  - AML genes: ranks 1-{len(aml_genes)} within AML class")
        
        topk_output = f_scores_df[['overall_rank', 'rank_within_class', 'gene_accession',
                                     'gene_description', 'f_score', 'favors_class']]
    else:
        # For non-balanced: just overall rank
        f_scores_df['rank'] = range(1, len(f_scores_df) + 1)
        topk_output = f_scores_df[['rank', 'gene_accession', 'gene_description', 
                                     'f_score', 'favors_class']]

    # FILE 1: Full details with rankings and scores
    topk_output.to_csv(out_path / f"topk_anova_f_{k}genes.csv", index=False)
    print(f"[INFO] Top-k details saved to {out_path / f'topk_anova_f_{k}genes.csv'}")

    # FILE 2: Selected genes only (just accession numbers, one per line, no header)
    # feature_names are now already accession numbers
    with open(out_path / f"selected_genes_anova_f_{k}genes.csv", 'w') as f:
        for acc in feature_names:
            f.write(f"{acc}\n")
    print(f"[INFO] Selected genes saved to {out_path / f'selected_genes_anova_f_{k}genes.csv'}")
    
    # FILE 3+: Data files based on validation strategy
    if validation_strategy is None or validation_strategy["method"] == "split":
        # Train/test split mode
        if validation_strategy:
            test_ratio = 1.0 - validation_strategy["ratio"]
            X_train_int, y_train_int, pids_train_int, X_test_int, y_test_int, pids_test_int = split_train_test(
                X_train[feature_names], y_train, patient_ids, test_ratio
            )

            # Internal training set
            train_int_df = X_train_int.copy()
            train_int_df["cancer"] = y_train_int.values
            train_int_df["patient"] = pids_train_int.values
            train_int_df.to_csv(out_path / f"train_internal_top_{k}_anova_f.csv", index=False)
            print(f"[INFO] Internal train CSV saved: {out_path / f'train_internal_top_{k}_anova_f.csv'}")

            # Internal test set
            test_int_df = X_test_int.copy()
            test_int_df["cancer"] = y_test_int.values
            test_int_df["patient"] = pids_test_int.values
            test_int_df.to_csv(out_path / f"test_internal_top_{k}_anova_f.csv", index=False)
            print(f"[INFO] Internal test CSV saved: {out_path / f'test_internal_top_{k}_anova_f.csv'}")
        else:
            # No internal split - output full training set (backward compatibility)
            train_topk = X_train[feature_names].copy()
            train_topk["cancer"] = y_train.values
            train_topk["patient"] = patient_ids.values
            train_topk.to_csv(out_path / f"train_top_{k}_anova_f.csv", index=False)
            print(f"[INFO] Train CSV saved: {out_path / f'train_top_{k}_anova_f.csv'}")

    elif validation_strategy["method"] == "cv":
        # Cross-validation mode
        n_folds = validation_strategy["folds"]
        folds = split_cross_validation(X_train[feature_names], y_train, patient_ids, n_folds)

        for fold_idx, (X_tr, y_tr, pids_tr, X_te, y_te, pids_te) in enumerate(folds, start=1):
            # Fold training set
            fold_train_df = X_tr.copy()
            fold_train_df["cancer"] = y_tr.values
            fold_train_df["patient"] = pids_tr.values
            fold_train_df.to_csv(out_path / f"fold_{fold_idx}_train_top_{k}_anova_f.csv", index=False)

            # Fold test set
            fold_test_df = X_te.copy()
            fold_test_df["cancer"] = y_te.values
            fold_test_df["patient"] = pids_te.values
            fold_test_df.to_csv(out_path / f"fold_{fold_idx}_test_top_{k}_anova_f.csv", index=False)

        print(f"[INFO] {n_folds}-fold CV sets saved (fold_1 through fold_{n_folds})")

    elif validation_strategy["method"] == "loocv":
        # Leave-One-Out Cross-validation mode
        print("[INFO] Generating LOOCV folds (this may take a moment)...")
        folds = split_loocv(X_train[feature_names], y_train, patient_ids)
        n_folds = len(folds)

        for fold_idx, (X_tr, y_tr, pids_tr, X_te, y_te, pids_te) in enumerate(folds, start=1):
            # Fold training set (n-1 samples)
            fold_train_df = X_tr.copy()
            fold_train_df["cancer"] = y_tr.values
            fold_train_df["patient"] = pids_tr.values
            fold_train_df.to_csv(out_path / f"fold_{fold_idx}_train_top_{k}_anova_f.csv", index=False)

            # Fold test set (1 sample)
            fold_test_df = X_te.copy()
            fold_test_df["cancer"] = y_te.values
            fold_test_df["patient"] = pids_te.values
            fold_test_df.to_csv(out_path / f"fold_{fold_idx}_test_top_{k}_anova_f.csv", index=False)

        print(f"[INFO] LOOCV sets saved: {n_folds} folds (fold_1 through fold_{n_folds})")
        print(f"[INFO] Each fold trains on {n_folds-1} samples and tests on 1 sample")

    # Independent test set (always generated if available)
    if input_ind is not None:
        if needs_preprocessing:
            X_ind, y_ind, _, patient_ids_ind = preprocess_raw_data(input_ind, labels_csv, start_patient_id=39)
        else:
            df_ind = pd.read_csv(input_ind)
            y_ind = df_ind["cancer"] if "cancer" in df_ind.columns else df_ind["label"]
            patient_ids_ind = df_ind["patient"] if "patient" in df_ind.columns else pd.Series(range(39, 39+len(df_ind)))
            cols_to_drop = [c for c in ["cancer", "patient", "label"] if c in df_ind.columns]
            X_ind = df_ind.drop(columns=cols_to_drop)

        ind_topk = X_ind[feature_names].copy()
        ind_topk["cancer"] = y_ind.values
        ind_topk["patient"] = patient_ids_ind.values
        ind_topk.to_csv(out_path / f"independent_top_{k}_anova_f.csv", index=False)
        print(f"[INFO] Independent CSV saved: {out_path / f'independent_top_{k}_anova_f.csv'}")


def main():
    parser = argparse.ArgumentParser(description="ANOVA F-test feature selection for Golub dataset")
    parser.add_argument(
        "--train_csv",
        default="data/raw/data_set_ALL_AML_train.csv",
        help="Path to training CSV (raw or with 'label' column)",
    )
    parser.add_argument(
        "--ind_csv",
        default="data/raw/data_set_ALL_AML_independent.csv",
        help="Path to independent CSV (optional)",
    )
    parser.add_argument(
        "--actual_csv",
        default=None,
        help="Path to actual test CSV (optional)",
    )
    parser.add_argument(
        "--labels_csv",
        default="data/raw/actual.csv",
        help="Path to labels CSV (required for raw data)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of genes/features to select",
    )
    parser.add_argument(
        "--out_dir",
        default="results",
        help="Output directory for processed CSVs",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced ALL/AML mean-difference selection instead of pure ANOVA F-test top-k",
    )
    parser.add_argument(
        "--use_all_data",
        action="store_true",
        help="Combine train and independent data for feature selection (outputs single file)",
    )
    args = parser.parse_args()

    run_feature_selection(
        input_train=args.train_csv,
        input_ind=args.ind_csv,
        input_actual=args.actual_csv,
        k=args.k,
        out_dir=args.out_dir,
        balanced=args.balanced,
        labels_csv=args.labels_csv,
        use_all_data=args.use_all_data,
    )


if __name__ == "__main__":
    main()
