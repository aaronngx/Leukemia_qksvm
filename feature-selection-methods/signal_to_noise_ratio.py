import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_snr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Calculate Signal-to-Noise Ratio (SNR) for each feature column.

    SNR_j = | mean_1_j - mean_0_j | / (std_0_j + std_1_j + eps)
    where class 0 and class 1 are defined by y == 0 and y == 1.
    """
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    mean_0 = class_0.mean()
    mean_1 = class_1.mean()
    std_0 = class_0.std()
    std_1 = class_1.std()

    std_combined = std_0 + std_1
    std_combined[std_combined == 0] = 1e-10  # avoid division by zero

    snr = (mean_1 - mean_0).abs() / std_combined
    return snr


def select_features_snr(X_train: pd.DataFrame, y_train: pd.Series, k: int) -> list[str]:
    """
    Select top-k features using SNR.
    Returns a list of column names.
    """
    snr_scores = calculate_snr(X_train, y_train)
    top_features = snr_scores.nlargest(k).index.tolist()
    return top_features


def select_features_snr_balanced(X_train: pd.DataFrame, y_train: pd.Series, k: int) -> list[str]:
    """
    Select a balanced set of features:
    half that are higher in class 0 (ALL) and half higher in class 1 (AML),
    based on mean expression. This does NOT use SNR for the final pick.
    """
    if k % 2 != 0:
        k += 1
        print(f"[INFO] k must be even for balanced selection. Adjusted to k={k}.")

    X_all = X_train[y_train == 0]
    X_aml = X_train[y_train == 1]

    all_means = X_all.mean()
    aml_means = X_aml.mean()

    # Higher in ALL
    all_biomarkers = (all_means - aml_means).nlargest(k // 2).index.tolist()
    # Higher in AML
    aml_biomarkers = (aml_means - all_means).nlargest(k // 2).index.tolist()

    balanced_features = all_biomarkers + aml_biomarkers
    print(f"[INFO] Selected {len(all_biomarkers)} ALL biomarkers and {len(aml_biomarkers)} AML biomarkers.")
    return balanced_features


def preprocess_raw_data(raw_csv: str, labels_csv: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess raw Golub dataset:
    - Transpose data (genes as columns, samples as rows)
    - Add labels from labels CSV
    """
    # Read raw data
    df_raw = pd.read_csv(raw_csv)

    # Keep only numeric columns (sample columns like '1', '2', '3', etc.)
    numeric_cols = []
    for col in df_raw.columns:
        if col not in ['Gene Description', 'Gene Accession Number']:
            if not col.startswith('call'):
                numeric_cols.append(col)

    print(f"[INFO] Found {len(numeric_cols)} sample columns in {raw_csv}")

    # Extract numeric data and transpose
    X = df_raw[numeric_cols].values.T  # Transpose: samples as rows, genes as columns

    # Create DataFrame with gene columns
    gene_names = [f"gene_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=gene_names)

    # Read labels
    df_labels = pd.read_csv(labels_csv)
    label_map = {'ALL': 0, 'AML': 1}
    labels = df_labels['cancer'].map(label_map).values

    # Match labels to samples
    n_samples = len(numeric_cols)
    if len(labels) >= n_samples:
        y = pd.Series(labels[:n_samples], name='label')
    else:
        raise ValueError(f"Not enough labels ({len(labels)}) for samples ({n_samples})")

    print(f"[INFO] Preprocessed shape: {X_df.shape}, Labels: ALL={sum(y==0)}, AML={sum(y==1)}")

    return X_df, y


def run_feature_selection(
    input_train: str,
    input_ind: str | None,
    input_actual: str | None,
    k: int,
    out_dir: str,
    balanced: bool = False,
    labels_csv: str | None = None,
) -> None:
    """
    Run feature selection on train set and apply the same feature set
    to independent and actual sets (if provided).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Check if input is raw data (needs preprocessing) or already processed
    df_peek = pd.read_csv(input_train, nrows=1)
    needs_preprocessing = 'label' not in df_peek.columns

    if needs_preprocessing:
        if labels_csv is None:
            raise ValueError("Raw data detected but no labels_csv provided!")
        print(f"[INFO] Preprocessing raw data from {input_train}")
        X_train, y_train = preprocess_raw_data(input_train, labels_csv)
    else:
        df_train = pd.read_csv(input_train)
        y_train = df_train["label"]
        X_train = df_train.drop(columns=["label"])

    if balanced:
        feature_names = select_features_snr_balanced(X_train, y_train, k)
    else:
        feature_names = select_features_snr(X_train, y_train, k)

    print(f"[INFO] Selected {len(feature_names)} features.")
    print("[INFO] First 10 features:", feature_names[:10])

    # Train subset
    train_topk = X_train[feature_names].copy()
    train_topk["label"] = y_train
    train_topk.to_csv(out_path / "train_topk_snr.csv", index=False)

    # Apply same features to other datasets (if available)
    if input_ind is not None:
        if needs_preprocessing:
            X_ind, y_ind = preprocess_raw_data(input_ind, labels_csv)
        else:
            df_ind = pd.read_csv(input_ind)
            y_ind = df_ind["label"]
            X_ind = df_ind.drop(columns=["label"])
        ind_topk = X_ind[feature_names].copy()
        ind_topk["label"] = y_ind
        ind_topk.to_csv(out_path / "independent_topk_snr.csv", index=False)

    if input_actual is not None:
        if needs_preprocessing:
            X_act, y_act = preprocess_raw_data(input_actual, labels_csv)
        else:
            df_act = pd.read_csv(input_actual)
            y_act = df_act["label"]
            X_act = df_act.drop(columns=["label"])
        act_topk = X_act[feature_names].copy()
        act_topk["label"] = y_act
        act_topk.to_csv(out_path / "actual_topk_snr.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="SNR-based feature selection for Golub dataset")
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
        default="data/processed",
        help="Output directory for processed CSVs",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced ALL/AML mean-difference selection instead of pure SNR top-k",
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
    )


if __name__ == "__main__":
    main()
