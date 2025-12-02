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


def preprocess_raw_data(raw_csv: str, labels_csv: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Preprocess raw Golub dataset: transpose to (samples x genes) and add labels."""
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
    # Use actual gene descriptions as column names
    X_df = pd.DataFrame(X, columns=gene_descriptions)

    metadata_df = pd.DataFrame({
        'gene_description': gene_descriptions,
        'gene_accession': gene_accessions
    })

    df_labels = pd.read_csv(labels_csv)
    # Keep labels as ALL/AML strings
    labels = df_labels['cancer'].values

    n_samples = len(numeric_cols)
    if len(labels) >= n_samples:
        y = pd.Series(labels[:n_samples], name='cancer')
    else:
        raise ValueError(f"Not enough labels ({len(labels)}) for samples ({n_samples})")

    print(f"[INFO] Preprocessed shape: {X_df.shape}, Labels: ALL={sum(y=='ALL')}, AML={sum(y=='AML')}")

    return X_df, y, metadata_df


def run_feature_selection(
    input_train: str,
    input_ind: str | None,
    input_actual: str | None,
    k: int,
    out_dir: str,
    balanced: bool = False,
    labels_csv: str | None = None,
) -> None:
    """Run ANOVA F-test feature selection on train set and apply to test sets."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_peek = pd.read_csv(input_train, nrows=1)
    needs_preprocessing = 'label' not in df_peek.columns and 'cancer' not in df_peek.columns

    metadata = None
    if needs_preprocessing:
        if labels_csv is None:
            raise ValueError("Raw data detected but no labels_csv provided!")
        print(f"[INFO] Preprocessing raw data from {input_train}")
        X_train, y_train, metadata = preprocess_raw_data(input_train, labels_csv)
    else:
        df_train = pd.read_csv(input_train)
        y_train = df_train["cancer"] if "cancer" in df_train.columns else df_train["label"]
        cols_to_drop = [c for c in ["cancer", "patient", "label"] if c in df_train.columns]
        X_train = df_train.drop(columns=cols_to_drop)

    if balanced:
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

    # Filter to only the selected features
    f_scores_selected = f_scores.loc[feature_names]
    favors_class_selected = favors_class.loc[feature_names]

    f_scores_df = pd.DataFrame({
        'gene_description': f_scores_selected.index,
        'f_score': f_scores_selected.values,
        'favors_class': favors_class_selected.values
    }).sort_values('f_score', ascending=False)

    # Add rankings based on mode
    if balanced:
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
    else:
        # For non-balanced: just overall rank
        f_scores_df['rank'] = range(1, len(f_scores_df) + 1)

    # Prepare output with metadata
    if metadata is not None:
        f_scores_df = f_scores_df.merge(metadata, on='gene_description', how='left')
        if balanced:
            topk_output = f_scores_df[['overall_rank', 'rank_within_class', 'gene_description',
                                         'gene_accession', 'f_score', 'favors_class']]
        else:
            topk_output = f_scores_df[['rank', 'gene_description', 'gene_accession',
                                         'f_score', 'favors_class']]
    else:
        if balanced:
            topk_output = f_scores_df[['overall_rank', 'rank_within_class', 'gene_description',
                                         'f_score', 'favors_class']]
        else:
            topk_output = f_scores_df[['rank', 'gene_description', 'f_score', 'favors_class']]

    # FILE 1: Full details with rankings and scores
    topk_output.to_csv(out_path / f"topk_anova_f_{k}genes.csv", index=False)
    print(f"[INFO] Top-k details saved to {out_path / f'topk_anova_f_{k}genes.csv'}")

    # FILE 2: Selected genes only (just gene description and accession)
    if metadata is not None:
        selected_genes = f_scores_df[['gene_description', 'gene_accession']].copy()
    else:
        selected_genes = pd.DataFrame({'gene_description': f_scores_df['gene_description']})

    selected_genes.to_csv(out_path / f"selected_genes_anova_f_{k}genes.csv", index=False)
    print(f"[INFO] Selected genes saved to {out_path / f'selected_genes_anova_f_{k}genes.csv'}")

    train_topk = X_train[feature_names].copy()
    train_topk["cancer"] = y_train.values
    train_topk["patient"] = [str(i+1) for i in range(len(train_topk))]
    train_topk.to_csv(out_path / f"train_top_{k}_anova_f.csv", index=False)
    print(f"[INFO] Train CSV saved to {out_path / f'train_top_{k}_anova_f.csv'}")

    if input_ind is not None:
        if needs_preprocessing:
            X_ind, y_ind, _ = preprocess_raw_data(input_ind, labels_csv)
        else:
            df_ind = pd.read_csv(input_ind)
            y_ind = df_ind["cancer"] if "cancer" in df_ind.columns else df_ind["label"]
            cols_to_drop = [c for c in ["cancer", "patient", "label"] if c in df_ind.columns]
            X_ind = df_ind.drop(columns=cols_to_drop)
        ind_topk = X_ind[feature_names].copy()
        ind_topk["cancer"] = y_ind.values
        ind_topk["patient"] = [str(i+39) for i in range(len(ind_topk))]  # Independent set is patients 39-72
        ind_topk.to_csv(out_path / f"independent_top_{k}_anova_f.csv", index=False)
        print(f"[INFO] Independent CSV saved to {out_path / f'independent_top_{k}_anova_f.csv'}")

    if input_actual is not None:
        if needs_preprocessing:
            X_act, y_act, _ = preprocess_raw_data(input_actual, labels_csv)
        else:
            df_act = pd.read_csv(input_actual)
            y_act = df_act["cancer"] if "cancer" in df_act.columns else df_act["label"]
            cols_to_drop = [c for c in ["cancer", "patient", "label"] if c in df_act.columns]
            X_act = df_act.drop(columns=cols_to_drop)
        act_topk = X_act[feature_names].copy()
        act_topk["cancer"] = y_act.values
        act_topk["patient"] = [str(i+1) for i in range(len(act_topk))]
        act_topk.to_csv(out_path / f"actual_top_{k}_anova_f.csv", index=False)
        print(f"[INFO] Actual CSV saved to {out_path / f'actual_top_{k}_anova_f.csv'}")


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
