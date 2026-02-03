"""
Main orchestrator for the feature selection experiment.

Implements stratified 5-fold cross-validation with:
- Feature selection inside each fold (no leakage)
- Fixed classifier across all methods
- Comprehensive metrics collection
- Fold-aggregated gene rankings for final output

Usage:
    python Preprocessing.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .config import (
    K_VALUES,
    CV_FOLDS,
    RANDOM_SEED,
    METHOD_NAMES,
    LABEL_MAP,
    POS_LABEL,
    CLASSIFIER_PARAMS,
    SCALER,
    DATA_DIR,
    TRAIN_DATA_FILE,
    LABELS_FILE,
    RESULT_DIR,
    TOP_GENES_DIR,
    METRICS_BY_RUN_FILE,
    BEST_K_SUMMARY_FILE,
)
from .feature_selection_methods import (
    SignalToNoiseSelector,
    AnovaFSelector,
    MutualInformationSelector,
    TTestSelector,
    EnsembleVotingSelector,
    SCADSelector,
)

warnings.filterwarnings('ignore')


@dataclass
class FoldResult:
    """Results from a single CV fold."""
    fold_id: int
    method: str
    k: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    selected_features: List[str]
    feature_scores: Dict[str, float]  # Scores from training fold
    best_lambda: Optional[float] = None  # For SCAD


class FeatureSelectionExperiment:
    """
    Main orchestrator for reproducible feature selection experiments.

    Implements stratified 5-fold CV with:
    - Feature selection inside each fold (no leakage)
    - Fixed classifier across all methods
    - Comprehensive metrics collection
    - Fold-aggregated gene rankings

    All fitting operations use training data ONLY.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str = RESULT_DIR,
        k_values: List[int] = None,
        cv_folds: int = CV_FOLDS,
        random_seed: int = RANDOM_SEED,
        scaler_type: str = SCALER,
    ):
        """
        Initialize experiment.

        Args:
            X: Feature matrix (samples x genes), gene names as columns
            y: Labels as pd.Series (strings or integers)
            output_dir: Where to save results
            k_values: List of K values to test
            cv_folds: Number of CV folds
            random_seed: For reproducibility
            scaler_type: "StandardScaler" or "MinMaxScaler"
        """
        self.X = X
        self.y = self._encode_labels(y)
        self.output_dir = Path(output_dir)
        self.top_genes_dir = Path(TOP_GENES_DIR)
        self.k_values = k_values or K_VALUES
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.scaler_type = scaler_type

        # Initialize selectors
        self.selectors: Dict[str, object] = {
            "signal_to_noise": SignalToNoiseSelector(),
            "anova_f_test": AnovaFSelector(),
            "mutual_information": MutualInformationSelector(random_state=random_seed),
            "t_test": TTestSelector(),
            "ensemble_voting": EnsembleVotingSelector(random_state=random_seed),
            "scad_regularization": SCADSelector(random_state=random_seed),
        }

        # Results storage
        self.fold_results: List[FoldResult] = []

        # For fold-aggregated gene rankings (RECHECK #1)
        # Structure: {method: {k: [list of (selected_features, scores_dict) per fold]}}
        self.fold_gene_data: Dict[str, Dict[int, List[Tuple[List[str], Dict[str, float]]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def _encode_labels(self, y: pd.Series) -> pd.Series:
        """Convert string labels to numeric using LABEL_MAP."""
        if y.dtype == 'object':
            return y.map(LABEL_MAP)
        return y

    def _get_scaler(self):
        """Create appropriate scaler instance."""
        if self.scaler_type == "StandardScaler":
            return StandardScaler()
        elif self.scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        decision_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Uses decision_function scores for AUROC (not predict_proba).
        Handles edge case where only one class is present.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            decision_scores: Decision function scores from classifier

        Returns:
            Dict with accuracy, precision, recall, f1, auroc
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
            'recall': recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
            'f1': f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        }

        # AUROC edge case: requires both classes (Contract §5.3)
        if len(np.unique(y_true)) == 2:
            try:
                metrics['auroc'] = roc_auc_score(y_true, decision_scores)
            except Exception:
                metrics['auroc'] = np.nan
        else:
            metrics['auroc'] = np.nan
            print(f"    Warning: Only one class in y_val, AUROC set to NaN")

        return metrics

    def _run_single_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold_id: int,
        method: str,
        k: int,
    ) -> FoldResult:
        """
        Execute a single fold of the experiment.

        CRITICAL: All fitting happens on training data only!

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            fold_id: Fold index (0-4)
            method: Method name
            k: Number of features to select

        Returns:
            FoldResult with metrics and selected features
        """
        selector = self.selectors[method]

        # 1. Feature selection on training data ONLY
        selected_features = selector.select_features(X_train, y_train, k)

        # 2. Get scores from training data (for fold aggregation - RECHECK #1)
        train_scores = selector.calculate_scores(X_train, y_train)
        feature_scores = {gene: train_scores.get(gene, 0.0) for gene in selected_features}

        # 3. Subset to selected features
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]

        # 4. Scale - fit on training only, transform both
        scaler = self._get_scaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)

        # 5. Train classifier
        clf = LinearSVC(**CLASSIFIER_PARAMS)
        clf.fit(X_train_scaled, y_train.values)

        # 6. Predict and evaluate
        y_pred = clf.predict(X_val_scaled)
        decision_scores = clf.decision_function(X_val_scaled)

        # 7. Calculate metrics
        metrics = self._compute_metrics(y_val.values, y_pred, decision_scores)

        # 8. Get best_lambda for SCAD
        best_lambda = None
        if method == "scad_regularization":
            best_lambda = selector.get_best_lambda()

        return FoldResult(
            fold_id=fold_id,
            method=method,
            k=k,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            auroc=metrics['auroc'],
            selected_features=selected_features,
            feature_scores=feature_scores,
            best_lambda=best_lambda,
        )

    def run_experiment(self, methods: List[str] = None) -> pd.DataFrame:
        """
        Run complete CV experiment for all methods and K values.

        Args:
            methods: Subset of methods to run (default: all)

        Returns:
            DataFrame with all fold results
        """
        methods = methods or list(self.selectors.keys())

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_genes_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CV splitter
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_seed
        )

        total_runs = len(methods) * len(self.k_values) * self.cv_folds
        run_count = 0

        print(f"\n{'='*70}")
        print("FEATURE SELECTION EXPERIMENT")
        print(f"{'='*70}")
        print(f"Methods: {methods}")
        print(f"K values: {self.k_values}")
        print(f"CV folds: {self.cv_folds}")
        print(f"Total runs: {total_runs}")
        print(f"{'='*70}\n")

        # Pre-compute fold splits (RECHECK #4: use self.X, self.y)
        fold_splits = list(skf.split(self.X, self.y))

        for method in methods:
            print(f"\n--- Method: {method} ---")

            for k in self.k_values:
                print(f"\n  K = {k}")

                for fold_id, (train_idx, val_idx) in enumerate(fold_splits):
                    run_count += 1

                    # Split data (RECHECK #4: use self.X, self.y)
                    X_train = self.X.iloc[train_idx].copy()
                    y_train = self.y.iloc[train_idx].copy()
                    X_val = self.X.iloc[val_idx].copy()
                    y_val = self.y.iloc[val_idx].copy()

                    # Run fold
                    result = self._run_single_fold(
                        X_train, y_train, X_val, y_val,
                        fold_id, method, k
                    )

                    # Store result
                    self.fold_results.append(result)

                    # Store for fold aggregation (RECHECK #1)
                    self.fold_gene_data[method][k].append(
                        (result.selected_features, result.feature_scores)
                    )

                    # Progress output
                    lambda_str = f" | lambda={result.best_lambda:.2f}" if result.best_lambda else ""
                    print(f"    Fold {fold_id}: Acc={result.accuracy:.4f} | F1={result.f1:.4f}{lambda_str}")

        print(f"\n{'='*70}")
        print("Experiment complete. Saving results...")
        print(f"{'='*70}\n")

        # Generate output files
        self._save_metrics_by_run()
        summary_df = self._compute_and_save_best_k_summary()
        self._export_top_genes(summary_df)

        return self._get_results_dataframe()

    def _get_results_dataframe(self) -> pd.DataFrame:
        """Convert fold results to DataFrame."""
        records = []
        for r in self.fold_results:
            record = {
                'method': r.method,
                'K': r.k,
                'fold_id': r.fold_id,
                'accuracy': r.accuracy,
                'precision': r.precision,
                'recall': r.recall,
                'f1': r.f1,
                'auroc': r.auroc,
            }
            if r.best_lambda is not None:
                record['best_lambda'] = r.best_lambda
            records.append(record)
        return pd.DataFrame(records)

    def _save_metrics_by_run(self):
        """Save metrics_by_run.csv."""
        df = self._get_results_dataframe()
        output_path = self.output_dir / METRICS_BY_RUN_FILE
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    def _compute_and_save_best_k_summary(self) -> pd.DataFrame:
        """
        Compute best_k_summary.csv.

        For each method, find best_K by mean accuracy with tie-breakers:
        1. Higher mean F1
        2. Higher mean AUROC
        3. Smaller K (simpler model)

        Returns:
            Summary DataFrame (also saved to file)
        """
        df = self._get_results_dataframe()

        summary_records = []
        for method in df['method'].unique():
            method_df = df[df['method'] == method]

            # Group by K and compute stats
            k_stats = method_df.groupby('K').agg({
                'accuracy': ['mean', 'std'],
                'f1': 'mean',
                'auroc': 'mean',
            }).reset_index()

            k_stats.columns = ['K', 'mean_accuracy', 'std_accuracy', 'mean_f1', 'mean_auroc']

            # Sort by tie-breaker rules (Contract §5.4):
            # accuracy desc, f1 desc, auroc desc, K asc
            k_stats_sorted = k_stats.sort_values(
                by=['mean_accuracy', 'mean_f1', 'mean_auroc', 'K'],
                ascending=[False, False, False, True]
            )

            best_row = k_stats_sorted.iloc[0]

            record = {
                'method': method,
                'best_K': int(best_row['K']),
                'mean_accuracy': best_row['mean_accuracy'],
                'std_accuracy': best_row['std_accuracy'],
                'mean_f1': best_row['mean_f1'],
                'mean_auroc': best_row['mean_auroc'],
            }

            # Add mean best_lambda for SCAD
            if method == "scad_regularization":
                scad_results = [r for r in self.fold_results
                               if r.method == method and r.k == int(best_row['K'])]
                lambdas = [r.best_lambda for r in scad_results if r.best_lambda is not None]
                if lambdas:
                    record['mean_best_lambda'] = np.mean(lambdas)

            summary_records.append(record)

        summary_df = pd.DataFrame(summary_records)
        output_path = self.output_dir / BEST_K_SUMMARY_FILE
        summary_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

        return summary_df

    def _export_top_genes(self, summary_df: pd.DataFrame):
        """
        Export top genes for each method at its best_K.

        Uses fold-aggregated rankings (RECHECK #1):
        - vote_count: number of folds where gene was selected
        - avg_rank: average rank across folds
        - score: average of training-fold scores

        Args:
            summary_df: DataFrame with best_K per method
        """
        for _, row in summary_df.iterrows():
            method = row['method']
            best_k = int(row['best_K'])

            # Get fold data for this method and K
            fold_data = self.fold_gene_data[method][best_k]

            # Aggregate across folds
            vote_count = defaultdict(int)
            rank_sums = defaultdict(float)
            rank_counts = defaultdict(int)
            score_sums = defaultdict(float)
            score_counts = defaultdict(int)

            for fold_genes, fold_scores in fold_data:
                for rank, gene in enumerate(fold_genes, start=1):
                    vote_count[gene] += 1
                    rank_sums[gene] += rank
                    rank_counts[gene] += 1

                    if gene in fold_scores:
                        score_sums[gene] += abs(fold_scores[gene])  # Use absolute score
                        score_counts[gene] += 1

            # Build final ranking
            all_genes = set(g for genes, _ in fold_data for g in genes)
            final_ranking = []

            for gene in all_genes:
                avg_rank = rank_sums[gene] / rank_counts[gene] if rank_counts[gene] > 0 else best_k + 1
                avg_score = score_sums[gene] / score_counts[gene] if score_counts[gene] > 0 else 0.0
                final_ranking.append((gene, vote_count[gene], avg_rank, avg_score))

            # Sort: vote_count desc, avg_rank asc
            final_ranking.sort(key=lambda x: (-x[1], x[2]))

            # Output top best_k
            output_data = final_ranking[:best_k]
            df = pd.DataFrame(output_data, columns=['gene_id', 'vote_count', 'avg_rank', 'score'])
            df.insert(0, 'rank', range(1, len(df) + 1))

            output_path = self.top_genes_dir / f"{method}_topk.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")


def load_golub_data(
    data_dir: str = DATA_DIR,
    train_file: str = TRAIN_DATA_FILE,
    labels_file: str = LABELS_FILE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Golub AML/ALL dataset.

    Args:
        data_dir: Directory containing data files
        train_file: Training data CSV filename
        labels_file: Labels CSV filename

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is labels Series
    """
    data_path = Path(data_dir)

    # Load training data
    train_path = data_path / train_file
    print(f"Loading data from: {train_path}")

    df = pd.read_csv(train_path)

    # The Golub dataset has genes as rows, patients as columns
    # First two columns are typically gene info
    # Transpose to get samples as rows

    # Identify gene description column
    gene_col = None
    for col in ['Gene Description', 'gene_description', 'Gene Accession Number']:
        if col in df.columns:
            gene_col = col
            break

    if gene_col is None:
        gene_col = df.columns[0]

    # Set gene names as index
    df = df.set_index(gene_col)

    # Remove non-numeric columns (like 'call' columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    # Transpose: rows=samples, columns=genes
    X = df.T

    # Load labels
    labels_path = data_path / labels_file
    labels_df = pd.read_csv(labels_path)

    # Find label column
    label_col = None
    for col in ['cancer', 'label', 'Cancer', 'Label']:
        if col in labels_df.columns:
            label_col = col
            break

    if label_col is None:
        label_col = labels_df.columns[-1]

    y = labels_df[label_col]

    # Ensure X and y are aligned
    if len(X) != len(y):
        print(f"Warning: X has {len(X)} samples, y has {len(y)} labels")
        # Try to align by taking min
        n = min(len(X), len(y))
        X = X.iloc[:n]
        y = y.iloc[:n]

    print(f"Loaded: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y.reset_index(drop=True)


def main():
    """Main entry point for the experiment."""
    print("\n" + "="*70)
    print("GOLUB AML/ALL FEATURE SELECTION BENCHMARK")
    print("="*70 + "\n")

    # Load data
    try:
        X, y = load_golub_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease ensure the data files are in the correct location:")
        print(f"  - {DATA_DIR}/{TRAIN_DATA_FILE}")
        print(f"  - {DATA_DIR}/{LABELS_FILE}")
        sys.exit(1)

    # Create experiment
    experiment = FeatureSelectionExperiment(
        X=X,
        y=y,
        output_dir=RESULT_DIR,
        k_values=K_VALUES,
        cv_folds=CV_FOLDS,
        random_seed=RANDOM_SEED,
    )

    # Run experiment
    results_df = experiment.run_experiment()

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nResults saved to: {RESULT_DIR}/")
    print(f"  - {METRICS_BY_RUN_FILE}")
    print(f"  - {BEST_K_SUMMARY_FILE}")
    print(f"  - top_genes/*.csv")

    # Print best_k summary
    summary_path = Path(RESULT_DIR) / BEST_K_SUMMARY_FILE
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        print("\nBest K per method:")
        print(summary.to_string(index=False))

    print("\n" + "="*70)
    print("DONE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
