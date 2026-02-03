"""
Configuration constants for the feature selection experiment.

This is the single source of truth for all hyperparameters and settings.
Changes here affect all experiments for reproducibility.
"""

from typing import List, Dict, Any

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Gene counts to test
K_VALUES: List[int] = [4, 8, 16, 24, 32, 50]

# Cross-validation settings
CV_FOLDS: int = 5
RANDOM_SEED: int = 42

# =============================================================================
# LABEL ENCODING (Contract §2.3)
# =============================================================================

# ALL is positive class (1), AML is negative class (0)
POS_LABEL: int = 1
LABEL_MAP: Dict[str, int] = {"ALL": 1, "AML": 0}

# =============================================================================
# SCAD PARAMETERS (Contract §4.6)
# =============================================================================

SCAD_LAMBDA_MIN: float = 0.3
SCAD_LAMBDA_MAX: float = 2.0
SCAD_LAMBDA_STEP: float = 0.1
SCAD_INNER_CV_FOLDS: int = 3
SCAD_A: float = 3.7  # SCAD shape parameter (Fan & Li, 2001)

# =============================================================================
# CLASSIFIER (Contract §5.2) - Fixed across all methods
# =============================================================================

CLASSIFIER: str = "LinearSVC"
CLASSIFIER_PARAMS: Dict[str, Any] = {
    "C": 1.0,
    "random_state": RANDOM_SEED,
    "max_iter": 10000,
    "dual": True,
}

# =============================================================================
# SCALER - Fixed across all methods
# =============================================================================

SCALER: str = "StandardScaler"

# =============================================================================
# METHOD NAMES (CoderPack §5)
# =============================================================================

METHOD_NAMES: List[str] = [
    "signal_to_noise",
    "anova_f_test",
    "mutual_information",
    "t_test",
    "ensemble_voting",
    "scad_regularization",
]

# =============================================================================
# DATA PATHS
# =============================================================================

DATA_DIR: str = "data/raw"
TRAIN_DATA_FILE: str = "data_set_ALL_AML_train.csv"
LABELS_FILE: str = "actual.csv"

# =============================================================================
# OUTPUT PATHS
# =============================================================================

RESULT_DIR: str = "preprocessing/Result"
TOP_GENES_DIR: str = "preprocessing/Result/top_genes"
METRICS_BY_RUN_FILE: str = "metrics_by_run.csv"
BEST_K_SUMMARY_FILE: str = "best_k_summary.csv"

# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

EPS: float = 1e-9  # Small constant to avoid division by zero
