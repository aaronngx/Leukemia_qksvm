"""
ANOVA F-test feature selection.

Implements the F-statistic:
    F = between_class_variance / within_class_variance

Higher F indicates better class discrimination.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from .base import FeatureSelector
from ..config import EPS


class AnovaFSelector(FeatureSelector):
    """
    ANOVA F-test feature selection.

    Computes the F-statistic for each feature:
        F = between_class_variance / within_class_variance

    Uses sklearn's f_classif for robust implementation.
    """

    METHOD_NAME = "anova_f_test"

    def __init__(self, eps: float = EPS):
        """
        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps

    def calculate_scores(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        """
        Calculate ANOVA F-statistics for each feature.

        Args:
            X_train: Training features (samples x genes)
            y_train: Training labels (0=AML, 1=ALL or strings)

        Returns:
            pd.Series of F-statistics indexed by gene names.
            Higher F = better discrimination (always non-negative).
        """
        # Convert labels to numeric
        y_numeric = self._convert_labels(y_train)

        # Use sklearn's f_classif for robust F-statistic calculation
        f_scores, _ = f_classif(X_train.values, y_numeric.values)

        # Handle NaN (can occur with zero variance)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

        return pd.Series(f_scores, index=X_train.columns)
