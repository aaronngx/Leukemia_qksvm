"""
T-test feature selection.

Uses Welch's t-test to compute the t-statistic for each feature.
Selection is based on |t| (absolute value).
"""

import numpy as np
import pandas as pd
from scipy import stats

from .base import FeatureSelector


class TTestSelector(FeatureSelector):
    """
    Welch's t-test feature selection.

    Computes the t-statistic for each feature comparing the two classes.
    Uses Welch's t-test (unequal variances assumed) by default.
    """

    METHOD_NAME = "t_test"

    def __init__(self, equal_var: bool = False):
        """
        Args:
            equal_var: If False (default), use Welch's t-test (unequal variances).
                      If True, use standard t-test (equal variances).
        """
        self.equal_var = equal_var

    def calculate_scores(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        """
        Calculate t-statistics for each feature.

        Args:
            X_train: Training features (samples x genes)
            y_train: Training labels (0=AML, 1=ALL or strings)

        Returns:
            pd.Series of t-statistics indexed by gene names.
            Returns absolute t-statistic for ranking purposes.
        """
        # Convert labels to numeric and reset index for alignment
        y_numeric = self._convert_labels(y_train).reset_index(drop=True)
        X_reset = X_train.reset_index(drop=True)

        # Separate classes
        mask_all = (y_numeric == 1).values
        mask_aml = (y_numeric == 0).values

        X_all = X_reset.iloc[mask_all]
        X_aml = X_reset.iloc[mask_aml]

        # Calculate t-statistic for each feature (using column index to handle duplicates)
        t_scores = []
        for i in range(len(X_train.columns)):
            col_all = X_all.iloc[:, i].values
            col_aml = X_aml.iloc[:, i].values
            t_stat, _ = stats.ttest_ind(
                col_all,
                col_aml,
                equal_var=self.equal_var,
                nan_policy='omit'
            )
            # Use absolute t-statistic for ranking
            t_val = float(t_stat) if np.isfinite(t_stat) else 0.0
            t_scores.append(t_val)

        return pd.Series(t_scores, index=X_train.columns)
