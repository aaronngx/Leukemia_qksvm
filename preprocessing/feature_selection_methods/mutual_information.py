"""
Mutual Information feature selection.

Uses sklearn's mutual_info_classif to estimate the mutual information
between each feature and the target variable.

Higher MI indicates stronger dependence between feature and class.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .base import FeatureSelector
from ..config import RANDOM_SEED


class MutualInformationSelector(FeatureSelector):
    """
    Mutual Information feature selection using sklearn.

    Estimates MI(X_i, Y) for each feature X_i and target Y.
    Uses k-nearest neighbors estimation for continuous features.
    """

    METHOD_NAME = "mutual_information"

    def __init__(self, n_neighbors: int = 3, random_state: int = RANDOM_SEED):
        """
        Args:
            n_neighbors: Number of neighbors for MI estimation
            random_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def calculate_scores(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        """
        Calculate mutual information for each feature.

        Args:
            X_train: Training features (samples x genes)
            y_train: Training labels (0=AML, 1=ALL or strings)

        Returns:
            pd.Series of MI scores indexed by gene names.
            MI >= 0 (always non-negative).
        """
        # Convert labels to numeric
        y_numeric = self._convert_labels(y_train)

        # Calculate MI using sklearn
        mi_scores = mutual_info_classif(
            X_train.values,
            y_numeric.values,
            discrete_features=False,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state
        )

        return pd.Series(mi_scores, index=X_train.columns)
