"""
Abstract base class for feature selection methods.

All selectors must implement the FeatureSelector interface.
This ensures consistent behavior and enables easy extension.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd

from ..config import LABEL_MAP


class FeatureSelector(ABC):
    """
    Abstract base class for all feature selection methods.

    All methods must:
    - Operate on training data ONLY (no leakage)
    - Return scores indexed by feature names
    - Use higher |score| = more important (unless overridden)

    Attributes:
        METHOD_NAME: Unique identifier for this method (lower_snake_case)
    """

    METHOD_NAME: str = "base"

    @abstractmethod
    def calculate_scores(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        """
        Calculate feature importance scores.

        CRITICAL: Must use ONLY training data. Never access validation data.

        Args:
            X_train: Feature matrix (samples x features), columns are gene names
            y_train: Labels as pd.Series (0/1 or 'ALL'/'AML')

        Returns:
            pd.Series indexed by feature names (X_train.columns).
            Higher absolute value = more important feature.
        """
        pass

    def select_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, k: int
    ) -> List[str]:
        """
        Select top-k features based on calculated scores.

        CRITICAL: Uses ONLY training data for feature selection.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            k: Number of features to select

        Returns:
            List of k feature names (strings), ordered by importance
        """
        scores = self.calculate_scores(X_train, y_train)
        scores = self._validate_scores(scores, X_train)

        # Select top-k by absolute score (descending)
        top_features = scores.abs().nlargest(k).index.tolist()
        return top_features

    def _validate_scores(
        self, scores: pd.Series, X_train: pd.DataFrame
    ) -> pd.Series:
        """
        Validate and clean scores for deterministic selection.

        Ensures:
        - Index is aligned to X_train.columns (RECHECK item #2)
        - No inf values
        - No NaN values (filled with 0)
        - Numeric dtype

        Args:
            scores: Raw scores from calculate_scores()
            X_train: Training data for reference columns

        Returns:
            Cleaned scores aligned to X_train.columns
        """
        # RECHECK #2: Reindex to ensure deterministic alignment
        # (set comparison alone doesn't guarantee order)
        scores = scores.reindex(X_train.columns)

        # Handle inf and NaN deterministically
        scores = scores.replace([np.inf, -np.inf], np.nan)
        scores = scores.fillna(0.0)

        # Ensure numeric dtype
        scores = scores.astype(float)

        return scores

    def _convert_labels(self, y: pd.Series) -> pd.Series:
        """
        Convert string labels to numeric if needed.

        Uses LABEL_MAP from config: ALL -> 1, AML -> 0

        Args:
            y: Labels as pd.Series (strings or integers)

        Returns:
            pd.Series with integer labels (0 or 1)
        """
        if y.dtype == 'object':
            return y.map(LABEL_MAP)
        return y

    def get_scores_dataframe(
        self, X_train: pd.DataFrame, y_train: pd.Series, k: int
    ) -> pd.DataFrame:
        """
        Get scores for top-k features with metadata.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            k: Number of features

        Returns:
            DataFrame with columns: rank, gene_id, score
        """
        scores = self.calculate_scores(X_train, y_train)
        scores = self._validate_scores(scores, X_train)

        # Get top-k
        top_k = scores.abs().nlargest(k)

        df = pd.DataFrame({
            'rank': range(1, k + 1),
            'gene_id': top_k.index.tolist(),
            'score': [scores[g] for g in top_k.index]
        })

        return df

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(METHOD_NAME='{self.METHOD_NAME}')"
