"""
Ensemble Voting feature selection.

Combines rankings from multiple base methods (SNR, MI, ANOVA, T-test)
using rank aggregation. Genes are ranked by average rank across methods.

Contract §4.5: For a given K, each base method returns a ranked list.
Compute avg_rank(g) across base methods; absent genes get rank = K+1.
Sort by avg_rank ascending; select top K.
"""

from typing import List, Dict
import numpy as np
import pandas as pd

from .base import FeatureSelector
from .signal_to_noise import SignalToNoiseSelector
from .anova_f_test import AnovaFSelector
from .mutual_information import MutualInformationSelector
from .t_test import TTestSelector
from ..config import RANDOM_SEED


class EnsembleVotingSelector(FeatureSelector):
    """
    Ensemble feature selection using rank aggregation.

    Combines 4 base methods: SNR, ANOVA, MI, T-test.
    For each gene, computes average rank across methods that selected it.
    Genes not selected by a method get rank = K+1.

    RECHECK #3: Overrides select_features() to use avg_rank directly.
    """

    METHOD_NAME = "ensemble_voting"

    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.selectors = [
            SignalToNoiseSelector(),
            AnovaFSelector(),
            MutualInformationSelector(random_state=random_state),
            TTestSelector(),
        ]

    def calculate_scores(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        """
        Calculate negative average rank for each feature.

        Returns -avg_rank so that "higher score = better feature"
        (consistent with base class selection logic).

        Args:
            X_train: Training features (samples x genes)
            y_train: Training labels

        Returns:
            pd.Series of -avg_rank scores indexed by gene names.
        """
        n_features = len(X_train.columns)

        # Collect ranks from each base method
        rank_matrix = pd.DataFrame(index=X_train.columns)

        for selector in self.selectors:
            scores = selector.calculate_scores(X_train, y_train)
            # Rank: 1 = best (highest |score|), n = worst
            ranks = scores.abs().rank(ascending=False, method='average')
            rank_matrix[selector.METHOD_NAME] = ranks

        # Average rank across methods
        avg_rank = rank_matrix.mean(axis=1)

        # Return -avg_rank so nlargest() gives lowest avg_rank
        return -avg_rank

    def select_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, k: int
    ) -> List[str]:
        """
        Select top-k features using rank aggregation.

        RECHECK #3: Override to properly handle avg_rank logic.

        For each K:
        1. Each base method ranks all features
        2. Compute avg_rank per gene across methods
        3. Select top K by lowest avg_rank

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            k: Number of features to select

        Returns:
            List of k feature names ordered by avg_rank (ascending)
        """
        # Get -avg_rank scores (higher = better = lower avg_rank)
        scores = self.calculate_scores(X_train, y_train)
        scores = self._validate_scores(scores, X_train)

        # Select top k (nlargest of -avg_rank = smallest avg_rank)
        return scores.nlargest(k).index.tolist()

    def get_individual_rankings(
        self, X_train: pd.DataFrame, y_train: pd.Series, k: int
    ) -> Dict[str, List[str]]:
        """
        Get top-k features from each individual base method.

        Useful for analysis and debugging.

        Args:
            X_train: Training features
            y_train: Training labels
            k: Number of features

        Returns:
            Dict mapping method name to list of top-k gene names
        """
        rankings = {}
        for selector in self.selectors:
            rankings[selector.METHOD_NAME] = selector.select_features(X_train, y_train, k)
        return rankings
