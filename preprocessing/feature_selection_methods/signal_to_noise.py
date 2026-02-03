"""
Signal-to-Noise Ratio (SNR) feature selection.

Implements the Golub et al. (1999) P-score method:
    P(g) = (mu_ALL - mu_AML) / (sigma_ALL + sigma_AML + eps)

Higher |P| indicates better class discrimination.
"""

import numpy as np
import pandas as pd

from .base import FeatureSelector
from ..config import EPS


class SignalToNoiseSelector(FeatureSelector):
    """
    Signal-to-Noise Ratio (Golub P-score) feature selection.

    Computes P-score for each gene:
        P(g) = (mu_ALL - mu_AML) / (sigma_ALL + sigma_AML + eps)

    Selection is based on |P| (absolute value).
    Positive P indicates gene is higher in ALL, negative P higher in AML.
    """

    METHOD_NAME = "signal_to_noise"

    def __init__(self, eps: float = EPS):
        """
        Args:
            eps: Small constant to avoid division by zero (default from config)
        """
        self.eps = eps

    def calculate_scores(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        """
        Calculate Golub P-scores for each feature.

        Args:
            X_train: Training features (samples x genes)
            y_train: Training labels (0=AML, 1=ALL or strings)

        Returns:
            pd.Series of P-scores indexed by gene names.
            Positive = higher in ALL, Negative = higher in AML.
        """
        # Convert labels to numeric and reset index for alignment
        y_numeric = self._convert_labels(y_train).reset_index(drop=True)
        X_reset = X_train.reset_index(drop=True)

        # Separate classes (ALL = 1, AML = 0)
        mask_all = (y_numeric == 1).values
        mask_aml = (y_numeric == 0).values

        X_all = X_reset.iloc[mask_all]
        X_aml = X_reset.iloc[mask_aml]

        # Calculate class-wise statistics
        mu_all = X_all.mean()
        mu_aml = X_aml.mean()
        sigma_all = X_all.std(ddof=1).clip(lower=self.eps)
        sigma_aml = X_aml.std(ddof=1).clip(lower=self.eps)

        # P-score = (mu_ALL - mu_AML) / (sigma_ALL + sigma_AML)
        p_scores = (mu_all - mu_aml) / (sigma_all + sigma_aml + self.eps)

        return p_scores
