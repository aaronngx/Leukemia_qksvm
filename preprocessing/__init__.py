"""
Preprocessing package for Golub AML/ALL feature selection experiments.

This package implements a reproducible, leakage-safe benchmark comparing
6 feature selection methods across different gene counts (K values).
"""

from .Preprocessing import FeatureSelectionExperiment
from . import config
from . import feature_selection_methods

__all__ = ['FeatureSelectionExperiment', 'config', 'feature_selection_methods']
