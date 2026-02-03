"""
SCAD Regularization feature selection.

Implements SCAD-SVM using iteratively reweighted L1 approximation.
Tunes lambda via nested cross-validation on training data only.

Contract §4.6:
- Operates on training fold only
- Tunes lambda via nested Stratified CV (3-fold)
- Outputs scores as abs(coef_) aligned to feature names
- Records chosen best_lambda
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from .base import FeatureSelector
from ..config import (
    RANDOM_SEED,
    SCAD_LAMBDA_MIN,
    SCAD_LAMBDA_MAX,
    SCAD_LAMBDA_STEP,
    SCAD_INNER_CV_FOLDS,
    SCAD_A,
    EPS,
)


class SCADSelector(FeatureSelector):
    """
    SCAD-regularized feature selection via iteratively reweighted L1-SVM.

    SCAD (Smoothly Clipped Absolute Deviation) provides a non-convex penalty
    that approximates hard thresholding while maintaining continuity.

    Since sklearn doesn't have native SCAD, we approximate using
    iteratively reweighted L1 optimization.

    Attributes:
        best_lambda_: The lambda value selected by nested CV (per call)
    """

    METHOD_NAME = "scad_regularization"

    def __init__(
        self,
        lambda_min: float = SCAD_LAMBDA_MIN,
        lambda_max: float = SCAD_LAMBDA_MAX,
        lambda_step: float = SCAD_LAMBDA_STEP,
        a: float = SCAD_A,
        cv_folds: int = SCAD_INNER_CV_FOLDS,
        max_iter: int = 5,
        random_state: int = RANDOM_SEED,
    ):
        """
        Args:
            lambda_min: Minimum lambda for grid search
            lambda_max: Maximum lambda for grid search
            lambda_step: Step size for lambda grid
            a: SCAD shape parameter (default 3.7, Fan & Li recommendation)
            cv_folds: Number of inner CV folds for lambda selection
            max_iter: Maximum iterations for reweighted L1
            random_state: Random seed for reproducibility
        """
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_step = lambda_step
        self.a = a
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.random_state = random_state
        self.best_lambda_: Optional[float] = None
        self._feature_importances: Optional[pd.Series] = None

    def _scad_weight(self, beta: float, lam: float) -> float:
        """
        Compute SCAD penalty derivative weight.

        SCAD penalty derivative:
        - |beta| <= lambda: weight = lambda
        - lambda < |beta| <= a*lambda: weight = (a*lambda - |beta|) / (a - 1)
        - |beta| > a*lambda: weight = 0 (no penalty, keeps large coefficients)

        Args:
            beta: Coefficient value
            lam: Lambda regularization parameter

        Returns:
            Penalty weight for this coefficient
        """
        abs_beta = abs(beta)
        if abs_beta <= lam:
            return lam
        elif abs_beta <= self.a * lam:
            return (self.a * lam - abs_beta) / (self.a - 1)
        else:
            return 0.0

    def _fit_scad_svm(
        self, X: np.ndarray, y: np.ndarray, lam: float
    ) -> np.ndarray:
        """
        Fit SCAD-penalized SVM using iteratively reweighted L1.

        Algorithm:
        1. Initialize weights = 1 for all features
        2. Scale features by 1/weights (adaptive penalty)
        3. Fit L1-SVM
        4. Update weights based on SCAD derivative
        5. Repeat until convergence or max_iter

        Args:
            X: Standardized features (n_samples, n_features)
            y: Labels (n_samples,)
            lam: Regularization parameter

        Returns:
            Coefficient vector (n_features,)
        """
        n_features = X.shape[1]
        weights = np.ones(n_features)
        coef = np.zeros(n_features)

        for iteration in range(self.max_iter):
            # Scale features by inverse weights (adaptive penalty)
            weights_safe = np.clip(weights, EPS, None)
            X_weighted = X / weights_safe

            # Fit L1-penalized SVM
            # C is inverse of regularization, so use 1/lam
            clf = LinearSVC(
                penalty='l1',
                dual=False,
                C=1.0 / (lam + EPS),
                max_iter=10000,
                random_state=self.random_state,
            )

            try:
                clf.fit(X_weighted, y)
                # Unscale coefficients
                coef = clf.coef_.ravel() / weights_safe
            except Exception:
                # Convergence issues - return current coefficients
                break

            # Update weights based on SCAD penalty
            new_weights = np.array([self._scad_weight(c, lam) for c in coef])

            # Check convergence
            if np.allclose(weights, new_weights, rtol=1e-3):
                break

            weights = new_weights

        return coef

    def _tune_lambda(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Tune lambda using nested cross-validation.

        Args:
            X: Standardized features
            y: Labels

        Returns:
            Best lambda value
        """
        lambdas = np.arange(
            self.lambda_min,
            self.lambda_max + self.lambda_step / 2,
            self.lambda_step
        )

        best_lambda = lambdas[0]
        best_score = -np.inf

        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        for lam in lambdas:
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                # Fit SCAD-SVM
                coef = self._fit_scad_svm(X_tr, y_tr, lam)

                # Predict using linear combination
                y_scores = X_val @ coef
                y_pred = np.where(y_scores > 0, 1, 0)

                # Accuracy
                scores.append(accuracy_score(y_val, y_pred))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_lambda = lam

        return best_lambda

    def calculate_scores(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        """
        Calculate feature importance using SCAD-SVM.

        Steps:
        1. Standardize features (fit on training only)
        2. Tune lambda via nested CV
        3. Fit SCAD-SVM with best lambda
        4. Return absolute coefficients as importance scores

        Args:
            X_train: Training features (samples x genes)
            y_train: Training labels

        Returns:
            pd.Series of importance scores (|coef|) indexed by gene names
        """
        # Convert labels to numeric
        y_numeric = self._convert_labels(y_train)

        # Standardize features (training only)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train.values)
        y_array = y_numeric.values

        # Tune lambda via nested CV
        self.best_lambda_ = self._tune_lambda(X_scaled, y_array)

        # Fit final model with best lambda
        coef = self._fit_scad_svm(X_scaled, y_array, self.best_lambda_)

        # Importance = absolute coefficient
        self._feature_importances = pd.Series(np.abs(coef), index=X_train.columns)

        return self._feature_importances

    def get_best_lambda(self) -> Optional[float]:
        """Return the best lambda from the most recent fit."""
        return self.best_lambda_
