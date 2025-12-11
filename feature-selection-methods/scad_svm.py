#!/usr/bin/env python3
"""
SCAD-SVM (Smoothly Clipped Absolute Deviation Support Vector Machine)
Feature Selection Method

SCAD regularization automatically performs feature selection by:
1. Pushing coefficients of weak genes toward zero
2. NOT over-penalizing important features (unlike L1/L2)
3. Providing sparse solutions ideal for high-dimensional biological data

Key Parameters:
- lambda (λ): Regularization strength
  - Higher λ: More aggressive feature elimination
  - Lower λ: Keeps more features, risks overfitting
- a: SCAD shape parameter (default 3.7, as recommended by Fan & Li, 2001)

References:
- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood.
  Journal of the American Statistical Association, 96(456), 1348-1360.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SCAD PENALTY FUNCTIONS
# =============================================================================

def scad_penalty(beta: np.ndarray, lam: float, a: float = 3.7) -> float:
    """
    Compute the SCAD penalty value.
    
    SCAD penalty has three regions:
    1. |β| ≤ λ: Linear penalty (like L1)
    2. λ < |β| ≤ aλ: Quadratic transition
    3. |β| > aλ: Constant (no further shrinkage)
    
    Args:
        beta: Coefficient vector
        lam: Regularization strength (λ)
        a: SCAD shape parameter (default 3.7)
    
    Returns:
        Total SCAD penalty value
    """
    abs_beta = np.abs(beta)
    penalty = np.zeros_like(abs_beta)
    
    # Region 1: |β| ≤ λ
    mask1 = abs_beta <= lam
    penalty[mask1] = lam * abs_beta[mask1]
    
    # Region 2: λ < |β| ≤ aλ
    mask2 = (abs_beta > lam) & (abs_beta <= a * lam)
    penalty[mask2] = -(abs_beta[mask2]**2 - 2*a*lam*abs_beta[mask2] + lam**2) / (2*(a-1))
    
    # Region 3: |β| > aλ
    mask3 = abs_beta > a * lam
    penalty[mask3] = (a + 1) * lam**2 / 2
    
    return np.sum(penalty)


def scad_derivative(beta: np.ndarray, lam: float, a: float = 3.7) -> np.ndarray:
    """
    Compute the derivative of SCAD penalty (subgradient for non-differentiable points).
    
    Args:
        beta: Coefficient vector
        lam: Regularization strength
        a: SCAD shape parameter
    
    Returns:
        Gradient vector
    """
    abs_beta = np.abs(beta)
    sign_beta = np.sign(beta)
    grad = np.zeros_like(beta)
    
    # Small threshold to avoid division issues
    eps = 1e-8
    abs_beta = np.maximum(abs_beta, eps)
    
    # Region 1: |β| ≤ λ
    mask1 = abs_beta <= lam
    grad[mask1] = lam * sign_beta[mask1]
    
    # Region 2: λ < |β| ≤ aλ  
    mask2 = (abs_beta > lam) & (abs_beta <= a * lam)
    grad[mask2] = (a * lam - abs_beta[mask2]) / (a - 1) * sign_beta[mask2]
    
    # Region 3: |β| > aλ (no penalty gradient)
    mask3 = abs_beta > a * lam
    grad[mask3] = 0
    
    return grad


# =============================================================================
# SCAD-SVM IMPLEMENTATION
# =============================================================================

class SCADSVM:
    """
    SCAD-regularized Support Vector Machine for feature selection.
    
    Uses SCAD penalty to automatically select important features while
    training an SVM classifier.
    
    Attributes:
        lam: Regularization strength (λ)
        a: SCAD shape parameter
        C: SVM regularization parameter
        feature_weights_: Learned feature importance weights
        selected_features_: Indices of selected features (non-zero weights)
        coef_: Final coefficient vector
    """
    
    def __init__(
        self,
        lam: float = 0.1,
        a: float = 3.7,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        """
        Initialize SCAD-SVM.
        
        Args:
            lam: Regularization strength (higher = more sparsity)
            a: SCAD shape parameter (default 3.7)
            C: SVM soft margin parameter
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
            verbose: Print optimization progress
        """
        self.lam = lam
        self.a = a
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        self.feature_weights_ = None
        self.selected_features_ = None
        self.coef_ = None
        self.intercept_ = None
        self.scaler_ = None
        
    def _hinge_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        """Compute hinge loss: max(0, 1 - y*(X@w + b))"""
        margins = y * (X @ w + b)
        return np.sum(np.maximum(0, 1 - margins))
    
    def _objective(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        SCAD-SVM objective function.
        
        Minimize: (1/n) * Σ hinge_loss + SCAD_penalty(w)
        """
        n_features = X.shape[1]
        w = params[:n_features]
        b = params[n_features]
        
        # Hinge loss
        loss = self._hinge_loss(X, y, w, b) / len(y)
        
        # SCAD penalty
        penalty = scad_penalty(w, self.lam, self.a)
        
        return self.C * loss + penalty
    
    def _gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function."""
        n_features = X.shape[1]
        w = params[:n_features]
        b = params[n_features]
        
        # Hinge loss gradient (subgradient)
        margins = y * (X @ w + b)
        mask = margins < 1
        
        grad_w = -self.C * (X[mask].T @ y[mask]) / len(y)
        grad_b = -self.C * np.sum(y[mask]) / len(y)
        
        # Add SCAD gradient
        grad_w += scad_derivative(w, self.lam, self.a)
        
        return np.concatenate([grad_w, [grad_b]])
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit SCAD-SVM and perform feature selection.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (+1/-1 or 0/1, will be converted to +1/-1)
        
        Returns:
            self
        """
        # Convert labels to +1/-1
        y = np.array(y)
        if set(np.unique(y)) == {0, 1}:
            y = 2 * y - 1
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        n_samples, n_features = X_scaled.shape
        
        # Initialize parameters
        w_init = np.zeros(n_features)
        b_init = 0.0
        params_init = np.concatenate([w_init, [b_init]])
        
        if self.verbose:
            print(f"SCAD-SVM: Optimizing with λ={self.lam}, a={self.a}")
        
        # Optimize using L-BFGS-B
        result = minimize(
            self._objective,
            params_init,
            args=(X_scaled, y),
            method='L-BFGS-B',
            jac=self._gradient,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'disp': self.verbose
            }
        )
        
        # Extract results
        self.coef_ = result.x[:n_features]
        self.intercept_ = result.x[n_features]
        
        # Feature weights are absolute coefficients
        self.feature_weights_ = np.abs(self.coef_)
        
        # Selected features are those with non-negligible weights
        threshold = 1e-6
        self.selected_features_ = np.where(self.feature_weights_ > threshold)[0]
        
        if self.verbose:
            print(f"  Selected {len(self.selected_features_)} / {n_features} features")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler_.transform(X)
        scores = X_scaled @ self.coef_ + self.intercept_
        return np.sign(scores)
    
    def get_feature_ranking(self) -> np.ndarray:
        """
        Get feature ranking based on absolute coefficient magnitude.
        
        Returns:
            Array of feature indices sorted by importance (descending)
        """
        return np.argsort(self.feature_weights_)[::-1]
    
    def select_top_k(self, k: int) -> np.ndarray:
        """
        Select top-k features based on SCAD coefficients.
        
        Args:
            k: Number of features to select
        
        Returns:
            Indices of top-k features
        """
        ranking = self.get_feature_ranking()
        return ranking[:k]


# =============================================================================
# SCAD-SVM FEATURE SELECTION PIPELINE
# =============================================================================

def tune_lambda_cv(
    X: np.ndarray,
    y: np.ndarray,
    lambda_range: list = None,
    cv: int = 5,
    verbose: bool = True
) -> tuple:
    """
    Tune SCAD lambda parameter using cross-validation.
    
    Args:
        X: Feature matrix
        y: Labels
        lambda_range: List of lambda values to try
        cv: Number of CV folds
        verbose: Print progress
    
    Returns:
        (best_lambda, cv_results_dict)
    """
    if lambda_range is None:
        lambda_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    results = {}
    
    for lam in lambda_range:
        if verbose:
            print(f"  Testing λ={lam}...", end=" ")
        
        try:
            model = SCADSVM(lam=lam, verbose=False)
            model.fit(X, y)
            
            # Get selected features
            n_selected = len(model.selected_features_)
            
            if n_selected == 0:
                if verbose:
                    print(f"0 features selected (too aggressive)")
                results[lam] = {'n_features': 0, 'accuracy': 0}
                continue
            
            # Train SVM on selected features and evaluate
            X_selected = X[:, model.selected_features_]
            
            svc = SVC(kernel='rbf', C=1.0)
            scores = cross_val_score(svc, X_selected, y, cv=cv, scoring='accuracy')
            
            mean_acc = np.mean(scores)
            results[lam] = {
                'n_features': n_selected,
                'accuracy': mean_acc,
                'std': np.std(scores)
            }
            
            if verbose:
                print(f"{n_selected} features, acc={mean_acc:.3f} ± {np.std(scores):.3f}")
                
        except Exception as e:
            if verbose:
                print(f"Failed: {e}")
            results[lam] = {'n_features': 0, 'accuracy': 0}
    
    # Find best lambda (maximize accuracy)
    best_lambda = max(results, key=lambda x: results[x]['accuracy'])
    
    return best_lambda, results


def run_scad_selection(
    k: int = 16,
    use_all_data: bool = False,
    lambda_value: float = None,
    auto_tune_lambda: bool = True,
    output_dir: str = "output",
    verbose: bool = True
):
    """
    Run SCAD-SVM feature selection on Golub leukemia dataset.
    
    Args:
        k: Number of features to select
        use_all_data: Use combined train + test data
        lambda_value: Fixed lambda value (if None and auto_tune=True, will tune)
        auto_tune_lambda: Automatically tune lambda via CV
        output_dir: Output directory
        verbose: Print progress
    
    Returns:
        dict with selected genes and performance metrics
    """
    # Load raw data
    raw_data_path = Path("data/raw")
    train_data_path = raw_data_path / "golub_train_data.csv"
    train_label_path = raw_data_path / "golub_train_labels.csv"
    test_data_path = raw_data_path / "golub_test_data.csv"
    test_label_path = raw_data_path / "golub_test_labels.csv"
    
    if verbose:
        print("="*60)
        print("SCAD-SVM FEATURE SELECTION")
        print("="*60)
    
    # Load data
    train_df = pd.read_csv(train_data_path, index_col=0)
    train_labels = pd.read_csv(train_label_path, index_col=0)
    test_df = pd.read_csv(test_data_path, index_col=0)
    test_labels = pd.read_csv(test_label_path, index_col=0)
    
    # Get gene names (accession numbers are in index)
    gene_names = train_df.index.tolist()
    
    # Transpose: rows = samples, columns = genes
    X_train = train_df.T.values
    X_test = test_df.T.values
    
    # Convert labels
    y_train = (train_labels.values.flatten() == 'AML').astype(int)
    y_test = (test_labels.values.flatten() == 'AML').astype(int)
    
    if use_all_data:
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        if verbose:
            print(f"Using all data: {X.shape[0]} samples, {X.shape[1]} genes")
    else:
        X = X_train
        y = y_train
        if verbose:
            print(f"Using training data: {X.shape[0]} samples, {X.shape[1]} genes")
    
    # Tune or use fixed lambda
    if auto_tune_lambda and lambda_value is None:
        if verbose:
            print("\nTuning lambda via cross-validation...")
        best_lambda, cv_results = tune_lambda_cv(X, y, verbose=verbose)
        lambda_value = best_lambda
        if verbose:
            print(f"\nBest λ = {best_lambda}")
    elif lambda_value is None:
        lambda_value = 0.1  # Default
    
    # Fit SCAD-SVM
    if verbose:
        print(f"\nFitting SCAD-SVM with λ={lambda_value}...")
    
    model = SCADSVM(lam=lambda_value, verbose=verbose)
    model.fit(X, y)
    
    # Get top-k features
    top_k_indices = model.select_top_k(k)
    top_k_genes = [gene_names[i] for i in top_k_indices]
    top_k_weights = model.feature_weights_[top_k_indices]
    
    if verbose:
        print(f"\nTop {k} genes selected by SCAD-SVM:")
        for i, (gene, weight) in enumerate(zip(top_k_genes, top_k_weights)):
            print(f"  {i+1}. {gene}: {weight:.4f}")
    
    # Create output dataframe
    results_df = pd.DataFrame({
        'rank': range(1, k+1),
        'gene_accession': top_k_genes,
        'scad_weight': top_k_weights,
        'feature_index': top_k_indices
    })
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save selected genes
    genes_file = output_path / f"selected_genes_scad_{k}genes.csv"
    results_df.to_csv(genes_file, index=False)
    
    # Create data files with selected genes
    X_train_selected = X_train[:, top_k_indices]
    X_test_selected = X_test[:, top_k_indices]
    
    train_out = pd.DataFrame(
        X_train_selected,
        columns=top_k_genes
    )
    train_out['cancer'] = ['ALL' if y == 0 else 'AML' for y in y_train]
    
    test_out = pd.DataFrame(
        X_test_selected,
        columns=top_k_genes
    )
    test_out['cancer'] = ['ALL' if y == 0 else 'AML' for y in y_test]
    
    if use_all_data:
        all_out = pd.concat([train_out, test_out], ignore_index=True)
        all_out.to_csv(output_path / f"all_top_{k}_scad.csv", index=False)
    else:
        train_out.to_csv(output_path / f"train_top_{k}_scad.csv", index=False)
        test_out.to_csv(output_path / f"independent_top_{k}_scad.csv", index=False)
    
    if verbose:
        print(f"\nResults saved to: {output_path}")
        print(f"  - {genes_file.name}")
    
    return {
        'selected_genes': top_k_genes,
        'weights': top_k_weights,
        'indices': top_k_indices,
        'lambda': lambda_value,
        'model': model,
        'results_df': results_df
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("SCAD-SVM Feature Selection for Golub Leukemia Dataset")
    print("="*60)
    
    # Example: Select 16 genes using SCAD-SVM
    results = run_scad_selection(
        k=16,
        use_all_data=False,
        auto_tune_lambda=True,
        output_dir="output/scad_k16",
        verbose=True
    )
    
    print("\n" + "="*60)
    print("SCAD-SVM Feature Selection Complete!")
    print("="*60)

