#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Quantum Machine Learning

This module implements the correct preprocessing chains for:
1. Amplitude Encoding: Split → Feature Selection → Scaler → Clipping → Padding → AmplitudeEmbedding(normalize=True)
2. Angle Encoding: Split → Feature Selection → Scaler → Clipping → (Optional L2) → Angle Mapping → Feature Map

Key Principles:
- Feature selection fitted on TRAIN only, applied to all sets
- Scaler fitted on TRAIN only, transform all sets
- No data leakage between train/val/test
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Optional, Dict, Any, Literal
from dataclasses import dataclass
import warnings


@dataclass
class PreprocessedData:
    """Container for preprocessed data with metadata."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    scaler: Optional[Any] = None
    selected_features: Optional[np.ndarray] = None
    n_qubits: Optional[int] = None
    encoding_type: Optional[str] = None
    clip_val: Optional[float] = None


# =============================================================================
# AMPLITUDE ENCODING PREPROCESSING
# =============================================================================

def preprocess_for_amplitude(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    scaler_type: Literal["standard", "minmax"] = "standard",
    clip_val: float = 3.0,
    verbose: bool = True,
) -> PreprocessedData:
    """
    Preprocess data for Amplitude Encoding.
    
    Pipeline:
    1. Scaler (fit on train, transform all)
    2. Clipping (optional, same bounds for all)
    3. Padding to 2^n (if needed)
    4. NO L2 normalization (handled by AmplitudeEmbedding(normalize=True))
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        X_test: Test features (optional)
        y_test: Test labels (optional)
        scaler_type: "standard" (Z-score) or "minmax"
        clip_val: Clipping threshold (default 3.0 for Z-score)
        verbose: Print preprocessing info
    
    Returns:
        PreprocessedData with all preprocessed arrays and metadata
    """
    n_features = X_train.shape[1]
    n_qubits = int(np.ceil(np.log2(n_features)))
    padded_size = 2 ** n_qubits
    
    if verbose:
        print(f"\n{'='*60}")
        print("AMPLITUDE ENCODING PREPROCESSING")
        print(f"{'='*60}")
        print(f"Features: {n_features} → Qubits: {n_qubits} (2^{n_qubits}={padded_size})")
    
    # Step 1: Fit scaler on training data ONLY
    if scaler_type == "standard":
        scaler = StandardScaler()
        if verbose:
            print(f"Scaler: StandardScaler (Z-score normalization)")
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if verbose:
            print(f"Scaler: MinMaxScaler (range [-1, 1])")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    if verbose:
        print(f"  Train mean after scaling: {X_train_scaled.mean():.4f}")
        print(f"  Train std after scaling: {X_train_scaled.std():.4f}")
    
    # Step 2: Clipping
    if clip_val is not None:
        X_train_clipped = np.clip(X_train_scaled, -clip_val, clip_val)
        X_val_clipped = np.clip(X_val_scaled, -clip_val, clip_val) if X_val_scaled is not None else None
        X_test_clipped = np.clip(X_test_scaled, -clip_val, clip_val) if X_test_scaled is not None else None
        if verbose:
            print(f"Clipping: [-{clip_val}, {clip_val}]")
    else:
        X_train_clipped = X_train_scaled
        X_val_clipped = X_val_scaled
        X_test_clipped = X_test_scaled
    
    # Step 3: Padding (if needed)
    if n_features < padded_size:
        pad_width = padded_size - n_features
        X_train_padded = np.pad(X_train_clipped, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        X_val_padded = np.pad(X_val_clipped, ((0, 0), (0, pad_width)), mode='constant', constant_values=0) if X_val_clipped is not None else None
        X_test_padded = np.pad(X_test_clipped, ((0, 0), (0, pad_width)), mode='constant', constant_values=0) if X_test_clipped is not None else None
        if verbose:
            print(f"Padding: {n_features} → {padded_size} (added {pad_width} zeros)")
    else:
        X_train_padded = X_train_clipped
        X_val_padded = X_val_clipped
        X_test_padded = X_test_clipped
        if verbose:
            print(f"Padding: Not needed ({n_features} = 2^{n_qubits})")
    
    # Step 4: NO L2 normalization here
    # AmplitudeEmbedding(normalize=True) handles this internally
    if verbose:
        print(f"L2 Normalization: Handled by AmplitudeEmbedding(normalize=True)")
        print(f"{'='*60}\n")
    
    return PreprocessedData(
        X_train=X_train_padded,
        y_train=y_train,
        X_val=X_val_padded,
        y_val=y_val,
        X_test=X_test_padded,
        y_test=y_test,
        scaler=scaler,
        n_qubits=n_qubits,
        encoding_type="amplitude",
        clip_val=clip_val,
    )


# =============================================================================
# ANGLE ENCODING PREPROCESSING
# =============================================================================

def preprocess_for_angle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    scaler_type: Literal["standard", "minmax"] = "standard",
    clip_val: float = 3.0,
    angle_range: Literal["0_pi", "0_2pi", "neg_pi_pi"] = "0_pi",
    use_l2: bool = False,
    verbose: bool = True,
) -> PreprocessedData:
    """
    Preprocess data for Angle Encoding (Simple RY, ZZ Feature Map, BPS).
    
    Pipeline:
    1. Scaler (fit on train, transform all)
    2. Clipping (optional, same bounds for all)
    3. Optional L2 normalization (for direction-only info)
    4. Map to angles (configurable range)
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        X_test: Test features (optional)
        y_test: Test labels (optional)
        scaler_type: "standard" (Z-score) or "minmax"
        clip_val: Clipping threshold (default 3.0)
        angle_range: Target angle range:
            - "0_pi": [0, π] - most common for RY gates
            - "0_2pi": [0, 2π] - full rotation range
            - "neg_pi_pi": [-π, π] - symmetric around zero
        use_l2: Apply L2 normalization (default False)
        verbose: Print preprocessing info
    
    Returns:
        PreprocessedData with all preprocessed arrays and metadata
    """
    n_features = X_train.shape[1]
    n_qubits = n_features  # Angle encoding: 1 qubit per feature
    
    if verbose:
        print(f"\n{'='*60}")
        print("ANGLE ENCODING PREPROCESSING")
        print(f"{'='*60}")
        print(f"Features: {n_features} → Qubits: {n_qubits} (1:1 mapping)")
    
    # Step 1: Fit scaler on training data ONLY
    if scaler_type == "standard":
        scaler = StandardScaler()
        if verbose:
            print(f"Scaler: StandardScaler (Z-score normalization)")
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if verbose:
            print(f"Scaler: MinMaxScaler (range [-1, 1])")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    if verbose:
        print(f"  Train mean after scaling: {X_train_scaled.mean():.4f}")
        print(f"  Train std after scaling: {X_train_scaled.std():.4f}")
    
    # Step 2: Clipping
    if clip_val is not None:
        X_train_clipped = np.clip(X_train_scaled, -clip_val, clip_val)
        X_val_clipped = np.clip(X_val_scaled, -clip_val, clip_val) if X_val_scaled is not None else None
        X_test_clipped = np.clip(X_test_scaled, -clip_val, clip_val) if X_test_scaled is not None else None
        if verbose:
            print(f"Clipping: [-{clip_val}, {clip_val}]")
    else:
        X_train_clipped = X_train_scaled
        X_val_clipped = X_val_scaled
        X_test_clipped = X_test_scaled
        clip_val = X_train_scaled.max()  # Use data max for angle mapping
    
    # Step 3: Optional L2 normalization
    if use_l2:
        def l2_normalize(X):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            return X / norms
        
        X_train_l2 = l2_normalize(X_train_clipped)
        X_val_l2 = l2_normalize(X_val_clipped) if X_val_clipped is not None else None
        X_test_l2 = l2_normalize(X_test_clipped) if X_test_clipped is not None else None
        if verbose:
            print(f"L2 Normalization: Applied (direction-only)")
    else:
        X_train_l2 = X_train_clipped
        X_val_l2 = X_val_clipped
        X_test_l2 = X_test_clipped
        if verbose:
            print(f"L2 Normalization: Skipped")
    
    # Step 4: Map to angles based on selected range
    ANGLE_RANGES = {
        "0_pi": (0, np.pi),           # [0, π]
        "0_2pi": (0, 2 * np.pi),      # [0, 2π]
        "neg_pi_pi": (-np.pi, np.pi), # [-π, π]
    }
    
    angle_min, angle_max = ANGLE_RANGES.get(angle_range, (0, np.pi))
    
    def map_to_angles(X, clip_val):
        """Map clipped values to angle range."""
        # Normalize to [0, 1]
        X_norm = (X + clip_val) / (2 * clip_val)
        # Map to angle range
        return angle_min + X_norm * (angle_max - angle_min)
    
    X_train_angles = map_to_angles(X_train_l2, clip_val)
    X_val_angles = map_to_angles(X_val_l2, clip_val) if X_val_l2 is not None else None
    X_test_angles = map_to_angles(X_test_l2, clip_val) if X_test_l2 is not None else None
    
    # Store angle range name for reference
    angle_range_name = angle_range
    
    if verbose:
        print(f"Angle Mapping: {angle_range} → [{angle_min:.2f}, {angle_max:.2f}]")
        print(f"  Train angle range: [{X_train_angles.min():.2f}, {X_train_angles.max():.2f}]")
        print(f"{'='*60}\n")
    
    return PreprocessedData(
        X_train=X_train_angles,
        y_train=y_train,
        X_val=X_val_angles,
        y_val=y_val,
        X_test=X_test_angles,
        y_test=y_test,
        scaler=scaler,
        n_qubits=n_qubits,
        encoding_type="angle",
        clip_val=clip_val,
    )


# =============================================================================
# UNIFIED PREPROCESSING FUNCTION
# =============================================================================

def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    encoding_type: Literal["amplitude", "angle"] = "amplitude",
    test_size: float = 0.3,
    val_size: float = 0.0,
    random_state: int = 42,
    stratify: bool = True,
    **kwargs
) -> PreprocessedData:
    """
    Unified preprocessing function for both encoding types.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels
        encoding_type: "amplitude" or "angle"
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining after test)
        random_state: Random seed
        stratify: Use stratified split
        **kwargs: Additional arguments for specific preprocessing
    
    Returns:
        PreprocessedData object
    """
    # Split data
    stratify_y = y if stratify else None
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    
    if val_size > 0:
        # Further split train into train/val
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_fraction, 
            random_state=random_state, stratify=y_train_full if stratify else None
        )
    else:
        X_train, y_train = X_train_full, y_train_full
        X_val, y_val = None, None
    
    # Apply encoding-specific preprocessing
    if encoding_type == "amplitude":
        return preprocess_for_amplitude(
            X_train, y_train, X_val, y_val, X_test, y_test, **kwargs
        )
    else:
        return preprocess_for_angle(
            X_train, y_train, X_val, y_val, X_test, y_test, **kwargs
        )


# =============================================================================
# CROSS-VALIDATION PREPROCESSING
# =============================================================================

def preprocess_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    encoding_type: Literal["amplitude", "angle"] = "amplitude",
    n_folds: int = 5,
    random_state: int = 42,
    **kwargs
) -> list:
    """
    Preprocess data for cross-validation.
    
    For each fold:
    - Fit scaler on train fold ONLY
    - Apply same transformation to val fold
    
    Args:
        X: Feature matrix
        y: Labels
        encoding_type: "amplitude" or "angle"
        n_folds: Number of CV folds
        random_state: Random seed
        **kwargs: Additional preprocessing arguments
    
    Returns:
        List of PreprocessedData objects, one per fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_data = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if encoding_type == "amplitude":
            preprocessed = preprocess_for_amplitude(
                X_train, y_train, X_val, y_val, verbose=False, **kwargs
            )
        else:
            preprocessed = preprocess_for_angle(
                X_train, y_train, X_val, y_val, verbose=False, **kwargs
            )
        
        fold_data.append(preprocessed)
    
    return fold_data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_amplitude_config():
    """Get default preprocessing config for amplitude encoding."""
    return {
        "scaler_type": "standard",  # Z-score (Golub-style)
        "clip_val": 3.0,            # ±3 standard deviations
        # No L2 - handled by AmplitudeEmbedding(normalize=True)
    }


def get_angle_config(angle_range: str = "0_pi"):
    """
    Get default preprocessing config for angle encoding.
    
    Args:
        angle_range: One of:
            - "0_pi": [0, π] - most common, good for RY gates
            - "0_2pi": [0, 2π] - full rotation
            - "neg_pi_pi": [-π, π] - symmetric around zero
    """
    return {
        "scaler_type": "standard",      # Z-score
        "clip_val": 3.0,                # ±3 standard deviations
        "angle_range": angle_range,     # "0_pi", "0_2pi", or "neg_pi_pi"
        "use_l2": False,                # Optional
    }


def print_preprocessing_summary(data: PreprocessedData):
    """Print summary of preprocessed data."""
    print(f"\nPreprocessing Summary ({data.encoding_type} encoding):")
    print(f"  Train: {data.X_train.shape}")
    if data.X_val is not None:
        print(f"  Val:   {data.X_val.shape}")
    if data.X_test is not None:
        print(f"  Test:  {data.X_test.shape}")
    print(f"  Qubits: {data.n_qubits}")
    print(f"  Clip value: {data.clip_val}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    
    # Generate sample data (16 features like Golub)
    n_samples = 50
    n_features = 16
    X = np.random.randn(n_samples, n_features) * 100 + 500  # Gene expression scale
    y = np.array([0] * 25 + [1] * 25)  # Binary labels
    
    print("="*70)
    print("DATA PREPROCESSING PIPELINE DEMO")
    print("="*70)
    
    # Amplitude encoding preprocessing
    print("\n>>> AMPLITUDE ENCODING")
    amp_data = preprocess_data(
        X, y, 
        encoding_type="amplitude",
        test_size=0.3,
        **get_amplitude_config()
    )
    print_preprocessing_summary(amp_data)
    
    # Angle encoding preprocessing - [0, π]
    print("\n>>> ANGLE ENCODING [0, π]")
    angle_data = preprocess_data(
        X, y,
        encoding_type="angle", 
        test_size=0.3,
        **get_angle_config("0_pi")
    )
    print_preprocessing_summary(angle_data)
    
    # Angle encoding preprocessing - [0, 2π]
    print("\n>>> ANGLE ENCODING [0, 2π]")
    angle_data_2pi = preprocess_data(
        X, y,
        encoding_type="angle", 
        test_size=0.3,
        **get_angle_config("0_2pi")
    )
    print_preprocessing_summary(angle_data_2pi)
    
    # Angle encoding preprocessing - [-π, π]
    print("\n>>> ANGLE ENCODING [-π, π]")
    angle_data_neg = preprocess_data(
        X, y,
        encoding_type="angle", 
        test_size=0.3,
        **get_angle_config("neg_pi_pi")
    )
    print_preprocessing_summary(angle_data_neg)
    
    # Cross-validation example
    print("\n>>> 5-FOLD CROSS-VALIDATION (Amplitude)")
    cv_folds = preprocess_cv_folds(X, y, encoding_type="amplitude", n_folds=5)
    print(f"Created {len(cv_folds)} folds")
    for i, fold in enumerate(cv_folds):
        print(f"  Fold {i+1}: Train={fold.X_train.shape[0]}, Val={fold.X_val.shape[0]}")

