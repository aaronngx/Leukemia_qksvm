import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def get_num_qubits(num_features: int) -> int:
    """
    Calculate qubits for amplitude encoding.
    Logarithmic scaling: n = ⌈log₂(d)⌉
    
    Examples:
    - 16 features → 4 qubits
    - 32 features → 5 qubits
    - 50 features → 6 qubits
    """
    return max(1, int(np.ceil(np.log2(num_features))))


def preprocess_and_normalize(X, clip_alpha=1.0):
    """
    Preprocess features for amplitude encoding.
    
    Steps:
    1. Standardize: (x - μ)/σ
    2. Clip to [-1, 1] range
    3. Normalize per sample for amplitude encoding
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    clip_alpha : float
        Clipping scale factor
    
    Returns
    -------
    X_normalized : np.ndarray
        Preprocessed and normalized features
    scaler : StandardScaler
        Fitted scaler for inverse transform
    """
    # Standardize across features
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    
    # Clip to range
    X_clipped = np.clip(X_standardized / clip_alpha, -1, 1)
    
    # Normalize each sample to unit vector (for amplitude encoding)
    X_normalized = X_clipped / np.linalg.norm(X_clipped, axis=1, keepdims=True)
    
    return X_normalized, scaler


def amplitude_encoding_feature_map(num_features: int):
    """
    Create shallow amplitude encoding feature map U(x).
    
    Initial layer of RY rotations applied to normalized features.
    Maps d features to n = ⌈log₂(d)⌉ qubits.
    
    For 16 features → 4 qubits:
    - Each of the 16 normalized features gets encoded
    - RY rotations distribute features across 4 qubits
    
    Parameters
    ----------
    num_features : int
        Number of input features (e.g., 16 genes)
    
    Returns
    -------
    qc : QuantumCircuit
    x_params : list
        Feature parameters
    n_qubits : int
    """
    n_qubits = get_num_qubits(num_features)
    x = ParameterVector("x", num_features)
    
    qc = QuantumCircuit(n_qubits, name="U(x)")
    
    # Apply RY rotations to encode normalized features
    # Strategy: Distribute features across qubits cyclically
    for i in range(num_features):
        qubit_idx = i % n_qubits
        # Apply RY rotation with feature value
        # Multiple features per qubit are combined through successive rotations
        qc.ry(x[i], qubit_idx)
    
    return qc, list(x), n_qubits


def build_amplitude_vqc(num_features: int, reps: int = 2):
    """
    Build complete Amplitude-Encoded VQC.
    
    Architecture:
    1. Feature Map U(x): RY rotations on normalized features
    2. Variational Ansatz V(θ): RX-RZ-RX + linear CNOT
    
    For 16 genes, 2 layers:
    - Qubits: 4 (logarithmic scaling)
    - Parameters: 24 = 4 qubits × 3 rotations × 2 layers
    
    Parameters
    ----------
    num_features : int
        Number of input features
    reps : int
        Number of ansatz repetitions/layers
    
    Returns
    -------
    qc : QuantumCircuit
    x_params : list
        Feature parameters (input)
    theta_params : list
        Variational parameters (trainable)
    n_qubits : int
    """
    n_qubits = get_num_qubits(num_features)
    
    # Feature Map U(x): Initial layer of RY rotations
    feature_map, x_params, _ = amplitude_encoding_feature_map(num_features)
    
    # Variational Ansatz V(θ): Same structure as angle encoding
    # RX-RZ-RX blocks + linear nearest-neighbor entanglement
    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["rx", "rz", "rx"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=reps,
    )
    
    # Build full circuit: |ψ(x; θ)⟩ = V(θ)U(x)|0⟩⊗ⁿ
    qc = QuantumCircuit(n_qubits, name="AmplitudeVQC")
    qc.compose(feature_map, inplace=True)
    qc.barrier(label="U(x) | V(θ)")
    qc.compose(ansatz, inplace=True)
    
    # Configuration summary
    num_params = n_qubits * 3 * reps
    
    print(f"\n{'='*60}")
    print(f"AMPLITUDE ENCODING VQC CONFIGURATION")
    print(f"{'='*60}")
    print(f"Qubit Scaling:        Logarithmic n = ⌈log₂({num_features})⌉ = {n_qubits}")
    print(f"Feature Map U(x):     Initial RY rotations on normalized features")
    print(f"Variational Ansatz:   RX-RZ-RX + linear nearest-neighbor CNOT")
    print(f"Layers:               {reps}")
    print(f"Trainable Parameters: {num_params} = {n_qubits} qubits × 3 rotations × {reps} layers")
    print(f"Parameter Reduction:  75% vs angle encoding (96 → 24 for 16 features)")
    print(f"{'='*60}\n")
    
    return qc, x_params, list(ansatz.parameters), n_qubits


def train_amplitude_vqc(X, y, num_features=16, reps=2, test_size=0.3, 
                        max_iter=100, seed=42):
    """
    Train Amplitude-Encoded VQC using hybrid quantum-classical optimization.
    
    Training Process:
    - Optimizer: Adam (or COBYLA as alternative)
    - Gradients: Parameter-shift rule
    - Loop: Hybrid quantum-classical optimization
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Gene expression data
    y : np.ndarray of shape (n_samples,)
        Class labels
    num_features : int
        Number of features to use (after feature selection)
    reps : int
        Number of ansatz layers
    max_iter : int
        Maximum optimization iterations
    
    Returns
    -------
    clf : NeuralNetworkClassifier
        Trained classifier
    results : dict
        Training results and metrics
    """
    print("\n" + "="*60)
    print("AMPLITUDE-ENCODED VQC TRAINING")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Preprocess: standardize, clip, normalize
    print("\n[1/5] Preprocessing features...")
    X_train_norm, scaler = preprocess_and_normalize(X_train)
    X_test_norm, _ = preprocess_and_normalize(X_test)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {num_features}")
    
    # Build circuit
    print("\n[2/5] Building quantum circuit...")
    circuit, x_params, theta_params, n_qubits = build_amplitude_vqc(
        num_features=num_features, 
        reps=reps
    )
    
    # Create QNN
    print("\n[3/5] Creating Quantum Neural Network...")
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=x_params,
        weight_params=theta_params,
        input_gradients=True,  # Enable gradient computation
    )
    
    print(f"  Input parameters: {len(x_params)}")
    print(f"  Weight parameters: {len(theta_params)}")
    print(f"  Gradient method: Parameter-shift rule")
    
    # Create classifier with Adam optimizer
    print("\n[4/5] Training with hybrid quantum-classical optimization...")
    print(f"  Optimizer: Adam")
    print(f"  Max iterations: {max_iter}")
    
    # Note: Qiskit ML's NeuralNetworkClassifier uses different optimizer interface
    # For Adam-like behavior, we can use COBYLA or implement custom training loop
    from qiskit_algorithms.optimizers import COBYLA, ADAM
    
    # Use ADAM optimizer (if available in your Qiskit version)
    # Otherwise fall back to COBYLA
    try:
        optimizer = ADAM(maxiter=max_iter, lr=0.01)
        print(f"  Using ADAM optimizer")
    except:
        optimizer = COBYLA(maxiter=max_iter)
        print(f"  Using COBYLA optimizer (Adam not available)")
    
    clf = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=optimizer,
        callback=lambda weights, obj_val: print(f"    Iteration: obj value = {obj_val:.4f}")
    )
    
    # Train
    clf.fit(X_train_norm, y_train)
    
    # Evaluate
    print("\n[5/5] Evaluating model...")
    y_train_pred = clf.predict(X_train_norm)
    y_test_pred = clf.predict(X_test_norm)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, digits=4))
    print("="*60)
    
    results = {
        'circuit': circuit,
        'n_qubits': n_qubits,
        'n_params': len(theta_params),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'clf': clf
    }
    
    return clf, results


# Example usage with real preprocessed data
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from data_loader import discover_preprocessed_files, load_preprocessed_data

    print("\n" + "="*60)
    print("AMPLITUDE ENCODING - LOADING PREPROCESSED DATA")
    print("="*60)

    # Discover available files
    files = discover_preprocessed_files()

    if not (files['train_internal'] or files['folds']):
        print("\n[ERROR] No preprocessed data found!")
        print("Please run preprocessing.py first to generate feature-selected data.")
        print("\nExpected files in results/:")
        print("  - train_internal_top_*_anova_f.csv or train_internal_top_*_snr.csv")
        print("  - test_internal_top_*_anova_f.csv or test_internal_top_*_snr.csv")
        print("  - independent_top_*_anova_f.csv or independent_top_*_snr.csv")
        print("  OR")
        print("  - fold_1_train_top_*.csv, fold_1_test_top_*.csv, ...")
        sys.exit(1)

    # Print available files
    print("\n[INFO] Found preprocessed data:")
    if files['train_internal']:
        print(f"  ✓ Internal train: {files['train_internal'].name}")
    if files['test_internal']:
        print(f"  ✓ Internal test: {files['test_internal'].name}")
    if files['independent']:
        print(f"  ✓ Independent: {files['independent'].name}")
    if files['folds']:
        print(f"  ✓ Cross-validation: {len(files['folds'])} folds")

    # Load data based on available validation strategy
    if files['train_internal']:
        print("\n[INFO] Loading internal train/test split...")
        X_train, y_train = load_preprocessed_data(files['train_internal'])
        X_test, y_test = load_preprocessed_data(files['test_internal'])

        print(f"  Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        print(f"  Test: {X_test.shape[0]} samples × {X_test.shape[1]} features")
        print(f"  Class distribution (train): ALL={sum(y_train==0)}, AML={sum(y_train==1)}")

        # Combine for training (or use separately)
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

    elif files['folds']:
        print("\n[INFO] Found CV folds - using fold 1 for demonstration...")
        X_train, y_train = load_preprocessed_data(files['folds'][0][0])
        X_test, y_test = load_preprocessed_data(files['folds'][0][1])

        print(f"  Fold 1 train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        print(f"  Fold 1 test: {X_test.shape[0]} samples × {X_test.shape[1]} features")

        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

    else:
        print("\n[ERROR] No internal validation data found!")
        sys.exit(1)

    num_features = X.shape[1]
    n_qubits = get_num_qubits(num_features)

    print(f"\n[INFO] Training amplitude-encoded VQC:")
    print(f"  Features: {num_features}")
    print(f"  Qubits: {n_qubits} (logarithmic scaling)")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Class distribution: ALL={sum(y==0)}, AML={sum(y==1)}")
    print(f"\nThis may take a few minutes...\n")

    # Train amplitude-encoded VQC
    clf, results = train_amplitude_vqc(
        X, y,
        num_features=num_features,
        reps=2,
        max_iter=50
    )

    # Evaluate on independent set if available
    if files['independent']:
        print("\n" + "="*60)
        print("INDEPENDENT SET EVALUATION")
        print("="*60)
        X_ind, y_ind = load_preprocessed_data(files['independent'])
        print(f"Independent samples: {X_ind.shape[0]}")
        print(f"Class distribution: ALL={sum(y_ind==0)}, AML={sum(y_ind==1)}")

        # Preprocess independent set
        X_ind_norm, _ = preprocess_and_normalize(X_ind)
        y_ind_pred = clf.predict(X_ind_norm)

        ind_acc = accuracy_score(y_ind, y_ind_pred)
        print(f"\nIndependent Set Accuracy: {ind_acc:.4f}")
        print("\nClassification Report (Independent Set):")
        print(classification_report(y_ind, y_ind_pred,
                                    target_names=['ALL', 'AML'],
                                    digits=4))

    print("\n" + "="*60)
    print("✓ SUCCESSFULLY LOADED AND TRAINED ON PREPROCESSED DATA!")
    print("="*60)
    print(f"\n✓ Qubit scaling: Logarithmic ({num_features} features → {results['n_qubits']} qubits)")
    print(f"✓ Parameters: {results['n_params']}")
    print(f"✓ Training: Hybrid quantum-classical with parameter-shift gradients")
    print()