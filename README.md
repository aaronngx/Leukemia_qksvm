# Quantum Kernel SVM for Leukemia Classification

A quantum machine learning project for classifying leukemia subtypes (ALL vs AML) using the Golub dataset. This project implements quantum-enhanced classification with multiple encoding strategies and supports **flexible backend configuration** for scaling from small (2-16 qubits) to large systems (32+ qubits).

## Overview

This project compares quantum machine learning approaches for gene expression-based leukemia classification:

- **Quantum Kernel SVM (QKSVM)**: Uses quantum kernels for classification
- **Variational Quantum Classifier (VQC)**: Trainable quantum circuit classifier
- **Feature Selection**: SNR and ANOVA F-test methods
- **Encoding Methods**: Angle encoding (K features → K qubits) and Amplitude encoding (K features → log₂(K) qubits)
- **Kernel Methods**: Statevector, Tensor Network (MPS), Swap Test, Hadamard Test
- **Flexible Backends**: Exact statevector or tensor network (MPS) simulation
- **Scalability**: From 2 qubits to 32+ qubits with appropriate backend selection

## Project Structure

```
Leukemia_qksvm/
├── Experiment.py                        # Unified experiment runner (interactive)
├── angle_encoding.py                    # Angle encoding circuit
├── amplitude_encoding.py                # Amplitude encoding circuit
├── backend_config.py                    # Backend configuration & kernel methods
├── qksvm_golub.py                       # QKSVM implementation
├── vqc_golub.py                         # VQC implementation
├── generate_figures.py                  # Result visualization & figures
├── visualize_circuits.py                # Quantum circuit visualization
│
├── feature-selection-methods/
│   ├── signal_to_noise.py               # SNR feature selection
│   ├── anova_f.py                       # ANOVA F-test feature selection
│   └── compare_feature_selection.py     # Compare feature selection methods
│
├── data/
│   ├── raw/                             # Raw Golub dataset
│   └── processed_*/                     # Processed data (various configurations)
│
├── results/                             # Experiment results
├── results_golub_panel/                 # Panel experiment results
└── figures/                             # Generated figures
```

## Feature Selection Methods

### 1. Signal-to-Noise Ratio (SNR)

**Formula:**
```
SNR_j = |μ₁ - μ₀| / (σ₀ + σ₁)
```

Where:
- μ₀, μ₁ = mean expression in class 0 (ALL) and class 1 (AML)
- σ₀, σ₁ = standard deviation in each class

**Usage:**
```bash
python feature-selection-methods/signal_to_noise.py \
    --k 8 \
    --out_dir data/processed_snr_k8
```

### 2. ANOVA F-test

**Formula:**
```
F_j = (between-class variance) / (within-class variance)
```

Where:
- **Between-class variance**: Variance of class means around overall mean
- **Within-class variance**: Pooled variance within classes

Larger F-scores indicate stronger class separation.

**Usage:**
```bash
python feature-selection-methods/anova_f.py \
    --k 8 \
    --out_dir data/processed_anova_f_k8
```

## Backend Configuration

### Available Backends

This project supports two simulation backends:

| Backend | Description | Max Qubits | Memory | Accuracy | Speed |
|---------|-------------|------------|--------|----------|-------|
| **statevector** | Reference statevector (Qiskit) | ~16-20 | O(2^n) | Exact | Medium |
| **tensor_network** | Matrix Product State (MPS) | 32-100+ | O(χ²n) | Approximate | Medium |

### Memory Requirements

| Qubits | State Dimension | Memory (GB) | Recommended Backend |
|--------|----------------|-------------|---------------------|
| 4 | 16 | <0.001 | statevector |
| 8 | 256 | <0.01 | statevector |
| 16 | 65,536 | ~0.5 | statevector |
| 20 | 1,048,576 | ~8 | statevector (if enough RAM) |
| 24 | 16,777,216 | ~134 | **tensor_network** (required) |
| 32 | 4,294,967,296 | ~34,000 | **tensor_network** (required) |
| 64+ | 2^64 | Infeasible | **tensor_network** (required) |

### When to Use Each Backend

**Use `statevector`:**
- n ≤ 20 qubits (depending on RAM)
- Want exact simulation results
- Educational/debugging purposes

**Use `tensor_network` (MPS):**
- n ≥ 20-24 qubits
- Limited RAM (< 32GB)
- Acceptable to have approximate results
- Bond dimension χ controls accuracy/speed tradeoff

### Getting Backend Recommendations

```bash
# Get recommendation for specific qubit count
python qksvm_golub.py --train_csv data/processed_k32/train_topk_snr.csv \
    --recommend_backend --available_memory 16

# Or for VQC
python vqc_golub.py --train_csv data/processed_k32/train_topk_snr.csv \
    --recommend_backend --available_memory 16
```

## Running Experiments

### Quick Start: Interactive Mode (Recommended)

The easiest way to run experiments is using the unified experiment runner:

```bash
python Experiment.py
```

This launches an interactive menu that guides you through:
1. **Feature Selection Method**: ANOVA F-test, SNR, or Both
2. **K Values**: Number of features to select (e.g., 4, 8, 16, 32, 50)
3. **Encoding Type**: Angle, Amplitude, or Both
4. **Quantum Model**: VQC, QKSVM, or Both
5. **Kernel Method** (QKSVM only): Statevector, Tensor Network, Swap Test, Hadamard Test

The runner automatically:
- Runs feature selection for each (method, K) combination
- Scales features appropriately for each encoding type
- Switches to tensor network backend for large circuits (>20 qubits)
- Reports metrics: Accuracy, AUROC, F1-Score, Recall
- Saves results to `results/experiment_results.csv`

### Manual Workflow

**Step 1: Feature Selection**
```bash
# Select top k features using ANOVA F-test
python feature-selection-methods/anova_f.py \
    --k 8 \
    --out_dir data/processed_k8

# Or using SNR
python feature-selection-methods/signal_to_noise.py \
    --k 8 \
    --out_dir data/processed_k8
```

**Step 2: Run QKSVM or VQC**
```bash
# QKSVM with exact simulation (8 qubits)
python qksvm_golub.py \
    --train_csv data/processed_k8/train_top_8_anova_f.csv \
    --ind_csv data/processed_k8/independent_top_8_anova_f.csv \
    --backend statevector \
    --output_dir results/qksvm_k8

# QKSVM with tensor network (32 qubits)
python qksvm_golub.py \
    --train_csv data/processed_k32/train_top_32_anova_f.csv \
    --ind_csv data/processed_k32/independent_top_32_anova_f.csv \
    --backend tensor_network \
    --max_bond_dimension 100 \
    --output_dir results/qksvm_k32_tnw

# VQC with tensor network
python vqc_golub.py \
    --train_csv data/processed_k32/train_top_32_snr.csv \
    --ind_csv data/processed_k32/independent_top_32_snr.csv \
    --backend tensor_network \
    --max_bond_dimension 100 \
    --reps 2 \
    --max_iter 150 \
    --output_dir results/vqc_k32
```

### Compare Feature Selection Methods

```bash
python feature-selection-methods/compare_feature_selection.py
```

## Encoding Methods

### Angle Encoding
- **Mapping**: K features → K qubits
- **Circuit**: Each feature encoded as RY(θ) rotation
- **Use case**: Direct feature-to-qubit mapping, full expressibility

### Amplitude Encoding
- **Mapping**: K features → ceil(log₂(K)) qubits
- **Circuit**: Features encoded in amplitude of quantum state
- **Use case**: Compact representation, fewer qubits needed

## Kernel Methods (QKSVM)

| Method | Description | Qubits | Notes |
|--------|-------------|--------|-------|
| **statevector** | Exact state simulation | ≤20 | Reference implementation |
| **tensor_network** | MPS approximation | 20-100+ | Scalable, approximate |
| **swap_test** | Measurement-based | Any | Uses ancilla qubit |
| **hadamard_test** | Measurement-based | Any | Alternative to swap test |

## Quantum Kernel SVM Details

### Angle Encoding

Features are encoded into quantum states using RY rotation gates:
```
|ψ(x)⟩ = RY(x₁) ⊗ RY(x₂) ⊗ ... ⊗ RY(xₙ) |0...0⟩
```

Features are scaled to [0, π] before encoding.

### Kernel Computation

The quantum kernel is computed as:
```
K(x, z) = |⟨ψ(x)|ψ(z)⟩|² = |⟨0...0| U(z)† U(x) |0...0⟩|²
```

This is evaluated using Qiskit's Statevector simulator.

### Classification

A classical SVM with precomputed quantum kernel is trained using scikit-learn.

## Output Files

Each QKSVM run produces:

```
results/qksvm_{method}_k{n}/
├── circuit_{timestamp}.png                # Quantum circuit diagram
├── circuit_{timestamp}.txt                # Circuit as text
├── qksvm_model_{timestamp}.pkl           # Trained SVM model
├── qksvm_results_{timestamp}.txt         # Performance summary
├── validation_predictions_{timestamp}.csv # Validation set predictions
└── independent_predictions_{timestamp}.csv # Test set predictions
```

ANOVA F-test also produces:
```
data/processed_anova_f_k{n}/
└── anova_f_scores.csv  # F-scores for all features
```

## Requirements

- Python 3.10+
- qiskit
- qiskit-machine-learning
- qiskit-algorithms
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

Install dependencies:
```bash
pip install qiskit qiskit-machine-learning qiskit-algorithms scikit-learn pandas numpy matplotlib joblib
```

## Key Parameters

### Feature Selection
- `--k`: Number of features to select (typically 2, 4, 8, or 16 for quantum experiments)
- `--balanced`: Use balanced class selection (equal features from ALL and AML)

### QKSVM
- `--test_size`: Validation split ratio (default: 0.3)
- `--seed`: Random seed for reproducibility (default: 42)

### VQC
- `--reps`: Number of ansatz repetitions (circuit depth, default: 2)

## Performance Metrics

All experiments report:
- **Accuracy**: Overall classification accuracy
- **AUROC**: Area Under ROC Curve
- **F1-Score**: Harmonic mean of precision and recall
- **Recall**: Sensitivity / True positive rate
- **Precision**: Positive predictive value

Results are provided for:
1. **Validation set**: Held-out portion of training data
2. **Independent test set**: Separate test dataset

## Comparing SNR vs ANOVA F-test

Both methods select features with strong class discrimination but use different criteria:

- **SNR**: Emphasizes difference in means relative to combined standard deviations
- **ANOVA F-test**: Compares variance between classes to variance within classes

ANOVA F-test is more statistically principled and commonly used in genomics, while SNR is simpler and computationally faster.

## Example Workflow

```bash
# Option 1: Interactive mode (recommended)
python Experiment.py

# Option 2: Manual workflow
# 1. Run feature selection
python feature-selection-methods/anova_f.py --k 16 --out_dir results

# 2. Run QKSVM
python qksvm_golub.py \
    --train_csv results/train_top_16_anova_f.csv \
    --ind_csv results/independent_top_16_anova_f.csv \
    --backend statevector \
    --output_dir results/qksvm_k16

# 3. Generate figures
python generate_figures.py
```

## References

- **Golub et al. (1999)**: "Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring"
- **Havlíček et al. (2019)**: "Supervised learning with quantum-enhanced feature spaces"
- Quantum feature encoding and kernel methods

