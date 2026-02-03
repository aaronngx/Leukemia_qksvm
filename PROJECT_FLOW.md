# Leukemia QKSVM Project Flow

## Complete Project Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LEUKEMIA QKSVM PROJECT                              │
│                   Quantum Machine Learning for Cancer Classification         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA INPUT                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw Golub Leukemia Dataset (1999)
    ├─ Training Set:     7,129 genes × 38 patients
    │  ├─ 27 ALL (Acute Lymphoblastic Leukemia)
    │  └─ 11 AML (Acute Myeloid Leukemia)
    │
    └─ Independent Set:  7,129 genes × 34 patients
       └─ Completely separate validation set

    Files:
    ├─ data/raw/data_set_ALL_AML_train.csv
    ├─ data/raw/data_set_ALL_AML_independent.csv
    └─ data/raw/actual.csv (labels)

                            ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: PREPROCESSING & FEATURE SELECTION                                   │
│ (Run: python preprocessing.py)                                              │
└─────────────────────────────────────────────────────────────────────────────┘

    USER CHOICES:
    ┌─────────────────────────────────────────────────────┐
    │ [1/4] Number of genes/features to select            │
    │       Options: k ∈ {4, 8, 16, 24, 32, 50}           │
    │       Gene balance: Yes (k/2 ALL + k/2 AML) or No   │
    ├─────────────────────────────────────────────────────┤
    │ [2/4] Feature selection method                      │
    │       1. ANOVA F-test                               │
    │       2. SNR (Signal-to-Noise Ratio)                │
    │       3. SCAD-SVM                                   │
    │       4. All methods                                │
    ├─────────────────────────────────────────────────────┤
    │ [3/4] Patient samples                               │
    │       1. All 38 patients (imbalanced)               │
    │       2. Balanced 22 (11 ALL + 11 AML)              │
    ├─────────────────────────────────────────────────────┤
    │ [4/4] Internal validation strategy                  │
    │       1a. 70/30 split                               │
    │       1b. 80/20 split                               │
    │       2a. 5-fold CV                                 │
    │       2b. 10-fold CV                                │
    │       2c. LOOCV (Leave-One-Out CV)                  │
    └─────────────────────────────────────────────────────┘

    DIMENSIONALITY REDUCTION:
    7,129 genes ──[Feature Selection]──> k genes (e.g., 16 genes)

    Methods:
    ┌──────────────┬─────────────────────────────────────────┐
    │ ANOVA F-test │ F = between_var / within_var            │
    ├──────────────┼─────────────────────────────────────────┤
    │ SNR          │ P(g) = (μ_ALL - μ_AML)/(σ_ALL + σ_AML) │
    ├──────────────┼─────────────────────────────────────────┤
    │ SCAD-SVM     │ Smoothly Clipped Absolute Deviation     │
    └──────────────┴─────────────────────────────────────────┘

    OUTPUT FILES (Example: k=16, ANOVA, 70/30 split):
    results/
    ├─ train_internal_top_16_anova_f.csv      (~15 samples × 16 genes)
    ├─ test_internal_top_16_anova_f.csv       (~7 samples × 16 genes)
    ├─ independent_top_16_anova_f.csv         (34 samples × 16 genes)
    ├─ topk_anova_f_16genes.csv               (rankings & scores)
    └─ selected_genes_anova_f_16genes.csv     (gene IDs)

                            ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: DATA LOADING                                                        │
│ (Run: python run_pipeline.py)                                               │
└─────────────────────────────────────────────────────────────────────────────┘

    Select preprocessed file from results/
    Load: X_train, y_train, X_test, y_test

    Data shape: (n_samples, k_genes)
    Example: (15, 16) for training

                            ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: QUANTUM ENCODING SELECTION                                          │
└─────────────────────────────────────────────────────────────────────────────┘

    Choose encoding method:
    ┌─────────────────┬──────────────┬─────────────────────────────┐
    │ Encoding        │ Qubits       │ Scaling & Normalization     │
    ├─────────────────┼──────────────┼─────────────────────────────┤
    │ Amplitude       │ ⌈log₂(k)⌉    │ StandardScaler + L2-norm    │
    │ (Logarithmic)   │ k=16 → 4     │ (x-μ)/σ then x/||x||        │
    ├─────────────────┼──────────────┼─────────────────────────────┤
    │ Angle           │ k            │ MinMaxScaler → [0,π]        │
    │ (Linear)        │ k=16 → 16    │ (x-min)/(max-min) × π       │
    │  - Simple RY    │              │                             │
    │  - ZZ Map       │              │                             │
    │  - BPS Circuit  │              │                             │
    └─────────────────┴──────────────┴─────────────────────────────┘

                ┌──────────────┴──────────────┐
                │                             │
                ↓                             ↓

    ┌───────────────────────┐     ┌───────────────────────┐
    │  AMPLITUDE ENCODING   │     │    ANGLE ENCODING     │
    └───────────────────────┘     └───────────────────────┘

    Amplitude: 16 features → 4 qubits
    ────────────────────────────────────
    q0: ─┤                            ├─
    q1: ─┤  Initialize(x[0:15])       ├─
    q2: ─┤  Mottonen Decomposition    ├─
    q3: ─┤                            ├─

    State: |ψ⟩ = Σᵢ xᵢ|i⟩
    Properties:
    • Logarithmic scaling
    • Deep circuit O(2^n)
    • Exponential data density

    Angle (Simple RY): 16 features → 16 qubits
    ────────────────────────────────────────────
    q0:  ─RY(x[0])─
    q1:  ─RY(x[1])─
    q2:  ─RY(x[2])─
    ...
    q15: ─RY(x[15])─

    Properties:
    • Linear scaling
    • Shallow circuit
    • No entanglement

    Angle (ZZ Feature Map): 16 features → 16 qubits
    ────────────────────────────────────────────────
    q0: ─H─RZ(2x[0])─●────────────●─
    q1: ─H─RZ(2x[1])─X─RZ(φ₀₁)─X─●────────
    q2: ─H─RZ(2x[2])───────────────X─RZ(φ₁₂)─X─
    ...

    φᵢⱼ = 2(π-xᵢ)(π-xⱼ)
    Properties:
    • Pairwise interactions
    • ZZ entanglement
    • Data re-uploading via reps

    Angle (BPS Circuit): 16 features → 16 qubits
    ─────────────────────────────────────────────
    q0: ─RZ(x[0])─RY(x[0])─●───────RZ(x[0])─
    q1: ─RZ(x[1])─RY(x[1])─X─●─────RZ(x[1])─
    q2: ─RZ(x[2])─RY(x[2])───X─●───RZ(x[2])─
    ...

    Properties:
    • Two-axis encoding (RZ+RY)
    • Linear CNOT chain
    • Explicit data re-uploading

                            ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: CLASSIFIER SELECTION & TRAINING                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    Choose classifier:
    ┌──────────┬────────────────────────────────────────────────────┐
    │ QKSVM    │ Quantum Kernel + Classical SVM                     │
    ├──────────┼────────────────────────────────────────────────────┤
    │ VQC      │ Variational Quantum Classifier                     │
    └──────────┴────────────────────────────────────────────────────┘

                ┌──────────────┴──────────────┐
                │                             │
                ↓                             ↓

┌───────────────────────────────────┐  ┌───────────────────────────────────┐
│        QKSVM WORKFLOW              │  │         VQC WORKFLOW              │
│   (qksvm_golub.py)                 │  │     (vqc_golub.py)                │
└───────────────────────────────────┘  └───────────────────────────────────┘

QKSVM:                                VQC:
──────                                ────
1. Build Feature Map                  1. Build Feature Map
   U(x) = encode(x)                      U(x) = encode(x)

2. Compute Quantum Kernel             2. Add Variational Ansatz
   K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|²             V(θ) = TwoLocal ansatz
                                         - Rotation: RX-RZ-RX
   For each pair (xᵢ, xⱼ):               - Entanglement: Linear CNOT
   ┌─────────────────────┐                - Reps: 1-3 layers
   │ U(xᵢ)† U(xⱼ) |0⟩    │
   │ Measure |⟨0|ψ⟩|²    │             3. Define Observable
   └─────────────────────┘               Parity = sum(qubits) mod 2

3. Build Kernel Matrix                4. Create QNN
   K_train: (n_train × n_train)          qnn = EstimatorQNN(
   K_test:  (n_test × n_train)             feature_map=U(x),
                                            ansatz=V(θ),
4. Train Classical SVM                     observable=Parity)
   sklearn.svm.SVC(
     kernel='precomputed')             5. Train with Optimizer
   Optimize: α, b                        COBYLA optimizer
                                         Max iterations: 50
5. Predict                               Loss: Squared error
   y = sign(Σᵢ αᵢyᵢK[i,x] + b)
                                      6. Predict
                                         Execute trained circuit
                                         Measure parity → {0,1}

Kernel Methods:                       Parameter Count:
├─ Statevector (default)              Amplitude (4 qubits, reps=1):
├─ Tensor Network (MPS)               - Feature params: 16
├─ Swap Test                          - Trainable params: 24
└─ Hadamard Test                      Total: 40 parameters

                            ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: EVALUATION                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

    INTERNAL TEST SET (from 70/30 split):
    ├─ Accuracy
    ├─ Precision, Recall, F1-score
    ├─ ROC-AUC
    └─ Confusion Matrix

    INDEPENDENT VALIDATION SET (CRITICAL):
    ├─ Load independent_top_k.csv (34 samples)
    ├─ Apply same scaler/preprocessing
    ├─ Predict with trained model
    └─ Report final unbiased performance

    Metrics calculated:
    ┌─────────────────┬──────────────────────────────────┐
    │ Accuracy        │ (TP + TN) / Total                │
    ├─────────────────┼──────────────────────────────────┤
    │ Precision       │ TP / (TP + FP)                   │
    ├─────────────────┼──────────────────────────────────┤
    │ Recall          │ TP / (TP + FN)                   │
    ├─────────────────┼──────────────────────────────────┤
    │ F1-Score        │ 2 × (Prec × Rec)/(Prec + Rec)    │
    ├─────────────────┼──────────────────────────────────┤
    │ ROC-AUC         │ Area Under ROC Curve             │
    └─────────────────┴──────────────────────────────────┘

                            ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: RESULTS & ANALYSIS                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    OUTPUT FILES:
    ├─ results_qksvm/ or results_vqc/
    │  ├─ metrics.json
    │  ├─ classification_report.txt
    │  ├─ confusion_matrix.png
    │  └─ model.pkl
    │
    └─ figures/
       ├─ accuracy_vs_k.png
       ├─ encoding_comparison.png
       └─ feature_selection_comparison.png

    ANALYSIS SCRIPTS:
    ├─ generate_figures.py          (Create visualizations)
    ├─ generate_analysis_plots.py   (Parameter efficiency)
    └─ run_comprehensive_test.py    (Test all combinations)

                            ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ OPTIONAL: COMPREHENSIVE TESTING                                             │
│ (Run: python run_comprehensive_test.py)                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    Test all combinations:
    ├─ K_VALUES = [4, 8, 16]
    ├─ FEATURE_SELECTION = [ANOVA, SNR]
    ├─ ANGLE_TYPES = [Simple RY, ZZ, BPS]
    ├─ PATIENT_BALANCE = [True, False]
    └─ GENE_BALANCE = [True, False]

    Total: 3 × 2 × 3 × 2 × 2 = 72 test runs

    Output: comprehensive_test_results.csv
    Plots: Comparison of all configurations

═══════════════════════════════════════════════════════════════════════════════
```

## Key Entry Points

### 1. **Interactive Pipeline** (Recommended)
```bash
python preprocessing.py    # Step 1: Feature selection
python run_pipeline.py     # Steps 2-6: Load, encode, train, evaluate
```

### 2. **Automated Testing**
```bash
python run_comprehensive_test.py    # Test all combinations
```

### 3. **Analysis & Visualization**
```bash
python generate_figures.py          # Create result plots
python generate_analysis_plots.py   # Parameter efficiency analysis
```

---

## Project Timeline Estimate

```
Feature Selection:        ~5-30 seconds
Data Loading:            <1 second
Encoding Setup:          ~1-2 seconds
QKSVM Training:          ~30 seconds - 10 minutes
VQC Training:            ~5-20 minutes
Evaluation:              ~1-5 minutes
Comprehensive Testing:   ~2-4 hours (72 runs)
```

---

## Data Flow Summary

```
Raw Data (7,129 genes)
    ↓
Feature Selection (→ k genes)
    ↓
Train/Test Split or CV
    ↓
Normalization/Scaling
    ↓
Quantum Encoding (Amplitude or Angle)
    ↓
Classification (QKSVM or VQC)
    ↓
Evaluation (Internal + Independent)
    ↓
Results & Analysis
```

---

## File Dependencies

```
preprocessing.py
    └─ feature-selection-methods/
       ├─ anova_f.py
       ├─ signal_to_noise.py
       └─ scad_svm.py

run_pipeline.py
    ├─ data_loader.py
    ├─ amplitude_encoding.py (if amplitude)
    ├─ angle_encoding.py (if angle)
    ├─ qksvm_golub.py (if QKSVM)
    │  └─ backend_config.py
    └─ vqc_golub.py (if VQC)
       ├─ ansatz.py
       └─ amplitude_encoding.py

generate_figures.py
    └─ results/ (CSV files)

run_comprehensive_test.py
    ├─ preprocessing (internally)
    ├─ qksvm_golub.py
    └─ angle_encoding.py
```

---

## Research Questions Addressed

1. **Quantum Advantage**: Does quantum encoding improve classification?
2. **Encoding Efficiency**: Amplitude (log) vs Angle (linear) scaling?
3. **Feature Selection**: Which method (ANOVA/SNR/SCAD) works best?
4. **Scalability**: How does performance scale with feature count?
5. **Generalization**: Do models generalize to independent data?

---

**Last Updated:** 2025-12-14
**Project:** Leukemia QKSVM - Quantum Machine Learning for Cancer Classification
