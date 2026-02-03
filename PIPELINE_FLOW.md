# Complete Pipeline Flow: Preprocessing → Encoding → VQC/QKSVM

## Overview Diagram

```
Raw Data (7129 genes × 72 patients)
    ↓
[STEP 1: PREPROCESSING - Feature Selection]
    ↓
Selected Features (k genes × patients)
    ↓
[STEP 2: DATA LOADING]
    ↓
Train/Test/Independent Sets
    ↓
[STEP 3: ENCODING - Quantum Feature Map]
    ↓
Quantum States (n qubits)
    ↓
[STEP 4A: VQC Path]          [STEP 4B: QKSVM Path]
    ↓                              ↓
Variational Ansatz           Quantum Kernel Matrix
    ↓                              ↓
Gradient Optimization        Classical SVM
    ↓                              ↓
Trained VQC Model            Trained QKSVM Model
    ↓                              ↓
[STEP 5: EVALUATION]
    ↓
Independent Test Set Results
```

---

## STEP 1: PREPROCESSING (Feature Selection)

### 1.1 Input
**File:** `data/raw/data_set_ALL_AML_train.csv`
- **Shape:** 7129 genes × 38 patients
- **Labels:** 27 ALL, 11 AML
- **Format:** CSV with gene expression values

### 1.2 User Interaction
**File:** `preprocessing.py`

```python
# User answers 4 questions:
[1/4] How many genes? (e.g., 16, 24, 32, 50)
      → User input: k = 16
      → Balance genes? (y/n) → User: yes

[2/4] Feature selection method?
      → 1. ANOVA F-test
      → 2. SNR (Signal-to-Noise Ratio)
      → 3. Both
      → User: 1 (ANOVA)

[3/4] Patient samples?
      → 1. All 38 patients (27 ALL + 11 AML)
      → 2. Balanced 22 patients (11 ALL + 11 AML)
      → User: 2 (balanced)

[4/4] Validation strategy?
      → 1a. 70/30 train/test split
      → 1b. 80/20 train/test split
      → 2a. 5-fold CV
      → 2b. 10-fold CV
      → User: 1a (70/30 split)
```

### 1.3 Feature Selection Process
**File:** `feature-selection-methods/anova_f.py`

```python
# Step 1: Load raw data
df_train = pd.read_csv("data/raw/data_set_ALL_AML_train.csv")
# Shape: (7129 genes, 38 patients)

# Step 2: Balance patient samples (if selected)
if use_balanced_patients:
    # Select 11 ALL + 11 AML = 22 patients
    X, y = balance_patient_samples(X_train_full, y_train_full)
    # Shape: (7129 genes, 22 patients)

# Step 3: Calculate F-scores for ALL genes
f_scores = []
for gene_idx in range(7129):
    gene_values = X[gene_idx, :]
    all_values = gene_values[y == 0]  # ALL samples
    aml_values = gene_values[y == 1]  # AML samples

    # ANOVA F-statistic
    f_score = f_oneway(all_values, aml_values)
    f_scores.append(f_score)

# Step 4: Rank genes by F-score
ranked_genes = np.argsort(f_scores)[::-1]  # Descending order

# Step 5: Select top k genes (with optional balancing)
if balanced_genes:
    # Select k/2 ALL-favoring + k/2 AML-favoring
    all_favoring = [g for g in ranked_genes if mean(ALL) > mean(AML)][:k//2]
    aml_favoring = [g for g in ranked_genes if mean(AML) > mean(ALL)][:k//2]
    selected_genes = all_favoring + aml_favoring  # k genes total
else:
    # Pure top-k by F-score
    selected_genes = ranked_genes[:k]  # k genes

# Step 6: Extract selected gene data
X_selected = X[selected_genes, :]  # (k genes, 22 patients)
```

### 1.4 Internal Validation Split
**File:** `anova_f.py` (lines 450-500)

```python
# Split 22 patients into train/test (stratified)
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    range(22),
    test_size=0.3,  # 70/30 split
    stratify=y,     # Maintain class balance
    random_state=42
)

# Create train/test sets
X_train = X_selected[:, train_idx]  # (16 genes, ~15 patients)
y_train = y[train_idx]

X_test = X_selected[:, test_idx]    # (16 genes, ~7 patients)
y_test = y[test_idx]
```

### 1.5 Output Files
**Directory:** `results/`

```
results/
├── train_internal_top_16_anova_f.csv
│   → Shape: (15 patients, 18 columns)
│   → Columns: [cancer, patient, gene_1, gene_2, ..., gene_16]
│
├── test_internal_top_16_anova_f.csv
│   → Shape: (7 patients, 18 columns)
│   → Columns: [cancer, patient, gene_1, gene_2, ..., gene_16]
│
└── independent_top_16_anova_f.csv
    → Shape: (34 patients, 18 columns)
    → Columns: [cancer, patient, gene_1, gene_2, ..., gene_16]
    → NOTE: Independent set uses SAME 16 genes selected from training
```

**Data Format Example:**
```csv
cancer,patient,gene_1,gene_2,gene_3,...,gene_16
ALL,1,234.5,678.9,123.4,...,567.8
ALL,2,345.6,789.0,234.5,...,678.9
AML,3,456.7,890.1,345.6,...,789.0
...
```

---

## STEP 2: DATA LOADING

### 2.1 Load Preprocessed Data
**File:** `data_loader.py`

```python
from data_loader import load_preprocessed_data

# Load training set
X_train, y_train = load_preprocessed_data(
    "results/train_internal_top_16_anova_f.csv"
)

# What happens inside:
# 1. Read CSV
df = pd.read_csv(file_path)

# 2. Detect label column ('cancer' or 'label')
label_col = 'cancer'  # Auto-detected

# 3. Convert labels: ALL → 0, AML → 1
y = df['cancer'].map({'ALL': 0, 'AML': 1}).values

# 4. Extract feature columns (exclude 'cancer', 'patient')
feature_cols = [c for c in df.columns if c not in ['cancer', 'patient']]
X = df[feature_cols].values

# 5. Return numpy arrays
return X, y
```

**Output:**
```python
X_train.shape  # (15, 16) - 15 patients × 16 genes
y_train.shape  # (15,)    - 15 labels (0 or 1)

X_test.shape   # (7, 16)  - 7 patients × 16 genes
y_test.shape   # (7,)     - 7 labels

X_ind.shape    # (34, 16) - 34 patients × 16 genes
y_ind.shape    # (34,)    - 34 labels
```

---

## STEP 3: ENCODING (Quantum Feature Map)

This is where classical data gets mapped to quantum states.

### Path A: "Amplitude" Encoding (Actually Angle Encoding)

**File:** `amplitude_encoding.py`

#### 3A.1 Preprocessing
```python
from amplitude_encoding import preprocess_and_normalize

X_train_norm, scaler = preprocess_and_normalize(X_train)

# Inside the function:
# Step 1: Standardize (z-score normalization)
scaler = StandardScaler()
X_std = scaler.fit_transform(X_train)
# Each feature: (x - mean) / std

# Step 2: Clip to [-1, 1]
X_clipped = np.clip(X_std / 1.0, -1, 1)

# Step 3: L2 normalize each sample (row)
X_norm = X_clipped / np.linalg.norm(X_clipped, axis=1, keepdims=True)

# Result:
X_train_norm.shape  # (15, 16)
# Each row is a unit vector: ||x|| = 1
```

#### 3A.2 Build Feature Map Circuit
```python
from amplitude_encoding import amplitude_encoding_feature_map

circuit, x_params, n_qubits = amplitude_encoding_feature_map(num_features=16)

# Inside:
n_qubits = ceil(log2(16)) = 4  # Logarithmic scaling

# Create parameter vector
x = ParameterVector("x", 16)  # 16 parameters (one per gene)

# Build circuit
qc = QuantumCircuit(4)

# Apply RY rotations cyclically
for i in range(16):
    qubit_idx = i % 4  # Cycles: 0, 1, 2, 3, 0, 1, 2, 3, ...
    qc.ry(x[i], qubit_idx)

# Result: 4-qubit circuit with 16 RY gates
# q0: RY(x[0]) → RY(x[4]) → RY(x[8]) → RY(x[12])
# q1: RY(x[1]) → RY(x[5]) → RY(x[9]) → RY(x[13])
# q2: RY(x[2]) → RY(x[6]) → RY(x[10]) → RY(x[14])
# q3: RY(x[3]) → RY(x[7]) → RY(x[11]) → RY(x[15])
```

**What this produces:**
```python
circuit.num_qubits   # 4
len(x_params)        # 16
circuit.depth()      # 4 (4 layers of RY gates)
```

#### 3A.3 Bind Data to Circuit
```python
# For a single sample (one patient)
sample = X_train_norm[0]  # Shape: (16,) - 16 gene expression values

# Create parameter binding dictionary
param_dict = {x_params[i]: sample[i] for i in range(16)}
# {x[0]: 0.123, x[1]: -0.456, ..., x[15]: 0.789}

# Bind parameters to circuit
bound_circuit = circuit.assign_parameters(param_dict)

# Now the circuit has concrete rotation angles:
# q0: RY(0.123) → RY(0.234) → RY(0.345) → RY(0.456)
# q1: RY(-0.456) → RY(-0.345) → RY(-0.234) → RY(-0.123)
# ...
```

**Quantum State After Encoding:**
```python
# Execute the circuit to get quantum state
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')
result = backend.run(bound_circuit).result()
statevector = result.get_statevector()

# State is a complex vector
statevector.shape  # (16,) - 2^4 = 16 complex amplitudes
# |ψ⟩ = α₀|0000⟩ + α₁|0001⟩ + ... + α₁₅|1111⟩
# where |αᵢ|² represents probability
```

### Path B: Angle Encoding

**File:** `angle_encoding.py`

```python
from angle_encoding import angle_encoding_circuit

circuit, x_params = angle_encoding_circuit(n_qubits=16)

# Build circuit
qc = QuantumCircuit(16)  # Linear scaling: 16 features → 16 qubits
x = ParameterVector("x", 16)

# Apply one RY per qubit
for i in range(16):
    qc.ry(x[i], i)

# Result:
# q0:  RY(x[0])
# q1:  RY(x[1])
# q2:  RY(x[2])
# ...
# q15: RY(x[15])
```

**Preprocessing for angle encoding:**
```python
from qksvm_golub import scale_to_angle

X_scaled, scaler = scale_to_angle(X_train)

# Inside:
scaler = MinMaxScaler(feature_range=(0, π))
X_scaled = scaler.fit_transform(X_train)
# Each feature: x_scaled = (x - min) / (max - min) * π

# Result:
X_scaled.shape  # (15, 16)
# All values in range [0, π]
```

---

## STEP 4A: VQC (Variational Quantum Classifier)

**File:** `amplitude_encoding.py` → `build_amplitude_vqc()`

### 4A.1 Build Complete VQC Circuit

```python
circuit, x_params, theta_params, n_qubits = build_amplitude_vqc(
    num_features=16,
    reps=2
)

# Circuit composition:
# 1. Feature map (from Step 3A.2)
# 2. Variational ansatz (TwoLocal)

# Inside:
# Step 1: Create feature map
feature_map, x_params, n_qubits = amplitude_encoding_feature_map(16)

# Step 2: Create variational ansatz
from qiskit.circuit.library import TwoLocal

ansatz = TwoLocal(
    num_qubits=4,
    rotation_blocks='ry',     # RY rotations
    entanglement_blocks='cx', # CNOT gates
    entanglement='linear',    # q0-q1, q1-q2, q2-q3
    reps=2                    # 2 repetitions
)

# Step 3: Compose circuits
circuit = feature_map.compose(ansatz)

# Final circuit structure:
# ┌─────────────────────────────────┐  ┌──────────────────────────────┐
# │   Feature Map (data upload)      │  │  Variational Ansatz (trainable) │
# │                                  │  │                                 │
# │ q0: RY(x[0])─RY(x[4])─...        │  │ RY(θ₀)─■─RY(θ₄)─■─...           │
# │ q1: RY(x[1])─RY(x[5])─...        │  │ RY(θ₁)─╳─RY(θ₅)─╳─...           │
# │ q2: RY(x[2])─RY(x[6])─...        │  │ RY(θ₂)─■─RY(θ₆)─■─...           │
# │ q3: RY(x[3])─RY(x[7])─...        │  │ RY(θ₃)─╳─RY(θ₇)─╳─...           │
# └─────────────────────────────────┘  └──────────────────────────────┘
#          16 input params                  20 trainable params
```

**Parameters:**
```python
len(x_params)      # 16 - input parameters (gene expression values)
len(theta_params)  # 20 - trainable parameters (optimized during training)
```

### 4A.2 Create Quantum Neural Network

```python
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Parity measurement (sum of all qubits mod 2)
def parity(x):
    return "{:b}".format(x).count("1") % 2

qnn = EstimatorQNN(
    circuit=circuit,
    input_params=x_params,      # 16 params bound to data
    weight_params=theta_params, # 20 params to optimize
    interpret=parity            # Observable: parity of measurement
)
```

### 4A.3 Create Classifier

```python
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import COBYLA

classifier = NeuralNetworkClassifier(
    neural_network=qnn,
    optimizer=COBYLA(maxiter=50),  # Classical optimizer
    loss='squared_error'
)
```

### 4A.4 Training Process

```python
# Train the VQC
classifier.fit(X_train_norm, y_train)

# What happens during training:
for iteration in range(50):
    # 1. For each training sample
    for i, (x_sample, y_label) in enumerate(zip(X_train_norm, y_train)):

        # 2. Bind input data to circuit
        param_dict = {x_params[j]: x_sample[j] for j in range(16)}
        param_dict.update({theta_params[j]: current_theta[j] for j in range(20)})

        # 3. Execute quantum circuit
        bound_circuit = circuit.assign_parameters(param_dict)
        result = estimator.run(bound_circuit).result()
        expectation_value = result.values[0]  # Parity expectation

        # 4. Compute loss
        prediction = 1 if expectation_value > 0 else 0
        loss = (prediction - y_label) ** 2

    # 5. Update theta parameters using classical optimizer
    theta_new = optimizer.step(loss, theta_old)

    # 6. Repeat until convergence
```

### 4A.5 Prediction

```python
# Predict on test set
y_pred = classifier.predict(X_test_norm)

# For each test sample:
# 1. Bind test data to x_params
# 2. Use learned theta_params
# 3. Execute circuit
# 4. Measure parity expectation
# 5. Classify: 0 (ALL) or 1 (AML)
```

---

## STEP 4B: QKSVM (Quantum Kernel SVM)

**File:** `qksvm_golub.py`

### 4B.1 Scale Data

```python
X_train_scaled, scaler = scale_to_angle(X_train)
X_test_scaled, _ = scaler.transform(X_test)

# Each feature scaled to [0, π] for angle encoding
```

### 4B.2 Build Feature Map

```python
from angle_encoding import angle_encoding_circuit

feature_map, x_params = angle_encoding_circuit(n_qubits=16)

# 16 features → 16 qubits → 16 RY gates
```

### 4B.3 Compute Quantum Kernel Matrix

This is the key step that makes QKSVM quantum!

```python
def build_kernel(XA, XB, feature_map, x_params):
    """
    Compute kernel matrix K where K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|²
    """
    n_samples_A = len(XA)
    n_samples_B = len(XB)
    K = np.zeros((n_samples_A, n_samples_B))

    for i in range(n_samples_A):
        for j in range(n_samples_B):
            # Compute kernel element K[i,j]
            K[i, j] = compute_kernel_element(
                XA[i], XB[j], feature_map, x_params
            )

    return K

def compute_kernel_element(x, z, feature_map, x_params):
    """
    Compute k(x,z) = |⟨φ(x)|φ(z)⟩|²

    Using the quantum trick:
    |⟨φ(x)|φ(z)⟩|² = |⟨0|U†(x)U(z)|0⟩|²
    """
    # Build circuit: U(x)† ⊗ U(z)
    qc = QuantumCircuit(16)

    # Apply U(z) - encode z
    param_dict_z = {x_params[k]: z[k] for k in range(16)}
    qc.assign_parameters(param_dict_z, inplace=True)

    # Apply U†(x) - encode x and take adjoint
    param_dict_x = {x_params[k]: -x[k] for k in range(16)}  # Negative for adjoint
    qc.assign_parameters(param_dict_x, inplace=True)

    # Execute circuit and measure overlap
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(qc).result()
    statevector = result.get_statevector()

    # Inner product with |0⟩ state
    overlap = abs(statevector[0]) ** 2

    return overlap
```

**Kernel Matrix Computation:**
```python
# Compute training kernel K_train
K_train = build_kernel(X_train_scaled, X_train_scaled, feature_map, x_params)
# K_train.shape = (15, 15)
# K_train[i,j] = quantum similarity between patient i and patient j

# Compute test kernel K_test
K_test = build_kernel(X_test_scaled, X_train_scaled, feature_map, x_params)
# K_test.shape = (7, 15)
# K_test[i,j] = quantum similarity between test patient i and train patient j
```

**Example Kernel Values:**
```
K_train:
        Patient 0  Patient 1  Patient 2  ...
Patient 0   1.000      0.823      0.654  ...
Patient 1   0.823      1.000      0.712  ...
Patient 2   0.654      0.712      1.000  ...
...

Diagonal = 1.0 (patient is identical to itself)
Off-diagonal = similarity (higher = more similar quantum states)
```

### 4B.4 Train Classical SVM on Quantum Kernel

```python
from sklearn.svm import SVC

# Create SVM with precomputed quantum kernel
svm = SVC(kernel='precomputed')

# Train on quantum kernel matrix
svm.fit(K_train, y_train)

# The SVM finds decision boundary in quantum feature space!
# α coefficients found by solving:
# min (1/2)α^T K α - α^T 1
# subject to: 0 ≤ α ≤ C, α^T y = 0
```

### 4B.5 Prediction

```python
# Predict on test set
y_pred = svm.predict(K_test)

# For each test sample:
# 1. Compute quantum kernel with all training samples (K_test row)
# 2. Use SVM decision function with learned α coefficients
# 3. decision(x) = Σᵢ αᵢ yᵢ k(xᵢ, x) + b
# 4. Classify based on sign of decision function
```

---

## STEP 5: EVALUATION

### 5.1 Evaluate on Test Set

```python
from sklearn.metrics import accuracy_score, classification_report

# Compute accuracy
test_acc = accuracy_score(y_test, y_pred)
# e.g., 0.857 (6/7 correct)

# Classification report
print(classification_report(y_test, y_pred, target_names=['ALL', 'AML']))
```

**Output:**
```
              precision    recall  f1-score   support

         ALL       0.80      1.00      0.89         4
         AML       1.00      0.67      0.80         3

    accuracy                           0.86         7
   macro avg       0.90      0.83      0.84         7
weighted avg       0.89      0.86      0.85         7
```

### 5.2 Evaluate on Independent Set

**CRITICAL:** Independent set was NEVER seen during:
- Feature selection
- Training
- Hyperparameter tuning

```python
# Load independent set
X_ind, y_ind = load_preprocessed_data(
    "results/independent_top_16_anova_f.csv"
)

# Preprocess (using TRAINING scaler!)
X_ind_norm = scaler.transform(X_ind)  # ← Use train scaler, don't fit new one

# For QKSVM:
K_ind = build_kernel(X_ind_scaled, X_train_scaled, feature_map, x_params)
y_ind_pred = svm.predict(K_ind)

# For VQC:
y_ind_pred = classifier.predict(X_ind_norm)

# Evaluate
ind_acc = accuracy_score(y_ind, y_ind_pred)
print(f"Independent Set Accuracy: {ind_acc:.4f}")
```

**This is the TRUE performance metric** - completely unbiased!

---

## COMPLETE DATA FLOW SUMMARY

### Data Shapes Through Pipeline

```
PREPROCESSING:
├─ Raw data:                  (7129 genes, 38 patients)
├─ Balanced patients:         (7129 genes, 22 patients)
├─ Selected features:         (16 genes, 22 patients)
├─ Train split:               (16 genes, 15 patients)
├─ Test split:                (16 genes, 7 patients)
└─ Independent:               (16 genes, 34 patients)

DATA LOADING:
├─ X_train:                   (15, 16) - 15 samples × 16 features
├─ y_train:                   (15,)    - 15 labels
├─ X_test:                    (7, 16)
├─ y_test:                    (7,)
├─ X_ind:                     (34, 16)
└─ y_ind:                     (34,)

ENCODING:
├─ Amplitude (angle):         4 qubits, 16 params, 20 trainable params
│   ├─ Feature map:           QuantumCircuit(4 qubits, 16 RY gates)
│   └─ Quantum state:         (16,) complex amplitudes
│
└─ Angle:                     16 qubits, 16 params
    ├─ Feature map:           QuantumCircuit(16 qubits, 16 RY gates)
    └─ Quantum state:         (65536,) complex amplitudes

QKSVM:
├─ K_train:                   (15, 15) - quantum kernel matrix
├─ K_test:                    (7, 15)
├─ SVM α coefficients:        (15,) - learned weights
└─ Predictions:               (7,) or (34,) - binary labels

VQC:
├─ Theta parameters:          (20,) - learned rotation angles
├─ Training iterations:       50
└─ Predictions:               (7,) or (34,) - binary labels
```

---

## KEY INSIGHTS

### 1. Feature Selection BEFORE Encoding
- Feature selection reduces 7129 genes → 16 genes
- This happens BEFORE quantum encoding
- Only selected genes are encoded into quantum states

### 2. Quantum Advantage
- **QKSVM:** Quantum kernel captures complex feature relationships
- **VQC:** Variational circuit learns optimal parameters

### 3. Data Flow is One-Way
```
Preprocessing → Encoding → Training → Evaluation
     ↓              ↓           ↓          ↓
  Classical    Quantum     Classical   Classical
   (Python)   (Qiskit)     (Sklearn)   (Metrics)
```

### 4. Critical: No Data Leakage
- Training scaler fit ONLY on training data
- Same scaler applied to test and independent sets
- Independent set completely unseen during preprocessing

---

## QUANTUM vs CLASSICAL COMPARISON

| Component | Classical | Quantum (Amplitude) | Quantum (Angle) |
|-----------|-----------|---------------------|-----------------|
| Feature selection | Classical statistics | Classical statistics | Classical statistics |
| Encoding | None | 4-qubit circuit | 16-qubit circuit |
| Feature map | Identity | RY gates (cyclic) | RY gates (1:1) |
| Kernel | RBF/Linear | Quantum overlap | Quantum overlap |
| Optimization | Gradient descent | Hybrid quantum-classical | Classical SVM |
| Prediction | Direct computation | Quantum measurement | Quantum measurement |

---

## FILES INVOLVED AT EACH STEP

```
STEP 1 (Preprocessing):
├─ preprocessing.py (main)
├─ feature-selection-methods/anova_f.py
└─ feature-selection-methods/signal_to_noise.py

STEP 2 (Data Loading):
└─ data_loader.py

STEP 3 (Encoding):
├─ amplitude_encoding.py
├─ angle_encoding.py
└─ backend_config.py

STEP 4A (VQC):
├─ amplitude_encoding.py (train_amplitude_vqc)
└─ vqc_golub.py

STEP 4B (QKSVM):
└─ qksvm_golub.py

STEP 5 (Evaluation):
└─ sklearn.metrics (accuracy_score, classification_report)
```

---

## ORCHESTRATION: run_pipeline.py

The `run_pipeline.py` script connects all these steps:

```python
# 1. Run preprocessing (or load existing)
run_preprocessing_step()

# 2. Select data file
data_file = select_from_available_files()

# 3. Load data
X, y, feature_cols, patient_ids = load_selected_data(data_file)

# 4. Select encoding
encoding_choice = select_encoding_method()  # Amplitude or Angle

# 5. Select classifier
classifier_choice = select_classifier()  # VQC or QKSVM

# 6. Train and evaluate
if classifier_choice == VQC:
    train_amplitude_vqc(X, y, num_features, reps=2)
elif classifier_choice == QKSVM:
    train_eval_qksvm(X, y, encoding_type, kernel_method)

# 7. Report results
print_results()
```

This completes the full pipeline from raw gene expression data to quantum-enhanced classification!
