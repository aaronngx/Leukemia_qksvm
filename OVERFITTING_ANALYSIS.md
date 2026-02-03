# Overfitting Prevention Analysis

## Executive Summary

**Overall Assessment:** ⚠️ **MIXED** - Has good practices but also significant overfitting risks

**Risk Level:** 🟡 **MODERATE-HIGH** (due to small dataset size and potential data leakage)

---

## ✅ Overfitting Prevention Mechanisms (GOOD PRACTICES)

### 1. **Independent Test Set (CRITICAL)**

```
Training Data:      38 patients (or 22 balanced)
                    ↓ Feature Selection
                    ↓ Model Training
                    ✗ NEVER seen by independent set

Independent Set:    34 patients (COMPLETELY SEPARATE)
                    ↓ ONLY used for final validation
                    ✓ True unbiased performance
```

**Files:** `preprocessing.py:172-178`
```python
print("  Data Strategy (Golub Methodology):")
print("  • Gene selection: Selected patients ONLY (38 or 22)")
print("  • Internal validation: From selected patients")
print("  • Independent test: Completely separate (34 samples)")
```

✓ **Correctly implements Golub et al. (1999) methodology**
✓ **No data leakage to independent set** (when `use_all_data=False`)

---

### 2. **Cross-Validation Options**

**Available Strategies:**
```
├─ 70/30 Train/Test Split (stratified)
├─ 80/20 Train/Test Split (stratified)
├─ 5-Fold Cross-Validation (stratified)
├─ 10-Fold Cross-Validation (stratified)
└─ LOOCV (Leave-One-Out CV) ← NEW
```

**Implementation:** `anova_f.py:187-242`, `signal_to_noise.py:337-397`

✓ **Stratified splitting** - maintains class balance
✓ **LOOCV available** - low bias for small datasets
✓ **Multiple validation options** - can assess robustness

---

### 3. **Regularization Mechanisms**

#### **SVM Regularization (C parameter)**

**Files:** `qksvm_golub.py:295`, `scad_svm.py:122`

```python
# QKSVM
clf = SVC(kernel="precomputed", C=C)  # Default C=1.0

# Controls margin vs misclassification trade-off
# Larger C → Harder margin (more overfitting risk)
# Smaller C → Softer margin (more regularization)
```

✓ **Tunable C parameter** - can adjust regularization strength
⚠️ **Default C=1.0** - not optimized per dataset

#### **SCAD Regularization**

**File:** `scad_svm.py:38-70`

```python
def scad_penalty(beta, lam, a=3.7):
    """
    SCAD penalty with three regions:
    1. |β| ≤ λ: Linear penalty (like L1)
    2. λ < |β| ≤ aλ: Quadratic transition
    3. |β| > aλ: Constant (no further shrinkage)
    """
    # Automatic feature selection via penalty
    # Less aggressive than L1 for large coefficients
```

✓ **Automatic feature selection** - reduces model complexity
✓ **Less aggressive than L1** - preserves important features
✓ **Auto-tuning available** - `auto_tune_lambda=True`

#### **StandardScaler Normalization**

**Files:** `amplitude_encoding.py:147-150`, `scad_svm.py:219-220`

```python
scaler = StandardScaler()  # Zero mean, unit variance
X_scaled = scaler.fit_transform(X_train)
```

✓ **Prevents feature dominance** - all features on same scale
✓ **Fit on training only** - no test set leakage

---

### 4. **Feature Selection Before Splitting**

**Files:** `anova_f.py:282-293`, `signal_to_noise.py:434-449`

```python
# CORRECT: Feature selection on training data only
X_train, y_train = load_training_data()  # 38 patients

# Feature selection uses ONLY training data
top_k_genes = select_features(X_train, y_train, k)

# THEN split for internal validation
X_train_int, X_test_int = split(X_train[top_k_genes])
```

✓ **No test set leakage** - feature selection on training only
✓ **Proper pipeline** - select → split → train

---

## ⚠️ OVERFITTING RISKS (CONCERNS)

### 1. **Small Dataset Size (MAJOR RISK)**

```
Training: 22-38 patients
Features: 4-50 genes
Independent: 34 patients

Risk Level: 🔴 HIGH
```

**Problems:**
- **High variance** - small sample size → unstable estimates
- **Limited generalization** - may not represent population
- **Model complexity** - quantum models can overfit easily

**Evidence:**
```python
# Example: 22 patients with 16 genes
n_samples = 22
n_features = 16
ratio = n_samples / n_features = 1.375  # Very low!

# Rule of thumb: Need n_samples >> n_features
# Typically want ratio > 10
```

⚠️ **Recommendation:** Use LOOCV or 10-fold CV for small datasets

---

### 2. **`use_all_data` Flag (DATA LEAKAGE RISK)**

**Files:** `anova_f.py:255-293`, `signal_to_noise.py:406-449`

```python
def run_feature_selection(
    ...
    use_all_data: bool = False,  # ⚠️ DANGER FLAG
):
    """
    If use_all_data=True:
        - Combines train + independent for feature selection
        - CREATES DATA LEAKAGE
        - Invalid for unbiased validation
    """
    if use_all_data and input_ind is not None:
        # ⚠️ LEAKAGE: Independent set used for feature selection
        X_train = concat([X_train, X_ind])
        y_train = concat([y_train, y_ind])
```

🔴 **CRITICAL WARNING:**
- If `use_all_data=True`, independent set is "seen" during feature selection
- This creates **data leakage** - overly optimistic results
- Final accuracy on independent set is **biased**

**Current Default:** `use_all_data=False` ✓ (Safe)

**In `preprocessing.py`:** ❌ **NOT EXPOSED TO USER**
- User cannot accidentally enable this flag
- Always defaults to `False` in interactive mode
- Only accessible via command-line argument

✓ **Good:** Default is safe
⚠️ **Risk:** Advanced users could enable via CLI

---

### 3. **No Hyperparameter Tuning by Default**

**Files:** `qksvm_golub.py:209-210`

```python
# Fixed hyperparameters (no tuning)
C: float = 1.0  # SVM regularization
```

⚠️ **Concerns:**
- Fixed C=1.0 may not be optimal
- No grid search or Bayesian optimization
- Quantum circuit depth/reps fixed
- Could be underfitting OR overfitting

**Mitigation:** ENSGA optimizer available
```python
# File: ensga_optimizer.py
# Multi-objective optimization of C and gamma
use_ensga=True  # Optimize hyperparameters
```

✓ **Available but not default**
⚠️ **Most users won't use it**

---

### 4. **VQC Model Complexity**

**File:** `vqc_golub.py:52-57`

```python
# VQC with TwoLocal ansatz
ansatz = TwoLocal(
    n_qubits=n_qubits,
    rotation_blocks=['rx', 'rz', 'rx'],  # 3 rotations per qubit
    entanglement='linear',
    reps=reps  # Number of repetitions
)

# Parameter count: n_qubits × 3 × (reps + 1)
# Example: 16 qubits, reps=2
# → 16 × 3 × 3 = 144 trainable parameters

# Training samples: 22
# Parameters: 144
# Ratio: 0.15 (parameters >> samples) ⚠️ OVERFITTING RISK
```

🔴 **HIGH RISK:** More parameters than samples

**Mitigation:**
- Amplitude encoding uses fewer qubits (log scaling)
  - 16 features → 4 qubits → 36 params (better ratio)
- Can reduce `reps` to decrease parameters

---

### 5. **No Early Stopping**

**File:** `vqc_golub.py` (training loop)

```python
# VQC training
optimizer = COBYLA(maxiter=50)

# ⚠️ No validation loss monitoring
# ⚠️ No early stopping
# ⚠️ Always runs full 50 iterations
```

⚠️ **Could overfit** - no stopping criterion based on validation

---

### 6. **Feature Selection Instability**

**Small sample problem:**
```
With 22 patients:
- ANOVA F-test may be unstable
- SNR scores have high variance
- Different train/test splits → different top genes
```

⚠️ **Recommendation:**
- Use ensemble feature selection
- Average rankings across multiple CV folds
- Check feature stability across splits

---

## 📊 Overfitting Risk Assessment by Configuration

| Configuration | Dataset Size | Feature/Sample Ratio | Risk Level | Recommendation |
|---------------|--------------|----------------------|------------|----------------|
| **22 patients, 4 genes, Amplitude** | Very Small | 4/22 = 0.18 | 🟢 LOW | Good choice |
| **22 patients, 16 genes, Amplitude** | Very Small | 16/22 = 0.73 | 🟡 MODERATE | Use LOOCV |
| **22 patients, 50 genes, Amplitude** | Very Small | 50/22 = 2.27 | 🔴 HIGH | Too many features |
| **38 patients, 16 genes, Amplitude** | Small | 16/38 = 0.42 | 🟡 MODERATE | Acceptable |
| **22 patients, 16 genes, Angle (VQC, reps=2)** | Very Small | 144/22 = 6.55 | 🔴 CRITICAL | Severe overfitting risk |
| **38 patients, 16 genes, Angle (VQC, reps=1)** | Small | 96/38 = 2.53 | 🔴 HIGH | Reduce reps or use QKSVM |

---

## 🎯 Best Practices to Avoid Overfitting

### Recommended Configuration

```python
# GOOD CONFIGURATION
{
    "patients": 38,              # Use all available data
    "patient_balance": False,    # Don't reduce samples
    "genes": 16,                 # Reasonable feature count
    "gene_balance": True,        # Balanced selection
    "validation": "loocv",       # Exhaustive validation
    "encoding": "amplitude",     # Fewer qubits (log scaling)
    "classifier": "qksvm",       # No trainable parameters
    "use_all_data": False,       # ✅ NO DATA LEAKAGE
}
```

### Step-by-Step Recommendations

1. **Use Maximum Available Training Data**
   ```bash
   # Choose 38 patients (option 1)
   # Don't reduce to 22 unless class balance critical
   ```

2. **Conservative Feature Selection**
   ```bash
   # Start with k ≤ 16 genes
   # Ensure n_samples / k > 2
   ```

3. **Use LOOCV for Small Datasets**
   ```bash
   # Select option 2c: LOOCV
   # Provides low-bias estimates
   ```

4. **Prefer QKSVM over VQC**
   ```bash
   # QKSVM: No trainable parameters (just kernel)
   # VQC: Many trainable parameters (high overfitting risk)
   ```

5. **Use Amplitude Encoding**
   ```bash
   # 16 genes → 4 qubits (log scaling)
   # vs 16 genes → 16 qubits (linear scaling)
   ```

6. **Never Use `use_all_data=True`**
   ```bash
   # ALWAYS keep independent set separate
   # Only use for final validation
   ```

7. **Report Multiple Metrics**
   ```bash
   # Internal CV accuracy (potentially optimistic)
   # Independent set accuracy (TRUE performance)
   # If gap is large → overfitting
   ```

---

## ✅ Validation Checklist

Before trusting results, verify:

- [ ] `use_all_data=False` in preprocessing
- [ ] Feature selection used ONLY training data
- [ ] Independent set NEVER used during:
  - [ ] Feature selection
  - [ ] Hyperparameter tuning
  - [ ] Model training
- [ ] Sample size reasonable: n_samples/n_features > 2
- [ ] Cross-validation used (not just single split)
- [ ] Results reported on independent set
- [ ] Performance gap checked: |CV_acc - Ind_acc| < 10%

---

## 📈 Detecting Overfitting

### Warning Signs

```python
# Example results:
Internal CV Accuracy:    95%  ✓
Independent Accuracy:    65%  ⚠️

Gap = 95% - 65% = 30%  🔴 SEVERE OVERFITTING
```

**Healthy Gap:** < 10%
**Moderate Gap:** 10-20%
**Severe Gap:** > 20%

### Common Causes

1. **Data leakage** - `use_all_data=True` was used
2. **Too many features** - k > n_samples/2
3. **Model too complex** - VQC with high reps
4. **Unstable feature selection** - different genes each fold
5. **Lucky train/test split** - use CV to verify

---

## 🔬 Comparison with Original Golub Study

| Aspect | Golub et al. (1999) | This Project | Assessment |
|--------|---------------------|--------------|------------|
| Training Size | 38 patients | 22-38 patients | ✓ Same |
| Test Size | 34 patients (independent) | 34 patients | ✓ Same |
| Feature Selection | P-score on train only | ANOVA/SNR on train only | ✓ Correct |
| Validation | Independent set | Independent set + CV | ✓ Better |
| Data Leakage Prevention | Yes | Yes (if use_all_data=False) | ✓ Good |
| Model Complexity | Weighted voting (simple) | Quantum (complex) | ⚠️ Higher risk |

---

## 📝 Summary

### ✅ What Works Well

1. **Independent test set** properly isolated
2. **Multiple CV options** including LOOCV
3. **Regularization available** (SVM C, SCAD)
4. **Stratified splits** maintain class balance
5. **Default safe** - `use_all_data=False`

### ⚠️ Areas of Concern

1. **Small dataset** (22-38 samples)
2. **High-dimensional risk** (k up to 50)
3. **VQC complexity** (many trainable parameters)
4. **No hyperparameter tuning** by default
5. **Potential data leakage** if `use_all_data=True` used

### 🎯 Final Recommendation

**The project CAN avoid overfitting IF:**
- ✅ Use 38 patients (not 22)
- ✅ Keep k ≤ 16 genes
- ✅ Use LOOCV or 10-fold CV
- ✅ Prefer QKSVM over VQC
- ✅ Use amplitude encoding
- ✅ NEVER enable `use_all_data=True`
- ✅ Report independent set results
- ✅ Check internal vs independent gap

**Risk Level with Recommended Settings:** 🟢 **LOW-MODERATE**

**Risk Level with Poor Settings:** 🔴 **HIGH**

---

**Last Updated:** 2025-12-14
**Project:** Leukemia QKSVM Overfitting Analysis
