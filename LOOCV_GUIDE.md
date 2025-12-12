# Leave-One-Out Cross-Validation (LOOCV) Guide

## Overview

LOOCV (Leave-One-Out Cross-Validation) has been added to the Leukemia QKSVM preprocessing pipeline as an additional validation strategy option.

## What is LOOCV?

Leave-One-Out Cross-Validation is an exhaustive cross-validation method where:
- The model is trained on **n-1** samples
- Testing is performed on the **1 remaining sample**
- This process is repeated **n times** (once for each data point)
- Each sample serves as the test set exactly once

## Characteristics

### Advantages
✓ **Low Bias**: Uses maximum training data (n-1 samples)
✓ **Deterministic**: No randomness in splits
✓ **Exhaustive**: Every sample tested independently
✓ **Ideal for Small Datasets**: Perfect for Golub dataset (22-38 samples)

### Disadvantages
✗ **High Variance**: Testing on single points can be unstable
✗ **Computationally Expensive**: Requires n training iterations
✗ **Outlier Sensitivity**: Single outliers can skew results

## When to Use LOOCV

**Recommended for:**
- Small datasets (n < 50)
- When you need unbiased performance estimates
- Research requiring reproducible splits
- Comparing with Golub et al. (1999) methodology

**Not recommended for:**
- Large datasets (training time becomes prohibitive)
- When computational resources are limited
- Datasets with many outliers

## How to Use LOOCV in This Project

### Step 1: Run Preprocessing

```bash
python preprocessing.py
```

### Step 2: Select Validation Strategy

When prompted for validation strategy:

```
[4/4] Internal validation strategy:
      1. Train/test split
         a. 70/30 split
         b. 80/20 split
      2. Cross-validation
         a. 5-fold CV
         b. 10-fold CV
         c. LOOCV (Leave-One-Out CV)  ← Choose this

      -> (1a/1b/2a/2b/2c): 2c
```

### Step 3: Generated Output Files

LOOCV will generate **n fold pairs** (where n = number of samples):

**For balanced dataset (22 patients):**
```
results/
├── fold_1_train_top_k_anova_f.csv    (21 samples)
├── fold_1_test_top_k_anova_f.csv     (1 sample)
├── fold_2_train_top_k_anova_f.csv    (21 samples)
├── fold_2_test_top_k_anova_f.csv     (1 sample)
...
├── fold_22_train_top_k_anova_f.csv   (21 samples)
└── fold_22_test_top_k_anova_f.csv    (1 sample)
```

**For all patients (38 patients):**
- Generates 38 fold pairs
- Each training set contains 37 samples
- Each test set contains 1 sample

## Example Use Case

### Scenario: Feature Selection with LOOCV

1. **Configuration:**
   - Genes: 16
   - Gene balance: No
   - Method: ANOVA F-test
   - Patients: Balanced 22 (11 ALL + 11 AML)
   - Validation: **LOOCV**

2. **Process:**
   - Selects top 16 genes from 22 patients
   - Generates 22 train/test fold pairs
   - Each fold trains on 21 patients, tests on 1

3. **Training Workflow:**
   ```python
   accuracies = []
   for fold in range(1, 23):
       train_data = load(f"fold_{fold}_train_top_16_anova_f.csv")
       test_data = load(f"fold_{fold}_test_top_16_anova_f.csv")

       model = train_qksvm(train_data)
       accuracy = evaluate(model, test_data)
       accuracies.append(accuracy)

   # Average performance across all folds
   mean_accuracy = np.mean(accuracies)
   std_accuracy = np.std(accuracies)
   ```

4. **Final Validation:**
   - Train on all 22 patients
   - Test on independent set (34 patients)
   - Report accuracy on independent set as final result

## Implementation Details

### scikit-learn Implementation

Both `anova_f.py` and `signal_to_noise.py` use:

```python
from sklearn.model_selection import LeaveOneOut

def split_loocv(X, y, patient_ids):
    """Split data into LOOCV sets."""
    loo = LeaveOneOut()
    folds = []

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        # ... (rest of split logic)
        folds.append((X_train, y_train, ..., X_test, y_test, ...))

    return folds
```

### Files Modified

1. **preprocessing.py**
   - Added LOOCV option (2c) to validation strategy menu
   - Updated configuration summary
   - Added LOOCV-specific next steps

2. **feature-selection-methods/anova_f.py**
   - Added `split_loocv()` function
   - Updated `run_feature_selection()` to handle LOOCV

3. **feature-selection-methods/signal_to_noise.py**
   - Added `split_loocv()` function
   - Updated `run_snr_selection()` to handle LOOCV

## Computational Considerations

### Time Complexity

For a dataset with **n samples**:

| Validation Method | Training Iterations | Time Complexity |
|-------------------|---------------------|-----------------|
| 70/30 Split       | 1                   | O(1)            |
| 5-fold CV         | 5                   | O(5)            |
| 10-fold CV        | 10                  | O(10)           |
| **LOOCV**         | **n**               | **O(n)**        |

### Golub Dataset Examples

| Patient Set | LOOCV Iterations | Expected Time* |
|-------------|------------------|----------------|
| Balanced (22) | 22 folds       | ~15-30 min     |
| All (38)      | 38 folds       | ~25-50 min     |

*Time estimates for QKSVM with 16 genes using statevector backend

### Memory Usage

- **LOOCV folds**: Each fold is stored as separate CSV
- **22 patients × 16 genes**: ~44 files (~10 KB each)
- **38 patients × 16 genes**: ~76 files (~15 KB each)
- **Total storage**: ~1-2 MB per feature selection run

## Comparison with Other Methods

### LOOCV vs K-Fold CV

| Aspect | LOOCV | 5-Fold CV | 10-Fold CV |
|--------|-------|-----------|------------|
| Bias | Lowest | Medium | Low |
| Variance | Highest | Low | Medium |
| Training Size | n-1 | 0.8n | 0.9n |
| Test Size | 1 | 0.2n | 0.1n |
| Iterations | n | 5 | 10 |
| Randomness | None | Yes | Yes |
| Reproducibility | Perfect | Seeded | Seeded |

### When to Choose Each

- **LOOCV (2c)**: Maximum data usage, perfect reproducibility, small n
- **10-fold CV (2b)**: Good balance, moderate computational cost
- **5-fold CV (2a)**: Fast, lower variance than LOOCV
- **70/30 Split (1a)**: Fastest, single evaluation, hyperparameter tuning

## References

1. **Golub et al. (1999)** - "Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring"
   - Original leukemia classification study
   - Used weighted voting with P-scores

2. **Hastie, Tibshirani, Friedman (2009)** - "The Elements of Statistical Learning"
   - Section 7.10: Cross-Validation
   - Discusses LOOCV bias-variance tradeoff

3. **scikit-learn Documentation** - LeaveOneOut
   - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html

## Notes

- LOOCV is **not stratified** (cannot stratify a single sample)
- For class imbalance, consider using all 38 patients instead of balanced 22
- LOOCV results can be averaged for robust performance estimation
- Always validate final model on independent set (34 samples)

---

**Last Updated:** 2025-12-11
**Version:** 1.0
**Feature Added to:** Leukemia QKSVM Preprocessing Pipeline
