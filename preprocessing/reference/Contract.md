# Contract — Conventions, Constraints, and Interfaces (English)

This document is the “rules of the road.” If any code violates these, results are not valid.

---

## 1) Non-Negotiables (Hard Constraints)

### 1.1 Training-only / Anti-leakage
**All fitting must use training data only.** Specifically:
- Feature ranking/selection must depend only on `(X_train, y_train)` for that fold
- Scaler must be `fit` on training fold, then applied to validation fold
- Classifier must be `fit` on training fold only
- Hyperparameter selection (e.g., SCAD lambda) must be done using **nested CV inside the training fold** only

✅ Allowed: `selector.select_features(X_train, y_train, k)`  
❌ Not allowed: any method reading `X_val` or `y_val` to choose features/params

**Naming guarantee:** `data_set_ALL_AML_train` is the raw dataset used to generate folds; *no other dataset may be used to fit selectors/scalers/models*.

---

## 2) Dataset & Label Contract

### 2.1 Input files
- `data/raw/data_set_ALL_AML_train.csv` → features matrix X
- `data/raw/actual.csv` → labels y

### 2.2 In-memory objects
- `X: pd.DataFrame` shape (n_samples, n_genes), column names are `gene_id`
- `y: pd.Series` length n_samples, binary labels representing `{ALL, AML}`

### 2.3 Label encoding
- Use a deterministic mapping, e.g.:
  - `ALL -> 1` (positive class)
  - `AML -> 0` (negative class)
- This mapping must be used consistently for:
  - precision/recall/F1 (binary)
  - AUROC (requires a score with higher = more ALL)

---

## 3) FeatureSelector Interface Contract

### 3.1 Required interface
Every selector must implement:
- `METHOD_NAME: str` (unique, lower_snake_case recommended)
- `calculate_scores(X_train, y_train) -> pd.Series`
  - Index MUST exactly match `X_train.columns`
  - Higher magnitude means more important
- `select_features(X_train, y_train, k) -> list[str]`
  - Returns exactly `k` unique feature names
  - Uses ONLY training data

### 3.2 Score semantics
- Base class will select by **absolute score** unless a method overrides selection:
  - Default: rank descending by `abs(score)`
- Missing/NaN scores must be handled (drop or set to 0) deterministically.

---

## 4) Method Definitions (Implementation Contract)

### 4.1 Signal-to-Noise Ratio (SNR)
Compute on training fold only:
- `snr(g) = (mu_ALL(g) - mu_AML(g)) / (sigma_ALL(g) + sigma_AML(g) + eps)`
- Use `abs(snr)` for ranking
n stability: `eps = 1e-9`

### 4.2 ANOVA F-test
Compute on training fold only:
- Use sklearn `f_classif(X_train, y_train)` or your adapted implementation
- Score = F statistic (non-negative)

### 4.3 Mutual Information
Compute on training fold only:
- `mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED)`
- Score = MI (>= 0)

### 4.4 T-test
Compute on training fold only:
- `ttest_ind(X_train[y==0], X_train[y==1], equal_var=False)`
- Score = `abs(t_stat)` (handle NaNs by 0)

### 4.5 Ensemble Voting (Rank aggregation)
Must specify deterministic recipe:
- Base methods included: SNR, MI, ANOVA, T-test
- For a given K, each base method returns a ranked list length K (train-only)
- Compute `avg_rank(g)` across base methods; absent genes get rank = K+1 (or a large constant)
- Sort by `avg_rank` ascending; select top K

> Note: This is “average rank” aggregation (not intersection). It yields exactly K genes.

### 4.6 SCAD Regularization (SCAD-SVM)
Hardest method. Contract requirements:
- Operates on training fold only
- Tunes lambda via **nested Stratified CV** (default 3-fold) using training fold only
- Outputs scores as `abs(coef_)` aligned to feature names
- Must record chosen `best_lambda` for each outer fold (and ideally overall)

**Determinism:** the nested CV must use `random_state=RANDOM_SEED`.

---

## 5) Experiment Protocol Contract

### 5.1 Cross-validation
- Use **StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)**
- Fold ids are 0..4 in the order produced by the splitter

### 5.2 Fixed model & preprocessing
- Classifier is fixed: `LinearSVC(C=1.0, random_state=RANDOM_SEED)`
- Scaler fixed to one of:
  - StandardScaler
  - MinMaxScaler
- Scaling must be fit on training fold only.

### 5.3 Metrics
For each fold:
- Accuracy: `accuracy_score(y_val, y_pred)`
- Precision/Recall/F1: binary with `pos_label=1`
- AUROC:
  - Use `clf.decision_function(X_val_scaled)` as score
  - `roc_auc_score(y_val, scores)` (requires both classes present; handle edge cases)

**Edge case policy (must implement):**
- If a fold contains only one class in y_val (rare with stratification but possible in tiny datasets), set AUROC = NaN and log a warning.

### 5.4 Best-K selection
For each method:
- Compute mean accuracy over folds for each K
- `best_K = argmax(mean_accuracy)`
- Tie-breakers (in order):
  1. higher mean F1
  2. higher mean AUROC
  3. smaller K (prefer simpler model)

---

## 6) Output Contract (CSV Schemas)

### 6.1 `metrics_by_run.csv`
Columns:
- `method, K, fold_id, accuracy, precision, recall, f1, auroc`
- Optional:
  - `best_lambda` for SCAD

### 6.2 `best_k_summary.csv`
Columns:
- `method, best_K, mean_accuracy, std_accuracy, mean_f1, mean_auroc`
- Optional:
  - `mean_best_lambda` for SCAD

### 6.3 `top_genes/<method>_topk.csv`
Columns:
- `rank, gene_id, score`
- Optional:
  - `vote_count, avg_rank` for ensemble
  - `score_signed` for SNR

---

## 7) Verification / Tests (Must Pass)
- Selector unit tests:
  - returns Series with correct index
  - returns exactly k features
- Anti-leakage test:
  - corrupt X_val and verify selected genes unchanged
- Reproducibility test:
  - run twice with same seed → identical CSV outputs (byte-for-byte if possible)

