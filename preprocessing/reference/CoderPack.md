# Coder Pack — Implementation Guide (English)

This is the “how to build it” checklist for an agent or a human coder.

---

## 0) Quick Review of Your Plan (Critical Notes)

### ✅ Strong points
- Clear module separation + base class
- Stratified 5-fold CV and fixed classifier
- Explicit anti-leakage guarantees
- Good output artifacts (metrics table + best_k + top_genes)

### ⚠️ Important adjustments to avoid silent bugs
1) **AUROC with LinearSVC**  
LinearSVC has no `predict_proba`. Use `decision_function` scores for AUROC.

2) **Nested CV for SCAD (cost + determinism)**  
SCAD nested CV inside each outer fold is correct but expensive. Cache intermediate results if needed.

3) **Top genes at best_K should be fold-aggregated**  
If you output top genes from a single fold, it’s unstable. Aggregate across folds:
- rank genes per fold → aggregate via frequency + average rank → final top-K list

4) **Consistent label encoding**  
Define `POS_LABEL = 1` for ALL (or AML) and use it everywhere.

---

## 1) Step-by-step Build Order (Recommended)

### Step 1 — `config.py`
Include:
- K_VALUES
- CV_FOLDS, RANDOM_SEED
- SCAD lambda grid
- SCALER choice string
- CLASSIFIER string + constants (C)

Also define:
- `POS_LABEL = 1`
- `LABEL_MAP = {"ALL": 1, "AML": 0}`

### Step 2 — `feature_selection_methods/base.py`
Implement:
- `FeatureSelector.calculate_scores()` abstract
- `FeatureSelector.select_features()` concrete:
  - calls calculate_scores
  - drops NaNs deterministically
  - sorts by abs(score) desc
  - returns top k feature names

Add a `validate_scores(scores, X_train)` helper:
- assert index matches X_train.columns
- coerce to float
- replace inf with NaN then fillna(0)

### Step 3 — Implement methods (one file per method)
**Common coding rules:**
- Inputs: `(X_train: pd.DataFrame, y_train: pd.Series)`
- Output: `pd.Series(scores, index=X_train.columns)`
- Never reference validation data
- Set all randomness via RANDOM_SEED

#### 3.1 signal_to_noise.py
- compute per-class means/stds for each gene column
- apply eps to denom
- return signed score (selection uses abs by default)

#### 3.2 anova_f_test.py
- simplest: sklearn `f_classif`
- return F scores

#### 3.3 mutual_information.py
- sklearn `mutual_info_classif`
- ensure X is numeric ndarray
- return MI scores

#### 3.4 t_test.py
- scipy `ttest_ind(..., equal_var=False, nan_policy='omit')`
- return abs(t_stat); fill NaN=0

#### 3.5 ensemble_voting.py
- depends on other selectors
- compute ranks for each base method (SNR/MI/ANOVA/T)
- average ranks
- return score = -avg_rank (so “higher is better”)
- selection still uses abs(score); alternatively override select_features to use avg_rank directly

#### 3.6 scad_regularization.py
Minimum viable SCAD approach (deterministic):
- Standardize X_train inside the selector (training-only)
- Nested CV over lambda grid:
  - inner splitter: StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
- For each lambda:
  - Fit an **iteratively reweighted L1 linear model** to approximate SCAD
  - Score lambda by inner mean accuracy
- Fit final model using best lambda on full training fold
- Return abs(coefs) as scores

> Practical note: true SCAD-SVM is nontrivial. If you cannot implement a full SCAD-SVM solver, document the approximation (iteratively reweighted L1) explicitly in code comments and in results metadata.

---

## 2) Orchestrator — `Preprocessing.py`

### 2.1 Recommended class skeleton
- `FeatureSelectionExperiment.__init__(X, y, output_dir, config)`
- `run_experiment(methods=None) -> pd.DataFrame`
- `summarize_best_k(df) -> pd.DataFrame`
- `export_top_genes(df, fold_rankings_cache)`

### 2.2 Implementation details
1) Create outer CV splitter:
- StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)

2) For each fold:
- slice X/y into train and val by indices
- for each method and K:
  - selected = selector.select_features(X_train, y_train, k)
  - subset train/val
  - scaler.fit(train); transform train/val
  - clf.fit(train_scaled, y_train)
  - y_pred = clf.predict(val_scaled)
  - scores = clf.decision_function(val_scaled)  # for AUROC
  - compute metrics dict
  - append row to results

3) Save `metrics_by_run.csv` after loop finishes.

4) Build `best_k_summary.csv`:
- groupby(method, K) → mean/std
- pick best_K per method per Contract tie-break rules

5) Export top genes:
- For each method:
  - collect per-fold ranked lists for the method’s best_K
  - aggregate ranks across folds:
    - `vote_count = number of folds where gene appears in top_K`
    - `avg_rank = mean rank among folds where it appears` (or treat missing as K+1)
  - sort by vote_count desc then avg_rank asc
  - output final top_K to csv

---

## 3) Metrics Helper (`metrics.py` inside preprocessing/ optional)
Create `compute_metrics(y_true, y_pred, score=None)`:
- accuracy_score
- precision_score(pos_label=1, zero_division=0)
- recall_score(pos_label=1, zero_division=0)
- f1_score(pos_label=1, zero_division=0)
- roc_auc_score if score is not None and y_true has both classes else NaN

---

## 4) Testing Pack (Minimal but strong)

### 4.1 Unit tests for selectors
- Synthetic dataset where first 3 genes separate classes strongly
- Assert top genes include those
- Assert output length = k and unique

### 4.2 Anti-leakage test
- run selection on (X_train, y_train) with an X_val copy
- corrupt X_val heavily and re-run selection
- assert selected genes unchanged

### 4.3 Reproducibility
- run full experiment twice with same seed
- compare DataFrames (or hash CSV) identical

---

## 5) Coding Conventions (Consistency)
- Methods use snake_case names
- METHOD_NAME values:
  - "signal_to_noise", "anova_f_test", "mutual_information", "t_test", "ensemble_voting", "scad_regularization"
- Always log:
  - fold_id, method, K
  - for SCAD: best_lambda (per fold) and optional aggregated lambda

---

## 6) Common Failure Modes (Checklist)
- [ ] Selector accidentally uses X_val (leakage)
- [ ] Scaler fit on full data (leakage)
- [ ] AUROC computed from predicted labels instead of decision scores
- [ ] Label mapping inconsistent across metrics
- [ ] NaNs/infs in scores cause unstable ranking
- [ ] Ensemble aggregation not deterministic

---

## 7) Definition of “Done”
- Running `python Preprocessing.py` produces:
  - `Result/metrics_by_run.csv` with rows = methods × K × folds
  - `Result/best_k_summary.csv` with 6 rows
  - `Result/top_genes/*.csv` with 6 files and correct K sizes
- Anti-leakage + reproducibility tests pass
