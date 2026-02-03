# Blueprint — Overall Architecture (English)

## 1) Goal
Build a **reproducible, leakage-safe** preprocessing + feature-selection benchmark for the **Golub AML/ALL** dataset to:

- Compare **accuracy** across **6 feature selection methods**
- Sweep gene counts **K ∈ {4, 8, 16, 24, 32, 50}**
- Use **Stratified 5-fold cross-validation** for model selection and reporting
- Choose **best_K per method** by **mean CV accuracy** (tie-breakers defined in Contract)
- Export the **final top-K gene list** for each method at its best_K

> Training-only rule: `data_set_ALL_AML_train` is the raw dataset variable used to derive *train folds* for fitting selectors/scalers/models.

---

## 2) Repository Layout
```
preprocessing/
  __init__.py
  Preprocessing.py                 # experiment orchestrator (CLI entry)
  config.py                        # constants & hyperparameters
  feature_selection_methods/
    __init__.py
    base.py                        # FeatureSelector abstract interface
    signal_to_noise.py             # SNR baseline (adapt existing)
    anova_f_test.py                # ANOVA F-test (adapt existing)
    mutual_information.py          # sklearn mutual_info_classif wrapper
    t_test.py                      # scipy.stats ttest_ind wrapper
    ensemble_voting.py             # rank aggregation across base methods
    scad_regularization.py         # SCAD-SVM (iteratively reweighted L1) + nested CV for lambda
  Result/
    metrics_by_run.csv             # per-fold metrics
    best_k_summary.csv             # per-method best_K summary (mean±std)
    top_genes/
      signal_to_noise_topk.csv
      mutual_information_topk.csv
      anova_f_test_topk.csv
      t_test_topk.csv
      ensemble_voting_topk.csv
      scad_regularization_topk.csv
```

---

## 3) High-Level Data Flow (Outer CV)
For each outer fold `(train_idx, val_idx)`:
1. Build `X_train, y_train` and `X_val, y_val` **from `data_set_ALL_AML_train` only**
2. For each method `m` and each `k` in K_VALUES:
   - `selected = selector[m].select_features(X_train, y_train, k)`  (**train-only**)
   - Subset: `X_train_sel = X_train[selected]`, `X_val_sel = X_val[selected]`
   - Scale: `scaler.fit(X_train_sel)` then transform both train/val (**fit train-only**)
   - Train: fixed classifier (LinearSVC) on train
   - Evaluate on val → store metrics
3. Aggregate results across folds:
   - For each method & K: mean±std metrics
   - Choose best_K per method
4. Produce final top genes per method at best_K using **fold-aggregation** rule (see Contract)

---

## 4) Component Responsibilities

### `config.py`
Single source of truth for all constants (K list, seeds, CV folds, lambda grid, classifier/scaler choices).

### `FeatureSelector` base class (`feature_selection_methods/base.py`)
Defines the stable interface all selectors must follow:
- `calculate_scores(X_train, y_train) -> pd.Series` (indexed by feature names)
- `select_features(X_train, y_train, k) -> List[str]` (top-k by |score|)

### Feature selection method modules
Each module implements one selector using the base interface. They are **pure functions on (X_train, y_train)** and must not access validation data.

### `Preprocessing.py`
The orchestrator:
- loads data
- builds folds
- loops methods × K × folds
- computes metrics
- writes CSV outputs

---

## 5) Outputs
- `Result/metrics_by_run.csv`: one row per (method, K, fold)
- `Result/best_k_summary.csv`: one row per method with best_K and summary metrics
- `Result/top_genes/<method>_topk.csv`: final gene list for each method at best_K

---

## 6) Key Design Choices (Why this architecture)
- **Comparability**: all methods share the same CV splits, scaler type, and classifier
- **Leakage safety**: selectors/scalers/models are fit only on training folds
- **Reproducibility**: centralized seed + deterministic pipelines
- **Extensibility**: adding a new method requires only a new selector module implementing the base interface
