# Preprocessing Comparison Report

Generated: 2025-12-15 21:59:59

## Configuration

- **K Values**: [2, 4, 8, 16, 24, 32, 50]
- **Methods**: ANOVA, SNR, SCAD-SVM
- **Gene Balance**: Balanced (k/2 each class) and Unbalanced (pure top-k)
- **Patient Balance**: Balanced (22) and Unbalanced (38)
- **Validation**: 70/30, 80/20, 5-Fold CV, 10-Fold CV, LOOCV

## Summary Statistics

- **Total Configurations**: 420
- **Successful**: 420
- **Failed**: 0
- **Total Time**: 13539.7 seconds

## Method Comparison

| Method | Avg Time (s) | Std | Min | Max | Runs |
|--------|--------------|-----|-----|-----|------|
| ANOVA | 0.24 | 0.10 | 0.12 | 0.64 | 140 |
| SNR | 1.17 | 0.39 | 0.70 | 2.20 | 140 |
| SCAD | 95.30 | 1050.72 | 2.05 | 12429.96 | 140 |

## Time by K Value

| K | ANOVA (s) | SNR (s) | SCAD (s) |
|---|-----------|---------|----------|
| 2 | 0.16 | 0.74 | 27.86 |
| 4 | 0.35 | 1.76 | 3.90 |
| 8 | 0.33 | 1.67 | 624.86 |
| 16 | 0.21 | 0.98 | 2.51 |
| 24 | 0.20 | 0.97 | 2.31 |
| 32 | 0.21 | 0.94 | 2.87 |
| 50 | 0.25 | 1.13 | 2.77 |

## Key Findings

1. **Fastest Method**: ANOVA (0.24s average)
2. **Slowest Method**: SCAD (95.30s average)
3. **Gene Balance Effect**: 56.90s difference
4. **Patient Balance Effect**: 61.38s difference

## Output Files

- `comparison_methods.csv` - Method comparison statistics
- `comparison_k_values.csv` - Time by K value
- `comparison_gene_balance.csv` - Gene balance effect
- `comparison_patient_balance.csv` - Patient balance effect
- `comparison_validation.csv` - Validation strategy comparison
- `comparison_full_summary.csv` - Full summary table
- `all_results.csv` - Raw results for all configurations
