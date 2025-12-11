#!/usr/bin/env python3
"""
Generate Analysis Plots and Tables for Quantum ML Experiments.

Creates:
1. Feature Efficiency Plot: AUROC vs. Number of Features (K) for all models
2. Parameter Efficiency Table: Model vs. Parameter Count vs. Peak AUROC

Usage:
    python generate_analysis_plots.py                    # Use existing experiment_results.csv
    python generate_analysis_plots.py --run-experiments  # Run experiments first, then plot
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from amplitude_encoding import get_num_qubits

# Output directory
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Styling
COLORS = {
    'QKSVM_angle': '#1f77b4',      # Blue
    'QKSVM_amplitude': '#ff7f0e',  # Orange
    'VQC_angle': '#2ca02c',        # Green
    'VQC_amplitude': '#d62728',    # Red (if implemented)
}

MARKERS = {
    'QKSVM_angle': 'o',
    'QKSVM_amplitude': 's',
    'VQC_angle': '^',
    'VQC_amplitude': 'D',
}

LINESTYLES = {
    'anova': '-',
    'snr': '--',
}


def load_experiment_results(results_path: str = "results/experiment_results.csv") -> Optional[pd.DataFrame]:
    """Load experiment results from CSV."""
    path = Path(results_path)
    if not path.exists():
        print(f"[ERROR] Results file not found: {path}")
        print("        Run experiments first with: python Experiment.py")
        print("        Or run this script with --run-experiments flag")
        return None
    
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} experiment results from {path}")
    return df


def compute_parameter_count(model: str, encoding: str, k: int, reps: int = 2) -> dict:
    """
    Compute parameter counts for a given model configuration.
    
    Returns dict with:
    - n_qubits: Number of qubits
    - feature_params: Number of feature map parameters (fixed)
    - trainable_params: Number of trainable parameters
    - total_params: Total parameters
    """
    if encoding == "amplitude":
        n_qubits = get_num_qubits(k)
    else:  # angle
        n_qubits = k
    
    # Feature map parameters (input data, not trainable)
    feature_params = k  # Always K features
    
    # Trainable parameters depend on model
    if model.upper() == "QKSVM":
        # QKSVM has no trainable quantum parameters
        # Only classical SVM parameters (support vectors, etc.)
        trainable_params = 0  # Quantum circuit has no trainable params
    else:  # VQC
        # TwoLocal ansatz: 3 rotations * n_qubits * (reps + 1)
        trainable_params = 3 * n_qubits * (reps + 1)
    
    return {
        "n_qubits": n_qubits,
        "feature_params": feature_params,
        "trainable_params": trainable_params,
        "total_params": feature_params + trainable_params,
    }


def fig_auroc_vs_features(df: pd.DataFrame, output_prefix: str = "fig_auroc_vs_k"):
    """
    Generate Feature Efficiency Plot: AUROC vs. Number of Features (K).
    
    Creates separate plots for ANOVA and SNR, or combined plot.
    """
    print("\n[Fig] AUROC vs Number of Features (K)...")
    
    # Filter out rows without AUROC
    df_valid = df.dropna(subset=['ind_auroc'])
    
    if len(df_valid) == 0:
        print("  [WARNING] No AUROC data available. Skipping plot.")
        return
    
    # Get unique feature selection methods
    fs_methods = df_valid['feature_selection'].unique()
    
    # Create combined plot
    fig, axes = plt.subplots(1, len(fs_methods), figsize=(7 * len(fs_methods), 6), squeeze=False)
    
    for idx, fs_method in enumerate(fs_methods):
        ax = axes[0, idx]
        df_fs = df_valid[df_valid['feature_selection'] == fs_method]
        
        # Group by model and encoding
        for (model, encoding), group in df_fs.groupby(['model', 'encoding']):
            key = f"{model}_{encoding}"
            color = COLORS.get(key, '#333333')
            marker = MARKERS.get(key, 'o')
            
            # Average AUROC for each K (in case of multiple kernel methods)
            k_auroc = group.groupby('k')['ind_auroc'].mean().reset_index()
            k_auroc = k_auroc.sort_values('k')
            
            ax.plot(k_auroc['k'], k_auroc['ind_auroc'], 
                   marker=marker, color=color, linewidth=2, markersize=8,
                   label=f"{model} ({encoding.capitalize()})")
        
        ax.set_xlabel('Number of Features (K)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Feature Efficiency: {fs_method.upper()} Selection', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.05)
        
        # Add K values as x-ticks
        k_values = sorted(df_fs['k'].unique())
        ax.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / output_prefix}.png/pdf")
    
    # Also create a single combined plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for (fs_method, model, encoding), group in df_valid.groupby(['feature_selection', 'model', 'encoding']):
        key = f"{model}_{encoding}"
        color = COLORS.get(key, '#333333')
        marker = MARKERS.get(key, 'o')
        linestyle = LINESTYLES.get(fs_method, '-')
        
        # Average AUROC for each K
        k_auroc = group.groupby('k')['ind_auroc'].mean().reset_index()
        k_auroc = k_auroc.sort_values('k')
        
        label = f"{model} ({encoding.capitalize()}) - {fs_method.upper()}"
        ax.plot(k_auroc['k'], k_auroc['ind_auroc'], 
               marker=marker, color=color, linestyle=linestyle,
               linewidth=2, markersize=8, label=label)
    
    ax.set_xlabel('Number of Features (K)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Feature Efficiency: AUROC vs Number of Features\n(All Models, ANOVA vs SNR)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    
    k_values = sorted(df_valid['k'].unique())
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_prefix}_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / f'{output_prefix}_combined.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / output_prefix}_combined.png/pdf")


def table_parameter_efficiency(df: pd.DataFrame, output_prefix: str = "table_param_efficiency"):
    """
    Generate Parameter Efficiency Table: Model vs. Parameter Count vs. Peak AUROC.
    """
    print("\n[Table] Parameter Efficiency Table...")
    
    # Filter out rows without AUROC
    df_valid = df.dropna(subset=['ind_auroc'])
    
    if len(df_valid) == 0:
        print("  [WARNING] No AUROC data available. Skipping table.")
        return
    
    # Build table data
    table_data = []
    
    for (fs_method, model, encoding), group in df_valid.groupby(['feature_selection', 'model', 'encoding']):
        # Find peak AUROC and corresponding K
        best_idx = group['ind_auroc'].idxmax()
        best_row = group.loc[best_idx]
        best_k = int(best_row['k'])
        best_auroc = best_row['ind_auroc']
        
        # Compute parameters
        params = compute_parameter_count(model, encoding, best_k, reps=2)
        
        table_data.append({
            'Feature Selection': fs_method.upper(),
            'Model': model.upper(),
            'Encoding': encoding.capitalize(),
            'Best K': best_k,
            'Qubits': params['n_qubits'],
            'Feature Params': params['feature_params'],
            'Trainable Params': params['trainable_params'],
            'Total Params': params['total_params'],
            'Peak AUROC': f"{best_auroc:.4f}",
            'Peak Accuracy': f"{best_row['ind_acc']:.4f}" if pd.notna(best_row['ind_acc']) else "N/A",
        })
    
    # Create DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = OUTPUT_DIR / f'{output_prefix}.csv'
    table_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Create figure table
    fig, ax = plt.subplots(figsize=(16, max(3, len(table_data) * 0.6 + 1)))
    ax.axis('off')
    
    # Create table
    columns = list(table_df.columns)
    cell_text = table_df.values.tolist()
    
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#4472C4'] * len(columns),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE5')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax.set_title('Parameter Efficiency: Model vs Parameter Count vs Peak AUROC',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / output_prefix}.png/pdf")
    
    # Print to console
    print("\n" + "=" * 120)
    print("PARAMETER EFFICIENCY TABLE")
    print("=" * 120)
    print(table_df.to_string(index=False))
    print("=" * 120)


def fig_accuracy_comparison(df: pd.DataFrame, output_prefix: str = "fig_accuracy_comparison"):
    """
    Generate Accuracy comparison bar chart for all models.
    """
    print("\n[Fig] Accuracy Comparison Bar Chart...")
    
    df_valid = df.dropna(subset=['ind_acc'])
    
    if len(df_valid) == 0:
        print("  [WARNING] No accuracy data available. Skipping plot.")
        return
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique combinations
    k_values = sorted(df_valid['k'].unique())
    models = df_valid.groupby(['model', 'encoding']).size().reset_index()[['model', 'encoding']].values.tolist()
    
    x = np.arange(len(k_values))
    width = 0.8 / len(models)
    
    for i, (model, encoding) in enumerate(models):
        key = f"{model}_{encoding}"
        color = COLORS.get(key, '#333333')
        
        accs = []
        for k in k_values:
            mask = (df_valid['model'] == model) & (df_valid['encoding'] == encoding) & (df_valid['k'] == k)
            acc = df_valid.loc[mask, 'ind_acc'].mean()
            accs.append(acc if pd.notna(acc) else 0)
        
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=f"{model} ({encoding.capitalize()})", color=color)
        
        # Add value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.annotate(f'{acc:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    ax.set_xlabel('Number of Features (K)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Comparison\n(Independent Test Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / output_prefix}.png/pdf")


def generate_all_analysis(results_path: str = "results/experiment_results.csv"):
    """Generate all analysis plots and tables."""
    print("=" * 60)
    print("GENERATING ANALYSIS PLOTS AND TABLES")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # Load results
    df = load_experiment_results(results_path)
    if df is None:
        return
    
    # Generate plots
    fig_auroc_vs_features(df)
    table_parameter_efficiency(df)
    fig_accuracy_comparison(df)
    
    print("\n" + "=" * 60)
    print("ALL ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  - fig_auroc_vs_k.png/pdf (per feature selection method)")
    print("  - fig_auroc_vs_k_combined.png/pdf (all methods)")
    print("  - table_param_efficiency.png/pdf/csv")
    print("  - fig_accuracy_comparison.png/pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots for QML experiments")
    parser.add_argument("--results", type=str, default="results/experiment_results.csv",
                       help="Path to experiment results CSV")
    parser.add_argument("--run-experiments", action="store_true",
                       help="Run experiments first before generating plots")
    
    args = parser.parse_args()
    
    if args.run_experiments:
        print("[INFO] Running experiments first...")
        from Experiment import interactive_mode
        interactive_mode()
    
    generate_all_analysis(args.results)


if __name__ == "__main__":
    main()

