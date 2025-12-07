#!/usr/bin/env python3
"""
Unified Pipeline: Preprocessing → Quantum Encoding → QKSVM

Connects feature selection with quantum encoding and classification:
1. Run preprocessing to select genes
2. Load selected gene data
3. Apply amplitude and/or angle encoding
4. Train QKSVM classifier
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add feature-selection-methods to path
sys.path.insert(0, str(Path(__file__).parent / "feature-selection-methods"))

from anova_f import run_feature_selection as run_anova
from signal_to_noise import run_snr_selection
from amplitude_encoding import (
    get_num_qubits,
    preprocess_and_normalize,
    build_amplitude_vqc,
    train_amplitude_vqc
)
from angle_encoding import angle_encoding_circuit, AngleEncodingType
from qksvm_golub import (
    train_eval_qksvm,
    EncodingType,
    scale_to_angle
)
from backend_config import BackendType, KernelMethod


# ============================================================
# Constants
# ============================================================
PATIENTS_TRAIN_BALANCED = 22   # 11 ALL + 11 AML
PATIENTS_ALL_BALANCED = 50     # 25 ALL + 25 AML


def print_banner():
    print()
    print("=" * 70)
    print("  QUANTUM MACHINE LEARNING PIPELINE")
    print("  Gene Expression → Feature Selection → Quantum Encoding")
    print("=" * 70)
    print()


def print_separator():
    print("=" * 70)


# ============================================================
# Step 1: Preprocessing (Feature Selection)
# ============================================================
def run_preprocessing_step():
    """Run interactive preprocessing or use existing results."""
    print("\n[STEP 1] FEATURE SELECTION")
    print("-" * 50)
    
    # Check if results already exist
    results_dir = Path("results")
    
    print("Options:")
    print("  1. Run new feature selection (interactive)")
    print("  2. Use existing results from 'results/' folder")
    
    while True:
        choice = input("  -> ").strip()
        if choice in ["1", "2"]:
            break
        print("  [ERROR] Please enter 1 or 2")
    
    if choice == "1":
        # Run interactive preprocessing
        from preprocessing import run_preprocessing
        run_preprocessing()
    else:
        # Check what files exist
        anova_files = list(results_dir.glob("all_top_*_anova_f.csv")) + list(results_dir.glob("train_top_*_anova_f.csv"))
        snr_files = list(results_dir.glob("all_top_*_snr.csv")) + list(results_dir.glob("train_top_*_snr.csv"))
        
        if not anova_files and not snr_files:
            print("\n  [WARNING] No existing results found. Running feature selection...")
            from preprocessing import run_preprocessing
            run_preprocessing()


def list_available_data():
    """List available preprocessed data files."""
    results_dir = Path("results")
    
    data_files = []
    
    # Find all data files
    for pattern in ["all_top_*.csv", "train_top_*.csv"]:
        for f in results_dir.glob(pattern):
            if "anova" in f.name or "snr" in f.name:
                # Parse filename to get info
                parts = f.stem.split("_")
                k = int(parts[2])  # e.g., "all_top_16_snr" -> 16
                method = "ANOVA" if "anova" in f.name else "SNR"
                data_type = "all" if f.name.startswith("all_") else "train"
                
                data_files.append({
                    "path": f,
                    "k": k,
                    "method": method,
                    "data_type": data_type,
                    "name": f.name
                })
    
    return data_files


# ============================================================
# Step 2: Load Selected Gene Data
# ============================================================
def load_selected_data(data_file: Path):
    """Load the selected gene expression data."""
    from data_loader import load_preprocessed_data

    # Load features and labels using unified loader
    X, y = load_preprocessed_data(data_file)

    # Also load patient IDs and feature column names for reference
    df = pd.read_csv(data_file)
    patient_ids = df["patient"].values if "patient" in df.columns else None

    # Get feature columns (everything except cancer/label and patient)
    exclude_cols = {"cancer", "label", "patient", "Patient", "patient_id"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\n  Loaded data from: {data_file.name}")
    print(f"  Samples: {len(X)}")
    print(f"  Features (genes): {len(feature_cols)}")
    print(f"  Classes: ALL={sum(y==0)}, AML={sum(y==1)}")

    return X, y, feature_cols, patient_ids


# ============================================================
# Step 3: Quantum Encoding & Classification
# ============================================================
def select_encoding_method():
    """Ask which encoding method to use."""
    print("\n[STEP 3] SELECT ENCODING METHOD")
    print("-" * 50)
    print("  1. Amplitude Encoding (logarithmic qubit scaling)")
    print("  2. Angle Encoding (linear qubit scaling)")
    print("  3. Both (compare methods)")
    
    while True:
        choice = input("  -> ").strip()
        if choice in ["1", "2", "3"]:
            return int(choice)
        print("  [ERROR] Please enter 1, 2, or 3")


def select_angle_encoding_type():
    """Ask which angle encoding circuit type to use."""
    print("\n[STEP 3b] SELECT ANGLE ENCODING TYPE")
    print("-" * 50)
    print("  1. Simple RY (no entanglement)")
    print("     - Single-axis: RY gates only")
    print("     - No entanglement between qubits")
    print()
    print("  2. ZZ Feature Map (ZZ interactions)")
    print("     - Single-axis: RZ gates")
    print("     - Entanglement: ZZ interactions (CNOT-RZ-CNOT)")
    print("     - Data re-uploading via repetitions")
    print()
    print("  3. BPS Circuit (two-axis encoding)")
    print("     - Two-axis: RZ + RY gates")
    print("     - Entanglement: Linear CNOT chain")
    print("     - Data re-uploading: Explicit RZ at end")
    
    while True:
        choice = input("  -> ").strip()
        if choice == "1":
            return AngleEncodingType.SIMPLE_RY, 1
        elif choice == "2":
            reps = ask_reps("ZZ Feature Map", default=2)
            return AngleEncodingType.ZZ_FEATURE_MAP, reps
        elif choice == "3":
            reps = ask_reps("BPS Circuit", default=1)
            return AngleEncodingType.BPS_CIRCUIT, reps
        print("  [ERROR] Please enter 1, 2, or 3")


def ask_reps(circuit_name: str, default: int) -> int:
    """Ask for number of repetitions."""
    print(f"\n  How many repetitions for {circuit_name}?")
    print(f"  (more reps = more expressivity but deeper circuit)")
    while True:
        choice = input(f"  -> (default {default}): ").strip()
        if choice == "":
            return default
        try:
            reps = int(choice)
            if reps >= 1:
                return reps
            print("  [ERROR] Reps must be >= 1")
        except ValueError:
            print("  [ERROR] Please enter a number")


def select_classifier():
    """Ask which classifier to use."""
    print("\n[STEP 4] SELECT CLASSIFIER")
    print("-" * 50)
    print("  1. QKSVM (Quantum Kernel SVM)")
    print("  2. VQC (Variational Quantum Classifier)")
    print("  3. Circuit visualization only (no training)")
    
    while True:
        choice = input("  -> ").strip()
        if choice in ["1", "2", "3"]:
            return int(choice)
        print("  [ERROR] Please enter 1, 2, or 3")


def run_amplitude_encoding(X, y, num_features):
    """Run amplitude encoding VQC."""
    print("\n" + "=" * 60)
    print("AMPLITUDE ENCODING")
    print("=" * 60)
    
    n_qubits = get_num_qubits(num_features)
    print(f"\nQubit scaling: {num_features} features → {n_qubits} qubits (logarithmic)")
    
    # Build circuit for visualization
    circuit, x_params, theta_params, n_qubits = build_amplitude_vqc(
        num_features=num_features,
        reps=2
    )
    
    print(f"\nCircuit built:")
    print(f"  - Input parameters: {len(x_params)}")
    print(f"  - Trainable parameters: {len(theta_params)}")
    
    # Ask if user wants to train
    print("\nTrain the model? (this may take a while)")
    train_choice = input("  -> (y/n): ").lower().strip()
    
    if train_choice in ["y", "yes"]:
        clf, results = train_amplitude_vqc(
            X, y,
            num_features=num_features,
            reps=2,
            max_iter=50
        )
        return results
    else:
        return {"circuit": circuit, "n_qubits": n_qubits, "n_params": len(theta_params)}


def run_angle_encoding(X, y, num_features):
    """Run angle encoding."""
    print("\n" + "=" * 60)
    print("ANGLE ENCODING")
    print("=" * 60)
    
    n_qubits = num_features  # Linear scaling
    print(f"\nQubit scaling: {num_features} features → {n_qubits} qubits (linear)")
    
    # Build circuit
    circuit, x_params = angle_encoding_circuit(n_qubits)
    
    print(f"\nCircuit built:")
    print(f"  - Qubits: {n_qubits}")
    print(f"  - Input parameters: {len(x_params)}")
    print(f"  - Structure: RY rotations on each qubit")
    
    return {"circuit": circuit, "n_qubits": n_qubits, "x_params": x_params}


def compare_encodings(num_features):
    """Compare amplitude vs angle encoding."""
    print("\n" + "=" * 60)
    print("ENCODING COMPARISON")
    print("=" * 60)
    
    amp_qubits = get_num_qubits(num_features)
    angle_qubits = num_features
    
    print(f"\nFor {num_features} features/genes:")
    print(f"\n{'Method':<20} {'Qubits':<10} {'Scaling':<15} {'Reduction':<15}")
    print("-" * 60)
    print(f"{'Amplitude':<20} {amp_qubits:<10} {'Logarithmic':<15} {f'{(1-amp_qubits/angle_qubits)*100:.0f}%':<15}")
    print(f"{'Angle':<20} {angle_qubits:<10} {'Linear':<15} {'Baseline':<15}")
    print()


def run_qksvm(
    X, y, num_features, encoding_type_str,
    angle_encoding_type: AngleEncodingType = AngleEncodingType.SIMPLE_RY,
    angle_reps: int = 2,
):
    """Run QKSVM with selected encoding."""
    print("\n" + "=" * 60)
    print("QUANTUM KERNEL SVM (QKSVM)")
    print("=" * 60)
    
    # Convert encoding string to EncodingType
    if encoding_type_str == "amplitude":
        encoding = EncodingType.AMPLITUDE
        n_qubits = get_num_qubits(num_features)
    else:
        encoding = EncodingType.ANGLE
        n_qubits = num_features
    
    print(f"\nEncoding: {encoding_type_str}")
    if encoding_type_str == "angle":
        print(f"Angle encoding type: {angle_encoding_type.value}")
        if angle_encoding_type != AngleEncodingType.SIMPLE_RY:
            print(f"Repetitions: {angle_reps}")
    print(f"Qubits: {n_qubits}")
    print(f"Samples: {len(X)}")
    print(f"Features: {num_features}")
    
    # Check if qubits is reasonable for simulation
    if n_qubits > 20:
        print(f"\n[WARNING] {n_qubits} qubits requires significant computation.")
        print("  Consider using amplitude encoding for fewer qubits.")
        proceed = input("  Continue anyway? (y/n): ").lower().strip()
        if proceed not in ["y", "yes"]:
            return None
    
    print("\nRunning QKSVM training...")
    print("This may take a few minutes depending on the data size.\n")
    
    try:
        train_eval_qksvm(
            X, y,
            ind_data=None,  # No separate test set for combined data
            test_size=0.3,
            seed=42,
            output_dir="results_qksvm",
            encoding_type=encoding,
            angle_encoding_type=angle_encoding_type,
            angle_reps=angle_reps,
            kernel_method=KernelMethod.STATEVECTOR,
            backend_type=BackendType.STATEVECTOR,
        )
        return {
            "success": True, 
            "encoding": encoding_type_str, 
            "n_qubits": n_qubits,
            "angle_type": angle_encoding_type.value if encoding_type_str == "angle" else None,
        }
    except Exception as e:
        print(f"\n[ERROR] QKSVM failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# Main Pipeline
# ============================================================
def run_pipeline():
    """Run the complete pipeline."""
    print_banner()
    
    # Step 1: Feature Selection
    run_preprocessing_step()
    
    # Step 2: Select data file
    print("\n[STEP 2] SELECT DATA FILE")
    print("-" * 50)
    
    data_files = list_available_data()
    
    if not data_files:
        print("  [ERROR] No data files found in results/")
        print("  Please run feature selection first (option 1 in step 1)")
        return
    
    print("\nAvailable data files:")
    for i, df in enumerate(data_files, 1):
        print(f"  {i}. {df['name']} ({df['k']} genes, {df['method']}, {df['data_type']} data)")
    
    while True:
        try:
            choice = int(input("  -> "))
            if 1 <= choice <= len(data_files):
                selected_file = data_files[choice - 1]
                break
            print(f"  [ERROR] Please enter 1-{len(data_files)}")
        except ValueError:
            print("  [ERROR] Please enter a valid number")
    
    # Load data
    X, y, feature_cols, patient_ids = load_selected_data(selected_file["path"])
    num_features = len(feature_cols)
    
    # Step 3: Encoding
    encoding_choice = select_encoding_method()
    
    # Ask for angle encoding type if using angle encoding
    angle_encoding_type = AngleEncodingType.SIMPLE_RY
    angle_reps = 2
    if encoding_choice in [2, 3]:  # Angle or Both
        angle_encoding_type, angle_reps = select_angle_encoding_type()
    
    # Step 4: Classifier
    classifier_choice = select_classifier()
    
    results = {}
    
    # Determine encoding type string
    if encoding_choice == 1:
        encoding_types = ["amplitude"]
    elif encoding_choice == 2:
        encoding_types = ["angle"]
    else:
        encoding_types = ["amplitude", "angle"]
    
    # Run based on classifier choice
    if classifier_choice == 1:  # QKSVM
        for enc_type in encoding_types:
            print(f"\n--- Running QKSVM with {enc_type} encoding ---")
            results[f"qksvm_{enc_type}"] = run_qksvm(
                X, y, num_features, enc_type,
                angle_encoding_type=angle_encoding_type,
                angle_reps=angle_reps,
            )
    
    elif classifier_choice == 2:  # VQC
        for enc_type in encoding_types:
            if enc_type == "amplitude":
                results["amplitude"] = run_amplitude_encoding(X, y, num_features)
            else:
                results["angle"] = run_angle_encoding(X, y, num_features)
    
    else:  # Visualization only
        if encoding_choice in [1, 3]:
            results["amplitude"] = run_amplitude_encoding(X, y, num_features)
        if encoding_choice in [2, 3]:
            results["angle"] = run_angle_encoding(X, y, num_features)
    
    if encoding_choice == 3:
        compare_encodings(num_features)
    
    # Summary
    print_separator()
    print("  PIPELINE COMPLETE!")
    print_separator()
    print(f"\nData: {selected_file['name']}")
    print(f"Features: {num_features} genes")
    print(f"Samples: {len(X)} patients")
    
    if "amplitude" in results:
        print(f"\nAmplitude Encoding:")
        print(f"  - Qubits: {results['amplitude']['n_qubits']}")
        if 'n_params' in results['amplitude']:
            print(f"  - Parameters: {results['amplitude']['n_params']}")
    
    if "angle" in results:
        print(f"\nAngle Encoding:")
        print(f"  - Qubits: {results['angle']['n_qubits']}")
    
    for key in results:
        if key.startswith("qksvm_") and results[key] and results[key].get("success"):
            enc = results[key]["encoding"]
            qubits = results[key]["n_qubits"]
            print(f"\nQKSVM ({enc}):")
            print(f"  - Qubits: {qubits}")
            print(f"  - Results saved to: results_qksvm/")
    
    print()
    
    return results


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\n[INFO] Cancelled by user\n")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

