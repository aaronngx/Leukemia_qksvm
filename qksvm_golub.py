import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import joblib

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from angle_encoding import angle_encoding_circuit


def scale_to_angle(X: np.ndarray):
    """Scale features to [0, π] for angle encoding."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def load_processed(train_csv: str, test_csv: str | None = None):
    """Load SNR-processed CSVs."""
    df_train = pd.read_csv(train_csv)
    y_train = df_train["label"].values
    X_train = df_train.drop(columns=["label"]).values

    if test_csv is not None:
        df_test = pd.read_csv(test_csv)
        y_test = df_test["label"].values
        X_test = df_test.drop(columns=["label"]).values
    else:
        X_test, y_test = None, None

    return (X_train, y_train), (X_test, y_test)


def build_kernel(
    XA: np.ndarray,
    XB: np.ndarray,
    feature_map: QuantumCircuit,
    x_params,
):
    """
    Build kernel matrix K where K[i, j] = |<φ(x_i)|φ(x_j)>|^2
    using angle encoding and U(x) U(z)† on a statevector simulator.
    """
    n_qubits = feature_map.num_qubits
    K = np.zeros((len(XA), len(XB)))

    for i, x in enumerate(XA):
        for j, z in enumerate(XB):
            qc = QuantumCircuit(n_qubits)

            # U(x)
            bind_x = {x_params[k]: float(x[k]) for k in range(n_qubits)}
            qc.compose(feature_map.assign_parameters(bind_x), inplace=True)

            # U(z)†
            bind_z = {x_params[k]: float(z[k]) for k in range(n_qubits)}
            qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)

            # Use Statevector to compute the kernel value
            sv = Statevector.from_instruction(qc)
            K[i, j] = float(np.abs(sv[0]) ** 2)  # prob(|0...0>)

    return K


def train_eval_qksvm(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
    output_dir: str = "results",
):
    """Train and evaluate QKSVM with angle-encoding quantum kernel."""
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    X_scaled, _ = scale_to_angle(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    n_features = X_train.shape[1]

    feature_map, x_params = angle_encoding_circuit(n_features)

    # Save circuit visualization
    print("[INFO] Saving circuit diagram...")
    fig = feature_map.draw(output='mpl', style='iqp')
    plt.savefig(out_path / f"circuit_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Also save circuit as text
    with open(out_path / f"circuit_{timestamp}.txt", 'w', encoding='utf-8') as f:
        f.write(str(feature_map.draw(output='text')))

    print("[INFO] Building training kernel...")
    K_train = build_kernel(X_train, X_train, feature_map, x_params)

    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)

    # Save trained model
    print("[INFO] Saving trained model...")
    joblib.dump(clf, out_path / f"qksvm_model_{timestamp}.pkl")

    print("[INFO] Building validation kernel...")
    K_val = build_kernel(X_val, X_train, feature_map, x_params)
    y_pred = clf.predict(K_val)

    acc = accuracy_score(y_val, y_pred)
    val_report = classification_report(y_val, y_pred, digits=4)

    print("\n=== QKSVM (validation) ===")
    print("Accuracy:", round(acc, 4))
    print(val_report)

    # Save validation predictions
    val_df = pd.DataFrame({
        'true_label': y_val,
        'predicted_label': y_pred
    })
    val_df.to_csv(out_path / f"validation_predictions_{timestamp}.csv", index=False)

    # Prepare results text
    results_text = []
    results_text.append("="*60)
    results_text.append("QUANTUM KERNEL SVM (QKSVM) RESULTS")
    results_text.append("="*60)
    results_text.append(f"\nTimestamp: {timestamp}")
    results_text.append(f"Number of features: {n_features}")
    results_text.append(f"Training samples: {len(X_train)}")
    results_text.append(f"Validation samples: {len(X_val)}")
    results_text.append(f"\n{'='*60}")
    results_text.append("VALIDATION SET RESULTS")
    results_text.append('='*60)
    results_text.append(f"Accuracy: {round(acc, 4)}")
    results_text.append(f"\n{val_report}")

    # Evaluate on independent set if provided
    if ind_data is not None and ind_data[0] is not None:
        X_ind, y_ind = ind_data
        X_ind_scaled, _ = scale_to_angle(X_ind)
        print("[INFO] Building independent-set kernel...")
        K_ind = build_kernel(X_ind_scaled, X_train, feature_map, x_params)
        y_ind_pred = clf.predict(K_ind)
        ind_acc = accuracy_score(y_ind, y_ind_pred)
        ind_report = classification_report(y_ind, y_ind_pred, digits=4)

        print("\n=== QKSVM (independent set) ===")
        print("Accuracy:", round(ind_acc, 4))
        print(ind_report)

        # Save independent predictions
        ind_df = pd.DataFrame({
            'true_label': y_ind,
            'predicted_label': y_ind_pred
        })
        ind_df.to_csv(out_path / f"independent_predictions_{timestamp}.csv", index=False)

        results_text.append(f"\n{'='*60}")
        results_text.append("INDEPENDENT TEST SET RESULTS")
        results_text.append('='*60)
        results_text.append(f"Test samples: {len(X_ind)}")
        results_text.append(f"Accuracy: {round(ind_acc, 4)}")
        results_text.append(f"\n{ind_report}")

    # Save all results to text file
    results_file = out_path / f"qksvm_results_{timestamp}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_text))

    print(f"\n[INFO] All results saved to: {out_path.absolute()}")
    print(f"  - Circuit diagram: circuit_{timestamp}.png")
    print(f"  - Circuit text: circuit_{timestamp}.txt")
    print(f"  - Trained model: qksvm_model_{timestamp}.pkl")
    print(f"  - Results summary: qksvm_results_{timestamp}.txt")
    print(f"  - Predictions: validation_predictions_{timestamp}.csv")
    if ind_data is not None and ind_data[0] is not None:
        print(f"  - Independent predictions: independent_predictions_{timestamp}.csv")


def main():
    parser = argparse.ArgumentParser(description="QKSVM (angle-encoding kernel) on Golub dataset.")
    parser.add_argument(
        "--train_csv",
        default="data/processed/train_topk_snr.csv",
        help="Path to processed train CSV (SNR-selected).",
    )
    parser.add_argument(
        "--ind_csv",
        default="data/processed/independent_topk_snr.csv",
        help="Path to processed independent CSV (SNR-selected).",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    (X_train, y_train), (X_ind, y_ind) = load_processed(args.train_csv, args.ind_csv)
    train_eval_qksvm(X_train, y_train, ind_data=(X_ind, y_ind), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
