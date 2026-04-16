#!/usr/bin/env python
"""Run a full QML experiment with the EUOS-SLAS dataset."""

from src.training import train_model
from src.angle_encoding_model import AngleEncodingModel
from src.data_preprocessing import load_euos_dataset, prepare_train_test_data
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print('=' * 60)
    print('QUANTUM ML - DRUG DISCOVERY (EUOS-SLAS)')
    print('=' * 60)

    # Uses the new 3-class EUOS-SLAS dataset loaded directly from Kaggle CSV and intercepts weights
    X, X_full, y, feature_cols, class_weights = load_euos_dataset()

    print(f'Selected descriptors: {feature_cols}')
    print(f'Feature count: {X.shape[1]}')
    print(f'Sample count for fast QML evaluation: {X.shape[0]}')

    data = prepare_train_test_data(X, X_full, y)
    print(f"Training samples: {data['X_train'].shape[0]}, Test samples: {data['X_test'].shape[0]}")

    n_qubits = min(len(feature_cols), 8)  # Angle model uses 8 qubits for 8 features
    print(f"\nTraining Angle Encoding ({n_qubits} qubits) for 99% Target...")
    angle_model = AngleEncodingModel(n_qubits=n_qubits)
    angle_results = train_model(
        angle_model,
        data['X_train'],
        data['X_full_train'],
        data['y_train'],
        data['X_test'],
        data['X_full_test'],
        data['y_test'],
        epochs=1,  # Increased epochs for higher accuracy penetration
        verbose=True,
        class_weights=class_weights
    )

    # print(f"\nTraining Amplitude Encoding (3 qubits for 8 features) for 99% Target...")
    # amp_model = AmplitudeEncodingModel(n_qubits=3)
    # amp_results = train_model(
    #     amp_model,
    #     data['X_train'],
    #     data['X_full_train'],
    #     data['y_train'],
    #     data['X_test'],
    #     data['X_full_test'],
    #     data['y_test'],
    #     epochs=1,
    #     verbose=True,
    #     class_weights=class_weights
    # )

    print('\n' + '=' * 60)
    print('FINAL RESULTS FOR ACCURACY')
    print('=' * 60)
    print(
        f"Angle Encoding - Train Acc: {angle_results['train_acc']*100:.2f}%, Test Acc: {angle_results['test_acc']*100:.2f}%, Best: {angle_results['best_accuracy']*100:.2f}%")
    # print(f"Amplitude Encoding - Train Acc: {amp_results['train_acc']*100:.2f}%, Test Acc: {amp_results['test_acc']*100:.2f}%, Best: {amp_results['best_accuracy']*100:.2f}%")

    # =========================================================
    # PREMIUM HYBRID QUANTUM ENSEMBLE
    # =========================================================
    from src.quantum_forest import QuantumForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    print('\n' + '=' * 60)
    print("Training High-Performance QuantumForest Ensemble...")

    q_forest = QuantumForestClassifier(n_estimators=250, n_qubits=n_qubits)
    q_forest.fit(data['X_train'], data['X_full_train'], data['y_train'])

    test_preds = q_forest.predict(data['X_test'], data['X_full_test'])
    qf_acc = accuracy_score(data['y_test'], test_preds)
    qf_f1 = f1_score(data['y_test'], test_preds, average='macro', zero_division=0)

    print(f"\n🚀 PREMIUM HYBRID MODEL RESULTS 🚀")
    print(f"QuantumForest Ensemble Test Accuracy: {qf_acc*100:.2f}%")
    print(f"QuantumForest Ensemble Test F1 Score: {qf_f1*100:.2f}%")
    print('=' * 60)


if __name__ == '__main__':
    main()
