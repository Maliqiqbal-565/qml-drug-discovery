#!/usr/bin/env python
"""Run a full QML experiment with the Delaney solubility dataset."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_delaney_dataset, preprocess_data, prepare_train_test_data
from src.angle_encoding_model import AngleEncodingModel
from src.amplitude_encoding_model import AmplitudeEncodingModel
from src.training import train_model


def main():
    print('=' * 60)
    print('QUANTUM ML - DRUG DISCOVERY')
    print('=' * 60)

    df = load_delaney_dataset()
    X, y, feature_cols = preprocess_data(df)

    print(f'Selected features: {feature_cols}')
    print(f'Feature count: {X.shape[1]}')
    print(f'Sample count: {X.shape[0]}')

    data = prepare_train_test_data(X, y)
    print(f"Training samples: {data['X_train'].shape[0]}, Test samples: {data['X_test'].shape[0]}")

    n_qubits = min(len(feature_cols), 4)
    print(f"\nTraining Angle Encoding ({n_qubits} qubits)...")
    angle_model = AngleEncodingModel(n_qubits=n_qubits)
    angle_results = train_model(
        angle_model,
        data['X_train'],
        data['y_train'],
        data['X_test'],
        data['y_test'],
        epochs=10,
        verbose=True,
    )

    print(f"\nTraining Amplitude Encoding (3 qubits)...")
    amp_model = AmplitudeEncodingModel(n_qubits=3)
    amp_results = train_model(
        amp_model,
        data['X_train'],
        data['y_train'],
        data['X_test'],
        data['y_test'],
        epochs=10,
        verbose=True,
    )

    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f"Angle Encoding - RMSE: {angle_results['test_rmse']:.4f}, R²: {angle_results['r2_score']:.4f}")
    print(f"Amplitude Encoding - RMSE: {amp_results['test_rmse']:.4f}, R²: {amp_results['r2_score']:.4f}")
    print('=' * 60)


if __name__ == '__main__':
    main()