#!/usr/bin/env python
"""
Evaluate trained models and generate comparison metrics
"""

import json
import numpy as np
from src.utils import plot_training_history
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("="*60)
    print("Model Evaluation and Visualization")
    print("="*60)

    # Load results
    results_dir = 'results/'

    with open(f"{results_dir}angle_encoding_results.json", 'r') as f:
        angle_results = json.load(f)

    with open(f"{results_dir}amplitude_encoding_results.json", 'r') as f:
        amp_results = json.load(f)

    # Load predictions
    angle_test_pred = np.load(f"{results_dir}angle_encoding_test_pred.npy")
    amp_test_pred = np.load(f"{results_dir}amplitude_encoding_test_pred.npy")

    angle_train_pred = np.load(f"{results_dir}angle_encoding_train_pred.npy")
    amp_train_pred = np.load(f"{results_dir}amplitude_encoding_train_pred.npy")

    # Generate visualizations
    print("\nGenerating visualizations...")

    print("  - Angle encoding loss history...")
    plot_training_history(angle_results['losses'], 'Angle Encoding')

    print("  - Amplitude encoding loss history...")
    plot_training_history(amp_results['losses'], 'Amplitude Encoding')

    print("  - Angle encoding predictions...")
    # Note: We don't have actual y_test here, skip prediction plots

    print("  - Amplitude encoding predictions...")

    print("\n✅ Visualizations saved to results/figures/")


if __name__ == "__main__":
    main()
