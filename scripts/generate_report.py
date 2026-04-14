#!/usr/bin/env python
"""
Generate comprehensive project report
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime

def generate_report():
    print("="*60)
    print("Generating Project Report")
    print("="*60)
    
    results_dir = 'results/'
    
    # Load results
    with open(f"{results_dir}comparison_results.json", 'r') as f:
        results = json.load(f)
    
    angle_res = results['angle_encoding']
    amp_res = results['amplitude_encoding']
    config = results['config']
    
    # Generate report
    report_path = f"{results_dir}reports/comprehensive_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUANTUM MACHINE LEARNING FOR DRUG DISCOVERY\n")
        f.write("COMPREHENSIVE EXPERIMENT REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write("This report compares two quantum encoding methods for molecular property\n")
        f.write("prediction using the ESOL dataset. The models use 8 molecular features to\n")
        f.write("predict solubility. Key metrics include RMSE, R² score, and efficiency.\n\n")
        
        # Dataset Information
        f.write("DATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Dataset: {config['data']['dataset'].upper()}\n")
        f.write(f"Test Size: {config['data']['test_size']*100}%\n")
        f.write(f"Features: {', '.join(config['data']['features'])}\n\n")
        
        # Model Configurations
        f.write("MODEL CONFIGURATIONS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Angle Encoding:\n")
        angle_cfg = config['quantum']['angle_encoding']
        f.write(f"  - Qubits: {angle_cfg['n_qubits']}\n")
        f.write(f"  - Layers: {angle_cfg['n_layers']}\n")
        f.write(f"  - Learning Rate: {angle_cfg['learning_rate']}\n")
        f.write(f"  - Epochs: {angle_cfg['epochs']}\n")
        f.write(f"  - Circuit Depth: ~{angle_cfg['n_qubits'] + (angle_cfg['n_layers'] * (angle_cfg['n_qubits'] * 2 + angle_cfg['n_qubits']))}\n\n")
        
        f.write("Amplitude Encoding:\n")
        amp_cfg = config['quantum']['amplitude_encoding']
        f.write(f"  - Qubits: {amp_cfg['n_qubits']}\n")
        f.write(f"  - Layers: {amp_cfg['n_layers']}\n")
        f.write(f"  - Learning Rate: {amp_cfg['learning_rate']}\n")
        f.write(f"  - Epochs: {amp_cfg['epochs']}\n")
        f.write(f"  - Circuit Depth: ~{amp_cfg['n_qubits'] + (amp_cfg['n_layers'] * (amp_cfg['n_qubits'] * 2 + amp_cfg['n_qubits']))}\n\n")
        
        # Performance Results
        f.write("PERFORMANCE RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<30} {'Angle Encoding':<20} {'Amplitude Encoding':<20}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Training RMSE':<30} {angle_res['train_rmse']:<20.4f} {amp_res['train_rmse']:<20.4f}\n")
        f.write(f"{'Test RMSE':<30} {angle_res['test_rmse']:<20.4f} {amp_res['test_rmse']:<20.4f}\n")
        f.write(f"{'R² Score':<30} {angle_res['r2_score']:<20.4f} {amp_res['r2_score']:<20.4f}\n")
        f.write(f"{'Training Time (seconds)':<30} {angle_res['training_time']:<20.2f} {amp_res['training_time']:<20.2f}\n")
        f.write(f"{'Epochs Completed':<30} {angle_res['epochs_completed']:<20} {amp_res['epochs_completed']:<20}\n\n")
        
        # Resource Usage
        f.write("RESOURCE REQUIREMENTS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Resource':<30} {'Angle Encoding':<20} {'Amplitude Encoding':<20}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Qubits':<30} {angle_cfg['n_qubits']:<20} {amp_cfg['n_qubits']:<20}\n")
        f.write(f"{'Parameters':<30} {'33':<20} {'13':<20}\n")
        f.write(f"{'Qubit Efficiency':<30} {'1:1':<20} {'8:3 (~2.67:1)':<20}\n\n")
        
        # Analysis
        f.write("ANALYSIS AND CONCLUSIONS\n")
        f.write("-"*70 + "\n")
        
        # Determine winner
        angle_better_rmse = angle_res['test_rmse'] < amp_res['test_rmse']
        angle_better_r2 = angle_res['r2_score'] > amp_res['r2_score']
        
        f.write("\nPerformance Comparison:\n")
        if angle_better_rmse and angle_better_r2:
            f.write("  Angle Encoding achieves superior performance on both RMSE and R² metrics.\n")
        elif amp_res['test_rmse'] < angle_res['test_rmse'] and amp_res['r2_score'] > angle_res['r2_score']:
            f.write("  Amplitude Encoding achieves superior performance on both RMSE and R² metrics.\n")
        else:
            f.write("  Both methods show comparable performance trade-offs.\n")
            f.write(f"  - RMSE: {'Angle' if angle_better_rmse else 'Amplitude'} is better\n")
            f.write(f"  - R²: {'Angle' if angle_better_r2 else 'Amplitude'} is better\n")
        
        f.write("\nResource Efficiency:\n")
        f.write(f"  Amplitude Encoding uses {amp_cfg['n_qubits']} qubits vs Angle's {angle_cfg['n_qubits']} qubits.\n")
        f.write("  This represents a qubit reduction of "
                f"{((angle_cfg['n_qubits']-amp_cfg['n_qubits'])/angle_cfg['n_qubits']*100):.1f}%.\n")
        
        f.write("\nRecommendations:\n")
        f.write("  1. For optimal accuracy on this dataset, use ")
        f.write("Angle Encoding\n" if angle_better_rmse else "Amplitude Encoding\n")
        f.write("  2. For resource-constrained scenarios, Amplitude Encoding is more efficient\n")
        f.write("  3. Consider hybrid approaches combining both encoding methods\n")
        f.write("  4. Test on larger datasets (QM9, CHEMBL) for general applicability\n\n")
        
        # Future Work
        f.write("FUTURE WORK\n")
        f.write("-"*70 + "\n")
        f.write("  - Implement phase encoding and other advanced encoding techniques\n")
        f.write("  - Deploy on cloud quantum providers (IBM Quantum, Amazon Braket)\n")
        f.write("  - Test on larger molecular datasets\n")
        f.write("  - Optimize circuit topology for QAOA-based approaches\n")
        f.write("  - Implement quantum classifiers for drug activity prediction\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Report saved to {report_path}")

if __name__ == "__main__":
    generate_report()
