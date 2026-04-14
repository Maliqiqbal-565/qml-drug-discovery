import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def calculate_metrics(y_true, y_pred):
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def plot_training_history(losses, model_name, save_dir='results/figures'):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses['train'], label='Training Loss')
    plt.plot(losses['test'], label='Test Loss')
    plt.legend()
    plt.title(f"{model_name} Training History")
    
    path = os.path.join(save_dir, f"{model_name}_loss.png")
    plt.savefig(path)
    plt.close()

def plot_predictions(y_test, y_pred, model_name, save_dir='results/figures'):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure()
    plt.scatter(y_test, y_pred)
    
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val])
    
    path = os.path.join(save_dir, f"{model_name}_pred.png")
    plt.savefig(path)
    plt.close()

def plot_residuals(y_test, y_pred, model_name, save_dir='results/figures'):
    os.makedirs(save_dir, exist_ok=True)
    
    residuals = y_test - y_pred
    
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    
    path = os.path.join(save_dir, f"{model_name}_residuals.png")
    plt.savefig(path)
    plt.close()

def create_comparison_report(results_dict, save_path='results/report.txt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        for name, res in results_dict.items():
            f.write(f"{name}: RMSE={res['test_rmse']:.4f}, R2={res['r2_score']:.4f}\n")

if __name__ == "__main__":
    print("Utils module working fine ✅")