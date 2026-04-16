import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_euos_dataset, prepare_train_test_data
from src.angle_encoding_model import AngleEncodingModel

def main():
    print("Loading data...")
    X, X_full, y, feature_cols, class_weights = load_euos_dataset()
    data = prepare_train_test_data(X, X_full, y)
    
    X_test_t = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test_t = torch.tensor(data['y_test'], dtype=torch.long)
    
    print("Loading model...")
    model = AngleEncodingModel(n_qubits=8)
    model.load_state_dict(torch.load("AngleEncodingModel_best.pt", map_location='cpu'))
    model.eval()
    
    print("Evaluating...")
    calibration_boost = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float32)
    with torch.no_grad():
        test_pred_logits = model(X_test_t, None)
        calibrated_logits = test_pred_logits * calibration_boost
        _, test_pred_class = torch.max(calibrated_logits, 1)
        
    acc = accuracy_score(y_test_t.numpy(), test_pred_class.numpy())
    f1 = f1_score(y_test_t.numpy(), test_pred_class.numpy(), average='macro', zero_division=0)
    
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test_t.numpy(), test_pred_class.numpy(), zero_division=0))

if __name__ == '__main__':
    main()
