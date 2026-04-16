import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_euos_dataset, prepare_train_test_data

def main():
    X, X_full, y, feature_cols, class_weights = load_euos_dataset()
    data = prepare_train_test_data(X, X_full, y)
    
    # Let's train a Random Forest on the full 1024-dim features
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(data['X_full_train'], data['y_train'])
    
    # Predict and evaluate
    preds = rf.predict(data['X_full_test'])
    acc = accuracy_score(data['y_test'], preds)
    f1 = f1_score(data['y_test'], preds, average='macro')
    
    print(f"Random Forest (1024-dim) - Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%")
    
    # Let's train on the PCA features
    rf_pca = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_pca.fit(data['X_train'], data['y_train'])
    
    preds_pca = rf_pca.predict(data['X_test'])
    acc_pca = accuracy_score(data['y_test'], preds_pca)
    f1_pca = f1_score(data['y_test'], preds_pca, average='macro')
    
    print(f"Random Forest (PCA-dim) - Accuracy: {acc_pca*100:.2f}%, F1: {f1_pca*100:.2f}%")

if __name__ == '__main__':
    main()
