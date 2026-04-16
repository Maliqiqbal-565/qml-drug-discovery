import numpy as np
import pandas as pd
from src.data_preprocessing import get_morgan_fingerprint, prepare_train_test_data

def test_get_morgan_fingerprint():
    # Simple test to check fingerprint generator
    fp = get_morgan_fingerprint("C1=CC=CC=C1")  # Benzene
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (1024,)

def test_prepare_train_test_data():
    X = np.random.rand(60, 8)
    X_full = np.random.rand(60, 1024)
    # Give it evenly distributed 3 classes
    y = np.array([0, 1, 2] * 20)
    
    data = prepare_train_test_data(X, X_full, y, test_size=0.4, random_state=42)

    assert 'X_train' in data
    assert 'X_test' in data
    assert 'X_full_train' in data
    assert 'X_full_test' in data
    assert 'y_train' in data
    assert 'y_test' in data
    assert 'feature_scaler' in data
    
    # The number of samples in train might be balanced via SMOTE, so we check columns and that elements exist
    assert data['X_train'].shape[1] == 8
    assert data['X_train'].shape[0] > 0
    assert data['X_full_test'].shape[1] == 1024
    assert data['y_test'].shape[0] > 0
