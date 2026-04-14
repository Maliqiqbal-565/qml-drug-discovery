import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_preprocessing import preprocess_data, prepare_train_test_data, preprocess_single_molecule


def test_preprocess_data():
    df = pd.DataFrame({
        'MolWt': [100.0, 120.0],
        'MolLogP': [1.2, 0.8],
        'NumHAcceptors': [2, 3],
        'NumHDonors': [1, 0],
        'NumRotatableBonds': [4, 5],
        'NumHeteroatoms': [3, 4],
        'HeavyAtomCount': [10, 12],
        'NumSaturatedHeterocycles': [0, 1],
        'measured log solubility in mols per litre': [ -1.2, -0.7 ]
    })
    feature_cols = ['MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors',
                    'NumRotatableBonds', 'NumHeteroatoms', 'HeavyAtomCount',
                    'NumSaturatedHeterocycles']
    X, y, selected_features = preprocess_data(df, feature_cols, 'measured log solubility in mols per litre')

    assert X.shape == (2, 8)
    assert y.shape == (2,)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert selected_features == feature_cols




def test_preprocess_data_with_dataset_columns():
    df = pd.DataFrame({
        'Compound ID': ['A', 'B'],
        'Molecular Weight': [180.16, 240.31],
        'Number of H-Bond Donors': [1, 2],
        'Number of Rings': [1, 0],
        'Number of Rotatable Bonds': [3, 4],
        'Polar Surface Area': [50.2, 65.1],
        'measured log solubility in mols per litre': [-1.0, -2.3]
    })

    X, y, selected_features = preprocess_data(df)

    assert X.shape[0] == 2
    assert X.shape[1] >= 1
    assert y.shape == (2,)
    assert 'Molecular Weight' in selected_features or 'Number of H-Bond Donors' in selected_features


def test_prepare_train_test_data():
    X = np.array([[0.1, 0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                  [0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    y = np.array([-1.0, -0.8])
    data = prepare_train_test_data(X, y, test_size=0.5, random_state=42)

    assert 'X_train' in data
    assert 'X_test' in data
    assert 'y_train' in data
    assert 'y_test' in data
    assert data['X_train'].shape[1] == 8
    assert data['X_test'].shape[1] == 8
    assert np.isfinite(data['X_train']).all()
    assert np.isfinite(data['y_train']).all()


def test_preprocess_single_molecule_with_feature_cols():
    feature_cols = [
        'Molecular Weight', 'MolLogP', 'Number of H-Bond Acceptors',
        'Number of H-Bond Donors', 'Number of Rotatable Bonds',
        'NumHeteroatoms', 'HeavyAtomCount', 'NumSaturatedHeterocycles'
    ]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(np.zeros((1, 8)))

    X_scaled = preprocess_single_molecule('O', feature_scaler=scaler, feature_cols=feature_cols)
    assert X_scaled.shape == (1, 8)
    assert np.isfinite(X_scaled).all()
