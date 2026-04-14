import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os


def load_delaney_dataset(data_path='data/raw/delaney-processed.csv'):
    """Load and return the Delaney solubility dataset"""
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Downloading dataset to {data_path}...")
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)
        print("Download complete!")
    else:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset from {data_path}")

    return df


def preprocess_data(df, feature_cols=None, target_col='measured log solubility in mols per litre'):
    """Preprocess the dataset and return feature/target arrays."""
    # Define feature synonyms and common dataset column names
    synonyms = {
        'MolWt': 'Molecular Weight',
        'MW': 'Molecular Weight',
        'MolLogP': 'MolLogP',
        'LogP': 'MolLogP',
        'NumHAcceptors': 'Number of H-Bond Acceptors',
        'NumHDonors': 'Number of H-Bond Donors',
        'NumRotatableBonds': 'Number of Rotatable Bonds',
        'NumHeteroatoms': 'NumHeteroatoms',
        'HeavyAtomCount': 'HeavyAtomCount',
        'NumSaturatedHeterocycles': 'NumSaturatedHeterocycles',
        'PolarSurfaceArea': 'Polar Surface Area',
        'NumRings': 'Number of Rings',
        'MinDegree': 'Minimum Degree'
    }

    if feature_cols is not None:
        requested = list(feature_cols)
        feature_cols = [col for col in requested if col in df.columns]
        for col in requested:
            if col not in feature_cols and col in synonyms and synonyms[col] in df.columns:
                feature_cols.append(synonyms[col])

    if feature_cols is None or not feature_cols:
        possible_columns = [
            'Molecular Weight',
            'MolWt',
            'MW',
            'MolLogP',
            'LogP',
            'Number of H-Bond Acceptors',
            'Number of H-Bond Donors',
            'NumHAcceptors',
            'NumHDonors',
            'Number of Rotatable Bonds',
            'NumRotatableBonds',
            'Polar Surface Area',
            'NumRings',
            'Number of Rings',
            'Minimum Degree',
            'MinDegree',
            'NumHeteroatoms',
            'HeavyAtomCount',
            'NumSaturatedHeterocycles'
        ]

        feature_cols = [col for col in possible_columns if col in df.columns]

        if not feature_cols:
            feature_cols = [
                col for col in df.columns
                if col != target_col and df[col].dtype in ['float64', 'float32', 'int64', 'int32']
            ]

        feature_cols = feature_cols[:8]
        print(f"Auto-detected features: {feature_cols}")
    else:
        print(f"Using provided feature columns: {feature_cols}")

    if target_col not in df.columns:
        alt_targets = [
            'measured log solubility in mols per litre',
            'ESOL predicted log solubility in mols per litre',
            'solubility',
            'exp',
            'log solubility',
            'target'
        ]
        for alt in alt_targets:
            if alt in df.columns:
                target_col = alt
                print(f"Using target column: {target_col}")
                break
        else:
            target_col = df.columns[-1]
            print(f"Using last column as target: {target_col}")

    if not feature_cols:
        raise ValueError("No feature columns could be detected. Please provide feature_cols explicitly.")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    return X, y, feature_cols


def prepare_train_test_data(X, y, test_size=0.2, random_state=42):
    """Split and scale features and target data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return {
        'X_train': X_train.astype(np.float32),
        'X_test': X_test.astype(np.float32),
        'y_train': y_train.astype(np.float32),
        'y_test': y_test.astype(np.float32),
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
    }


def preprocess_single_molecule(smiles, feature_scaler=None, feature_cols=None):
    """Preprocess a single molecule SMILES string and return scaled features."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
        
    desc_mapping = {
        'Molecular Weight': 'MolWt',
        'MolWt': 'MolWt',
        'MW': 'MolWt',
        'MolLogP': 'MolLogP',
        'LogP': 'MolLogP',
        'Number of H-Bond Acceptors': 'NumHAcceptors',
        'NumHAcceptors': 'NumHAcceptors',
        'Number of H-Bond Donors': 'NumHDonors',
        'NumHDonors': 'NumHDonors',
        'Number of Rotatable Bonds': 'NumRotatableBonds',
        'NumRotatableBonds': 'NumRotatableBonds',
        'NumHeteroatoms': 'NumHeteroatoms',
        'HeavyAtomCount': 'HeavyAtomCount',
        'NumSaturatedHeterocycles': 'NumSaturatedHeterocycles',
    }
    
    features = []
    if feature_cols is None:
        feature_cols = ['Molecular Weight', 'MolLogP', 'Number of H-Bond Acceptors',
                        'Number of H-Bond Donors', 'Number of Rotatable Bonds',
                        'NumHeteroatoms', 'HeavyAtomCount', 'NumSaturatedHeterocycles']
    
    for col in feature_cols:
        desc_name = desc_mapping.get(col, col)
        if hasattr(Descriptors, desc_name):
            val = getattr(Descriptors, desc_name)(mol)
        else:
            val = 0.0
        features.append(val)
        
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    
    if feature_scaler is not None:
        X = feature_scaler.transform(X)
        
    return X.astype(np.float32)
