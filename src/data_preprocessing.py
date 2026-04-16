import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import torch


def get_morgan_fingerprint(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(1024)
        # Radius 2, 1024 bit vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(list(fp), dtype=np.float32)
    except Exception as e:
        return np.zeros(1024, dtype=np.float32)


def load_euos_dataset(data_path='data/raw/dataset/euos-slas/train.csv', max_samples=4000):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Retain natural distribution without arbitrarily isolating majority classes
    if max_samples:
        df = df.sample(n=min(len(df), max_samples), random_state=42)

    print(f"Executing deep Morgan Fingerprinting on {len(df)} discrete samples...")
    features = []
    labels = []

    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = int(row['sol_category'])

        fp = get_morgan_fingerprint(smiles)
        features.append(fp)
        labels.append(label)

    X_fingerprints = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Condense 1024-bit graph representations down to 8 hyper-dense principal components
    print("Condensing 1024 features down to 8 Principal Components via PCA...")
    pca = PCA(n_components=8, random_state=42)
    X = pca.fit_transform(X_fingerprints)

    # Compute dynamic class weights to pass directly into PyTorch CrossEntropyLoss
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weights = np.sqrt(weights)  # Smooth the severe gradient divergence
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Mathematical Class Weights mapped securely: {class_weights}")

    return X, X_fingerprints, y, [f'PCA_{i}' for i in range(8)], class_weights


def prepare_train_test_data(X, X_full, y, test_size=0.2, random_state=42):
    X_train, X_test, X_full_train, X_full_test, y_train, y_test = train_test_split(
        X, X_full, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Scale only the PCA features (for quantum circuit input)
    # X_full (fingerprints) stay as binary [0,1] — no scaling needed
    robust = RobustScaler()
    X_train = robust.fit_transform(X_train)
    X_test = robust.transform(X_test)

    feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    # ── Oversample minority classes in TRAINING set only ────────────────────
    # Use SMOTE to synthesize new samples in the combined feature space
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=random_state)

    # Combine PCA and Full features for SMOTE so the synthetic points are continuous across both spaces
    X_combined = np.hstack((X_train, X_full_train))
    X_combined_bal, y_train_bal = smote.fit_resample(X_combined, y_train)

    # Split back into X_train_bal (PCA dims) and X_full_train_bal (1024 dims)
    pca_dims = X_train.shape[1]
    X_train_bal = X_combined_bal[:, :pca_dims].astype(np.float32)
    X_full_train_bal = X_combined_bal[:, pca_dims:].astype(np.float32)

    # Shuffle the balanced training set
    shuffle_idx = np.random.RandomState(random_state).permutation(len(y_train_bal))
    X_train_bal = X_train_bal[shuffle_idx]
    X_full_train_bal = X_full_train_bal[shuffle_idx]
    y_train_bal = y_train_bal[shuffle_idx]

    print(f"SMOTE Balanced training set: {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")

    return {
        'X_train': X_train_bal,
        'X_test': X_test.astype(np.float32),
        'X_full_train': X_full_train_bal,
        'X_full_test': X_full_test.astype(np.float32),
        'y_train': y_train_bal,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': None,
    }
