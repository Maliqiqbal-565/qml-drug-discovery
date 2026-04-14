# create_all_files.py
import os

# Create directories
directories = [
    'src',
    'webapp',
    'data/raw',
    'data/processed',
    'models',
    'results/figures',
    'results/reports',
    'scripts'
]

for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# File contents
files = {
    'check_columns.py': '''import pandas as pd

# Load the dataset
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
df = pd.read_csv(url)

print("Actual column names in dataset:")
print(df.columns.tolist())
print("\\n" + "="*60)
print("First 5 rows:")
print(df.head())
print("\\n" + "="*60)
print(f"Dataset shape: {df.shape}")
''',

    'src/__init__.py': '# Initialize src package\n',

    'src/data_preprocessing.py': '''import pandas as pd
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
    
    print(f"Dataset columns: {df.columns.tolist()}")
    return df

def get_feature_columns(df):
    """Get available feature columns from the dataset"""
    # These are the actual column names in the ESOL dataset
    feature_cols = []
    
    # Check for typical column names
    possible_features = ['MolWt', 'MW', 'MolLogP', 'LogP', 'NumHAcceptors', 'NumHDonors',
                         'NumRotatableBonds', 'NumHeteroatoms', 'HeavyAtomCount', 'NumSaturatedHeterocycles']
    
    for col in possible_features:
        if col in df.columns:
            feature_cols.append(col)
    
    # If no features found, use all numeric columns except target
    if not feature_cols:
        target_col = 'measured log solubility in mols per litre'
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['float64', 'int64']]
    
    return feature_cols[:8]  # Limit to 8 features

def preprocess_data(df, feature_cols=None, target_col='measured log solubility in mols per litre'):
    """Preprocess features and target"""
    
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
        print(f"Auto-detected feature columns: {feature_cols}")
    
    if target_col not in df.columns:
        alt_targets = ['measured log solubility in mols per litre', 'solubility', 'exp']
        for alt in alt_targets:
            if alt in df.columns:
                target_col = alt
                print(f"Using target column: {target_col}")
                break
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    
    return X, y, feature_cols

def prepare_train_test_data(X, y, test_size=0.2, random_state=42):
    """Split and scale data"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    return {
        'X_train': X_train_scaled.astype(np.float32),
        'X_test': X_test_scaled.astype(np.float32),
        'y_train': y_train_scaled.astype(np.float32),
        'y_test': y_test_scaled.astype(np.float32),
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
''',

    'src/angle_encoding_model.py': '''import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class AngleEncodingModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, device='default.qubit'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.dev = qml.device(device, wires=n_qubits)
        self.qnode = self._create_qnode()
        
        self.params = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 2))
        self.classical_layer = nn.Linear(1, 1)
    
    def _circuit(self, params, inputs):
        for i in range(self.n_qubits):
            qml.RX(inputs[i] if i < len(inputs) else 0, wires=i)
        
        for layer in range(self.n_layers):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits-1, 0])
            
            for i in range(self.n_qubits):
                qml.RY(params[layer][i][0], wires=i)
                qml.RZ(params[layer][i][1], wires=i)
        
        return qml.expval(qml.PauliZ(0))
    
    def _create_qnode(self):
        return qml.QNode(self._circuit, self.dev, interface='torch')
    
    def forward(self, x):
        predictions = []
        for sample in x:
            sample_np = sample.detach().cpu().numpy()
            # Pad or truncate to match n_qubits
            if len(sample_np) < self.n_qubits:
                sample_np = np.pad(sample_np, (0, self.n_qubits - len(sample_np)))
            else:
                sample_np = sample_np[:self.n_qubits]
            result = self.qnode(self.params, sample_np)
            predictions.append(result)
        
        predictions = torch.tensor(predictions, dtype=torch.float32).view(-1, 1)
        return self.classical_layer(predictions).flatten()
''',

    'src/amplitude_encoding_model.py': '''import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class AmplitudeEncodingModel(nn.Module):
    def __init__(self, n_qubits=3, n_layers=2, device='default.qubit'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = 2 ** n_qubits
        
        self.dev = qml.device(device, wires=n_qubits)
        self.qnode = self._create_qnode()
        
        self.params = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 2))
        self.classical_layer = nn.Linear(1, 1)
    
    def _circuit(self, params, inputs):
        inputs_np = inputs.detach().cpu().numpy() if hasattr(inputs, 'detach') else inputs
        
        if len(inputs_np) < self.n_features:
            inputs_np = np.pad(inputs_np, (0, self.n_features - len(inputs_np)))
        else:
            inputs_np = inputs_np[:self.n_features]
        
        norm = np.linalg.norm(inputs_np)
        if norm > 0:
            inputs_np = inputs_np / norm
        
        qml.AmplitudeEmbedding(features=inputs_np, wires=range(self.n_qubits), normalize=False)
        
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(params[layer][i][0], wires=i)
                qml.RZ(params[layer][i][1], wires=i)
            
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits-1, 0])
        
        return qml.expval(qml.PauliZ(0))
    
    def _create_qnode(self):
        return qml.QNode(self._circuit, self.dev, interface='torch')
    
    def forward(self, x):
        predictions = []
        for sample in x:
            result = self.qnode(self.params, sample)
            predictions.append(result)
        
        predictions = torch.tensor(predictions, dtype=torch.float32).view(-1, 1)
        return self.classical_layer(predictions).flatten()
''',

    'src/training.py': '''import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=20, lr=0.01, verbose=True):
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    losses = {'train': [], 'test': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train_tensor)
        train_loss = loss_fn(train_pred, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = loss_fn(test_pred, y_test_tensor)
        
        losses['train'].append(train_loss.item())
        losses['test'].append(test_loss.item())
        
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Test Loss = {test_loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        train_pred_final = model(X_train_tensor).numpy()
        test_pred_final = model(X_test_tensor).numpy()
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred_final))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred_final))
    r2 = r2_score(y_test, test_pred_final)
    
    results = {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'r2_score': float(r2),
        'training_time': training_time,
        'epochs_completed': epoch + 1,
        'losses': losses,
        'predictions': {
            'train': train_pred_final.tolist(),
            'test': test_pred_final.tolist()
        }
    }
    
    return results
''',

    'webapp/app.py': '''import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import load_delaney_dataset, preprocess_data, prepare_train_test_data
from src.angle_encoding_model import AngleEncodingModel
from src.amplitude_encoding_model import AmplitudeEncodingModel
from src.training import train_model

import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="QML Drug Discovery", layout="wide")

st.title("🧪 Quantum Machine Learning for Drug Discovery")
st.markdown("### Comparing Angle vs Amplitude Encoding")

with st.sidebar:
    st.header("⚙️ Configuration")
    encoding_option = st.selectbox("Select Encoding Method", ["Compare Both", "Angle Encoding Only", "Amplitude Encoding Only"])
    epochs = st.slider("Training Epochs", 5, 20, 10)
    learning_rate = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05], value=0.01)

if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.training_done = False

if st.button("🚀 Start Training", type="primary"):
    with st.spinner("Loading data and training models..."):
        df = load_delaney_dataset()
        X, y, feature_cols = preprocess_data(df)
        data = prepare_train_test_data(X, y)
        
        results_dict = {}
        
        if encoding_option in ["Angle Encoding Only", "Compare Both"]:
            st.info("Training Angle Encoding...")
            n_qubits = min(len(feature_cols), 4)
            model = AngleEncodingModel(n_qubits=n_qubits, n_layers=2)
            results = train_model(model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=epochs, lr=learning_rate, verbose=False)
            results_dict['Angle Encoding'] = results
            st.success("Angle Encoding complete!")
        
        if encoding_option in ["Amplitude Encoding Only", "Compare Both"]:
            st.info("Training Amplitude Encoding...")
            model = AmplitudeEncodingModel(n_qubits=3, n_layers=2)
            results = train_model(model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=epochs, lr=learning_rate, verbose=False)
            results_dict['Amplitude Encoding'] = results
            st.success("Amplitude Encoding complete!")
        
        st.session_state.results = results_dict
        st.session_state.training_done = True
        st.success("Training complete! Check the Results tab.")

if st.session_state.training_done and st.session_state.results:
    st.subheader("📊 Results Comparison")
    
    metrics_data = []
    for model_name, results in st.session_state.results.items():
        metrics_data.append({
            'Model': model_name,
            'Test RMSE': f"{results['test_rmse']:.4f}",
            'R² Score': f"{results['r2_score']:.4f}",
            'Training Time': f"{results['training_time']:.2f}s"
        })
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    st.subheader("Training Loss Comparison")
    fig = go.Figure()
    for model_name, results in st.session_state.results.items():
        fig.add_trace(go.Scatter(y=results['losses']['train'], name=f"{model_name} - Train", mode='lines'))
        fig.add_trace(go.Scatter(y=results['losses']['test'], name=f"{model_name} - Test", mode='lines', line=dict(dash='dash')))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=500)
    st.plotly_chart(fig, use_container_width=True)
''',

    'scripts/run_experiments.py': '''import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_delaney_dataset, preprocess_data, prepare_train_test_data
from src.angle_encoding_model import AngleEncodingModel
from src.amplitude_encoding_model import AmplitudeEncodingModel
from src.training import train_model

def main():
    print("="*60)
    print("QUANTUM ML - DRUG DISCOVERY")
    print("="*60)
    
    df = load_delaney_dataset()
    X, y, feature_cols = preprocess_data(df)
    print(f"Features: {len(feature_cols)}")
    
    data = prepare_train_test_data(X, y)
    print(f"Training: {len(data['X_train'])} samples, Test: {len(data['X_test'])} samples")
    
    n_qubits = min(len(feature_cols), 4)
    print(f"\\nTraining Angle Encoding ({n_qubits} qubits)...")
    angle_model = AngleEncodingModel(n_qubits=n_qubits)
    angle_results = train_model(angle_model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=10, verbose=True)
    
    print(f"\\nTraining Amplitude Encoding (3 qubits)...")
    amp_model = AmplitudeEncodingModel(n_qubits=3)
    amp_results = train_model(amp_model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=10, verbose=True)
    
    print("\\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Angle Encoding - RMSE: {angle_results['test_rmse']:.4f}, R²: {angle_results['r2_score']:.4f}")
    print(f"Amplitude Encoding - RMSE: {amp_results['test_rmse']:.4f}, R²: {amp_results['r2_score']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
'''
}

# Create all files
for file_path, content in files.items():
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created file: {file_path}")

print("\n" + "="*60)
print("All files created successfully!")
print("="*60)
print("\nNext steps:")
print("1. Run: python check_columns.py")
print("2. Run: python scripts/run_experiments.py")
print("3. Run: streamlit run webapp/app.py")