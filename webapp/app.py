import streamlit as st
import sys
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import torch
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import load_delaney_dataset, preprocess_data, prepare_train_test_data
from src.angle_encoding_model import AngleEncodingModel
from src.amplitude_encoding_model import AmplitudeEncodingModel
from src.training import train_model

st.set_page_config(page_title="QuantumML Drug Discovery", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

st.markdown('<h1 class="main-header">⚛️ Quantum Nexus: Molecular Drug Discovery</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    epochs = st.slider("Training Epochs", 5, 20, 10)
    learning_rate = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05], value=0.01)

if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.training_done = False
    st.session_state.prediction_history = []

if st.button("🚀 Start Quantum Experiment", type="primary"):
    with st.spinner("Training quantum models... This may take 1-2 minutes..."):
        df = load_delaney_dataset()
        X, y, feature_cols = preprocess_data(df)
        X = X[:100]
        y = y[:100]
        data = prepare_train_test_data(X, y)
        
        angle_model = AngleEncodingModel(n_qubits=4, n_layers=2)
        angle_results = train_model(angle_model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=epochs, lr=learning_rate, verbose=False)
        
        amp_model = AmplitudeEncodingModel(n_qubits=3, n_layers=2)
        amp_results = train_model(amp_model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=epochs, lr=learning_rate, verbose=False)
        
        st.session_state.results = {'Angle Encoding': angle_results, 'Amplitude Encoding': amp_results}
        st.session_state.angle_model = angle_model
        st.session_state.amp_model = amp_model
        st.session_state.feature_scaler = data['feature_scaler']
        st.session_state.target_scaler = data['target_scaler']
        st.session_state.feature_cols = feature_cols
        
        st.session_state.training_done = True
        st.success("Training Complete!")
        st.balloons()

if st.session_state.training_done and st.session_state.results:
    angle_res = st.session_state.results['Angle Encoding']
    amp_res = st.session_state.results['Amplitude Encoding']
    
    st.markdown("---")
    st.markdown("## 🧪 Predict Custom Molecule")
    st.markdown("Enter a common molecule name (e.g., 'Aspirin', 'Caffeine', 'Benzene') to predict its solubility.")
    
    mol_input = st.text_input("Molecule Name:")
    
    if st.button("Predict Solubility", type="primary"):
        if not mol_input:
            st.warning("Please enter a molecule name.")
        else:
            with st.spinner(f"Predicting solubility for {mol_input}..."):
                try:
                    smiles = mol_input
                    mol = Chem.MolFromSmiles(mol_input)
                    if mol is None:
                        # Try PubChem API
                        api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{mol_input}/property/CanonicalSMILES,IsomericSMILES/JSON"
                        res = requests.get(api_url)
                        if res.status_code == 200:
                            props = res.json().get("PropertyTable", {}).get("Properties", [])
                            if props:
                                p = props[0]
                                fetched_smiles = p.get("CanonicalSMILES") or p.get("SMILES") or p.get("IsomericSMILES") or p.get("ConnectivitySMILES")
                                if fetched_smiles:
                                    smiles = fetched_smiles
                        
                        if smiles == mol_input:
                            raise ValueError(f"Could not find or interpret SMILES for name '{mol_input}'")
                    
                    from src.data_preprocessing import preprocess_single_molecule
                    X_input = preprocess_single_molecule(smiles, feature_scaler=st.session_state.feature_scaler, feature_cols=st.session_state.feature_cols)
                    X_tensor = torch.tensor(X_input, dtype=torch.float32)
                    
                    angle_model = st.session_state.angle_model
                    amp_model = st.session_state.amp_model
                    angle_model.eval()
                    amp_model.eval()
                    
                    with torch.no_grad():
                        angle_pred_scaled = angle_model(X_tensor).numpy()
                        amp_pred_scaled = amp_model(X_tensor).numpy()
                    
                    # Inverse transform
                    angle_pred = st.session_state.target_scaler.inverse_transform(angle_pred_scaled.reshape(-1, 1)).flatten()[0]
                    amp_pred = st.session_state.target_scaler.inverse_transform(amp_pred_scaled.reshape(-1, 1)).flatten()[0]
                    
                    st.markdown(f"### Prediction for **{mol_input}**")
                    colA, colB = st.columns(2)
                    colA.metric("Angle Encoding Prediction", f"{angle_pred:.4f} mol/L")
                    colB.metric("Amplitude Encoding Prediction", f"{amp_pred:.4f} mol/L")
                    
                    st.session_state.prediction_history.append({
                        "Molecule": mol_input,
                        "Angle Encoding (mol/L)": round(float(angle_pred), 4),
                        "Amplitude Encoding (mol/L)": round(float(amp_pred), 4)
                    })
                    
                except Exception as e:
                    st.error(f"Error predicting solubility: {str(e)}")

    if st.session_state.get('prediction_history'):
        st.info("📂 View your full prediction history in the **Comparison** page from the sidebar.")

        st.markdown("---")
        st.markdown("## 📊 Model Training Benchmark Highlights")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Angle R² Score", f"{angle_res['r2_score']:.4f}")
            st.metric("Angle RMSE", f"{angle_res['test_rmse']:.4f}")
        with col2:
            st.metric("🌀 Amplitude R² Score", f"{amp_res['r2_score']:.4f}")
            st.metric("Amplitude RMSE", f"{amp_res['test_rmse']:.4f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=angle_res['losses']['train'], name='Angle Train', line=dict(color='#FF6B6B', width=2)))
        fig.add_trace(go.Scatter(y=amp_res['losses']['train'], name='Amplitude Train', line=dict(color='#4ECDC4', width=2)))
        fig.update_layout(title="Training Loss Comparison", xaxis_title="Epoch", yaxis_title="Loss", height=500, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        if angle_res['r2_score'] > amp_res['r2_score']:
            st.success(f"🏆 Winner: Angle Encoding with R² = {angle_res['r2_score']:.4f}")
        else:
            st.success(f"🏆 Winner: Amplitude Encoding with R² = {amp_res['r2_score']:.4f}")
