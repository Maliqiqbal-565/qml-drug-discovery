import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training import train_model
from src.quantum_forest import QuantumForestClassifier
from src.amplitude_encoding_model import AmplitudeEncodingModel
from src.angle_encoding_model import AngleEncodingModel
from src.data_preprocessing import load_euos_dataset, prepare_train_test_data, get_morgan_fingerprint
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from rdkit import Chem
from sklearn.decomposition import PCA

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
    epochs = st.slider("Quantum Warmup Epochs", 1, 5, 1)

    if 'angle_acc' in st.session_state and 'amp_acc' in st.session_state:
        st.markdown("---")
        st.markdown("### 🏆 Live Accuracies")
        st.metric("Angle Encoding", f"{st.session_state.angle_acc:.2f}%")
        st.metric("Amplitude Encoding", f"{st.session_state.amp_acc:.2f}%")

if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.training_done = False
    st.session_state.prediction_history = []

if st.button("🚀 Start Quantum Experiment", type="primary"):
    with st.spinner("Training Premium Hybrid Ensemble... This requires ~2 minutes..."):
        X, X_full, y, feature_cols, class_weights = load_euos_dataset(max_samples=1000)
        data = prepare_train_test_data(X, X_full, y)

        angle_model = AngleEncodingModel(n_qubits=8)
        angle_results = train_model(angle_model, data['X_train'], data['X_full_train'], data['y_train'], data['X_test'],
                                    data['X_full_test'], data['y_test'], epochs=epochs, verbose=False, class_weights=class_weights)

        amplitude_model = AmplitudeEncodingModel(n_qubits=3)
        amplitude_results = train_model(amplitude_model, data['X_train'], data['X_full_train'], data['y_train'],
                                        data['X_test'], data['X_full_test'], data['y_test'], epochs=epochs, verbose=False, class_weights=class_weights)

        # Save encoding info into session state for the sidebar
        st.session_state.angle_acc = angle_results['test_acc'] * 100
        st.session_state.amp_acc = amplitude_results['test_acc'] * 100

        q_forest = QuantumForestClassifier(n_estimators=100, n_qubits=8)
        q_forest.fit(data['X_train'], data['X_full_train'], data['y_train'])

        from sklearn.metrics import accuracy_score, f1_score
        test_preds = q_forest.predict(data['X_test'], data['X_full_test'])
        qf_acc = accuracy_score(data['y_test'], test_preds)
        qf_f1 = f1_score(data['y_test'], test_preds, average='macro', zero_division=0)

        st.session_state.results = {
            'Angle Encoding': angle_results,
            'Amplitude Encoding': amplitude_results,
            'QuantumForest Ensemble': {'test_acc': qf_acc, 'test_f1': qf_f1}
        }

        # Fit a full PCA on the dataset again just to be able to transform single instances
        st.session_state.pca = PCA(n_components=8, random_state=42)
        st.session_state.pca.fit(X_full)
        st.session_state.feature_scaler = data['feature_scaler']

        st.session_state.q_forest = q_forest
        st.session_state.feature_cols = feature_cols

        st.session_state.training_done = True
        st.success("Training Complete!")
        st.balloons()

if st.session_state.training_done and st.session_state.results:
    angle_res = st.session_state.results['Angle Encoding']
    amp_res = st.session_state.results['Amplitude Encoding']
    qf_res = st.session_state.results['QuantumForest Ensemble']

    st.markdown("---")

    st.markdown("## 📊 Encoding Techniques Comparison")
    angle_acc = angle_res['test_acc'] * 100
    amp_acc = amp_res['test_acc'] * 100

    col_g, col_metrics = st.columns([2, 1])

    with col_g:
        fig = go.Figure(data=[
            go.Bar(name='Test Accuracy', x=['Angle Encoding', 'Amplitude Encoding'],
                   y=[angle_acc, amp_acc], marker_color=['#1f77b4', '#ff7f0e'])
        ])
        fig.update_layout(title="Amplitude vs Angle Encoding Accuracy",
                          yaxis_title="Accuracy (%)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.metric("Angle Encoding Accuracy", f"{angle_acc:.2f}%")
        st.metric("Amplitude Encoding Accuracy", f"{amp_acc:.2f}%")

        winner_name = "Angle Encoding" if angle_acc > amp_acc else "Amplitude Encoding"
        winner_acc = max(angle_acc, amp_acc)
        st.success(f"🏆 Winner: **{winner_name}** ({winner_acc:.2f}%)")

    st.markdown("---")
    st.markdown("## 🧪 Predict Custom Molecule")
    st.markdown("Enter a common molecule name (e.g., 'Aspirin', 'Caffeine', 'Benzene') to predict its solubility class (0: Low, 1: Medium, 2: High).")

    mol_input = st.text_input("Molecule Name:")

    if st.button("Predict Solubility", type="primary"):
        if not mol_input:
            st.warning("Please enter a molecule name.")
        else:
            with st.spinner(f"Predicting solubility for {mol_input}..."):
                try:
                    smiles = None
                    mol = Chem.MolFromSmiles(mol_input)
                    if mol is None:
                        api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{mol_input}/property/CanonicalSMILES,IsomericSMILES/JSON"
                        res = requests.get(api_url)
                        if res.status_code == 200:
                            props = res.json().get("PropertyTable", {}).get("Properties", [])
                            if props:
                                p = props[0]
                                fetched_smiles = p.get("CanonicalSMILES") or p.get("SMILES") or p.get(
                                    "IsomericSMILES") or p.get("ConnectivitySMILES")
                                if fetched_smiles:
                                    smiles = fetched_smiles

                        if not smiles:
                            raise ValueError(f"Could not find or interpret SMILES for name '{mol_input}'")
                    else:
                        smiles = mol_input

                    fp = get_morgan_fingerprint(smiles)
                    X_full_input = fp.reshape(1, -1)
                    X_pca_input = st.session_state.pca.transform(X_full_input)
                    X_scaled = st.session_state.feature_scaler.transform(X_pca_input)

                    q_forest = st.session_state.q_forest
                    predicted_class = q_forest.predict(X_scaled.astype(np.float32), X_full_input.astype(np.float32))[0]

                    st.markdown(f"### Prediction for **{mol_input}**")
                    colA, colB = st.columns(2)
                    colA.metric("Predicted Class", f"{predicted_class}")

                    if predicted_class == 0:
                        colB.metric("Solubility", "Low")
                    elif predicted_class == 1:
                        colB.metric("Solubility", "Medium")
                    else:
                        colB.metric("Solubility", "High")

                    st.session_state.prediction_history.append({
                        "Molecule": mol_input,
                        "Predicted Class": predicted_class
                    })

                except Exception as e:
                    st.error(f"Error predicting solubility: {str(e)}")

    if st.session_state.get('prediction_history'):
        st.info("📂 View your full prediction history in the **Comparison** page from the sidebar.")

        st.markdown("---")
        st.markdown("## 🌐 QuantumForest Benchmark vs Pure Quantum")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 QuantumForest Acc", f"{qf_res['test_acc']*100:.2f}%")
        with col2:
            st.metric("🌀 Angle Encoding Best Acc", f"{angle_res['best_accuracy']*100:.2f}%")
        with col3:
            st.metric("💠 Amplitude Encoding Best Acc", f"{amp_res['best_accuracy']*100:.2f}%")

        st.success(f"🌟 Overall Top Model: QuantumForest Ensemble with Accuracy = {qf_res['test_acc']*100:.2f}%")
