# Quantum Machine Learning for Drug Discovery

## 🧪 Project Overview

This project explores **Quantum Machine Learning (QML)** techniques for molecular property prediction using angle and amplitude encoding methods. We compare the performance of both encoding strategies on the ESOL (Delaney) solubility dataset.

### 🎯 Objectives

- Design and implement QML models using Qiskit and PennyLane
- Compare angle encoding vs amplitude encoding for molecular features
- Build a web interface for interactive model comparison
- Evaluate quantum advantage in feature encoding efficiency

### 📊 Team
 **Mohammad Maliq Iqbal Hussain** (25215A0520)
- **Jatla Radhika** (24211A05M6)
- **Katla Ravitrayini** (24211A05R9)
-

**Guide:** Dr. Swathi Mummadi  
**Institution:** B V Raju Institute of Technology - Department of CSE

---

## 📁 Project Structure

```
qml-drug-discovery/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Dependencies
├── ⚙️ config.yaml                 # Configuration parameters
│
├── 📁 data/
│   ├── raw/                       # Original datasets
│   └── processed/                 # Preprocessed data
│
├── 📁 src/                        # Core modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading & preprocessing
│   ├── angle_encoding_model.py    # Angle encoding QML model
│   ├── amplitude_encoding_model.py # Amplitude encoding QML model
│   ├── training.py                # Training functions
│   └── utils.py                   # Helper utilities
│
├── 📁 notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_angle_encoding_analysis.ipynb
│   ├── 03_amplitude_encoding_analysis.ipynb
│   └── 04_comparison_results.ipynb
│
├── 📁 webapp/                     # Streamlit web application
│   ├── app.py                     # Main Streamlit app
│   └── pages/                     # Additional pages
│
├── 📁 scripts/                    # Utility scripts
│   ├── run_experiments.py         # Main experiment runner
│   ├── evaluate_models.py         # Model evaluation
│   └── generate_report.py         # Report generation
│
├── 📁 tests/                      # Unit tests
├── 📁 models/                     # Saved trained models
├── 📁 results/                    # Outputs and results
│   ├── figures/                   # Generated plots
│   ├── reports/                   # Text reports
│   └── logs/                      # Training logs
│
└── 📁 docs/                       # Documentation
```

---

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Clone or navigate to project
cd qml-drug-discovery

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Run the Experiment**
```bash
python scripts/run_experiments.py
```

### 4. **Launch Web Interface**
```bash
streamlit run webapp/app.py
```

Then open your browser to `http://localhost:8501`

---

## 🔬 Quantum Encoding Methods

### **Angle Encoding**
- Encodes classical data as rotation angles
- Direct mapping: classical value → qubit rotation angle
- Better for small feature sets
- Qubits needed: Equal to number of features

```
Input: [x₁, x₂, ..., xₙ]
→ RX(x₁) ⊗ RX(x₂) ⊗ ... ⊗ RX(xₙ)
```

### **Amplitude Encoding**
- Encodes data as quantum state amplitudes
- Requires normalization and padding
- More qubit-efficient: 2^N features with N qubits
- Better for high-dimensional data

```
Input: [a₀, a₁, ..., aₙ₋₁] (normalized)
→ |ψ⟩ = a₀|0⟩ + a₁|1⟩ + ... + aₙ₋₁|n-1⟩
```

---

## 📊 Dataset

**ESOL Dataset (Delaney et al.)**
- **Molecules:** 1,128 compounds
- **Features:** 8 molecular descriptors
  - Molecular Weight (MolWt)
  - Log Partition Coefficient (MolLogP)
  - H-Bond Acceptors (NumHAcceptors)
  - H-Bond Donors (NumHDonors)
  - Rotatable Bonds (NumRotatableBonds)
  - Heteroatoms (NumHeteroatoms)
  - Heavy Atom Count
  - Saturated Heterocycles
- **Target:** Log solubility in mol/L

---

## 🏗️ Key Components

### **Data Preprocessing** (`src/data_preprocessing.py`)
- Loads and preprocesses molecular data
- Handles missing values
- Normalizes features to [0, π] for angle encoding
- Standardizes targets

### **Quantum Models** (`src/angle_encoding_model.py`, `src/amplitude_encoding_model.py`)
- PyTorch-based QML models
- PennyLane quantum circuits
- Trainable rotation layers
- Classical output head

### **Training** (`src/training.py`)
- Adam optimizer
- MSE loss function
- Early stopping with patience
- Metrics calculation (RMSE, R²)

### **Web Interface** (`webapp/app.py`)
- Interactive Streamlit dashboard
- Real-time hyperparameter tuning
- Live visualization of results
- Model comparison metrics

---

## 🎯 Expected Results

| Metric | Angle Encoding | Amplitude Encoding |
|--------|----------------|-------------------|
| **Qubits** | 8 | 3 |
| **Circuit Depth** | ~40 | ~30 |
| **Parameters** | 33 | 13 |
| **Test RMSE** | ~0.8-1.0 | ~0.8-1.0 |
| **R² Score** | ~0.4-0.6 | ~0.4-0.6 |

---

## 📚 Future Enhancements

- [ ] Test on larger datasets (QM9, CHEMBL)
- [ ] Implement phase encoding
- [ ] Deploy on cloud quantum providers (IBM Quantum, Amazon Braket)
- [ ] Add variational classifier for drug activity prediction
- [ ] Optimize circuit topology for QAOA approach
- [ ] Hybrid classical-quantum optimization

---

## 📖 References

1. Tudisco, A. (2022). "Encoding techniques for Quantum Machine Learning"
2. Rath, M., & Date, H. (2024). "Quantum data encoding: a comparative analysis." EPJ Quantum Technology
3. Schuld, M., & Killoran, N. (2022). "Quantum machine learning for chemistry." Nature Chemistry
4. Henderson, M., Shakya, S., Pradhan, S., & Cook, T. (2020). "Quanvolutional neural networks"

---

## 🔧 Configuration

Edit `config.yaml` to adjust:
- Quantum parameters (qubits, layers, learning rate)
- Data preprocessing settings
- Training hyperparameters
- Output directory paths

---

## 💻 System Requirements

- Python 3.8+
- PyTorch 2.0+
- Qiskit 1.0+
- PennyLane 0.35+
- 4GB+ RAM
- ~2GB disk space for datasets and models

---

## ⚠️ Troubleshooting

### **Import Errors**
```bash
pip install --upgrade qiskit pennylane torch
```

### **Streamlit Not Loading**
```bash
pip install --upgrade streamlit
streamlit run webapp/app.py --logger.level=debug
```

### **Memory Issues**
- Reduce batch size in `config.yaml`
- Use fewer epochs for testing
- Reduce feature dimensions

---

## 📝 License

This project is for educational purposes at B V Raju Institute of Technology.

---

## 🤝 Contributing

For modifications or improvements:
1. Create a new branch
2. Make changes in your feature branch
3. Test thoroughly
4. Submit pull request with documentation

---

**Last Updated:** April 2026  
**Status:** Active Development
