import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_delaney_dataset, preprocess_data, prepare_train_test_data
from src.angle_encoding_model import AngleEncodingModel
from src.amplitude_encoding_model import AmplitudeEncodingModel
from src.training import train_model

print("="*60)
print("QUANTUM MACHINE LEARNING - DRUG DISCOVERY")
print("Angle vs Amplitude Encoding Comparison")
print("="*60)

print("\n[1/4] Loading dataset...")
df = load_delaney_dataset()
X, y, features = preprocess_data(df)
X = X[:100]
y = y[:100]
print(f"    Loaded {len(X)} molecules with {len(features)} features")

print("\n[2/4] Preparing data...")
data = prepare_train_test_data(X, y)
print(f"    Training: {len(data['X_train'])} samples")
print(f"    Test: {len(data['X_test'])} samples")

print("\n[3/4] Training Angle Encoding Model...")
angle_model = AngleEncodingModel(n_qubits=4, n_layers=2)
angle_results = train_model(angle_model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=10, verbose=True)

print("\n[4/4] Training Amplitude Encoding Model...")
amp_model = AmplitudeEncodingModel(n_qubits=3, n_layers=2)
amp_results = train_model(amp_model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=10, verbose=True)

print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"{'Metric':<25} {'Angle Encoding':<20} {'Amplitude Encoding':<20}")
print("-"*65)
print(f"{'Test RMSE':<25} {angle_results['test_rmse']:<20.4f} {amp_results['test_rmse']:<20.4f}")
print(f"{'R² Score':<25} {angle_results['r2_score']:<20.4f} {amp_results['r2_score']:<20.4f}")
print(f"{'Training Time':<25} {angle_results['training_time']:<20.2f}s {amp_results['training_time']:<20.2f}s")
print("="*60)

if angle_results['r2_score'] > amp_results['r2_score']:
    print(f"\n🏆 WINNER: Angle Encoding (R² = {angle_results['r2_score']:.4f})")
else:
    print(f"\n🏆 WINNER: Amplitude Encoding (R² = {amp_results['r2_score']:.4f})")
