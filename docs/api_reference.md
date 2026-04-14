# API Reference

## `src.data_preprocessing`

- `load_delaney_dataset(data_path='data/raw/delaney-processed.csv')`
- `preprocess_data(df, feature_cols, target_col)`
- `prepare_train_test_data(X, y, test_size=0.2, random_state=42)`

## `src.angle_encoding_model`

- `AngleEncodingModel(n_qubits=8, n_layers=2, device='default.qubit')`

## `src.amplitude_encoding_model`

- `AmplitudeEncodingModel(n_qubits=3, n_layers=2, device='default.qubit')`

## `src.training`

- `train_model(model, X_train, y_train, X_test, y_test, epochs=50, lr=0.01, verbose=True)`
- `save_results(results, model_name, save_dir='results/')`
