import torch
from src.amplitude_encoding_model import AmplitudeEncodingModel

def test_amplitude_encoding_forward():
    model = AmplitudeEncodingModel(n_qubits=3, n_rounds=2)
    x_pca = torch.rand((2, 8), dtype=torch.float32)
    x_full = torch.rand((2, 1024), dtype=torch.float32)
    output = model(x_pca, x_full)

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
