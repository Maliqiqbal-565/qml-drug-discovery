import torch

from src.amplitude_encoding_model import AmplitudeEncodingModel


def test_amplitude_encoding_forward():
    model = AmplitudeEncodingModel(n_qubits=3, n_layers=2)
    x = torch.rand((1, 8), dtype=torch.float32)
    output = model(x)

    assert output.shape == (1,)
    assert torch.isfinite(output).all()
