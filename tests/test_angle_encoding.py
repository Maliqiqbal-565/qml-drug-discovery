import torch
from src.angle_encoding_model import AngleEncodingModel

def test_angle_encoding_forward():
    model = AngleEncodingModel(n_qubits=8, n_rounds=2)
    x_pca = torch.rand((2, 8), dtype=torch.float32)
    x_full = torch.rand((2, 1024), dtype=torch.float32)
    output = model(x_pca, x_full)

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
