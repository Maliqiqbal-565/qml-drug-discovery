import torch

from src.angle_encoding_model import AngleEncodingModel


def test_angle_encoding_forward():
    model = AngleEncodingModel(n_qubits=8, n_layers=2)
    x = torch.rand((1, 8), dtype=torch.float32)
    output = model(x)

    assert output.shape == (1,)
    assert torch.isfinite(output).all()
