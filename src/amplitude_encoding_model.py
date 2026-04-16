import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class AmplitudeVQC(nn.Module):
    """
    Hybrid Quantum Classifier using Amplitude Encoding + Data Re-Uploading.

    Amplitude encoding packs 2^n features into n qubits.
    With 3 qubits we encode 8 features as quantum amplitudes.
    Then 4 re-uploading rounds of variational layers expand expressiveness.

    Architecture:
    - 3 qubits (encodes 8 PCA features as amplitudes)
    - 4 re-uploading rounds: AmplitudeEmbed -> StronglyEntangling, repeated
    - Measure 3 qubits -> Linear(3, 3) -> output

    Classical post-processing uses the 1024-bit fingerprints
    projected to 16 dims to supplement the quantum answer.
    """
    def __init__(self, n_qubits=3, n_rounds=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_rounds = n_rounds
        self.n_features = 2 ** n_qubits   # = 8

        dev = qml.device('lightning.qubit', wires=n_qubits)

        @qml.transforms.broadcast_expand
        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, params):
            for r in range(n_rounds):
                qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
                qml.StronglyEntanglingLayers(params[r], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"params": (n_rounds, 1, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Small classical supplement using raw fingerprints
        self.fp_proj = nn.Sequential(
            nn.Linear(1024, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )

        # Output: quantum (3) + fp projection (16) = 19
        self.output = nn.Sequential(
            nn.LayerNorm(n_qubits + 16),
            nn.Linear(n_qubits + 16, 3)
        )

    def forward(self, x_pca, x_full):
        if x_pca.dim() == 1:
            x_pca = x_pca.unsqueeze(0)
            x_full = x_full.unsqueeze(0)

        # Pad PCA to 2^n_qubits
        if x_pca.shape[1] < self.n_features:
            x_pad = F.pad(x_pca, (0, self.n_features - x_pca.shape[1]))
        else:
            x_pad = x_pca[:, :self.n_features]

        q_out = self.qlayer(x_pad)
        fp_out = self.fp_proj(x_full)
        combined = torch.cat([q_out, fp_out], dim=1)
        return self.output(combined)


AmplitudeEncodingModel = AmplitudeVQC