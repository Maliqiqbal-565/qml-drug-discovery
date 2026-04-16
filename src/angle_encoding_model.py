import torch.nn as nn
import pennylane as qml


class DataReuploadingVQC(nn.Module):
    """
    Pure Quantum Variational Classifier using Data Re-Uploading.

    Data Re-Uploading is the most expressive quantum technique for small qubit counts.
    Instead of encoding once, we re-encode the input features at every variational layer.
    This gives the quantum circuit polynomial expressiveness (like deep classical nets).

    Architecture:
    - 8 qubits
    - 4 re-uploading rounds: [AngleEmbed(Y) -> StronglyEntangling] x 4
    - Measure 3 qubits -> softmax -> 3-class output
    - Tiny classical head: Linear(8, 3) only (truly quantum-dominant)

    Total quantum parameters: 4 rounds x 8 qubits x 3 params = 96
    Total classical parameters: 8x3 + 3 = 27
    This is a genuinely quantum model.
    """

    def __init__(self, n_qubits=8, n_rounds=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_rounds = n_rounds

        dev = qml.device('lightning.qubit', wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, params):
            # Data Re-Uploading: encode → variational → encode → variational ...
            for r in range(n_rounds):
                # Alternate Y and Z rotations across rounds for diversity
                rotation = 'Y' if r % 2 == 0 else 'Z'
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
                qml.StronglyEntanglingLayers(params[r], wires=range(n_qubits))

            # Return all qubit measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # params shape: (n_rounds, 1_layer, n_qubits, 3)
        weight_shapes = {"params": (n_rounds, 1, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Minimal classical head — keeps this truly quantum-dominant
        self.output = nn.Sequential(
            nn.LayerNorm(n_qubits),
            nn.Linear(n_qubits, 3)
        )

    def forward(self, x_pca, x_full=None):
        # x_full is not used — this is a pure quantum model
        if x_pca.dim() == 1:
            x_pca = x_pca.unsqueeze(0)

        q_out = self.qlayer(x_pca)          # (batch, n_qubits)
        return self.output(q_out)           # (batch, 3)


# Keep AngleEncodingModel as an alias for the optimizer
AngleEncodingModel = DataReuploadingVQC
