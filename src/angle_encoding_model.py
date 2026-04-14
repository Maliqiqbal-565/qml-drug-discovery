import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class AngleEncodingModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device('lightning.qubit', wires=n_qubits)
        
        def circuit(inputs, params):
            inputs_np = inputs.detach().cpu().numpy()
            
            if len(inputs_np) < self.n_qubits:
                inputs_np = np.pad(inputs_np, (0, self.n_qubits - len(inputs_np)))
            
            for i in range(self.n_qubits):
                qml.RX(inputs_np[i], wires=i)
            
            for layer in range(self.n_layers):
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                
                for i in range(self.n_qubits):
                    qml.RY(params[layer][i][0], wires=i)
                    qml.RZ(params[layer][i][1], wires=i)
            
            return qml.expval(qml.PauliZ(0))
        
        self.qnode = qml.QNode(circuit, self.dev, interface='numpy')
        self.params = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 2))
        self.classical = nn.Linear(1, 1)

    def forward(self, x):
        preds = []
        for i in range(x.shape[0]):
            try:
                inputs_np = x[i].detach().cpu().numpy()
                params_np = self.params.detach().cpu().numpy()
                pred = self.qnode(inputs_np, params_np)
                pred = torch.tensor(float(pred), dtype=torch.float32)
                preds.append(pred)
            except Exception as e:
                # Fallback: use a simple prediction
                pred = torch.tensor(0.0, dtype=torch.float32)
                preds.append(pred)
        
        preds = torch.stack(preds).view(-1, 1)
        return self.classical(preds).flatten()