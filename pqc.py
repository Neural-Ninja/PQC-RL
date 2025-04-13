import pennylane as qml
import torch
import torch.nn as nn

class PQC(nn.Module):
    def __init__(self, n_qubits, n_depth):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_depth = n_depth
        self.s = nn.Parameter(torch.randn(n_qubits))
        self.theta = nn.Parameter(torch.randn(n_qubits))
        self.w = nn.Parameter(torch.randn(n_qubits))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, s):
        @qml.qnode(dev, interface="torch")
        def circuit(s, theta, w):
            for depth in range(self.n_depth):
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(s[i], wires=i)
                    qml.RY(s[i], wires=i)
                for i in range(self.n_qubits):
                    qml.RX(theta[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        raw_expectations = circuit(s, self.theta, self.w)
        
        raw_expectations_tensor = torch.stack(raw_expectations)
        weighted_expectations = raw_expectations_tensor * self.w
        
        softmax_output = torch.softmax(weighted_expectations, dim=0)

        return raw_expectations_tensor, softmax_output

n_qubits = 4
n_depth = 3
dev = qml.device("lightning.gpu", wires=n_qubits)

pqc = PQC(n_qubits=n_qubits, n_depth=n_depth)

s = torch.tensor([0.5, -0.2, 0.1, -0.3], dtype=torch.float32)

raw_pqc_output, softmax_pqc_output = pqc(s)

print("RAW-PQC Output (Expectation of Pauli-Z operators):", raw_pqc_output)
print("SOFTMAX-PQC Output (Softmax of weighted Hermitian operators):", softmax_pqc_output)