import pennylane as qml
import torch
import numpy as np

class PQC:
    def __init__(self, n_qubits, depth=1):
        """
        Initialize the Parameterized Quantum Circuit (PQC)
        Args:
            n_qubits: Number of qubits in the circuit
            depth: Depth of the encoding layers (D_enc)
        """
        self.n_qubits = n_qubits
        self.depth = depth
        
        # Calculate number of parameters
        self.n_var_params = 2 * n_qubits  # 2 rotation gates per qubit in variational layer
        self.n_enc_params = 2 * n_qubits * depth  # 2 rotation gates per qubit per encoding layer
        
        # Initialize the quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def circuit(state, var_params, enc_params):
            # Initial Hadamard layer
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # Variational layer U_var(φ_0)
            for i in range(n_qubits):
                qml.RZ(var_params[2*i], wires=i)
                qml.RY(var_params[2*i + 1], wires=i)
            
            # Encoding layers U_enc(s, λ)
            for d in range(depth):
                # Entangling layer
                for i in range(n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
                
                # Rotation layer with state and scaling parameters
                for i in range(n_qubits):
                    qml.RY(state[i] * enc_params[2*i], wires=i)
                    qml.RZ(state[i] * enc_params[2*i + 1], wires=i)
            
            # Final variational layer U_var(φ_1)
            for i in range(n_qubits):
                qml.RZ(var_params[2*i], wires=i)
                qml.RY(var_params[2*i + 1], wires=i)
            
            # Return expectation values for all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = circuit
    
    def __call__(self, state, var_params, enc_params):
        """
        Execute the PQC with given parameters
        Args:
            state: Input state vector
            var_params: Variational parameters φ
            enc_params: Encoding parameters λ
        Returns:
            Quantum measurement expectations
        """
        return np.array(self.circuit(state, var_params, enc_params))
    
    def parameter_shift(self, state, var_params, enc_params, param_index, param_type='var'):
        """
        Compute parameter shift gradient for a given parameter
        Args:
            state: Input state vector
            var_params: Variational parameters φ
            enc_params: Encoding parameters λ
            param_index: Index of parameter to compute gradient for
            param_type: Type of parameter ('var' or 'enc')
        Returns:
            Gradient estimate using parameter-shift rule
        """
        shift = np.pi/2
        params = var_params if param_type == 'var' else enc_params
        
        # Create shifted parameter sets
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[param_index] += shift
        params_minus[param_index] -= shift
        
        # Compute shifted expectations
        if param_type == 'var':
            exp_plus = np.array(self.circuit(state, params_plus, enc_params))
            exp_minus = np.array(self.circuit(state, params_minus, enc_params))
        else:
            exp_plus = np.array(self.circuit(state, var_params, params_plus))
            exp_minus = np.array(self.circuit(state, var_params, params_minus))
        
        return 0.5 * (exp_plus - exp_minus) 