import torch
import numpy as np
from pqc_architecture import PQC

class BasePQCPolicy:
    def __init__(self, n_qubits, n_actions, depth=1):
        """
        Base class for PQC-based policies
        Args:
            n_qubits: Number of qubits in the circuit
            n_actions: Number of possible actions
            depth: Depth of encoding layers
        """
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.pqc = PQC(n_qubits, depth)
        
        # Initialize parameters
        self.var_params = np.random.uniform(0, 2*np.pi, self.pqc.n_var_params)
        self.enc_params = np.random.uniform(0, 2*np.pi, self.pqc.n_enc_params)
        
    def preprocess_state(self, state):
        """Normalize state to [-1, 1] range"""
        return 2 * (state - np.min(state)) / (np.max(state) - np.min(state)) - 1

class RawPQCPolicy(BasePQCPolicy):
    def __init__(self, n_qubits, n_actions, depth=1):
        super().__init__(n_qubits, n_actions, depth)
        
    def __call__(self, state):
        """
        Compute RAW-PQC policy probabilities
        Args:
            state: Environment state
        Returns:
            Action probabilities
        """
        state = self.preprocess_state(state)
        expectations = self.pqc(state, self.var_params, self.enc_params)
        
        # Convert expectations to probabilities (P_a)
        probs = np.array([(1 + exp) / 2 for exp in expectations])
        
        # Normalize to ensure sum to 1
        return probs[:self.n_actions] / np.sum(probs[:self.n_actions])
    
    def compute_gradient(self, state, action):
        """
        Compute gradient of log policy with respect to parameters
        Args:
            state: Environment state
            action: Taken action
        Returns:
            Gradient for variational and encoding parameters
        """
        state = self.preprocess_state(state)
        probs = self.__call__(state)
        
        grad_var = np.zeros_like(self.var_params)
        grad_enc = np.zeros_like(self.enc_params)
        
        # Compute gradients using parameter-shift rule
        for i in range(len(self.var_params)):
            grad_var[i] = self.pqc.parameter_shift(state, self.var_params, self.enc_params, i, 'var')[action]
        
        for i in range(len(self.enc_params)):
            grad_enc[i] = self.pqc.parameter_shift(state, self.var_params, self.enc_params, i, 'enc')[action]
        
        return grad_var / probs[action], grad_enc / probs[action]

class SoftmaxPQCPolicy(BasePQCPolicy):
    def __init__(self, n_qubits, n_actions, depth=1, beta=1.0):
        super().__init__(n_qubits, n_actions, depth)
        self.beta = beta  # Inverse temperature parameter
        
        # Initialize observable weights
        self.weights = np.random.normal(0, 0.1, (n_actions, n_qubits))
    
    def compute_observables(self, state):
        """
        Compute observable expectations O_a for each action
        Args:
            state: Environment state
        Returns:
            Observable expectations for each action
        """
        state = self.preprocess_state(state)
        expectations = self.pqc(state, self.var_params, self.enc_params)
        
        # Compute weighted sum of expectations for each action
        observables = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            observables[a] = np.sum(self.weights[a] * expectations)
        
        return observables
    
    def __call__(self, state):
        """
        Compute SOFTMAX-PQC policy probabilities
        Args:
            state: Environment state
        Returns:
            Action probabilities
        """
        observables = self.compute_observables(state)
        logits = self.beta * observables
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def compute_gradient(self, state, action):
        """
        Compute gradient of log policy with respect to parameters
        Args:
            state: Environment state
            action: Taken action
        Returns:
            Gradient for variational parameters, encoding parameters, and weights
        """
        state = self.preprocess_state(state)
        probs = self.__call__(state)
        observables = self.compute_observables(state)
        
        grad_var = np.zeros_like(self.var_params)
        grad_enc = np.zeros_like(self.enc_params)
        grad_weights = np.zeros_like(self.weights)
        
        # Compute gradients using parameter-shift rule
        expectations = self.pqc(state, self.var_params, self.enc_params)
        
        for i in range(len(self.var_params)):
            shifted_exp = self.pqc.parameter_shift(state, self.var_params, self.enc_params, i, 'var')
            for a in range(self.n_actions):
                grad_var[i] += self.beta * (int(a == action) - probs[a]) * np.sum(self.weights[a] * shifted_exp)
        
        for i in range(len(self.enc_params)):
            shifted_exp = self.pqc.parameter_shift(state, self.var_params, self.enc_params, i, 'enc')
            for a in range(self.n_actions):
                grad_enc[i] += self.beta * (int(a == action) - probs[a]) * np.sum(self.weights[a] * shifted_exp)
        
        # Gradient for weights
        for a in range(self.n_actions):
            grad_weights[a] = self.beta * (int(a == action) - probs[a]) * expectations
        
        return grad_var, grad_enc, grad_weights 