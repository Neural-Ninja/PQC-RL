# Quantum Reinforcement Learning with PQC Policies

This project implements Quantum Reinforcement Learning using Parameterized Quantum Circuits (PQCs) as policies. It includes both RAW-PQC and SOFTMAX-PQC implementations and compares their performance on various Gymnasium environments.

## Environments

The implementation supports the following environments:
- CartPole-v1
- MountainCar-v0
- Acrobot-v1

## Features

- Parameterized Quantum Circuit (PQC) implementation using Pennylane
- RAW-PQC and SOFTMAX-PQC policy implementations
- REINFORCE algorithm with value function baseline
- Experiment framework for comparing policy types
- Training visualization with confidence intervals

## Requirements

- Python 3.7+
- PyTorch
- Pennylane
- Gymnasium
- NumPy
- Matplotlib

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run experiments on all environments:
```bash
python main.py
```

This will:
1. Train both RAW-PQC and SOFTMAX-PQC policies on each environment
2. Run multiple independent trials
3. Generate training curves with confidence intervals
4. Save results as PNG files

## Project Structure

- `pqc_architecture.py`: Implementation of the quantum circuit
- `policies.py`: RAW-PQC and SOFTMAX-PQC policy implementations
- `train.py`: Training algorithm implementation
- `main.py`: Experiment runner
- `requirements.txt`: Project dependencies

## Results

The training results for each environment will be saved as PNG files:
- `cartpole-v1_results.png`
- `mountaincar-v0_results.png`
- `acrobot-v1_results.png`

## References

This implementation is based on the quantum reinforcement learning architecture described in recent literature, using parameterized quantum circuits for policy representation in reinforcement learning tasks. 
