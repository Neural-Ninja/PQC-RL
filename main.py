import numpy as np
from train import PQCTrainer
import matplotlib.pyplot as plt
from collections import defaultdict

def run_experiment(env_name, policy_types=['raw', 'softmax'], n_qubits=4, depth=1,
                  n_episodes=1000, batch_size=16, n_runs=3):
    """
    Run experiment comparing RAW-PQC and SOFTMAX-PQC on a given environment
    Args:
        env_name: Name of the Gymnasium environment
        policy_types: List of policy types to compare
        n_qubits: Number of qubits in the circuit
        depth: Depth of encoding layers
        n_episodes: Number of episodes to train for
        batch_size: Number of episodes per batch
        n_runs: Number of independent runs
    Returns:
        Dictionary containing results for each policy type
    """
    results = defaultdict(list)
    
    print("\n" + "="*70)
    print(f"Starting experiments on {env_name}")
    print(f"Configuration:")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Circuit depth: {depth}")
    print(f"  Episodes per run: {n_episodes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of runs: {n_runs}")
    print("="*70)
    
    for policy_type in policy_types:
        print(f"\n{'-'*70}")
        print(f"Training {policy_type.upper()}-PQC Policy")
        print(f"{'-'*70}")
        
        for run in range(n_runs):
            print(f"\nStarting Run {run + 1}/{n_runs}")
            print(f"{'-'*30}")
            
            # Initialize and train policy
            trainer = PQCTrainer(
                env_name=env_name,
                policy_type=policy_type,
                n_qubits=n_qubits,
                depth=depth,
                learning_rate=0.01,
                gamma=0.99,
                beta=1.0 if policy_type == 'softmax' else None
            )
            
            # Train and collect rewards
            rewards = trainer.train(n_episodes=n_episodes, batch_size=batch_size)
            results[policy_type].append(rewards)
            
            print(f"\nCompleted Run {run + 1}/{n_runs}")
            print(f"Final Average Reward (last 100 episodes): {np.mean(rewards):.2f}")
    
    return results

def plot_results(results, env_name, window_size=100):
    """Plot training curves with confidence intervals"""
    print(f"\nGenerating plots for {env_name}...")
    
    plt.figure(figsize=(10, 6))
    
    colors = {'raw': 'blue', 'softmax': 'red'}
    labels = {'raw': 'RAW-PQC', 'softmax': 'SOFTMAX-PQC'}
    
    for policy_type in results:
        # Convert to numpy array
        rewards = np.array(results[policy_type])
        
        # Calculate mean and std across runs
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        
        # Smooth curves
        smoothed_mean = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_std = np.convolve(std_rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Plot mean and confidence interval
        x = np.arange(len(smoothed_mean))
        plt.plot(x, smoothed_mean, color=colors[policy_type], label=labels[policy_type])
        plt.fill_between(x, 
                        smoothed_mean - smoothed_std, 
                        smoothed_mean + smoothed_std,
                        color=colors[policy_type], alpha=0.2)
    
    plt.title(f'Training Curves on {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    filename = f'{env_name.lower()}_results.png'
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

def main():
    # Environment configurations
    env_configs = [
        {
            'name': 'CartPole-v1',
            'n_qubits': 4,
            'depth': 1,
            'n_episodes': 500,
            'batch_size': 16
        },
        {
            'name': 'MountainCar-v0',
            'n_qubits': 4,
            'depth': 2,
            'n_episodes': 1000,
            'batch_size': 32
        },
        {
            'name': 'Acrobot-v1',
            'n_qubits': 6,
            'depth': 2,
            'n_episodes': 1000,
            'batch_size': 32
        }
    ]
    
    print("\n" + "="*70)
    print("Starting PQC Policy Training Experiments")
    print("="*70)
    
    # Run experiments for each environment
    for i, config in enumerate(env_configs, 1):
        print(f"\nEnvironment {i}/{len(env_configs)}: {config['name']}")
        
        results = run_experiment(
            env_name=config['name'],
            n_qubits=config['n_qubits'],
            depth=config['depth'],
            n_episodes=config['n_episodes'],
            batch_size=config['batch_size']
        )
        
        plot_results(results, config['name'])
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main() 