import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
from policies import RawPQCPolicy, SoftmaxPQCPolicy

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class PQCTrainer:
    def __init__(self, env_name, policy_type='raw', n_qubits=4, depth=1, 
                 learning_rate=0.01, gamma=0.99, beta=1.0):
        """
        Initialize PQC trainer
        Args:
            env_name: Name of the Gymnasium environment
            policy_type: Type of PQC policy ('raw' or 'softmax')
            n_qubits: Number of qubits in the circuit
            depth: Depth of encoding layers
            learning_rate: Learning rate for policy optimization
            gamma: Discount factor
            beta: Inverse temperature (only for softmax policy)
        """
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        
        # Initialize policy
        if policy_type == 'raw':
            self.policy = RawPQCPolicy(n_qubits, self.n_actions, depth)
        else:
            self.policy = SoftmaxPQCPolicy(n_qubits, self.n_actions, depth, beta)
        
        # Initialize value network
        self.value_net = ValueNetwork(self.state_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
    
    def collect_episode(self):
        """
        Collect a single episode following the current policy
        Returns:
            List of (state, action, reward) tuples
        """
        state, _ = self.env.reset()
        done = False
        episode = []
        
        while not done:
            # Get action probabilities and sample action
            probs = self.policy(state)
            action = np.random.choice(self.n_actions, p=probs)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def compute_returns(self, rewards):
        """Compute discounted returns for each timestep"""
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def update_value_network(self, states, returns):
        """Update value network to better predict returns"""
        states = torch.FloatTensor(states)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        for _ in range(5):  # Multiple updates for better fitting
            values = self.value_net(states)
            value_loss = nn.MSELoss()(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def train(self, n_episodes=1000, batch_size=16):
        """
        Train the PQC policy using REINFORCE with baseline
        Args:
            n_episodes: Number of episodes to train for
            batch_size: Number of episodes per batch
        """
        episode_rewards = deque(maxlen=100)  # For tracking progress
        best_reward = float('-inf')
        
        print("\nStarting training...")
        print(f"Episodes: {n_episodes}, Batch size: {batch_size}")
        print("=" * 50)
        
        for episode in range(n_episodes):
            # Collect batch of episodes
            total_reward = 0
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print("-" * 30)
            
            print("Collecting episodes for batch...")
            batch_episodes = []
            for b in range(batch_size):
                episodes = self.collect_episode()
                batch_episodes.append(episodes)
                batch_reward = sum(step[2] for step in episodes)
                total_reward += batch_reward
                print(f"  Batch {b + 1}/{batch_size} - Episode Length: {len(episodes)}, Reward: {batch_reward:.2f}")
            
            # Process episodes
            states = []
            actions = []
            returns = []
            
            for episodes in batch_episodes:
                ep_states = np.array([step[0] for step in episodes])
                ep_actions = np.array([step[1] for step in episodes])
                ep_rewards = np.array([step[2] for step in episodes])
                
                ep_returns = self.compute_returns(ep_rewards)
                
                states.extend(ep_states)
                actions.extend(ep_actions)
                returns.extend(ep_returns)
                
                episode_rewards.append(np.sum(ep_rewards))
            
            states = np.array(states)
            actions = np.array(actions)
            returns = np.array(returns)
            
            # Update value network
            print("\nUpdating value network...")
            self.update_value_network(states, returns)
            
            # Compute advantages
            with torch.no_grad():
                values = self.value_net(torch.FloatTensor(states)).numpy().flatten()
            advantages = returns - values
            
            # Compute policy gradients
            print("Computing policy gradients...")
            policy_grad_var = np.zeros_like(self.policy.var_params)
            policy_grad_enc = np.zeros_like(self.policy.enc_params)
            
            if isinstance(self.policy, SoftmaxPQCPolicy):
                policy_grad_weights = np.zeros_like(self.policy.weights)
            
            # Accumulate gradients over batch
            for state, action, advantage in zip(states, actions, advantages):
                if isinstance(self.policy, RawPQCPolicy):
                    grad_var, grad_enc = self.policy.compute_gradient(state, action)
                    policy_grad_var += advantage * grad_var
                    policy_grad_enc += advantage * grad_enc
                else:
                    grad_var, grad_enc, grad_weights = self.policy.compute_gradient(state, action)
                    policy_grad_var += advantage * grad_var
                    policy_grad_enc += advantage * grad_enc
                    policy_grad_weights += advantage * grad_weights
            
            # Update policy parameters
            print("Updating policy parameters...")
            self.policy.var_params += self.learning_rate * policy_grad_var / batch_size
            self.policy.enc_params += self.learning_rate * policy_grad_enc / batch_size
            
            if isinstance(self.policy, SoftmaxPQCPolicy):
                self.policy.weights += self.learning_rate * policy_grad_weights / batch_size
            
            # Print progress
            avg_reward = total_reward / batch_size
            avg_last_100 = np.mean(episode_rewards)
            best_reward = max(best_reward, avg_reward)
            
            print("\nTraining Statistics:")
            print(f"  Average Reward this batch: {avg_reward:.2f}")
            print(f"  Average Reward (last 100): {avg_last_100:.2f}")
            print(f"  Best Average Reward: {best_reward:.2f}")
            print(f"  Average Episode Length: {len(states) // batch_size:.1f}")
        
        self.env.close()
        return episode_rewards 