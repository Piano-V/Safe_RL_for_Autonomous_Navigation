import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Standard experience replay buffer storing state transitions, actions, 
    rewards, next states, safety cost metrics, and termination flags.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, cost, done):
        """Saves a single experience transition to the cyclic buffer."""
        self.buffer.append((state, action, reward, next_state, cost, done))
    
    def sample(self, batch_size):
        """Samples a uniform batch of transitions for neural network training."""
        state, action, reward, next_state, cost, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(cost, dtype=np.float32), np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class CriticNetwork(nn.Module):
    """
    Twin Critic Network mapping (state, action) pairs to Q-values.
    Dual Q-networks help prevent overestimation bias in SAC.
    """
    def __init__(self, obs_dim, action_dim):
        super(CriticNetwork, self).__init__()
        # Twin Q-Network 1
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q1  = nn.Linear(256, 1)
        
        # Twin Q-Network 2
        self.fc3 = nn.Linear(obs_dim + action_dim, 256)
        self.fc4 = nn.Linear(256, 256)
        self.q2  = nn.Linear(256, 1)
        
    def forward(self, state, action):
        """Calculates Q1 and Q2 values for a given state-action batch."""
        xu = torch.cat([state, action], dim=1)
        
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)
        
        x2 = F.relu(self.fc3(xu))
        x2 = F.relu(self.fc4(x2))
        q2 = self.q2(x2)
        
        return q1, q2


class ActorNetwork(nn.Module):
    """
    Gaussian Policy Actor Network parameterizing continuous actions.
    Outputs action means and standard deviations to enable policy exploration.
    """
    def __init__(self, obs_dim, action_dim, max_action=1.0):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        """Computes mean and clamped log standard deviation of the action distribution."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample_action(self, state, reparameterize=True):
        """Samples an action from the policy distribution and squashes it using tanh."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = Normal(mean, std)
        
        if reparameterize:
            x_t = normal.rsample()
        else:
            x_t = normal.sample()
            
        action = torch.tanh(x_t)
        
        # Enforce squashing correction for the entropy term calculation
        log_prob = normal.log_prob(x_t) - torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action * self.max_action, log_prob