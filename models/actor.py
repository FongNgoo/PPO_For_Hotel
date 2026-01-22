# models/actor.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Gaussian Policy for PPO
    π(a|s) = N(μ(s), σ)
    """

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, 1)

        # log_std is a learnable parameter (state-independent)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)

        return mean, std

    def sample(self, state):
        """
        Used during rollout
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        action = dist.sample()
        action = torch.clamp(action, 0.5, 2.0)  # Giả sử range hợp lý cho alpha
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate(self, state, action):
        """
        Used during PPO update
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
