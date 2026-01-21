# models/actor_critic.py
import torch
import torch.nn as nn

from models.actor import Actor
from models.critic import Critic


class ActorCritic(nn.Module):
    """
    Actor-Critic wrapper for PPO

    - Actor: πθ(a|s)
    - Critic: Vφ(s)
    """

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()

        self.actor = Actor(state_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

    def act(self, state):
        """
        Used during rollout collection (no gradient needed)

        Returns:
        - action: sampled from πθ(a|s)
        - log_prob: log πθ(a|s)
        - value: Vφ(s)
        - entropy: H(πθ(.|s))
        """
        with torch.no_grad():
            action, log_prob, entropy = self.actor.sample(state)
            value = self.critic(state)

        return action, log_prob, value, entropy

    def evaluate(self, state, action):
        """
        Used during PPO update (with gradient)

        Returns:
        - log_prob: log πθ(a|s)
        - entropy: H(πθ(.|s))
        - value: Vφ(s)
        """
        log_prob, entropy = self.actor.evaluate(state, action)
        value = self.critic(state)

        return log_prob, entropy, value
