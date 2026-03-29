# models/actor_critic.py
"""
Multi-Head Actor-Critic Network for Multi-Room Pricing.

Architecture:
- Shared backbone: MLP (state_dim → 256 → 128)
- 7 Actor heads: Each outputs Beta distribution parameters for one room
- 1 Critic head: Outputs state value V(s)

Each Actor head uses Beta distribution for bounded action in [alpha_min, alpha_max].
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class SharedBackbone(nn.Module):
    """
    Shared feature extractor for all heads.
    
    Architecture: Linear → ReLU → Linear → ReLU
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActorHead(nn.Module):
    """
    Actor head for a single room type using Squashed Gaussian.
    
    Outputs mean for Normal distribution. Log_std is a learnable parameter.
    Action is mapped through Tanh and scaled to [alpha_min, alpha_max].
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        alpha_min: float = 0.8,
        alpha_max: float = 1.3
    ):
        super().__init__()
        
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_range = alpha_max - alpha_min
        self.alpha_scale = self.alpha_range / 2.0
        self.alpha_bias = (alpha_max + alpha_min) / 2.0
        
        # Network outputs mean
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # State-independent log standard deviation (commonly used in PPO)
        self.log_std = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize hidden layer
        nn.init.orthogonal_(self.network[0].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.network[0].bias)
        # Initialize final layer to 0 for neutral starting mean
        nn.init.zeros_(self.network[2].weight)
        nn.init.zeros_(self.network[2].bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean and standard deviation for Normal distribution.
        """
        mean = self.network(features).squeeze(-1)
        # log_std is broadcasted to batch size
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(
        self,
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from Tanh-Squashed Gaussian distribution.
        
        Returns:
            action: Price multiplier in [alpha_min, alpha_max]
            log_prob: Log probability of the action
        """
        mean, std = self.forward(features)
        
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            u = mean
        else:
            u = dist.rsample()  # Reparameterized sample
            
        log_prob_u = dist.log_prob(u)
        
        # Apply Tanh squashing
        action_tanh = torch.tanh(u)
        
        # Enforcing Action Bounds log probability correction
        # log p(action) = log p(u) - log(1 - tanh(u)^2 + epsilon)
        log_prob = log_prob_u - torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        
        # Scale to [alpha_min, alpha_max]
        action = self.alpha_bias + action_tanh * self.alpha_scale
        
        return action, log_prob
    
    def evaluate(
        self,
        features: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Args:
            features: State features
            action: Actions in [alpha_min, alpha_max]
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of distribution
        """
        mean, std = self.forward(features)
        dist = torch.distributions.Normal(mean, std)
        
        # Inverse mapping: action -> action_tanh -> u
        action_tanh = (action - self.alpha_bias) / self.alpha_scale
        
        # Clip to prevent nan in atanh
        action_tanh = action_tanh.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(action_tanh)
        
        log_prob_u = dist.log_prob(u)
        log_prob = log_prob_u - torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        
        # Plain Gaussian entropy (standard in PPO for squashed actions)
        entropy = dist.entropy()
        
        return log_prob, entropy


class CriticHead(nn.Module):
    """
    Critic head for state value estimation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        
        # Last layer with smaller weights
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class MultiRoomActorCritic(nn.Module):
    """
    Multi-Head Actor-Critic for Multi-Room Pricing.
    
    Architecture:
    - Shared backbone processes state features
    - 7 Actor heads (one per room) output price multipliers
    - 1 Critic head outputs state value
    
    Args:
        state_dim: Dimension of state features
        room_types: List of room type identifiers
        hidden_dims: Dimensions for shared backbone
        actor_hidden_dim: Hidden dimension for actor heads
        critic_hidden_dim: Hidden dimension for critic head
        alpha_min: Minimum price multiplier
        alpha_max: Maximum price multiplier
    """
    
    def __init__(
        self,
        state_dim: int,
        room_types: List[str],
        hidden_dims: List[int] = [256, 128],
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 64,
        alpha_min: float = 0.8,
        alpha_max: float = 1.3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.room_types = room_types
        self.n_rooms = len(room_types)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Shared backbone
        self.backbone = SharedBackbone(state_dim, hidden_dims)
        backbone_output_dim = self.backbone.output_dim
        
        # Actor heads (one per room)
        self.actor_heads = nn.ModuleDict({
            room: ActorHead(
                input_dim=backbone_output_dim,
                hidden_dim=actor_hidden_dim,
                alpha_min=alpha_min,
                alpha_max=alpha_max
            )
            for room in room_types
        })
        
        # Critic head
        self.critic = CriticHead(backbone_output_dim, critic_hidden_dim)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def get_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract features from state using shared backbone."""
        return self.backbone(state)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value from critic."""
        features = self.get_features(state)
        return self.critic(features)
    
    def get_actions(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Sample actions for all rooms.
        
        Returns:
            actions: Dict[room_type -> action tensor]
            log_probs: Dict[room_type -> log_prob tensor]
        """
        features = self.get_features(state)
        
        actions = {}
        log_probs = {}
        
        for room in self.room_types:
            action, log_prob = self.actor_heads[room].sample(features, deterministic)
            actions[room] = action
            log_probs[room] = log_prob
        
        return actions, log_probs
    
    def get_actions_tensor(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions and return as stacked tensors.
        
        Returns:
            actions: Tensor of shape (batch, n_rooms)
            log_probs: Tensor of shape (batch, n_rooms)
            value: Tensor of shape (batch,)
        """
        features = self.get_features(state)
        
        actions_list = []
        log_probs_list = []
        
        for room in self.room_types:
            action, log_prob = self.actor_heads[room].sample(features, deterministic)
            actions_list.append(action)
            log_probs_list.append(log_prob)
        
        # Stack along room dimension
        actions = torch.stack(actions_list, dim=-1)
        log_probs = torch.stack(log_probs_list, dim=-1)
        value = self.critic(features)
        
        return actions, log_probs, value
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.
        
        Args:
            state: State tensor (batch, state_dim)
            actions: Action tensor (batch, n_rooms)
            
        Returns:
            log_probs: Sum of log probs across rooms (batch,)
            entropy: Sum of entropies across rooms (batch,)
            value: State value (batch,)
        """
        features = self.get_features(state)
        
        log_probs_list = []
        entropies_list = []
        
        for i, room in enumerate(self.room_types):
            room_action = actions[:, i] if actions.dim() > 1 else actions[i:i+1]
            log_prob, entropy = self.actor_heads[room].evaluate(features, room_action)
            log_probs_list.append(log_prob)
            entropies_list.append(entropy)
        
        # Sum across rooms
        log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies_list, dim=-1).mean(dim=-1)
        value = self.critic(features)
        
        return log_probs, entropy, value
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: sample actions and get value.
        
        Returns:
            actions: (batch, n_rooms)
            log_probs: (batch, n_rooms)
            value: (batch,)
        """
        return self.get_actions_tensor(state, deterministic)


if __name__ == '__main__':
    # Test the network
    room_types = ['A', 'D', 'E', 'F', 'G', 'C', 'H']
    state_dim = 48
    
    model = MultiRoomActorCritic(
        state_dim=state_dim,
        room_types=room_types
    )
    
    print(f"Model parameters: {model.n_params:,}")
    
    # Test forward pass
    batch_size = 32
    state = torch.randn(batch_size, state_dim)
    
    actions, log_probs, values = model(state)
    
    print(f"Actions shape: {actions.shape}")  # (32, 7)
    print(f"Log probs shape: {log_probs.shape}")  # (32, 7)
    print(f"Values shape: {values.shape}")  # (32,)
    
    print(f"\nAction ranges:")
    print(f"  Min: {actions.min().item():.3f}")
    print(f"  Max: {actions.max().item():.3f}")
    print(f"  Mean: {actions.mean().item():.3f}")
    
    # Test evaluation
    log_probs_eval, entropy, values_eval = model.evaluate_actions(state, actions)
    print(f"\nEntropy mean: {entropy.mean().item():.4f}")
