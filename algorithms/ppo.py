# algorithms/ppo.py
"""
Proximal Policy Optimization (PPO) for Multi-Room Hotel Pricing.

Key features:
- Clipped surrogate objective
- Multi-room action handling (7 rooms)
- Strong entropy bonus to prevent collapse
- Target KL early stopping
- Entropy coefficient decay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional


class RolloutBuffer:
    """
    Buffer to store rollout data for PPO updates.
    
    Stores:
    - states: (T, state_dim)
    - actions: (T, n_rooms)
    - log_probs: (T,)
    - rewards: (T,)
    - values: (T,)
    - dones: (T,)
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.states.append(torch.FloatTensor(state))
        self.actions.append(torch.FloatTensor(action))
        self.log_probs.append(torch.FloatTensor([log_prob]))
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)


class PPO:
    """
    PPO-Clip algorithm for multi-room continuous action pricing.
    
    Args:
        model: MultiRoomActorCritic network
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_eps: PPO clipping parameter
        value_coef: Critic loss coefficient
        entropy_coef: Initial entropy bonus coefficient
        entropy_coef_min: Minimum entropy coefficient
        entropy_decay: Decay rate per iteration
        target_kl: Target KL for early stopping (None = no early stopping)
        epochs: PPO epochs per update
        batch_size: Mini-batch size
        max_grad_norm: Gradient clipping threshold
    """
    
    def __init__(
        self,
        model,
        lr: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.1,
        entropy_coef_min: float = 0.02,
        entropy_decay: float = 0.9995,
        target_kl: Optional[float] = 0.015,
        epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_decay = entropy_decay
        self.target_kl = target_kl
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Current entropy coefficient (decays over time)
        self.current_entropy_coef = entropy_coef
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            returns: Target values for critic
            advantages: Advantage estimates for actor
        """
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        
        gae = 0.0
        next_value = last_value
        
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)
    
    def update(
        self,
        buffer: RolloutBuffer,
        last_value: float = 0.0
    ) -> Dict:
        """
        Perform PPO update on collected rollout data.
        
        Args:
            buffer: RolloutBuffer with collected transitions
            last_value: Value estimate for final state
            
        Returns:
            Dictionary with training metrics
        """
        # Compute returns and advantages
        returns, advantages = self.compute_gae(
            buffer.rewards,
            buffer.values,
            buffer.dones,
            last_value
        )
        
        # Stack rollout data
        states = torch.stack(buffer.states).to(self.device)
        actions = torch.stack(buffer.actions).to(self.device)
        old_log_probs = torch.cat(buffer.log_probs).detach().to(self.device)
        
        returns = returns.detach().to(self.device)
        advantages = advantages.detach().to(self.device)
        
        dataset_size = len(buffer)
        
        # Track metrics
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []
        approx_kls = []
        early_stopped = False
        epochs_run = 0
        
        # PPO update epochs
        for epoch in range(self.epochs):
            epochs_run = epoch + 1
            
            # Shuffle indices
            indices = torch.randperm(dataset_size)
            epoch_kls = []
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Normalize advantages per mini-batch
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )
                
                # Forward pass
                log_probs, entropy, values = self.model.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # PPO ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps
                ) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # Total loss
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.current_entropy_coef * entropy.mean()
                )
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Track metrics
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())
                
                # Clip fraction
                clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                clip_fractions.append(clip_frac)
                
                # Approximate KL
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                    approx_kls.append(approx_kl)
                    epoch_kls.append(approx_kl)
            
            # Early stopping based on KL divergence
            mean_epoch_kl = np.mean(epoch_kls)
            if self.target_kl is not None and mean_epoch_kl > self.target_kl:
                early_stopped = True
                break
        
        # Decay entropy coefficient
        self.current_entropy_coef = max(
            self.entropy_coef_min,
            self.current_entropy_coef * self.entropy_decay
        )
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
            'approx_kl': np.mean(approx_kls),
            'entropy_coef': self.current_entropy_coef,
            'early_stopped': early_stopped,
            'epochs_run': epochs_run
        }
    
    def reset_entropy_coef(self):
        """Reset entropy coefficient to initial value."""
        self.current_entropy_coef = self.entropy_coef


if __name__ == '__main__':
    # Test PPO
    import sys
    sys.path.insert(0, '.')
    
    from models.actor_critic import MultiRoomActorCritic
    
    room_types = ['A', 'D', 'E', 'F', 'G', 'C', 'H']
    state_dim = 45
    
    model = MultiRoomActorCritic(state_dim=state_dim, room_types=room_types)
    ppo = PPO(model=model)
    
    # Create dummy buffer
    buffer = RolloutBuffer()
    for _ in range(128):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(0.8, 1.3, size=7).astype(np.float32)
        buffer.add(
            state=state,
            action=action,
            log_prob=np.random.randn(),
            reward=np.random.randn(),
            value=np.random.randn(),
            done=False
        )
    
    # Test update
    metrics = ppo.update(buffer)
    print("PPO Update Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
