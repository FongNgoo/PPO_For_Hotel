import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    """
    Proximal Policy Optimization (PPO-Clip)
    Suitable for continuous action pricing problems
    """

    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs=10,
        batch_size=64,
        max_grad_norm=0.5
    ):
        self.model = model
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def update(self, buffer, returns, advantages):
        """
        buffer.states      : list[Tensor]
        buffer.actions     : list[Tensor]
        buffer.log_probs   : list[Tensor]
        returns            : Tensor
        advantages         : Tensor
        """

        # ===== Stack rollout data =====
        states = torch.stack(buffer.states)
        actions = torch.stack(buffer.actions)
        old_log_probs = torch.stack(buffer.log_probs).detach()

        returns = returns.detach()
        advantages = advantages.detach()

        # ===== Advantage normalization =====
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        actor_losses = []
        critic_losses = []
        entropies = []

        dataset_size = states.size(0)
        indices = torch.randperm(dataset_size)

        # ===== PPO update =====
        for _ in range(self.epochs):
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Evaluate current policy
                log_probs, entropy, values = self.model.evaluate(
                    batch_states, batch_actions
                )

                # ===== PPO ratio =====
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # ===== Clipped surrogate objective =====
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps
                ) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                # ===== Value loss =====
                critic_loss = nn.MSELoss()(values, batch_returns)

                # ===== Total loss =====
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "entropy": sum(entropies) / len(entropies),
        }
