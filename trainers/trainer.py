import torch
import numpy as np
from algorithms.buffer import RolloutBuffer
from utils.plotting import (
    plot_rewards,
    plot_losses,
    plot_entropy,
    plot_price_distribution
)


class Trainer:
    def __init__(self, env, model, ppo, steps_per_iter=256):
        self.env = env
        self.model = model
        self.ppo = ppo
        self.steps = steps_per_iter

        # ===== Logs =====
        self.reward_log = []
        self.actor_loss_log = []
        self.critic_loss_log = []
        self.entropy_log = []
        self.price_log = []

    def train(self, iterations=500):
        for it in range(iterations):
            buffer = RolloutBuffer()
            state = torch.tensor(self.env.reset(), dtype=torch.float32)

            total_reward = 0

            for _ in range(self.steps):
                action, log_prob, value, entropy = self.model.act(state)
                next_state, reward, done, info = self.env.step(action.item())

                total_reward += reward
                self.price_log.append(info["price"])

                buffer.add(
                    state,
                    action,
                    torch.tensor(reward, dtype=torch.float32),
                    value.detach(),
                    log_prob.detach()
                )

                state = torch.tensor(next_state, dtype=torch.float32)

            returns, advantages = buffer.compute_returns_advantages()

            metrics = self.ppo.update(buffer, returns, advantages)

            actor_loss = metrics["actor_loss"]
            critic_loss = metrics["critic_loss"]
            entropy_mean = metrics["entropy"]

            # ===== Log =====
            self.reward_log.append(total_reward)
            self.actor_loss_log.append(actor_loss)
            self.critic_loss_log.append(critic_loss)
            self.entropy_log.append(entropy_mean)

            buffer.clear()

            if it % 50 == 0:
                print(
                    f"Iter {it} | "
                    f"Reward: {total_reward:.2f} | "
                    f"Actor Loss: {actor_loss:.4f} | "
                    f"Critic Loss: {critic_loss:.4f} | "
                    f"Entropy: {entropy_mean:.4f}"
                )

        # ===== Plot after training =====
        plot_rewards(self.reward_log)
        plot_losses(self.actor_loss_log, self.critic_loss_log)
        plot_entropy(self.entropy_log)
        plot_price_distribution(self.price_log)
