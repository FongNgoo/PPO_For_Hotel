#algorithms/buffer
import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_values = []  # Mới thêm

    def add(self, state, action, reward, value, log_prob, done, next_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_values.append(next_value)

    def compute_returns_advantages(self, gamma=0.99, lambda_=0.95):
        advantages = []
        returns = []
        gae = 0

        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                delta = self.rewards[i] - self.values[i]
                gae = delta
            else:
                # Lấy next_value từ buffer
                delta = self.rewards[i] + gamma * self.next_values[i] - self.values[i]
                gae = delta + gamma * lambda_ * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def clear(self):
        self.states.clear()