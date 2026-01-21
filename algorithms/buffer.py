import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def add(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_advantages(self, gamma=0.99):
        returns = []
        advantages = []

        G = 0
        for r, v in zip(reversed(self.rewards), reversed(self.values)):
            G = r + gamma * G
            returns.insert(0, G)
            advantages.insert(0, G - v)

        return (
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32)
        )

    def clear(self):
        self.__init__()
