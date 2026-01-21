import os
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ==============================
# 1. Reward per iteration
# ==============================
def plot_rewards(rewards, save_dir="plots"):
    ensure_dir(save_dir)

    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward")
    plt.grid()

    plt.savefig(os.path.join(save_dir, "reward_curve.png"))
    plt.close()


# ==============================
# 2. Actor & Critic Loss
# ==============================
def plot_losses(actor_losses, critic_losses, save_dir="plots"):
    ensure_dir(save_dir)

    plt.figure(figsize=(8, 4))
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Actor & Critic Loss")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()


# ==============================
# 3. Entropy (exploration)
# ==============================
def plot_entropy(entropies, save_dir="plots"):
    ensure_dir(save_dir)

    plt.figure(figsize=(8, 4))
    plt.plot(entropies)
    plt.xlabel("Iteration")
    plt.ylabel("Entropy")
    plt.title("Policy Entropy")
    plt.grid()

    plt.savefig(os.path.join(save_dir, "entropy_curve.png"))
    plt.close()


# ==============================
# 4. Price distribution (Hotel-specific)
# ==============================
def plot_price_distribution(prices, save_dir="plots"):
    ensure_dir(save_dir)

    plt.figure(figsize=(6, 4))
    plt.hist(prices, bins=30)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.title("Price Distribution Learned by PPO")

    plt.savefig(os.path.join(save_dir, "price_distribution.png"))
    plt.close()
