# trainers/trainer.py
"""
Trainer for Multi-Room Hotel Pricing PPO.

Handles:
- Rollout collection
- PPO updates
- Logging and checkpointing
- Evaluation
"""

import torch
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class Trainer:
    """
    PPO Trainer for Multi-Room Pricing.
    
    Args:
        env: MultiRoomPricingEnv
        model: MultiRoomActorCritic
        ppo: PPO algorithm
        steps_per_iter: Rollout steps per iteration
        log_interval: Iterations between logging
        save_dir: Directory for checkpoints
    """
    
    def __init__(
        self,
        env,
        model,
        ppo,
        steps_per_iter: int = 256,
        log_interval: int = 10,
        save_dir: str = 'checkpoints',
        device: torch.device = torch.device("cpu")
    ):
        self.env = env
        self.model = model
        self.ppo = ppo
        self.steps_per_iter = steps_per_iter
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.device = device
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'rewards': [],
            'revenues': [],
            'bookings': [],
            'entropies': [],
            'actor_losses': [],
            'critic_losses': [],
            'prices': {room: [] for room in env.room_types}
        }
        
        # Best model tracking
        self.best_reward = float('-inf')
    
    def collect_rollout(self) -> Tuple:
        """
        Collect rollout data for PPO update.
        
        Returns:
            buffer: RolloutBuffer with transitions
            episode_rewards: List of episode total rewards
            episode_revenues: List of episode total revenues
            episode_bookings: List of episode total bookings
        """
        from algorithms.ppo import RolloutBuffer
        
        buffer = RolloutBuffer()
        episode_rewards = []
        episode_revenues = []
        episode_bookings = []
        
        state = self.env.reset()
        episode_reward = 0.0
        
        for step in range(self.steps_per_iter):
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                actions, log_probs, value = self.model(state_tensor)
                
                action = actions.cpu().squeeze(0).numpy()
                log_prob = log_probs.sum().item()  # Sum log probs across rooms
                value = value.item()
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            
            # Store transition
            buffer.add(
                state=state,
                action=action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done
            )
            
            # Track prices
            for i, room in enumerate(self.env.room_types):
                self.history['prices'][room].append(action[i] * self.env.adr_refs[room])
            
            if done:
                # Episode finished
                summary = self.env.get_episode_summary()
                episode_rewards.append(episode_reward)
                episode_revenues.append(summary['total_revenue'])
                episode_bookings.append(summary['total_bookings'])
                
                # Reset
                state = self.env.reset()
                episode_reward = 0.0
            else:
                state = next_state
        
        # Get value of last state for GAE
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, last_value = self.model(state_tensor)
            last_value = last_value.item()
        
        return buffer, episode_rewards, episode_revenues, episode_bookings, last_value
    
    def train(
        self,
        iterations: int,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Train the agent.
        
        Args:
            iterations: Number of training iterations
            save_best: Whether to save best model
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Starting Multi-Room PPO Training")
            print("=" * 60)
            print(f"Steps per iteration: {self.steps_per_iter}")
            print(f"Total iterations: {iterations}")
            print(f"Room types: {self.env.room_types}")
            print("=" * 60)
        
        for iteration in range(iterations):
            # Collect rollout
            buffer, ep_rewards, ep_revenues, ep_bookings, last_value = self.collect_rollout()
            
            # PPO update
            metrics = self.ppo.update(buffer, last_value)
            
            # Track history
            mean_reward = np.mean(ep_rewards) if ep_rewards else 0
            mean_revenue = np.mean(ep_revenues) if ep_revenues else 0
            mean_bookings = np.mean(ep_bookings) if ep_bookings else 0
            
            self.history['rewards'].append(mean_reward)
            self.history['revenues'].append(mean_revenue)
            self.history['bookings'].append(mean_bookings)
            self.history['entropies'].append(metrics['entropy'])
            self.history['actor_losses'].append(metrics['actor_loss'])
            self.history['critic_losses'].append(metrics['critic_loss'])
            
            # Save best model
            if save_best and mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.save_checkpoint('best_model.pth')
            
            # Logging
            if verbose and (iteration % self.log_interval == 0 or iteration == iterations - 1):
                # Compute average prices
                avg_prices = {}
                for room in self.env.room_types:
                    recent_prices = self.history['prices'][room][-self.steps_per_iter:]
                    avg_prices[room] = np.mean(recent_prices) if recent_prices else 0
                
                print(f"Iter {iteration:4d} | "
                      f"Reward: {mean_reward:8.2f} | "
                      f"Revenue: €{mean_revenue:8.0f} | "
                      f"Bookings: {mean_bookings:5.0f} | "
                      f"Entropy: {metrics['entropy']:.4f} | "
                      f"KL: {metrics['approx_kl']:.4f}")
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print(f"Best reward: {self.best_reward:.2f}")
            print("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'entropy_coef': self.ppo.current_entropy_coef
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        # weights_only=False for PyTorch 2.6+ compatibility
        # Safe here because we're loading our own saved checkpoints
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.ppo.current_entropy_coef = checkpoint.get('entropy_coef', self.ppo.entropy_coef)
    
    def evaluate(
        self,
        eval_env,
        n_episodes: int = 5,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate the trained policy.
        
        Args:
            eval_env: Environment for evaluation
            n_episodes: Number of episodes to run
            deterministic: Use deterministic actions
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_revenues = []
        episode_bookings = []
        all_prices = {room: [] for room in eval_env.room_types}
        
        for ep in range(n_episodes):
            state = eval_env.reset(start_idx=ep * eval_env.episode_length)
            episode_reward = 0.0
            done = False
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    actions, _, _ = self.model(state_tensor, deterministic=deterministic)
                    action = actions.cpu().squeeze(0).numpy()
                
                state, reward, done, info = eval_env.step(action)
                episode_reward += reward
                
                # Track prices
                for i, room in enumerate(eval_env.room_types):
                    all_prices[room].append(action[i] * eval_env.adr_refs[room])
            
            summary = eval_env.get_episode_summary()
            episode_rewards.append(episode_reward)
            episode_revenues.append(summary['total_revenue'])
            episode_bookings.append(summary['total_bookings'])
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_revenue': np.mean(episode_revenues),
            'mean_bookings': np.mean(episode_bookings),
            'price_stats': {
                room: {
                    'mean': np.mean(prices),
                    'std': np.std(prices),
                    'min': np.min(prices),
                    'max': np.max(prices)
                }
                for room, prices in all_prices.items()
            }
        }


if __name__ == '__main__':
    # Quick test
    print("Trainer module loaded successfully!")
