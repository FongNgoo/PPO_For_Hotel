#!/usr/bin/env python
# main.py
"""
Multi-Room Hotel Pricing with PPO

Main entry point for training and evaluation.

Usage:
    python main.py              # Train with default settings
    python main.py --eval       # Evaluate saved model
    python main.py --baselines  # Evaluate baselines only

UPDATED: Uses auto-detected room types from data instead of hardcoding
"""

import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
import random
import os

# ============================================
# CONFIGURATION (MDP & PPO Hyperparameters)
# ============================================

# MDP Hyperparameters
# MDP Hyperparameters
ALPHA_MIN = 0.9          # Min price multiplier
ALPHA_MAX = 2.0          # Max price multiplier
LAMBDA_OCCUPANCY = 0.0   # Tắt thưởng lấp đầy
LAMBDA_STABILITY = 0.1   # Price stability penalty
LAMBDA_BOOKING = 0.0     # Tắt thưởng booking
LAMBDA_UNDERPRICE = 5.0  # Under-price penalty
REWARD_SCALE = 0.01
EPISODE_LENGTH = 60      # Days per episode

# PPO Hyperparameters
HIDDEN_DIMS = [256, 128]
ACTOR_HIDDEN = 64
CRITIC_HIDDEN = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99             # Discount factor (multi-day horizon)
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.2       # Higher for more exploration
ENTROPY_COEF_MIN = 0.02
ENTROPY_DECAY = 0.9999   # Decays slower for extended exploration
TARGET_KL = 0.015
PPO_EPOCHS = 10
BATCH_SIZE = 64

# Training settings
# Training settings
ITERATIONS = 300
STEPS_PER_ITER = 512
LOG_INTERVAL = 10
SEED = 42

# Data settings
HOTEL_DATA_PATH = 'data/resort_hotel_data.csv'  # Can be changed to city_hotel_data.csv
TRENDS_DATA_PATH = 'data/google_trends_portugal_algarve_daily.csv'
MIN_BOOKINGS_PER_ROOM = 50  # Minimum bookings to include a room type


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train(hotel_path: str = HOTEL_DATA_PATH):
    """Train the multi-room pricing agent."""
    
    print("=" * 60)
    print("MULTI-ROOM HOTEL PRICING WITH PPO")
    print("=" * 60)
    
    set_seed(SEED)
    
    # ========================================
    # PHASE 1: Data Loading & Demand Models
    # ========================================
    print("\n[PHASE 1] Loading data and training demand models...")
    print("-" * 60)
    
    from data.load_data import load_and_prepare_all, get_feature_columns
    from models.demand_models import prepare_and_train_demand_models
    
    # Load data with AUTO-DETECTED room types
    data = load_and_prepare_all(
        hotel_path=hotel_path,
        trends_path=TRENDS_DATA_PATH,
        min_bookings=MIN_BOOKINGS_PER_ROOM
    )
    
    # Get auto-detected room types, ADR refs, and capacities
    room_types = data['room_types']
    adr_refs = data['adr_refs']
    room_capacities = data['room_capacities']
    
    print(f"\nUsing {len(room_types)} room types: {room_types}")
    
    # Train demand models
    demand_models = prepare_and_train_demand_models(
        hotel_df=data['hotel_df'],
        daily_df=data['train_df'],
        room_types=room_types,
        adr_refs=adr_refs
    )
    
    # Verify price sensitivity
    demand_models.verify_price_sensitivity()
    
    # ========================================
    # PHASE 2: Setup RL Environment
    # ========================================
    print("\n[PHASE 2] Setting up RL environment...")
    print("-" * 60)
    
    from envs.multi_room_env import MultiRoomPricingEnv
    
    numerical_features, categorical_features = get_feature_columns(room_types)
    feature_columns = numerical_features + categorical_features
    
    env = MultiRoomPricingEnv(
        daily_df=data['train_df'],
        demand_models=demand_models,
        room_types=room_types,
        adr_refs=adr_refs,
        room_capacities=room_capacities,
        preprocessor=data['preprocessor'],
        feature_columns=feature_columns,
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX,
        lambda_occupancy=LAMBDA_OCCUPANCY,
        lambda_stability=LAMBDA_STABILITY,
        lambda_booking=LAMBDA_BOOKING,
        lambda_underprice=LAMBDA_UNDERPRICE,
        reward_scale=REWARD_SCALE,
        episode_length=EPISODE_LENGTH,
        seed=SEED
    )
    
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.n_rooms} (one per room type)")
    print(f"Episode length: {EPISODE_LENGTH} days")
    
    # ========================================
    # PHASE 3: Create Model and PPO
    # ========================================
    print("\n[PHASE 3] Creating Actor-Critic and PPO...")
    print("-" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from models.actor_critic import MultiRoomActorCritic
    from algorithms.ppo import PPO
    
    model = MultiRoomActorCritic(
        state_dim=env.state_dim,
        room_types=room_types,
        hidden_dims=HIDDEN_DIMS,
        actor_hidden_dim=ACTOR_HIDDEN,
        critic_hidden_dim=CRITIC_HIDDEN,
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX
    ).to(device)
    
    print(f"Model parameters: {model.n_params:,}")
    
    ppo = PPO(
        model=model,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_eps=CLIP_EPS,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        entropy_coef_min=ENTROPY_COEF_MIN,
        entropy_decay=ENTROPY_DECAY,
        target_kl=TARGET_KL,
        epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
        device=device
    )
    
    # ========================================
    # PHASE 4: Training
    # ========================================
    print("\n[PHASE 4] Training...")
    print("-" * 60)
    
    from trainers.trainer import Trainer
    
    trainer = Trainer(
        env=env,
        model=model,
        ppo=ppo,
        steps_per_iter=STEPS_PER_ITER,
        log_interval=LOG_INTERVAL,
        save_dir='checkpoints',
        device=device
    )
    
    history = trainer.train(
        iterations=ITERATIONS,
        save_best=True,
        verbose=True
    )
    
    # ========================================
    # PHASE 5: Evaluation
    # ========================================
    print("\n[PHASE 5] Evaluating on test set...")
    print("-" * 60)
    
    # Create test environment
    test_env = MultiRoomPricingEnv(
        daily_df=data['test_df'],
        demand_models=demand_models,
        room_types=room_types,
        adr_refs=adr_refs,
        room_capacities=room_capacities,
        preprocessor=data['preprocessor'],
        feature_columns=feature_columns,
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX,
        lambda_occupancy=LAMBDA_OCCUPANCY,
        lambda_stability=LAMBDA_STABILITY,
        lambda_booking=LAMBDA_BOOKING,
        lambda_underprice=LAMBDA_UNDERPRICE,
        reward_scale=REWARD_SCALE,
        episode_length=EPISODE_LENGTH,
        seed=SEED + 1
    )
    
    # Load best model
    trainer.load_checkpoint('best_model.pth')
    
    # Evaluate
    eval_results = trainer.evaluate(test_env, n_episodes=5, deterministic=True)
    
    print("\nTest Set Results:")
    print(f"  Mean Reward:   {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean Revenue:  €{eval_results['mean_revenue']:.0f}")
    print(f"  Mean Bookings: {eval_results['mean_bookings']:.0f}")
    
    print("\nPrice Statistics by Room:")
    for room, stats in eval_results['price_stats'].items():
        print(f"  Room {room}: €{stats['mean']:.0f} ± {stats['std']:.1f} "
              f"(range: €{stats['min']:.0f} - €{stats['max']:.0f})")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return history, eval_results, data


def evaluate_baseline(hotel_path: str = HOTEL_DATA_PATH):
    """Evaluate baseline strategies for comparison."""
    
    print("\n[BASELINES] Evaluating baseline strategies...")
    print("-" * 60)
    
    from data.load_data import load_and_prepare_all, get_feature_columns
    from models.demand_models import prepare_and_train_demand_models
    from envs.multi_room_env import MultiRoomPricingEnv
    
    set_seed(SEED)
    
    # Load data with auto-detected room types
    data = load_and_prepare_all(
        hotel_path=hotel_path,
        trends_path=TRENDS_DATA_PATH,
        min_bookings=MIN_BOOKINGS_PER_ROOM
    )
    
    room_types = data['room_types']
    adr_refs = data['adr_refs']
    room_capacities = data['room_capacities']
    n_rooms = len(room_types)
    
    # Train demand models
    demand_models = prepare_and_train_demand_models(
        hotel_df=data['hotel_df'],
        daily_df=data['train_df'],
        room_types=room_types,
        adr_refs=adr_refs
    )
    
    numerical_features, categorical_features = get_feature_columns(room_types)
    feature_columns = numerical_features + categorical_features
    
    test_env = MultiRoomPricingEnv(
        daily_df=data['test_df'],
        demand_models=demand_models,
        room_types=room_types,
        adr_refs=adr_refs,
        room_capacities=room_capacities,
        preprocessor=data['preprocessor'],
        feature_columns=feature_columns,
        episode_length=30,
        seed=SEED
    )
    
    baselines = {
        'Fixed (α=1.0)': lambda: np.ones(n_rooms),
        'Low (α=0.9)': lambda: np.full(n_rooms, 0.9),
        'High (α=1.2)': lambda: np.full(n_rooms, 1.2),
        'Random': lambda: np.random.uniform(0.85, 1.15, n_rooms)
    }
    
    results = {}
    for name, policy in baselines.items():
        rewards = []
        revenues = []
        bookings = []
        
        for ep in range(5):
            state = test_env.reset(start_idx=ep * 30)
            ep_reward = 0
            done = False
            
            while not done:
                action = policy()
                state, reward, done, info = test_env.step(action)
                ep_reward += reward
            
            summary = test_env.get_episode_summary()
            rewards.append(ep_reward)
            revenues.append(summary['total_revenue'])
            bookings.append(summary['total_bookings'])
        
        results[name] = {
            'reward': np.mean(rewards),
            'revenue': np.mean(revenues),
            'bookings': np.mean(bookings)
        }
        
        print(f"  {name}: Reward={np.mean(rewards):.2f}, "
              f"Revenue=€{np.mean(revenues):.0f}, "
              f"Bookings={np.mean(bookings):.0f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Room Hotel Pricing PPO')
    parser.add_argument('--eval', action='store_true', help='Evaluate only (no training)')
    parser.add_argument('--baselines', action='store_true', help='Evaluate baselines only')
    parser.add_argument('--iterations', type=int, default=ITERATIONS, help='Training iterations')
    parser.add_argument('--hotel', type=str, default=HOTEL_DATA_PATH, 
                        help='Path to hotel data CSV (e.g., data/city_hotel_data.csv)')
    
    args = parser.parse_args()
    
    if args.baselines:
        evaluate_baseline(hotel_path=args.hotel)
    elif args.eval:
        print("Evaluation mode not yet implemented. Run training first.")
    else:
        # Update iterations if specified
        ITERATIONS = args.iterations
        history, results, data = train(hotel_path=args.hotel)
        
        # Also run baselines for comparison
        print("\n" + "=" * 60)
        baseline_results = evaluate_baseline(hotel_path=args.hotel)
