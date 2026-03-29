# envs/multi_room_env.py
"""
Multi-Room Daily Pricing Environment.

This environment simulates daily hotel operations:
- Each step represents one day
- Agent sets prices for all 7 room types
- Reward is based on daily revenue and occupancy

MDP Design:
- State: Daily context (temporal, trend, historical demand)
- Action: 7 price multipliers (one per room)
- Reward: Normalized revenue + occupancy bonus - stability penalty
- Transition: Move to next day, update historical features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random


class MultiRoomPricingEnv:
    """
    Reinforcement Learning environment for multi-room hotel pricing.
    
    Args:
        daily_df: DataFrame with daily features and targets
        demand_models: MultiRoomDemandModel for P(booking|context, price)
        room_types: List of room types
        adr_refs: Reference ADR for each room type
        preprocessor: sklearn preprocessor for state encoding
        feature_columns: List of feature column names
        alpha_min: Minimum price multiplier
        alpha_max: Maximum price multiplier
        lambda_occupancy: Occupancy bonus coefficient
        lambda_stability: Price stability penalty coefficient
        reward_scale: Reward scaling factor
        episode_length: Number of days per episode (None = full dataset)
        seed: Random seed
    """
    
    def __init__(
        self,
        daily_df: pd.DataFrame,
        demand_models,  # MultiRoomDemandModel
        room_types: List[str],
        adr_refs: Dict[str, float],
        room_capacities: Dict[str, int],
        preprocessor,
        feature_columns: List[str],
        alpha_min: float = 0.8,
        alpha_max: float = 1.5,
        lambda_occupancy: float = 20.0,
        lambda_stability: float = 0.5,
        lambda_booking: float = 5.0,
        lambda_underprice: float = 10.0,
        reward_scale: float = 0.01,
        episode_length: Optional[int] = 30,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.daily_df = daily_df.reset_index(drop=True)
        self.demand_models = demand_models
        self.room_types = room_types
        self.n_rooms = len(room_types)
        self.adr_refs = adr_refs
        self.room_capacities = room_capacities
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lambda_occupancy = lambda_occupancy
        self.lambda_stability = lambda_stability
        self.lambda_booking = lambda_booking
        self.lambda_underprice = lambda_underprice
        self.reward_scale = reward_scale
        self.episode_length = episode_length or len(daily_df)
        
        # Compute expected daily bookings per room (for normalization)
        self.expected_bookings = {}
        for room in room_types:
            col = f'target_bookings_{room}'
            if col in daily_df.columns:
                self.expected_bookings[room] = daily_df[col].mean()
            else:
                self.expected_bookings[room] = 1.0
        
        # State dimension
        sample_row = daily_df.iloc[0:1][feature_columns]
        sample_state = preprocessor.transform(sample_row)
        self.state_dim = sample_state.shape[1] + self.n_rooms
        
        # Episode tracking
        self.current_day_idx = 0
        self.episode_start_idx = 0
        self.episode_step = 0
        
        # Episode statistics
        self.episode_stats = self._reset_stats()
    
    def _reset_stats(self) -> Dict:
        """Reset episode statistics."""
        return {
            'total_revenue': 0.0,
            'total_bookings': 0,
            'bookings_per_room': {room: 0 for room in self.room_types},
            'revenue_per_room': {room: 0.0 for room in self.room_types},
            'prices': {room: [] for room in self.room_types},
            'days_completed': 0
        }
    
    def reset(
        self,
        start_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Reset environment to start of a new episode.
        
        Args:
            start_idx: Starting day index (None = random)
            
        Returns:
            Initial state vector
        """
        # Choose starting point
        max_start = len(self.daily_df) - self.episode_length
        
        if start_idx is not None:
            self.episode_start_idx = min(start_idx, max_start)
        else:
            self.episode_start_idx = random.randint(0, max(0, max_start))
        
        self.current_day_idx = self.episode_start_idx
        self.episode_step = 0
        self.episode_stats = self._reset_stats()
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        row = self.daily_df.iloc[self.current_day_idx:self.current_day_idx+1]
        state = self.preprocessor.transform(row[self.feature_columns])
        state_flat = state.flatten().astype(np.float32)
        
        # Add unconstrained forecast (occupancy proxy) to state
        occ_rates = []
        for room in self.room_types:
            expected = row[f'target_bookings_{room}'].values[0]
            cap = self.room_capacities[room]
            occ_rates.append(expected / max(cap, 1))
            
        return np.concatenate([state_flat, occ_rates]).astype(np.float32)
    
    def _get_context_for_demand(self) -> np.ndarray:
        """Get context features for demand model (without price)."""
        row = self.daily_df.iloc[self.current_day_idx:self.current_day_idx+1]
        # Use same features but demand model will add price internally
        context = self.preprocessor.transform(row[self.feature_columns])
        return context.flatten()
    
    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one day step.
        
        Args:
            actions: Array of 7 price multipliers (one per room)
            
        Returns:
            next_state: State for next day
            reward: Daily reward
            done: Whether episode is finished
            info: Detailed information
        """
        # Ensure actions are in valid range
        actions = np.clip(actions, self.alpha_min, self.alpha_max)
        
        # Get current context for demand prediction
        context = self._get_context_for_demand()
        
        # Compute prices and simulate bookings
        daily_revenue = 0.0
        daily_bookings = 0
        room_info = {}
        
        for i, room in enumerate(self.room_types):
            alpha = actions[i]
            price = alpha * self.adr_refs[room]
            
            # Get booking probability from demand model
            p_book = self.demand_models.predict_proba(room, context, price)
            
            # Get expected bookings for this room today (from data)
            target_col = f'target_bookings_{room}'
            expected_today = self.daily_df.iloc[self.current_day_idx][target_col]
            
            # Simulate actual bookings (Poisson-like based on expected * p_book)
            expected_with_price = expected_today * p_book
            actual_bookings = np.random.poisson(max(0.1, expected_with_price))
            
            # Apply capacity constraint
            actual_bookings = min(actual_bookings, self.room_capacities[room])
            
            # Compute revenue
            room_revenue = actual_bookings * price
            
            # Update stats
            daily_revenue += room_revenue
            daily_bookings += actual_bookings
            
            self.episode_stats['bookings_per_room'][room] += actual_bookings
            self.episode_stats['revenue_per_room'][room] += room_revenue
            self.episode_stats['prices'][room].append(price)
            
            room_info[room] = {
                'alpha': alpha,
                'price': price,
                'p_book': p_book,
                'expected': expected_today,
                'actual_bookings': actual_bookings,
                'revenue': room_revenue
            }
        
        # Update episode totals
        self.episode_stats['total_revenue'] += daily_revenue
        self.episode_stats['total_bookings'] += daily_bookings
        self.episode_stats['days_completed'] += 1
        
        # ========================================
        # COMPUTE REWARD
        # ========================================
        
        # Revenue thuần (không bị pha loãng bởi occupancy)
        ref_revenue = sum(self.adr_refs[room] * self.room_capacities[room] 
                          for room in self.room_types)
        revenue_ratio = daily_revenue / max(ref_revenue, 1.0)
        
        # Chỉ giữ stability penalty nhẹ
        alpha_deviations = [(a - 1.0) ** 2 for a in actions]
        stability_penalty = self.lambda_stability * sum(alpha_deviations)
        
        # Under-pricing penalty
        underprice_deviations = [max(0.0, 1.0 - a) ** 2 for a in actions]
        underprice_penalty = self.lambda_underprice * sum(underprice_deviations)
        
        # Reward đơn giản và rõ ràng
        reward = (revenue_ratio * 100 - stability_penalty - underprice_penalty) * self.reward_scale
        
        # Move to next day
        self.current_day_idx += 1
        self.episode_step += 1
        
        # Check if episode is done
        done = (
            self.episode_step >= self.episode_length or
            self.current_day_idx >= len(self.daily_df)
        )
        
        # Get next state
        next_state = None if done else self._get_state()
        
        # Info dict
        info = {
            'date': self.daily_df.iloc[self.current_day_idx - 1]['date'],
            'daily_revenue': daily_revenue,
            'daily_bookings': daily_bookings,
            'revenue_ratio': revenue_ratio,
            'stability_penalty': stability_penalty,
            'underprice_penalty': underprice_penalty,
            'room_details': room_info,
            'episode_stats': self.episode_stats.copy() if done else None
        }
        
        return next_state, reward, done, info
    
    def get_current_date(self) -> pd.Timestamp:
        """Get current date."""
        return self.daily_df.iloc[self.current_day_idx]['date']
    
    def get_episode_summary(self) -> Dict:
        """Get summary of current/last episode."""
        stats = self.episode_stats
        
        return {
            'total_revenue': stats['total_revenue'],
            'total_bookings': stats['total_bookings'],
            'avg_daily_revenue': stats['total_revenue'] / max(stats['days_completed'], 1),
            'avg_daily_bookings': stats['total_bookings'] / max(stats['days_completed'], 1),
            'days_completed': stats['days_completed'],
            'avg_prices': {
                room: np.mean(prices) if prices else 0
                for room, prices in stats['prices'].items()
            }
        }


class EvaluationEnv(MultiRoomPricingEnv):
    """
    Evaluation environment that runs through data sequentially.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['episode_length'] = None  # Full dataset
        super().__init__(*args, **kwargs)
    
    def reset(self, start_idx: int = 0) -> np.ndarray:
        """Reset to specific starting point (default: beginning)."""
        return super().reset(start_idx=start_idx)


if __name__ == '__main__':
    # Test environment
    import sys
    sys.path.insert(0, '.')
    
    from data.load_data import load_and_prepare_all, get_feature_columns
    from models.demand_models import prepare_and_train_demand_models
    
    print("Loading data...")
    data = load_and_prepare_all()
    
    print("\nTraining demand models...")
    demand_models = prepare_and_train_demand_models(
        hotel_df=data['hotel_df'],
        daily_df=data['train_df'],
        room_types=data['room_types'],
        adr_refs=data['adr_refs']
    )
    
    print("\nCreating environment...")
    numerical_features, categorical_features = get_feature_columns()
    feature_columns = numerical_features + categorical_features
    
    env = MultiRoomPricingEnv(
        daily_df=data['train_df'],
        demand_models=demand_models,
        room_types=data['room_types'],
        adr_refs=data['adr_refs'],
        room_capacities=data['room_capacities'],
        preprocessor=data['preprocessor'],
        feature_columns=feature_columns,
        episode_length=30
    )
    
    print(f"State dimension: {env.state_dim}")
    print(f"Number of rooms: {env.n_rooms}")
    
    # Test episode
    print("\nRunning test episode...")
    state = env.reset()
    total_reward = 0
    
    for step in range(30):
        # Random actions
        actions = np.random.uniform(0.9, 1.1, size=env.n_rooms)
        next_state, reward, done, info = env.step(actions)
        total_reward += reward
        
        if done:
            break
    
    summary = env.get_episode_summary()
    print(f"\nEpisode Summary:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total revenue: €{summary['total_revenue']:.0f}")
    print(f"  Total bookings: {summary['total_bookings']}")
    print(f"  Days completed: {summary['days_completed']}")
