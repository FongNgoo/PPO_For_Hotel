# evaluation/evaluate_model.py
"""
Comprehensive Evaluation Module for Multi-Room Hotel Pricing.

This module provides rigorous scientific evaluation including:
1. Multiple baseline strategies for comparison
2. Statistical evaluation metrics with confidence intervals
3. Hypothesis testing (t-tests, Welch's t-test)
4. Per-room and aggregate analysis
5. Reproducible evaluation pipeline

Metrics computed:
- Expected Revenue per Booking (€/booking)
- Revenue Lift (%) vs baseline
- Expected Retention Rate (bookings/day)
- Price Deviation from ADR reference
- Price Variance (stability measure)
- Occupancy proxy metrics

Statistical Analysis:
- 95% Confidence Intervals
- Independent t-tests
- Effect size (Cohen's d)
- Multiple comparison awareness
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import torch
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================
# DATA CLASSES FOR STRUCTURED RESULTS
# ============================================

@dataclass
class EpisodeResult:
    """Results from a single evaluation episode."""
    total_reward: float
    total_revenue: float
    total_bookings: int
    daily_revenues: List[float]
    daily_bookings: List[int]
    daily_alphas: Dict[str, List[float]]
    daily_prices: Dict[str, List[float]]
    episode_length: int


@dataclass
class StrategyMetrics:
    """Computed metrics for a strategy."""
    name: str
    n_episodes: int
    
    # Revenue metrics
    mean_revenue: float
    std_revenue: float
    revenue_ci_lower: float
    revenue_ci_upper: float
    total_revenue: float
    
    # Booking metrics
    mean_bookings: float
    std_bookings: float
    total_bookings: int
    
    # Efficiency metrics
    revenue_per_booking: float
    daily_retention_rate: float  # bookings per day
    
    # Reward metrics
    mean_reward: float
    std_reward: float
    
    # Per-room metrics
    room_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Raw data for statistical tests
    all_revenues: List[float] = field(default_factory=list)
    all_bookings: List[int] = field(default_factory=list)
    all_rewards: List[float] = field(default_factory=list)


@dataclass
class StatisticalTest:
    """Results from statistical hypothesis test."""
    strategy_name: str
    baseline_name: str
    
    # Revenue comparison
    revenue_lift_pct: float
    revenue_t_statistic: float
    revenue_p_value: float
    revenue_significant: bool
    revenue_effect_size: float  # Cohen's d
    
    # Booking comparison
    booking_diff: float
    booking_t_statistic: float
    booking_p_value: float
    booking_significant: bool
    
    # Retention comparison
    retention_diff: float
    
    # Interpretation
    interpretation: str


# ============================================
# BASELINE STRATEGIES
# ============================================

class BaselineStrategy:
    """Base class for pricing strategies."""
    
    def __init__(self, name: str, room_types: List[str], adr_refs: Dict[str, float]):
        self.name = name
        self.room_types = room_types
        self.adr_refs = adr_refs
        self.n_rooms = len(room_types)
    
    def get_action(self, state: np.ndarray, day_info: Dict = None) -> np.ndarray:
        """Return price multipliers (alphas) for all rooms."""
        raise NotImplementedError
    
    def reset(self):
        """Reset strategy state (if any)."""
        pass


class FixedPriceStrategy(BaselineStrategy):
    """Fixed price multiplier for all rooms (static pricing)."""
    
    def __init__(self, alpha: float, room_types: List[str], adr_refs: Dict[str, float]):
        super().__init__(f"Fixed (α={alpha})", room_types, adr_refs)
        self.alpha = alpha
    
    def get_action(self, state: np.ndarray, day_info: Dict = None) -> np.ndarray:
        return np.full(self.n_rooms, self.alpha)


class SegmentedPriceStrategy(BaselineStrategy):
    """
    Segmented pricing: different fixed prices for room segments.
    
    Budget rooms: lower multiplier
    Standard rooms: base multiplier  
    Premium rooms: higher multiplier
    """
    
    def __init__(self, room_types: List[str], adr_refs: Dict[str, float],
                 budget_alpha: float = 0.92,
                 standard_alpha: float = 1.0,
                 premium_alpha: float = 1.08):
        super().__init__("Segmented Pricing", room_types, adr_refs)
        
        # Classify rooms by ADR into 3 segments
        sorted_rooms = sorted(room_types, key=lambda r: adr_refs[r])
        n = len(sorted_rooms)
        n_budget = n // 3
        n_premium = n // 3
        
        self.alphas = {}
        for i, room in enumerate(sorted_rooms):
            if i < n_budget:
                self.alphas[room] = budget_alpha
            elif i >= n - n_premium:
                self.alphas[room] = premium_alpha
            else:
                self.alphas[room] = standard_alpha
    
    def get_action(self, state: np.ndarray, day_info: Dict = None) -> np.ndarray:
        return np.array([self.alphas[room] for room in self.room_types])


class WeekendWeekdayStrategy(BaselineStrategy):
    """
    Time-based pricing: higher prices on weekends.
    
    Uses day_info to determine if weekend.
    """
    
    def __init__(self, room_types: List[str], adr_refs: Dict[str, float],
                 weekday_alpha: float = 0.95,
                 weekend_alpha: float = 1.12):
        super().__init__("Weekend/Weekday", room_types, adr_refs)
        self.weekday_alpha = weekday_alpha
        self.weekend_alpha = weekend_alpha
        self._day_counter = 0
    
    def get_action(self, state: np.ndarray, day_info: Dict = None) -> np.ndarray:
        # Check if weekend from day_info or state
        is_weekend = False
        if day_info and 'is_weekend' in day_info:
            is_weekend = day_info['is_weekend']
        elif day_info and 'day_of_week' in day_info:
            is_weekend = day_info['day_of_week'] >= 5
        else:
            # Fallback: simulate weekly pattern
            is_weekend = self._day_counter % 7 >= 5
            self._day_counter += 1
        
        alpha = self.weekend_alpha if is_weekend else self.weekday_alpha
        return np.full(self.n_rooms, alpha)
    
    def reset(self):
        self._day_counter = 0


class SeasonalHeuristicStrategy(BaselineStrategy):
    """
    Demand-based heuristic: adjust prices based on demand signals in state.
    
    Uses rolling booking averages from state to estimate demand.
    """
    
    def __init__(self, room_types: List[str], adr_refs: Dict[str, float],
                 low_demand_alpha: float = 0.88,
                 high_demand_alpha: float = 1.15):
        super().__init__("Demand Heuristic", room_types, adr_refs)
        self.low_alpha = low_demand_alpha
        self.high_alpha = high_demand_alpha
    
    def get_action(self, state: np.ndarray, day_info: Dict = None) -> np.ndarray:
        # Estimate demand from state features
        # Assume normalized features, higher values = higher demand
        if len(state) > 10:
            # Use mean of demand-related features (first portion of state)
            demand_signal = np.clip(np.mean(state[5:15]), 0, 1)
        else:
            demand_signal = 0.5
        
        # Linear interpolation between low and high alpha
        alpha = self.low_alpha + (self.high_alpha - self.low_alpha) * demand_signal
        return np.full(self.n_rooms, alpha)


class DynamicRuleBasedStrategy(BaselineStrategy):
    """
    Rule-based dynamic pricing combining multiple factors.
    
    Considers: weekday/weekend, demand level, room segment.
    """
    
    def __init__(self, room_types: List[str], adr_refs: Dict[str, float]):
        super().__init__("Rule-Based Dynamic", room_types, adr_refs)
        
        # Classify rooms
        sorted_rooms = sorted(room_types, key=lambda r: adr_refs[r])
        n = len(sorted_rooms)
        self.room_segment = {}
        for i, room in enumerate(sorted_rooms):
            if i < n // 3:
                self.room_segment[room] = 'budget'
            elif i >= n - n // 3:
                self.room_segment[room] = 'premium'
            else:
                self.room_segment[room] = 'standard'
        
        self._day_counter = 0
    
    def get_action(self, state: np.ndarray, day_info: Dict = None) -> np.ndarray:
        # Determine factors
        is_weekend = self._day_counter % 7 >= 5
        self._day_counter += 1
        
        demand_signal = np.clip(np.mean(state[5:15]), 0, 1) if len(state) > 10 else 0.5
        
        alphas = []
        for room in self.room_types:
            # Base alpha by segment
            if self.room_segment[room] == 'budget':
                base = 0.95
            elif self.room_segment[room] == 'premium':
                base = 1.05
            else:
                base = 1.0
            
            # Weekend adjustment
            if is_weekend:
                base += 0.05
            
            # Demand adjustment
            base += (demand_signal - 0.5) * 0.1
            
            alphas.append(np.clip(base, 0.8, 1.2))
        
        return np.array(alphas)
    
    def reset(self):
        self._day_counter = 0


class RandomStrategy(BaselineStrategy):
    """Random pricing within bounds (noise baseline)."""
    
    def __init__(self, room_types: List[str], adr_refs: Dict[str, float],
                 alpha_min: float = 0.85, alpha_max: float = 1.15):
        super().__init__("Random", room_types, adr_refs)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
    
    def get_action(self, state: np.ndarray, day_info: Dict = None) -> np.ndarray:
        return np.random.uniform(self.alpha_min, self.alpha_max, self.n_rooms)


class PPOStrategy(BaselineStrategy):
    """Trained PPO policy."""
    
    def __init__(self, model, room_types: List[str], adr_refs: Dict[str, float],
                 deterministic: bool = True):
        super().__init__("PPO (Ours)", room_types, adr_refs)
        self.model = model
        self.deterministic = deterministic
        self.model.eval()
    
    def get_action(self, state: np.ndarray, day_info: Dict = None) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            actions, _, _ = self.model(state_tensor, deterministic=self.deterministic)
            return actions.squeeze(0).numpy()


# ============================================
# EVALUATION ENGINE
# ============================================

class EvaluationEngine:
    """
    Core evaluation engine for running simulations and computing metrics.
    """
    
    def __init__(
        self,
        env,
        room_types: List[str],
        adr_refs: Dict[str, float],
        episode_length: int = 30
    ):
        self.env = env
        self.room_types = room_types
        self.adr_refs = adr_refs
        self.episode_length = episode_length
    
    def run_episode(
        self,
        strategy: BaselineStrategy,
        start_idx: int = 0
    ) -> EpisodeResult:
        """Run a single evaluation episode."""
        
        strategy.reset()
        state = self.env.reset(start_idx=start_idx)
        
        daily_revenues = []
        daily_bookings = []
        daily_alphas = {room: [] for room in self.room_types}
        daily_prices = {room: [] for room in self.room_types}
        
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Get day info if available
            day_info = {}
            if hasattr(self.env, 'current_day_info'):
                day_info = self.env.current_day_info
            
            # Get action
            action = strategy.get_action(state, day_info)
            
            # Store prices and alphas
            for i, room in enumerate(self.room_types):
                daily_alphas[room].append(action[i])
                daily_prices[room].append(action[i] * self.adr_refs[room])
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Store daily metrics
            daily_revenues.append(info.get('daily_revenue', 0))
            daily_bookings.append(info.get('daily_bookings', 0))
            
            state = next_state
            step += 1
        
        # Get episode summary
        summary = self.env.get_episode_summary()
        
        return EpisodeResult(
            total_reward=total_reward,
            total_revenue=summary['total_revenue'],
            total_bookings=summary['total_bookings'],
            daily_revenues=daily_revenues,
            daily_bookings=daily_bookings,
            daily_alphas=daily_alphas,
            daily_prices=daily_prices,
            episode_length=step
        )
    
    def evaluate_strategy(
        self,
        strategy: BaselineStrategy,
        n_episodes: int = 10,
        verbose: bool = False
    ) -> StrategyMetrics:
        """Evaluate a strategy over multiple episodes."""
        
        episodes = []
        
        for ep in range(n_episodes):
            # Vary starting point for diversity
            max_start = max(1, len(self.env.daily_df) - self.episode_length - 1)
            start_idx = (ep * self.episode_length) % max_start
            
            result = self.run_episode(strategy, start_idx)
            episodes.append(result)
        
        # Aggregate results
        all_revenues = [ep.total_revenue for ep in episodes]
        all_bookings = [ep.total_bookings for ep in episodes]
        all_rewards = [ep.total_reward for ep in episodes]
        
        # Compute confidence interval (95%)
        n = len(all_revenues)
        mean_rev = np.mean(all_revenues)
        std_rev = np.std(all_revenues, ddof=1)
        se_rev = std_rev / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower = mean_rev - t_crit * se_rev
        ci_upper = mean_rev + t_crit * se_rev
        
        # Per-room metrics
        room_metrics = {}
        for room in self.room_types:
            all_alphas = []
            all_prices = []
            for ep in episodes:
                all_alphas.extend(ep.daily_alphas[room])
                all_prices.extend(ep.daily_prices[room])
            
            room_metrics[room] = {
                'mean_price': np.mean(all_prices),
                'std_price': np.std(all_prices),
                'min_price': np.min(all_prices),
                'max_price': np.max(all_prices),
                'mean_alpha': np.mean(all_alphas),
                'std_alpha': np.std(all_alphas),
                'alpha_variance': np.var(all_alphas),
                'price_deviation': np.mean(all_alphas) - 1.0,
                'adr_ref': self.adr_refs[room]
            }
        
        total_bookings = sum(all_bookings)
        total_days = sum(ep.episode_length for ep in episodes)
        
        return StrategyMetrics(
            name=strategy.name,
            n_episodes=n_episodes,
            mean_revenue=mean_rev,
            std_revenue=std_rev,
            revenue_ci_lower=ci_lower,
            revenue_ci_upper=ci_upper,
            total_revenue=sum(all_revenues),
            mean_bookings=np.mean(all_bookings),
            std_bookings=np.std(all_bookings),
            total_bookings=total_bookings,
            revenue_per_booking=sum(all_revenues) / max(total_bookings, 1),
            daily_retention_rate=total_bookings / max(total_days, 1),
            mean_reward=np.mean(all_rewards),
            std_reward=np.std(all_rewards),
            room_metrics=room_metrics,
            all_revenues=all_revenues,
            all_bookings=all_bookings,
            all_rewards=all_rewards
        )


# ============================================
# STATISTICAL ANALYSIS
# ============================================

class StatisticalAnalyzer:
    """
    Perform rigorous statistical analysis comparing strategies.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Significance level
    
    def compute_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def compare_strategies(
        self,
        strategy_metrics: StrategyMetrics,
        baseline_metrics: StrategyMetrics
    ) -> StatisticalTest:
        """
        Compare a strategy against baseline using statistical tests.
        
        Uses Welch's t-test (unequal variances assumed).
        """
        
        # Revenue comparison
        t_stat_rev, p_val_rev = stats.ttest_ind(
            strategy_metrics.all_revenues,
            baseline_metrics.all_revenues,
            equal_var=False  # Welch's t-test
        )
        
        revenue_lift = (
            (strategy_metrics.mean_revenue - baseline_metrics.mean_revenue) 
            / baseline_metrics.mean_revenue * 100
        )
        
        effect_size_rev = self.compute_cohens_d(
            strategy_metrics.all_revenues,
            baseline_metrics.all_revenues
        )
        
        # Booking comparison
        t_stat_book, p_val_book = stats.ttest_ind(
            strategy_metrics.all_bookings,
            baseline_metrics.all_bookings,
            equal_var=False
        )
        
        booking_diff = strategy_metrics.mean_bookings - baseline_metrics.mean_bookings
        
        # Retention comparison
        retention_diff = (
            strategy_metrics.daily_retention_rate - 
            baseline_metrics.daily_retention_rate
        )
        
        # Interpretation
        if p_val_rev < self.alpha and revenue_lift > 0:
            interpretation = f"Significantly BETTER than baseline (p={p_val_rev:.4f}, lift={revenue_lift:+.1f}%)"
        elif p_val_rev < self.alpha and revenue_lift < 0:
            interpretation = f"Significantly WORSE than baseline (p={p_val_rev:.4f}, lift={revenue_lift:+.1f}%)"
        else:
            interpretation = f"No significant difference from baseline (p={p_val_rev:.4f})"
        
        return StatisticalTest(
            strategy_name=strategy_metrics.name,
            baseline_name=baseline_metrics.name,
            revenue_lift_pct=revenue_lift,
            revenue_t_statistic=t_stat_rev,
            revenue_p_value=p_val_rev,
            revenue_significant=p_val_rev < self.alpha,
            revenue_effect_size=effect_size_rev,
            booking_diff=booking_diff,
            booking_t_statistic=t_stat_book,
            booking_p_value=p_val_book,
            booking_significant=p_val_book < self.alpha,
            retention_diff=retention_diff,
            interpretation=interpretation
        )
    
    def run_all_comparisons(
        self,
        all_metrics: Dict[str, StrategyMetrics],
        baseline_name: str = "Fixed (α=1.0)"
    ) -> Dict[str, StatisticalTest]:
        """Compare all strategies against baseline."""
        
        if baseline_name not in all_metrics:
            raise ValueError(f"Baseline '{baseline_name}' not found")
        
        baseline = all_metrics[baseline_name]
        results = {}
        
        for name, metrics in all_metrics.items():
            if name != baseline_name:
                results[name] = self.compare_strategies(metrics, baseline)
        
        return results


# ============================================
# MAIN EVALUATOR CLASS
# ============================================

class ComprehensiveEvaluator:
    """
    Main class for comprehensive model evaluation.
    
    Orchestrates:
    1. Strategy creation
    2. Simulation runs
    3. Metrics computation
    4. Statistical analysis
    5. Report generation
    """
    
    def __init__(
        self,
        env,
        room_types: List[str],
        adr_refs: Dict[str, float],
        ppo_model=None,
        n_episodes: int = 10,
        seed: int = 42
    ):
        self.env = env
        self.room_types = room_types
        self.adr_refs = adr_refs
        self.ppo_model = ppo_model
        self.n_episodes = n_episodes
        self.seed = seed
        
        np.random.seed(seed)
        
        self.engine = EvaluationEngine(
            env=env,
            room_types=room_types,
            adr_refs=adr_refs,
            episode_length=env.episode_length
        )
        
        self.analyzer = StatisticalAnalyzer(alpha=0.05)
        
        # Results storage
        self.all_metrics: Dict[str, StrategyMetrics] = {}
        self.statistical_tests: Dict[str, StatisticalTest] = {}
    
    def create_baseline_strategies(self) -> List[BaselineStrategy]:
        """Create all baseline strategies for comparison."""
        
        strategies = [
            # Static pricing baselines
            FixedPriceStrategy(1.0, self.room_types, self.adr_refs),
            FixedPriceStrategy(0.9, self.room_types, self.adr_refs),
            FixedPriceStrategy(1.1, self.room_types, self.adr_refs),
            
            # Segment-based pricing
            SegmentedPriceStrategy(self.room_types, self.adr_refs),
            
            # Time-based pricing
            WeekendWeekdayStrategy(self.room_types, self.adr_refs),
            
            # Demand-based heuristics
            SeasonalHeuristicStrategy(self.room_types, self.adr_refs),
            DynamicRuleBasedStrategy(self.room_types, self.adr_refs),
            
            # Random baseline
            RandomStrategy(self.room_types, self.adr_refs),
        ]
        
        # Add PPO if available
        if self.ppo_model is not None:
            strategies.insert(0, PPOStrategy(
                self.ppo_model, self.room_types, self.adr_refs,
                deterministic=True
            ))
        
        return strategies
    
    def run_evaluation(self, verbose: bool = True) -> Dict[str, StrategyMetrics]:
        """Run evaluation for all strategies."""
        
        strategies = self.create_baseline_strategies()
        
        if verbose:
            print("\n" + "=" * 75)
            print("COMPREHENSIVE MODEL EVALUATION")
            print("=" * 75)
            print(f"Episodes per strategy: {self.n_episodes}")
            print(f"Episode length: {self.env.episode_length} days")
            print(f"Room types: {self.room_types}")
            print(f"Total strategies: {len(strategies)}")
            print("-" * 75)
        
        for strategy in strategies:
            if verbose:
                print(f"\nEvaluating: {strategy.name}...")
            
            metrics = self.engine.evaluate_strategy(
                strategy,
                n_episodes=self.n_episodes,
                verbose=verbose
            )
            
            self.all_metrics[strategy.name] = metrics
            
            if verbose:
                print(f"  Revenue: €{metrics.mean_revenue:,.0f} ± €{metrics.std_revenue:,.0f}")
                print(f"  95% CI: [€{metrics.revenue_ci_lower:,.0f}, €{metrics.revenue_ci_upper:,.0f}]")
                print(f"  Bookings: {metrics.mean_bookings:.1f} ± {metrics.std_bookings:.1f}")
                print(f"  Rev/Booking: €{metrics.revenue_per_booking:.2f}")
                print(f"  Retention: {metrics.daily_retention_rate:.2f} bookings/day")
        
        return self.all_metrics
    
    def run_statistical_analysis(
        self,
        baseline_name: str = "Fixed (α=1.0)",
        verbose: bool = True
    ) -> Dict[str, StatisticalTest]:
        """Run statistical tests comparing all strategies to baseline."""
        
        self.statistical_tests = self.analyzer.run_all_comparisons(
            self.all_metrics,
            baseline_name
        )
        
        if verbose:
            print("\n" + "=" * 75)
            print(f"STATISTICAL ANALYSIS (vs {baseline_name})")
            print("=" * 75)
            print(f"Significance level: α = 0.05")
            print("-" * 75)
            
            for name, test in self.statistical_tests.items():
                sig_marker = "✓" if test.revenue_significant else " "
                effect_interp = self.analyzer.interpret_effect_size(test.revenue_effect_size)
                
                print(f"\n{name}:")
                print(f"  Revenue Lift: {test.revenue_lift_pct:+.2f}%")
                print(f"  t-statistic: {test.revenue_t_statistic:.3f}")
                print(f"  p-value: {test.revenue_p_value:.4f} [{sig_marker}]")
                print(f"  Effect size (Cohen's d): {test.revenue_effect_size:.3f} ({effect_interp})")
                print(f"  Retention diff: {test.retention_diff:+.3f} bookings/day")
                print(f"  → {test.interpretation}")
        
        return self.statistical_tests
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary comparison table."""
        
        rows = []
        baseline_name = "Fixed (α=1.0)"
        baseline_rev = self.all_metrics.get(baseline_name, 
                                            list(self.all_metrics.values())[0]).mean_revenue
        
        for name, metrics in self.all_metrics.items():
            row = {
                'Strategy': name,
                'Mean Revenue (€)': f"{metrics.mean_revenue:,.0f}",
                'Std Revenue': f"±{metrics.std_revenue:,.0f}",
                '95% CI': f"[{metrics.revenue_ci_lower:,.0f}, {metrics.revenue_ci_upper:,.0f}]",
                'Mean Bookings': f"{metrics.mean_bookings:.1f}",
                'Rev/Booking (€)': f"{metrics.revenue_per_booking:.2f}",
                'Retention (book/day)': f"{metrics.daily_retention_rate:.2f}",
            }
            
            # Add statistical test results
            if name in self.statistical_tests:
                test = self.statistical_tests[name]
                row['Revenue Lift'] = f"{test.revenue_lift_pct:+.1f}%"
                row['p-value'] = f"{test.revenue_p_value:.4f}"
                row['Significant'] = "✓" if test.revenue_significant else ""
            elif name == baseline_name:
                row['Revenue Lift'] = "baseline"
                row['p-value'] = "-"
                row['Significant'] = "-"
            else:
                row['Revenue Lift'] = "N/A"
                row['p-value'] = "N/A"
                row['Significant'] = ""
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_room_analysis(self) -> pd.DataFrame:
        """Generate per-room analysis table."""
        
        rows = []
        
        for strategy_name, metrics in self.all_metrics.items():
            for room, rm in metrics.room_metrics.items():
                rows.append({
                    'Strategy': strategy_name,
                    'Room': room,
                    'ADR Ref (€)': f"{rm['adr_ref']:.0f}",
                    'Mean Price (€)': f"{rm['mean_price']:.0f}",
                    'Std Price': f"±{rm['std_price']:.1f}",
                    'Mean α': f"{rm['mean_alpha']:.3f}",
                    'α Std': f"{rm['std_alpha']:.3f}",
                    'Price Deviation': f"{rm['price_deviation']:+.3f}",
                    'α Variance': f"{rm['alpha_variance']:.4f}"
                })
        
        return pd.DataFrame(rows)
    
    def save_results(self, output_dir: str = "evaluation_results"):
        """Save all results to files."""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Summary table
        summary_df = self.generate_summary_table()
        summary_df.to_csv(f"{output_dir}/summary_{timestamp}.csv", index=False)
        
        # Room analysis
        room_df = self.generate_room_analysis()
        room_df.to_csv(f"{output_dir}/room_analysis_{timestamp}.csv", index=False)
        
        # Detailed JSON results
        detailed = {}
        for name, metrics in self.all_metrics.items():
            # Convert room_metrics to JSON-serializable format
            room_metrics_json = {}
            for room, rm in metrics.room_metrics.items():
                room_metrics_json[room] = {
                    k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                    for k, v in rm.items()
                }
            
            detailed[name] = {
                'mean_revenue': float(metrics.mean_revenue),
                'std_revenue': float(metrics.std_revenue),
                'revenue_ci': [float(metrics.revenue_ci_lower), float(metrics.revenue_ci_upper)],
                'mean_bookings': float(metrics.mean_bookings),
                'revenue_per_booking': float(metrics.revenue_per_booking),
                'daily_retention_rate': float(metrics.daily_retention_rate),
                'room_metrics': room_metrics_json
            }
            
            if name in self.statistical_tests:
                test = self.statistical_tests[name]
                detailed[name]['statistical_test'] = {
                    'revenue_lift_pct': float(test.revenue_lift_pct),
                    'p_value': float(test.revenue_p_value),
                    'significant': bool(test.revenue_significant),
                    'effect_size': float(test.revenue_effect_size),
                    'interpretation': test.interpretation
                }
        
        with open(f"{output_dir}/detailed_results_{timestamp}.json", 'w') as f:
            json.dump(detailed, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")
        
        return summary_df, room_df


# ============================================
# MAIN FUNCTION
# ============================================

def run_comprehensive_evaluation(
    hotel_path: str = 'data/resort_hotel_data.csv',
    trends_path: str = 'data/google_trends_portugal_algarve_daily.csv',
    checkpoint_path: str = 'checkpoints/best_model.pth',
    n_episodes: int = 10,
    output_dir: str = 'evaluation_results',
    seed: int = 42
) -> Tuple[Dict[str, StrategyMetrics], Dict[str, StatisticalTest], pd.DataFrame]:
    """
    Run complete evaluation pipeline.
    
    Returns:
        all_metrics: Dict of strategy metrics
        statistical_tests: Dict of statistical test results
        summary_df: Summary comparison table
    """
    
    print("=" * 75)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("=" * 75)
    
    # Imports
    from data.load_data import load_and_prepare_all, get_feature_columns
    from models.demand_models import prepare_and_train_demand_models
    from envs.multi_room_env import MultiRoomPricingEnv
    from models.actor_critic import MultiRoomActorCritic
    
    # Load data
    print("\n[1/6] Loading data...")
    data = load_and_prepare_all(hotel_path, trends_path)
    room_types = data['room_types']
    adr_refs = data['adr_refs']
    
    print(f"  Room types: {room_types}")
    print(f"  Test set: {len(data['test_df'])} days")
    
    # Train demand models
    print("\n[2/6] Training demand models...")
    demand_models = prepare_and_train_demand_models(
        hotel_df=data['hotel_df'],
        daily_df=data['train_df'],
        room_types=room_types,
        adr_refs=adr_refs
    )
    
    # Create test environment
    print("\n[3/6] Creating test environment...")
    numerical_features, categorical_features = get_feature_columns(room_types)
    feature_columns = numerical_features + categorical_features
    
    test_env = MultiRoomPricingEnv(
        daily_df=data['test_df'],
        demand_models=demand_models,
        room_types=room_types,
        adr_refs=adr_refs,
        preprocessor=data['preprocessor'],
        feature_columns=feature_columns,
        episode_length=30,
        seed=seed
    )
    
    # Load PPO model
    ppo_model = None
    if os.path.exists(checkpoint_path):
        print(f"\n[4/6] Loading PPO model from {checkpoint_path}...")
        ppo_model = MultiRoomActorCritic(
            state_dim=test_env.state_dim,
            room_types=room_types
        )
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            ppo_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            ppo_model.load_state_dict(checkpoint)
        
        ppo_model.eval()
        print("  ✓ Model loaded successfully")
    else:
        print(f"\n[4/6] No checkpoint at {checkpoint_path}, evaluating baselines only")
    
    # Run evaluation
    print("\n[5/6] Running evaluation...")
    evaluator = ComprehensiveEvaluator(
        env=test_env,
        room_types=room_types,
        adr_refs=adr_refs,
        ppo_model=ppo_model,
        n_episodes=n_episodes,
        seed=seed
    )
    
    all_metrics = evaluator.run_evaluation(verbose=True)
    statistical_tests = evaluator.run_statistical_analysis(verbose=True)
    
    # Generate and save results
    print("\n[6/6] Generating reports...")
    summary_df, room_df = evaluator.save_results(output_dir)
    
    # Print final summary
    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)
    
    return all_metrics, statistical_tests, summary_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--hotel', type=str, default='data/resort_hotel_data.csv')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--output', type=str, default='evaluation_results')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    run_comprehensive_evaluation(
        hotel_path=args.hotel,
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        output_dir=args.output,
        seed=args.seed
    )
