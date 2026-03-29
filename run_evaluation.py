#!/usr/bin/env python
# run_evaluation.py
"""
Standalone Evaluation Script for Multi-Room Hotel Pricing.

This script runs comprehensive evaluation of trained PPO model
against multiple baseline strategies.

Usage:
    python run_evaluation.py                    # Full evaluation
    python run_evaluation.py --episodes 20     # More episodes
    python run_evaluation.py --no-ppo          # Baselines only

Output:
    - evaluation_results/summary_TIMESTAMP.csv
    - evaluation_results/room_analysis_TIMESTAMP.csv
    - evaluation_results/detailed_results_TIMESTAMP.json
    - evaluation_results/*.png (visualization plots)
"""

import sys
sys.path.insert(0, '.')

import argparse
import os
import torch
import numpy as np
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Model Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_evaluation.py
    python run_evaluation.py --hotel data/city_hotel_data.csv
    python run_evaluation.py --episodes 20 --output my_results
    python run_evaluation.py --no-ppo  # Baselines only
        """
    )
    
    parser.add_argument('--hotel', type=str, default='data/city_hotel_data.csv',
                        help='Path to hotel data CSV')
    parser.add_argument('--trends', type=str, 
                        default='data/google_trends_portugal_algarve_daily.csv',
                        help='Path to trends data CSV')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes per strategy')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-ppo', action='store_true',
                        help='Skip PPO evaluation (baselines only)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 75)
    print("MULTI-ROOM HOTEL PRICING - COMPREHENSIVE EVALUATION")
    print("=" * 75)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hotel data: {args.hotel}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output}")
    print("=" * 75)
    
    # Import modules
    from data.load_data import load_and_prepare_all, get_feature_columns
    from models.demand_models import prepare_and_train_demand_models
    from envs.multi_room_env import MultiRoomPricingEnv
    from models.actor_critic import MultiRoomActorCritic
    from evaluation.evaluate_model import ComprehensiveEvaluator
    from evaluation.visualize import generate_all_visualizations
    
    # Step 1: Load data
    print("\n[1/6] Loading and preparing data...")
    data = load_and_prepare_all(
        hotel_path=args.hotel,
        trends_path=args.trends
    )
    
    room_types = data['room_types']
    adr_refs = data['adr_refs']
    room_capacities = data['room_capacities']
    
    print(f"  Room types: {room_types}")
    print(f"  ADR refs: {adr_refs}")
    print(f"  Test set: {len(data['test_df'])} days")
    
    # Step 2: Train demand models
    print("\n[2/6] Training demand models...")
    demand_models = prepare_and_train_demand_models(
        hotel_df=data['hotel_df'],
        daily_df=data['train_df'],
        room_types=room_types,
        adr_refs=adr_refs
    )
    
    # Step 3: Create test environment
    print("\n[3/6] Creating test environment...")
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
        seed=args.seed
    )
    
    print(f"  State dim: {test_env.state_dim}")
    print(f"  Episode length: {test_env.episode_length} days")
    
    # Step 4: Load PPO model (if available and requested)
    ppo_model = None
    if not args.no_ppo and os.path.exists(args.checkpoint):
        print(f"\n[4/6] Loading PPO model from {args.checkpoint}...")
        
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        
        # Handle different checkpoint formats to extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            best_reward = checkpoint.get('best_reward', 'N/A')
        else:
            state_dict = checkpoint
            best_reward = 'N/A'
            
        # Infer checkpoint configuration
        chkpt_rooms = [k.split('.')[1] for k in state_dict.keys() if k.startswith('actor_heads.') and k.endswith('.network.0.weight')]
        chkpt_state_dim = state_dict['backbone.network.0.weight'].shape[1]
        
        # Validate compatibility
        if set(chkpt_rooms) != set(room_types):
            print(f"\n[ERROR] Model checkpoint mismatch!")
            print(f"Checkpoint was trained on room types: {chkpt_rooms} (State dim: {chkpt_state_dim})")
            print(f"Current evaluation environment has: {room_types} (State dim: {test_env.state_dim})")
            print("\nThis usually happens when trying to evaluate a model trained on one hotel (e.g., City Hotel)")
            print("using the data from another hotel (e.g., Resort Hotel).")
            print("\nPlease specify the correct dataset. For example:")
            print("  python run_evaluation.py --hotel data/city_hotel_data.csv")
            sys.exit(1)
            
        if chkpt_state_dim != test_env.state_dim:
            print(f"\n[ERROR] State dimension mismatch!")
            print(f"Checkpoint state dim: {chkpt_state_dim}, Env state dim: {test_env.state_dim}")
            print("The environment features have changed. Please retrain the model.")
            sys.exit(1)
            
        ppo_model = MultiRoomActorCritic(
            state_dim=test_env.state_dim,
            room_types=room_types
        )
        
        ppo_model.load_state_dict(state_dict)
        
        ppo_model.eval()
        
        print("  ✓ PPO model loaded successfully")
        print(f"  Best training reward: {best_reward}")
    else:
        print(f"\n[4/6] Skipping PPO model (--no-ppo or checkpoint not found)")
    
    # Step 5: Run comprehensive evaluation
    print("\n[5/6] Running comprehensive evaluation...")
    
    evaluator = ComprehensiveEvaluator(
        env=test_env,
        room_types=room_types,
        adr_refs=adr_refs,
        ppo_model=ppo_model,
        n_episodes=args.episodes,
        seed=args.seed
    )
    
    # Run evaluation
    all_metrics = evaluator.run_evaluation(verbose=True)
    
    # Run statistical analysis
    statistical_tests = evaluator.run_statistical_analysis(verbose=True)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    summary_df, room_df = evaluator.save_results(args.output)
    
    # Step 6: Generate visualizations
    if not args.no_plots:
        print("\n[6/6] Generating visualizations...")
        
        generate_all_visualizations(
            metrics=all_metrics,
            statistical_tests=statistical_tests,
            room_types=room_types,
            adr_refs=adr_refs,
            output_dir=args.output
        )
    else:
        print("\n[6/6] Skipping visualizations (--no-plots)")
    
    # Final summary
    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)
    
    print("\nSUMMARY TABLE:")
    print("-" * 75)
    print(summary_df.to_string(index=False))
    
    print(f"\n✓ Results saved to: {args.output}/")
    print("  - summary_*.csv")
    print("  - room_analysis_*.csv")
    print("  - detailed_results_*.json")
    if not args.no_plots:
        print("  - *.png (visualization plots)")
    
    # Highlight key findings
    print("\n" + "=" * 75)
    print("KEY FINDINGS")
    print("=" * 75)
    
    if ppo_model and 'PPO (Ours)' in statistical_tests:
        ppo_test = statistical_tests['PPO (Ours)']
        print(f"\nPPO vs Fixed (α=1.0) Baseline:")
        print(f"  Revenue Lift: {ppo_test.revenue_lift_pct:+.2f}%")
        print(f"  p-value: {ppo_test.revenue_p_value:.4f}")
        print(f"  Significant: {'Yes ✓' if ppo_test.revenue_significant else 'No'}")
        print(f"  Effect Size: {ppo_test.revenue_effect_size:.3f}")
        print(f"  → {ppo_test.interpretation}")
    
    # Find best strategy
    best_strategy = max(all_metrics.items(), 
                        key=lambda x: x[1].mean_revenue)
    print(f"\nBest Strategy by Revenue: {best_strategy[0]}")
    print(f"  Mean Revenue: €{best_strategy[1].mean_revenue:,.0f}")
    
    print("\n" + "=" * 75)
    
    return all_metrics, statistical_tests, summary_df


if __name__ == '__main__':
    main()
