#!/usr/bin/env python3
"""
Modular CFR+ Training Script
============================

This script replaces the original train_cfr_.py with the new modular CFR+ implementation.
It can train CFR+ strategies for multiple card games using a unified interface.
"""

import argparse
from spiral.cfr.trainer import CFRPlusTrainer


def main():
    parser = argparse.ArgumentParser(description="Train CFR+ strategies using modular implementation")
    
    parser.add_argument("game", choices=["kuhn_poker", "kuhn", "leduc_holdem", "leduc"],
                       help="Game to train CFR+ strategy for", default="kuhn_poker")
    parser.add_argument("-n", "--iterations", type=int, default=100000,
                       help="Number of training iterations")
    parser.add_argument("--delay", type=int, default=0,
                       help="CFR+ averaging delay parameter")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--log-every", type=int, default=10000,
                       help="How often to log progress")
    parser.add_argument("--save", type=str, default="cfr_plus_strategy.pkl",
                       help="Path to save the trained strategy")
    parser.add_argument("--save-with-alpha-0", action="store_true",
                       help="Also save Kuhn Poker strategy with alpha=0 GTO overrides")
    
    args = parser.parse_args()
    
    print(f"Training CFR+ for {args.game}")
    print(f"Parameters: iterations={args.iterations:,}, delay={args.delay}, seed={args.seed}")
    print("=" * 60)
    
    # Create trainer
    trainer = CFRPlusTrainer(args.game, delay=args.delay)
    
    # Train
    results = trainer.train(
        iterations=args.iterations,
        seed=args.seed,
        log_every=args.log_every,
        save_path=args.save,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Game: {results['game_name']}")
    print(f"Total iterations: {results['iterations']:,}")
    print(f"Information sets created: {results['num_infosets']}")
    print(f"Strategy saved to: {args.save}")
    
    # Save alpha=0 version for Kuhn Poker if requested
    if args.save_with_alpha_0 and args.game in ["kuhn_poker", "kuhn"]:
        print("\nCreating alpha=0 GTO strategy...")
        alpha_0_overrides = {
            "J:": [1.0, 0.0],    # Player 1 with Jack: always pass (never bet)
            "K:": [0.0, 1.0],    # Player 1 with King: always bet
            "Q:": [1.0, 0.0],    # Player 1 with Queen: always check
            "Q:b": [2/3, 1/3],   # Player 2 with Queen: fold 2/3, call 1/3
            "J:b": [1.0, 0.0],   # Player 2 with Jack: always fold
            "K:b": [0.0, 1.0],   # Player 2 with King: always call
        }
        
        alpha_0_path = args.save.replace('.pkl', '_alpha_0.pkl')
        trainer.save_with_overrides(alpha_0_path, alpha_0_overrides)
        print(f"Alpha=0 GTO strategy saved to: {alpha_0_path}")
        
    print("\nFinal strategy:")
    trainer.print_strategy()


if __name__ == "__main__":
    main()