#!/usr/bin/env python3
"""
CFR+ Trainer - High-level interface for training CFR+ strategies
===============================================================

This module provides a convenient high-level interface for training 
CFR+ strategies for different games.
"""

import argparse
from typing import Optional, Dict, Any
import numpy as np

from .base import CFRPlusBase
from .games import KuhnPokerGame, LeducHoldemGame


class CFRPlusTrainer:
    """High-level trainer for CFR+ strategies."""
    
    SUPPORTED_GAMES = {
        "kuhn_poker": KuhnPokerGame,
        "kuhn": KuhnPokerGame,  # Alias
        "leduc_holdem": LeducHoldemGame,
        "leduc": LeducHoldemGame,  # Alias
    }
    
    def __init__(self, game_name: str, delay: int = 0, **game_kwargs):
        """
        Initialize trainer for a specific game.
        
        Args:
            game_name: Name of the game ("kuhn_poker", "leduc_holdem")
            delay: CFR+ averaging delay parameter
            **game_kwargs: Additional arguments for game initialization
        """
        if game_name not in self.SUPPORTED_GAMES:
            raise ValueError(f"Unsupported game: {game_name}. Supported: {list(self.SUPPORTED_GAMES.keys())}")
        
        game_class = self.SUPPORTED_GAMES[game_name]
        self.game = game_class(**game_kwargs)
        self.trainer = CFRPlusBase(self.game, delay=delay)
        self.game_name = game_name
    
    def train(
        self, 
        iterations: int,
        seed: Optional[int] = 42,
        log_every: int = 10000,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the CFR+ strategy.
        
        Args:
            iterations: Number of training iterations
            seed: Random seed for reproducibility
            log_every: How often to log progress and strategy
            save_path: Optional path to save the final strategy
        
        Returns:
            Dictionary containing training results and final strategy
        """
        print(f"Training CFR+ for {self.game_name}")
        print(f"Iterations: {iterations:,}")
        print(f"Delay: {self.trainer.delay}")
        print("=" * 50)
        
        # Train the strategy
        self.trainer.train(iterations, seed=seed, log_every=log_every)
        
        print("=" * 50)
        print("Training completed!")
        
        # Get final strategy
        final_strategy = self.trainer.get_strategy_dict()
        
        # Save if requested
        if save_path:
            self.trainer.save_strategy(save_path)
        
        return {
            "game_name": self.game_name,
            "iterations": iterations,
            "delay": self.trainer.delay,
            "num_infosets": len(self.trainer.infosets),
            "strategy": final_strategy,
        }
    
    def save_strategy(self, filepath: str) -> None:
        """Save the current strategy to a file."""
        self.trainer.save_strategy(filepath)
    
    def load_strategy(self, filepath: str) -> None:
        """Load a strategy from a file."""
        self.trainer.load_strategy(filepath)
    
    def get_strategy(self) -> Dict[str, np.ndarray]:
        """Get the current average strategy."""
        return self.trainer.get_strategy_dict()
    
    def print_strategy(self) -> None:
        """Print the current strategy."""
        self.trainer.print_strategy()
    
    def save_with_overrides(self, filepath: str, overrides: Optional[Dict[str, list]] = None) -> None:
        """
        Save strategy with optional manual overrides.
        
        Args:
            filepath: Path to save the strategy
            overrides: Dictionary of infoset_key -> strategy_list overrides
        """
        strategy = self.get_strategy()
        
        if overrides:
            print(f"Applying {len(overrides)} strategy overrides...")
            for key, override_strategy in overrides.items():
                strategy[key] = np.array(override_strategy)
                print(f"  {key}: {override_strategy}")
        
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(strategy, f)
        print(f"Strategy with overrides saved to {filepath}")


def main():
    """Command-line interface for CFR+ training."""
    parser = argparse.ArgumentParser(description="Train CFR+ strategies for card games")
    
    parser.add_argument("game", choices=list(CFRPlusTrainer.SUPPORTED_GAMES.keys()),
                       help="Game to train on")
    parser.add_argument("-n", "--iterations", type=int, default=100000,
                       help="Number of training iterations")
    parser.add_argument("--delay", type=int, default=0,
                       help="CFR+ averaging delay")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log-every", type=int, default=10000,
                       help="Logging frequency")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save the trained strategy")
    parser.add_argument("--save-with-alpha-0", action="store_true",
                       help="Save Kuhn Poker strategy with alpha=0 overrides")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CFRPlusTrainer(args.game, delay=args.delay)
    
    # Train
    results = trainer.train(
        iterations=args.iterations,
        seed=args.seed,
        log_every=args.log_every,
        save_path=args.save,
    )
    
    # Print final results
    print("\nFinal Results:")
    print(f"Game: {results['game_name']}")
    print(f"Iterations: {results['iterations']:,}")
    print(f"Information sets: {results['num_infosets']}")
    
    # Save with alpha=0 overrides for Kuhn Poker if requested
    if args.save_with_alpha_0 and args.game in ["kuhn_poker", "kuhn"]:
        alpha_0_overrides = {
            "J:": [1.0, 0.0],    # Always check with Jack
            "K:": [0.0, 1.0],    # Always bet with King  
            "Q:": [1.0, 0.0],    # Always check with Queen
            "Q:b": [2/3, 1/3],   # Fold 2/3, call 1/3 with Queen facing bet
            "J:b": [1.0, 0.0],   # Always fold with Jack facing bet
            "K:b": [0.0, 1.0],   # Always call with King facing bet
        }
        
        alpha_0_path = args.save.replace('.pkl', '_alpha_0.pkl') if args.save else 'cfr_plus_alpha_0.pkl'
        trainer.save_with_overrides(alpha_0_path, alpha_0_overrides)
        print(f"Alpha=0 GTO strategy saved to {alpha_0_path}")


if __name__ == "__main__":
    main()