#!/usr/bin/env python3
"""
Modular CFR+ Implementation - Base Classes
=========================================

This module provides a modular CFR+ implementation that can be easily extended
to support different card games while maintaining the core CFR+ algorithm.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pickle


class InfoSet:
    """A single information-set node with CFR+ updates."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.regret_sum = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions)
        self.strategy = np.full(n_actions, 1.0 / n_actions)

    def get_strategy(
        self,
        self_reach: float,
        iteration: int,
        delay: int = 0,
    ) -> np.ndarray:
        """Get current strategy using positive regret-matching with linear averaging."""
        # CFR+: positive regret-matching
        pos_regrets = np.maximum(self.regret_sum, 0.0)
        norm = pos_regrets.sum()
        
        if norm > 1e-12:
            self.strategy = pos_regrets / norm
        else:
            self.strategy = np.full(self.n_actions, 1.0 / self.n_actions)

        # Linear averaging with delay
        weight = max(iteration - delay, 0)
        self.strategy_sum += weight * self_reach * self.strategy
        
        return self.strategy

    def add_regrets(
        self, 
        action_utilities: np.ndarray, 
        node_utility: float, 
        opponent_reach: float
    ) -> None:
        """Add regrets with CFR+ positive clipping."""
        self.regret_sum = np.maximum(
            self.regret_sum + opponent_reach * (action_utilities - node_utility),
            0.0
        )

    def get_average_strategy(self) -> np.ndarray:
        """Get the average strategy over all iterations."""
        norm = self.strategy_sum.sum()
        if norm > 1e-12:
            return self.strategy_sum / norm
        else:
            return np.full(self.n_actions, 1.0 / self.n_actions)


class GameInterface(ABC):
    """Abstract interface for different card games."""

    @abstractmethod
    def get_num_actions(self) -> int:
        """Return the number of actions available in this game."""
        pass

    @abstractmethod
    def is_terminal(self, history: str) -> bool:
        """Check if a game history represents a terminal state."""
        pass

    @abstractmethod
    def get_payoff(self, cards: List[int], history: str, player: int) -> float:
        """Get the payoff for a player given cards, history, and player position."""
        pass

    @abstractmethod
    def get_valid_actions(self, history: str) -> List[int]:
        """Get the valid actions for the current game state."""
        pass

    @abstractmethod
    def get_infoset_key(self, player_card: int, history: str) -> str:
        """Generate a unique key for the information set."""
        pass

    @abstractmethod
    def get_all_possible_deals(self) -> List[Tuple[int, ...]]:
        """Get all possible card deals for chance sampling."""
        pass

    @abstractmethod
    def action_to_history_char(self, action: int) -> str:
        """Convert an action index to a character for history tracking."""
        pass


class CFRPlusBase:
    """Base CFR+ trainer that works with any GameInterface implementation."""

    def __init__(self, game: GameInterface, delay: int = 0):
        self.game = game
        self.delay = delay
        self.infosets: Dict[str, InfoSet] = {}
        self.num_actions = game.get_num_actions()

    def get_or_create_infoset(self, key: str) -> InfoSet:
        """Get an existing infoset or create a new one."""
        if key not in self.infosets:
            self.infosets[key] = InfoSet(self.num_actions)
        return self.infosets[key]

    def cfr(
        self,
        cards: List[int],
        history: str,
        reach_probabilities: List[float],
        traversing_player: int,
        iteration: int,
    ) -> float:
        """Core CFR recursion."""
        if self.game.is_terminal(history):
            return self.game.get_payoff(cards, history, traversing_player)

        # Determine acting player based on history length
        acting_player = len(history) % len(cards)
        
        # Get information set
        infoset_key = self.game.get_infoset_key(cards[acting_player], history)
        infoset = self.get_or_create_infoset(infoset_key)

        # Get strategy for the acting player
        strategy = infoset.get_strategy(
            reach_probabilities[acting_player], 
            iteration, 
            self.delay
        )

        # Calculate action utilities
        valid_actions = self.game.get_valid_actions(history)
        action_utilities = np.zeros(self.num_actions)
        node_utility = 0.0

        for action in valid_actions:
            # Build next history
            next_history = history + self.game.action_to_history_char(action)
            
            # Update reach probabilities
            next_reach = reach_probabilities.copy()
            next_reach[acting_player] *= strategy[action]
            
            # Recursive call
            action_utilities[action] = self.cfr(
                cards, next_history, next_reach, traversing_player, iteration
            )
            node_utility += strategy[action] * action_utilities[action]

        # Update regrets only for the traversing player
        if acting_player == traversing_player:
            opponent_reach = np.prod([reach_probabilities[i] for i in range(len(cards)) if i != acting_player])
            infoset.add_regrets(action_utilities, node_utility, opponent_reach)

        return node_utility

    def train(
        self,
        iterations: int,
        seed: Optional[int] = 42,
        log_every: int = 10000,
    ) -> None:
        """Train the CFR+ strategy."""
        if seed is not None:
            np.random.seed(seed)

        all_deals = self.game.get_all_possible_deals()
        
        for iteration in range(iterations):
            # Chance sampling: iterate over all possible deals
            for deal in all_deals:
                # External sampling: train each player separately
                for player in range(len(deal)):
                    initial_reach = [1.0] * len(deal)
                    self.cfr(list(deal), "", initial_reach, player, iteration)

            if (iteration + 1) % log_every == 0:
                print(f"Iteration {iteration + 1:,}/{iterations:,}")
                self.print_strategy()
                print()

    def print_strategy(self) -> None:
        """Print the current average strategy."""
        print("Average Strategy:")
        sorted_keys = sorted(self.infosets.keys())
        for key in sorted_keys:
            strategy = self.infosets[key].get_average_strategy()
            strategy_str = ", ".join(f"{prob:.3f}" for prob in strategy)
            print(f"{key:>8}: [{strategy_str}]")

    def save_strategy(self, filepath: str) -> None:
        """Save the average strategy to a file."""
        strategy_dict = {}
        for key, infoset in self.infosets.items():
            strategy_dict[key] = infoset.get_average_strategy()
        
        with open(filepath, "wb") as f:
            pickle.dump(strategy_dict, f)
        print(f"Strategy saved to {filepath}")

    def load_strategy(self, filepath: str) -> None:
        """Load a strategy from a file."""
        with open(filepath, "rb") as f:
            strategy_dict = pickle.load(f)
        
        self.infosets = {}
        for key, strategy in strategy_dict.items():
            infoset = InfoSet(self.num_actions)
            # Set strategy_sum to recreate the average strategy
            infoset.strategy_sum = strategy * 1000  # Arbitrary scaling
            self.infosets[key] = infoset
        print(f"Strategy loaded from {filepath}")

    def get_strategy_dict(self) -> Dict[str, np.ndarray]:
        """Get the current average strategy as a dictionary."""
        return {
            key: infoset.get_average_strategy() 
            for key, infoset in self.infosets.items()
        }