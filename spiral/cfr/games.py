#!/usr/bin/env python3
"""
Game-specific implementations for CFR+
======================================

This module contains concrete implementations of the GameInterface 
for different card games.
"""

import itertools
from typing import List, Tuple
from .base import GameInterface


class KuhnPokerGame(GameInterface):
    """Kuhn Poker game implementation for CFR+."""
    
    # Action constants
    PASS = 0
    BET = 1
    
    # Card constants 
    JACK = 0
    QUEEN = 1
    KING = 2
    
    def __init__(self):
        self.num_actions = 2
        self.cards = [self.JACK, self.QUEEN, self.KING]
    
    def get_num_actions(self) -> int:
        return self.num_actions
    
    def is_terminal(self, history: str) -> bool:
        """Check if history represents a terminal state."""
        return history in ("pp", "bp", "bb", "pbp", "pbb")
    
    def get_payoff(self, cards: List[int], history: str, player: int) -> float:
        """Get payoff for a player in Kuhn Poker."""
        me, opponent = player, 1 - player
        my_card, opp_card = cards[me], cards[opponent]
        
        if history == "pp":  # Both pass - showdown with ante only
            return 1 if my_card > opp_card else -1
        elif history == "bp":  # First bets, second passes (folds)
            return 1 if player == 0 else -1
        elif history == "pbp":  # First passes, second bets, first passes (folds)
            return -1 if player == 0 else 1
        elif history == "bb":  # Both bet - showdown with bets
            return 2 if my_card > opp_card else -2
        elif history == "pbb":  # Pass, bet, bet (call) - showdown with bets
            return 2 if my_card > opp_card else -2
        else:
            raise ValueError(f"Unknown terminal history: {history}")
    
    def get_valid_actions(self, history: str) -> List[int]:
        """Get valid actions for the current state."""
        if history in ("", "p"):  # Can check or bet
            return [self.PASS, self.BET]
        elif history in ("b", "pb"):  # Can fold or call
            return [self.PASS, self.BET]
        else:
            return []  # Terminal state
    
    def get_infoset_key(self, player_card: int, history: str) -> str:
        """Generate information set key."""
        card_name = "JQK"[player_card]
        return f"{card_name}:{history}"
    
    def get_all_possible_deals(self) -> List[Tuple[int, ...]]:
        """Get all possible 2-player card deals."""
        return list(itertools.permutations(self.cards, 2))
    
    def action_to_history_char(self, action: int) -> str:
        """Convert action to history character."""
        return "p" if action == self.PASS else "b"


class LeducHoldemGame(GameInterface):
    """Leduc Hold'em game implementation for CFR+."""
    
    # Action constants
    FOLD = 0
    CALL_CHECK = 1  
    BET_RAISE = 2
    
    # Card constants (same as Kuhn for simplicity)
    JACK = 0
    QUEEN = 1  
    KING = 2
    
    def __init__(self):
        self.num_actions = 3
        # Deck has 2 of each rank (6 cards total)
        self.deck = [self.JACK, self.JACK, self.QUEEN, self.QUEEN, self.KING, self.KING]
    
    def get_num_actions(self) -> int:
        return self.num_actions
    
    def is_terminal(self, history: str) -> bool:
        """Check if history represents a terminal state."""
        # Terminal states in Leduc Hold'em are more complex
        # This is a simplified version - in full Leduc, we need to track rounds
        
        # Fold scenarios
        if "f" in history:
            return True
            
        # Two betting rounds completed with calls/checks
        if len(history) >= 4:
            # Simple heuristic: if we have 4+ actions, consider terminal
            return True
            
        return False
    
    def get_payoff(self, cards: List[int], history: str, player: int) -> float:
        """Get payoff for a player in Leduc Hold'em."""
        # This is a simplified version - full Leduc Hold'em payoff calculation
        # would need to consider the public card and hand rankings
        
        if "f" in history:
            # Someone folded - determine who and assign payoff
            fold_position = history.find("f")
            folding_player = fold_position % 2
            if folding_player == player:
                return -1  # This player folded, loses ante
            else:
                return 1   # Opponent folded, this player wins
        
        # Showdown - simplified (would need public card in real implementation)
        me, opponent = player, 1 - player
        my_card, opp_card = cards[me], cards[opponent]
        
        # Count bets to determine pot size
        bet_count = history.count("b") + history.count("r")
        pot_multiplier = 1 + bet_count
        
        return pot_multiplier if my_card > opp_card else -pot_multiplier
    
    def get_valid_actions(self, history: str) -> List[int]:
        """Get valid actions for current state."""
        if not history:
            # First action: can check or bet
            return [self.CALL_CHECK, self.BET_RAISE]
        
        last_action = history[-1]
        if last_action in "br":  # Last action was bet or raise
            return [self.FOLD, self.CALL_CHECK, self.BET_RAISE]
        else:  # Last action was check or call
            return [self.CALL_CHECK, self.BET_RAISE]
    
    def get_infoset_key(self, player_card: int, history: str) -> str:
        """Generate information set key."""
        card_name = "JQK"[player_card]
        return f"{card_name}:{history}"
    
    def get_all_possible_deals(self) -> List[Tuple[int, ...]]:
        """Get all possible 3-card deals (2 private + 1 public)."""
        # For simplicity, we'll generate 2-player deals like Kuhn Poker
        # Full Leduc Hold'em would need to include the public card
        all_deals = []
        for p1_card in self.deck:
            for p2_card in self.deck:
                if p1_card != p2_card:  # Can't deal the same physical card
                    all_deals.append((p1_card, p2_card))
        return all_deals
    
    def action_to_history_char(self, action: int) -> str:
        """Convert action to history character."""
        if action == self.FOLD:
            return "f"
        elif action == self.CALL_CHECK:
            return "c"
        else:  # BET_RAISE
            return "b"