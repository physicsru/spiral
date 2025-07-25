"""
CFR+ Module for SPIRAL
======================

A modular CFR+ implementation for training Game-Theoretic Optimal (GTO) 
strategies in various card games.

Main components:
- CFRPlusBase: Core CFR+ algorithm
- GameInterface: Abstract interface for different games
- KuhnPokerGame, LeducHoldemGame: Specific game implementations
- CFRPlusAgent: Agent that uses trained CFR+ strategies
"""

from .base import CFRPlusBase, GameInterface, InfoSet
from .games import KuhnPokerGame, LeducHoldemGame
from .agent import CFRPlusAgent
from .trainer import CFRPlusTrainer

__all__ = [
    "CFRPlusBase",
    "GameInterface", 
    "InfoSet",
    "KuhnPokerGame",
    "LeducHoldemGame", 
    "CFRPlusAgent",
    "CFRPlusTrainer",
]