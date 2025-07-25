#!/usr/bin/env python3
"""
CFR+ Agent for playing games using trained strategies
====================================================

This module provides agents that can play games using CFR+ strategies
trained with the modular CFR+ implementation.
"""

import pickle
import random
import re
from typing import Dict, List, Optional, Union
import numpy as np


class CFRPlusAgent:
    """Agent that plays using a trained CFR+ strategy."""
    
    def __init__(self, strategy_path: Optional[str] = None, strategy_dict: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize CFR+ agent with either a file path or strategy dictionary.
        
        Args:
            strategy_path: Path to pickled strategy file
            strategy_dict: Dictionary mapping infoset keys to strategy arrays
        """
        if strategy_path is not None:
            self.load_strategy(strategy_path)
        elif strategy_dict is not None:
            self.strategy = strategy_dict
        else:
            raise ValueError("Must provide either strategy_path or strategy_dict")
    
    def load_strategy(self, filepath: str) -> None:
        """Load strategy from a pickle file."""
        with open(filepath, "rb") as f:
            self.strategy = pickle.load(f)
    
    def get_action_probabilities(self, infoset_key: str) -> np.ndarray:
        """Get action probabilities for a given information set."""
        if infoset_key in self.strategy:
            return self.strategy[infoset_key]
        else:
            # Return uniform distribution if infoset not found
            # This assumes 2 actions (works for Kuhn Poker)
            return np.array([0.5, 0.5])
    
    def select_action(self, infoset_key: str, valid_actions: List[int]) -> int:
        """Select an action based on the strategy."""
        probs = self.get_action_probabilities(infoset_key)
        
        # Filter probabilities for valid actions only
        valid_probs = [probs[action] for action in valid_actions]
        total = sum(valid_probs)
        
        if total > 0:
            normalized_probs = [p / total for p in valid_probs]
            return random.choices(valid_actions, weights=normalized_probs)[0]
        else:
            # Fallback to uniform random if all probabilities are 0
            return random.choice(valid_actions)


class KuhnPokerCFRAgent(CFRPlusAgent):
    """CFR+ agent specifically for Kuhn Poker with TextArena integration."""
    
    # Action mapping for TextArena format
    ACTION_MAP = {0: "[Check]", 1: "[Bet]"}
    
    def __init__(self, strategy_path: Optional[str] = None, strategy_dict: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(strategy_path, strategy_dict)
    
    @staticmethod
    def parse_card(observation: str) -> int:
        """Parse the player's card from the observation."""
        match = re.search(r"Your card is:\s*([JQK])", observation, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not parse card from observation: {observation}")
        card_str = match.group(1).upper()
        return {"J": 0, "Q": 1, "K": 2}[card_str]
    
    @staticmethod
    def extract_history(observation: str) -> str:
        """Extract the betting history from the observation."""
        history = ""
        lines = observation.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for action patterns
            if re.match(r"\[(Check|Bet|Call|Fold)\]", line, re.IGNORECASE):
                action = line.strip("[]").lower()
                if action in ["check"]:
                    history += "p"
                elif action in ["bet"]:
                    history += "b"
                elif action in ["call"]:
                    history += "b"  # In Kuhn Poker, call is treated as bet
                elif action in ["fold"]:
                    history += "p"  # In Kuhn Poker, fold is treated as pass
        
        return history
    
    def __call__(self, observation: str) -> str:
        """Play an action based on the observation (TextArena format)."""
        try:
            # Parse card and history
            card = self.parse_card(observation)
            history = self.extract_history(observation)
            
            # Create infoset key
            card_name = "JQK"[card]
            infoset_key = f"{card_name}:{history}"
            
            # Determine valid actions from observation
            valid_actions = self.get_valid_actions_from_observation(observation)
            
            # Map TextArena actions to CFR+ actions
            cfr_valid_actions = []
            action_mapping = {}
            
            for ta_action in valid_actions:
                if "[Check]" in ta_action:
                    cfr_valid_actions.append(0)
                    action_mapping[0] = "[Check]"
                elif "[Bet]" in ta_action:
                    cfr_valid_actions.append(1) 
                    action_mapping[1] = "[Bet]"
                elif "[Call]" in ta_action:
                    cfr_valid_actions.append(1)
                    action_mapping[1] = "[Call]"
                elif "[Fold]" in ta_action:
                    cfr_valid_actions.append(0)
                    action_mapping[0] = "[Fold]"
            
            # Select action using CFR+ strategy
            if cfr_valid_actions:
                cfr_action = self.select_action(infoset_key, cfr_valid_actions)
                return action_mapping[cfr_action]
            else:
                # Fallback if no valid actions found
                return "[Check]"
                
        except Exception as e:
            print(f"Error in CFR+ agent: {e}")
            print(f"Observation: {observation}")
            # Fallback to random valid action
            valid_actions = self.get_valid_actions_from_observation(observation)
            if valid_actions:
                return random.choice(valid_actions)
            else:
                return "[Check]"
    
    @staticmethod
    def get_valid_actions_from_observation(observation: str) -> List[str]:
        """Extract valid actions from the observation."""
        valid_actions = []
        
        # Look for "Your available actions" line
        for line in observation.split('\n'):
            if "available actions" in line.lower():
                # Extract actions in bracket format
                actions = re.findall(r'\[([^\]]+)\]', line)
                valid_actions = [f"[{action}]" for action in actions]
                break
        
        # Fallback if no actions found
        if not valid_actions:
            valid_actions = ["[Check]", "[Bet]"]
        
        return valid_actions


class LeducHoldemCFRAgent(CFRPlusAgent):
    """CFR+ agent specifically for Leduc Hold'em with TextArena integration."""
    
    def __init__(self, strategy_path: Optional[str] = None, strategy_dict: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(strategy_path, strategy_dict)
    
    def __call__(self, observation: str) -> str:
        """Play an action based on the observation (TextArena format)."""
        # This would need to be implemented similar to KuhnPokerCFRAgent
        # but adapted for Leduc Hold'em rules and action space
        
        # For now, return a placeholder implementation
        valid_actions = self.get_valid_actions_from_observation(observation)
        return random.choice(valid_actions) if valid_actions else "[Check]"
    
    @staticmethod
    def get_valid_actions_from_observation(observation: str) -> List[str]:
        """Extract valid actions from the observation."""
        valid_actions = []
        
        for line in observation.split('\n'):
            if "available actions" in line.lower():
                actions = re.findall(r'\[([^\]]+)\]', line)
                valid_actions = [f"[{action}]" for action in actions]
                break
        
        if not valid_actions:
            valid_actions = ["[Check]", "[Bet]", "[Call]", "[Fold]"]
        
        return valid_actions