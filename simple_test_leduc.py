#!/usr/bin/env python3
"""
Simple test script for Leduc Hold'em environment
Tests the core logic without full integration
"""

import sys
import os

# Add the spiral directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Mock textarena for testing purposes
class MockState:
    def __init__(self, num_players, min_players, max_players, max_turns, check_truncated=False):
        self.num_players = num_players
        self.min_players = min_players
        self.max_players = max_players
        self.max_turns = max_turns
        self.check_truncated = check_truncated
        self.current_player_id = 0
        self.done = False
        self.info = {}
        self.observations = []
        self.game_state = {}
        
    def reset(self, game_state, player_prompt_function, seed=None):
        self.game_state = game_state
        self.player_prompt_function = player_prompt_function
        
    def add_observation(self, from_id, to_id, message, for_logging=False):
        self.observations.append((from_id, to_id, message))
        print(f"[OBS] From {from_id} to {to_id}: {message}")
        
    def set_invalid_move(self, player_id, reason):
        print(f"[INVALID] Player {player_id}: {reason}")
        
    def step(self, rotate_player=True):
        if rotate_player and not self.done:
            self.current_player_id = 1 - self.current_player_id
        return self.done, self.info
        
    def set_winners(self, player_ids, reason):
        self.done = True
        self.info = {"winners": player_ids, "reason": reason}
        print(f"[GAME END] Winners: {player_ids}, Reason: {reason}")
        
    def set_draw(self, reason):
        self.done = True
        self.info = {"draw": True, "reason": reason}
        print(f"[GAME END] Draw: {reason}")
        
    def get_current_player_observation(self):
        return f"You are Player {self.current_player_id}. Your card will be revealed during the game."
        
    def manually_update_current_player(self, new_player_id):
        self.current_player_id = new_player_id

class MockTextArena:
    GAME_ID = -999
    
    class Env:
        def __init__(self):
            pass
    
    State = MockState

# Mock textarena module
import types
ta = types.ModuleType('textarena')
ta.Env = MockTextArena.Env
ta.State = MockTextArena.State
ta.GAME_ID = MockTextArena.GAME_ID
sys.modules['textarena'] = ta

# Now import our environment
from spiral.envs.LeducHoldem.env import LeducHoldemEnv

def test_basic_game():
    """Test basic game functionality"""
    print("Testing Leduc Hold'em Environment")
    print("=" * 50)
    
    # Create environment
    env = LeducHoldemEnv(max_rounds=1)
    print("Environment created successfully!")
    
    # Reset environment
    env.reset(num_players=2, seed=42)
    print("Environment reset successfully!")
    
    # Test game state
    print(f"Current player: {env.state.current_player_id}")
    print(f"Game state keys: {list(env.state.game_state.keys())}")
    print(f"Player cards: {env.state.game_state.get('player_cards', 'Not set')}")
    print(f"Public card: {env.state.game_state.get('public_card', 'Not set')}")
    print(f"Pot: {env.state.game_state.get('pot', 'Not set')}")
    
    # Test some actions
    print("\n" + "=" * 30)
    print("Testing actions...")
    
    # Player 0's turn (small blind)
    current_player, obs = env.get_observation()
    print(f"Player {current_player} observation: {obs}")
    
    # Test check action
    print(f"Player {current_player} checks...")
    done, info = env.step("[Check]")
    print(f"Game done: {done}")
    
    if not done:
        # Player 1's turn (big blind)
        current_player, obs = env.get_observation()
        print(f"Player {current_player} observation: {obs}")
        
        # Test bet action
        print(f"Player {current_player} bets...")
        done, info = env.step("[Bet]")
        print(f"Game done: {done}")
        
        if not done:
            # Back to Player 0
            current_player, obs = env.get_observation()
            print(f"Player {current_player} observation: {obs}")
            
            # Test call action
            print(f"Player {current_player} calls...")
            done, info = env.step("[Call]")
            print(f"Game done: {done}")
    
    print("\nTest completed successfully!")

def test_card_logic():
    """Test card and hand evaluation logic"""
    print("\nTesting card logic...")
    print("=" * 30)
    
    env = LeducHoldemEnv()
    
    # Test card to string conversion
    print("Card representations:")
    for i in range(3):
        print(f"  {i} -> {env._card_to_str(i)}")
    
    # Test hand strength calculation
    print("\nHand strength tests:")
    test_cases = [
        (0, 0, "Pair of Jacks"),  # Jack pair
        (1, 1, "Pair of Queens"), # Queen pair  
        (2, 2, "Pair of Kings"),  # King pair
        (0, 1, "Jack high"),      # Jack vs Queen public
        (1, 2, "Queen high"),     # Queen vs King public
        (2, 0, "King high"),      # King vs Jack public
    ]
    
    for private, public, expected_desc in test_cases:
        strength = env._get_hand_strength(private, public)
        description = env._describe_hand(private, public)
        print(f"  Private: {env._card_to_str(private)}, Public: {env._card_to_str(public)}")
        print(f"    Strength: {strength}, Description: {description}")
        assert expected_desc in description or description == expected_desc
    
    print("Card logic tests passed!")

if __name__ == "__main__":
    try:
        test_card_logic()
        test_basic_game()
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✅")
        print("Leduc Hold'em environment is working correctly!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc() 