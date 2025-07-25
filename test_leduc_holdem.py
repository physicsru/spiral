#!/usr/bin/env python3
"""
Test script for Leduc Hold'em environment
"""

import random
from spiral.envs import make_env


def test_leduc_holdem():
    """Test the Leduc Hold'em environment with random actions"""
    print("Testing Leduc Hold'em environment...")
    
    # Create environment
    env = make_env("LeducHoldem-v1", use_llm_obs_wrapper=False)
    
    # Reset environment
    env.reset(num_players=2, seed=42)
    
    print("Environment created and reset successfully!")
    print("=" * 50)
    
    game_count = 0
    while game_count < 3:  # Play 3 games
        game_count += 1
        print(f"\n=== GAME {game_count} ===")
        
        done = False
        turn_count = 0
        
        while not done and turn_count < 50:  # Safety limit
            turn_count += 1
            
            # Get current player and observation
            current_player, obs = env.get_observation()
            print(f"\nTurn {turn_count} - Player {current_player}'s turn")
            print(f"Observation: {obs}")
            
            # Choose a random valid action
            valid_actions = ["[Check]", "[Bet]", "[Call]", "[Raise]", "[Fold]"]
            action = random.choice(valid_actions)
            
            print(f"Player {current_player} action: {action}")
            
            # Take action
            done, info = env.step(action)
            
            if done:
                print("\nGame finished!")
                print(f"Game info: {info}")
                break
        
        if not done:
            print("Game reached turn limit!")
        
        print("-" * 30)
    
    print("\nAll tests completed successfully!")


def manual_test_game():
    """Manual test where you can input actions"""
    print("Manual Leduc Hold'em test - you can input actions manually")
    print("Available actions: [Check], [Bet], [Call], [Raise], [Fold]")
    print("=" * 50)
    
    # Create environment
    env = make_env("LeducHoldem-v1", use_llm_obs_wrapper=False)
    env.reset(num_players=2, seed=None)
    
    done = False
    turn_count = 0
    
    while not done and turn_count < 50:
        turn_count += 1
        
        # Get current player and observation
        current_player, obs = env.get_observation()
        print(f"\nTurn {turn_count} - Player {current_player}'s turn")
        print(f"Observation:\n{obs}")
        print("-" * 30)
        
        # Get action from user
        action = input(f"Enter action for Player {current_player}: ").strip()
        if not action:
            action = "[Check]"  # Default action
        
        print(f"Player {current_player} action: {action}")
        
        # Take action
        done, info = env.step(action)
        
        if done:
            print("\nGame finished!")
            print(f"Game info: {info}")
            break
    
    if not done:
        print("Game reached turn limit!")


if __name__ == "__main__":
    print("Leduc Hold'em Environment Test")
    print("1. Random test")
    print("2. Manual test")
    
    choice = input("Choose test type (1 or 2): ").strip()
    
    if choice == "2":
        manual_test_game()
    else:
        test_leduc_holdem() 