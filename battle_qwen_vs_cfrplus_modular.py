#!/usr/bin/env python3
"""
Modular Battle: Qwen LLM vs CFR+ Strategy
=========================================

This script replaces the original battle_qwen_vs_cfrplus.py with the new modular CFR+ implementation.
It can battle against CFR+ strategies trained for different card games.
"""

import argparse
import json
import random
import re
import time
from typing import Dict, List, Tuple, Callable
import pickle
import vllm

from spiral.envs import make_env
from spiral.cfr.agent import KuhnPokerCFRAgent, LeducHoldemCFRAgent


# ---------- Game State Management (same as original) -----------------------------------------
class GameState:
    """Class to maintain game state and history with context length limits."""

    def __init__(self, max_context_length: int = 32768, max_turns: int = 50):
        self.history = []
        self.max_context_length = max_context_length
        self.max_turns = max_turns
        self.turn_count = 0

    def add_interaction(self, player_id: int, observation: str, action: str) -> None:
        """Add a turn interaction to the game history."""
        self.history.append((player_id, observation, action))
        self.turn_count += 1

    def get_full_history_text(self) -> str:
        """Get the full game history as text."""
        history_text = []
        for player_id, observation, action in self.history:
            history_text.append(f"Player {player_id} observed:\n{observation}")
            history_text.append(f"Player {player_id} action:\n{action}")

        full_text = "\n".join(history_text)
        if len(full_text) > self.max_context_length:
            excess = len(full_text) - self.max_context_length
            full_text = full_text[excess:]

        return full_text

    def is_truncated(self) -> bool:
        """Check if game should be truncated due to max turns."""
        return self.turn_count >= self.max_turns


# ---------- Qwen Agent (same as original) -------------------------------------------------
class QwenAgent:
    def __init__(self, model_path: str, temperature: float = 0.7, max_context_length: int = 32768):
        self.model_path = model_path
        self.max_context_length = max_context_length
        self.llm = vllm.LLM(model=model_path, dtype="bfloat16")
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature, top_p=0.9, max_tokens=32, n=1
        )
        self.answer_re = re.compile(r"\\boxed\{([^}]*)\}")

    def _truncate_observation(self, obs: str) -> str:
        """Truncate observation to fit within context length limits."""
        template_overhead = 200
        generation_space = 32 * 4
        reserved_space = template_overhead + generation_space + 200
        
        max_obs_length = max(100, self.max_context_length - reserved_space)
        
        if len(obs) <= max_obs_length:
            return obs
        
        print(f"Warning: Truncating observation from {len(obs)} to {max_obs_length} characters")
        
        truncated = obs[-max_obs_length:]
        
        newline_pos = truncated.find('\n')
        if newline_pos > 0 and newline_pos < 200:
            truncated = truncated[newline_pos + 1:]
        
        return truncated

    def _template(self, obs: str) -> str:
        """Template for Qwen prompt"""
        truncated_obs = self._truncate_observation(obs)
        return f"""You are playing a card game. Reply ONLY with one legal action in bracket form.
Opponent and system messages:
{truncated_obs}
"""

    def _extract_action(self, text: str, legal_actions: List[str]) -> str:
        m = self.answer_re.search(text)
        if m:
            action = m.group(1).strip()
        else:
            action = text.strip()

        action = "[" + action.strip("[] ").capitalize() + "]"
        if action not in legal_actions:
            action = random.choice(legal_actions)
        return action

    def __call__(self, observation: str) -> str:
        prompt = self._template(observation)
        out = self.llm.generate(prompt, self.sampling_params)
        raw = out[0].outputs[0].text
        
        legal_found = re.findall(r"\[(Check|Bet|Call|Fold|Raise)\]", observation, re.IGNORECASE)
        legal_actions = ["[" + x.capitalize() + "]" for x in set(legal_found)]
        if not legal_actions:
            legal_actions = ["[Check]", "[Bet]", "[Call]", "[Fold]"]
        return self._extract_action(raw, legal_actions)


# ---------- Battle Driver -------------------------------------------------
def play_episode(env, agents: Dict[int, Callable], max_context_length: int = 32768, max_turns: int = 50) -> Dict[int, float]:
    env.reset(num_players=2, seed=random.randint(0, 1 << 31))
    
    game_state = GameState(max_context_length=max_context_length, max_turns=max_turns)
    
    done = False
    while not done:
        pid, obs = env.get_observation()
        action = agents[pid](obs)
        
        game_state.add_interaction(pid, obs, action)
        
        if game_state.is_truncated():
            print(f"Game truncated after {game_state.turn_count} turns (max: {max_turns})")
            return {0: 0.0, 1: 0.0}
        
        done, _ = env.step(action)
    
    return env.close()


def get_cfr_agent_for_game(game_name: str, strategy_path: str):
    """Get the appropriate CFR+ agent for the game."""
    if game_name.lower() in ["kuhn_poker", "kuhn", "kuhnpoker-v1"]:
        return KuhnPokerCFRAgent(strategy_path=strategy_path)
    elif game_name.lower() in ["leduc_holdem", "leduc", "leducholdem-v1"]:
        return LeducHoldemCFRAgent(strategy_path=strategy_path)
    else:
        raise ValueError(f"Unsupported game: {game_name}. Supported: kuhn_poker, leduc_holdem")


def get_env_id_for_game(game_name: str) -> str:
    """Get TextArena environment ID for the game."""
    if game_name.lower() in ["kuhn_poker", "kuhn"]:
        return "KuhnPoker-v1"
    elif game_name.lower() in ["leduc_holdem", "leduc"]:
        return "LeducHoldem-v1"
    else:
        raise ValueError(f"Unsupported game: {game_name}")


def main():
    parser = argparse.ArgumentParser(description="Qwen vs CFR+ battle using modular implementation")
    parser.add_argument("--qwen_path", required=True, help="Path to Qwen model")
    parser.add_argument("--cfr_strategy", required=True, help="Path to CFR+ strategy file")
    parser.add_argument("--game", choices=["kuhn_poker", "kuhn", "leduc_holdem", "leduc"], 
                       default="kuhn_poker", help="Game to play")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--qwen_first", action="store_true", help="Whether Qwen always plays first")
    parser.add_argument("--max_context_length", type=int, default=32768, help="Maximum context length")
    parser.add_argument("--max_turns", type=int, default=100, help="Maximum turns per game")
    parser.add_argument("--temperature", type=float, default=0.7, help="Qwen sampling temperature")
    args = parser.parse_args()

    print(f"Battle Configuration:")
    print(f"  Game: {args.game}")
    print(f"  Qwen model: {args.qwen_path}")
    print(f"  CFR+ strategy: {args.cfr_strategy}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Qwen always first: {args.qwen_first}")
    print(f"  Max context length: {args.max_context_length}")
    print(f"  Max turns per game: {args.max_turns}")
    print("=" * 60)

    # Initialize agents
    qwen_agent = QwenAgent(args.qwen_path, temperature=args.temperature, 
                          max_context_length=args.max_context_length)
    cfr_agent = get_cfr_agent_for_game(args.game, args.cfr_strategy)
    
    # Initialize environment
    env_id = get_env_id_for_game(args.game)
    env = make_env(env_id, use_llm_obs_wrapper=True)

    # Battle statistics
    win, loss, draw = 0, 0, 0

    start_time = time.time()
    for ep in range(args.episodes):
        # Determine positions
        if args.qwen_first:
            qwen_pos = 0
        else:
            qwen_pos = ep % 2
        
        cfr_pos = 1 - qwen_pos
        agents = {qwen_pos: qwen_agent, cfr_pos: cfr_agent}

        rewards = play_episode(env, agents, args.max_context_length, args.max_turns)
        qwen_reward = rewards[qwen_pos]
        cfr_reward = rewards[cfr_pos]

        if qwen_reward > cfr_reward:
            win += 1
            result = "W"
        elif qwen_reward < cfr_reward:
            loss += 1
            result = "L"
        else:
            draw += 1
            result = "D"

        if ep % 100 == 0 or ep < 10:
            print(f"Episode {ep+1:4d}/{args.episodes}: "
                  f"Qwen@P{qwen_pos} {result} "
                  f"({qwen_reward:.1f}:{cfr_reward:.1f})")

    elapsed = time.time() - start_time
    win_rate = win / args.episodes

    print("\n" + "=" * 60)
    print("BATTLE RESULTS")
    print("=" * 60)
    print(f"Game: {args.game.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/args.episodes:.3f}s per episode)")
    print(f"")
    print(f"Qwen Results:")
    print(f"  Wins:  {win:4d} ({win/args.episodes:.1%})")
    print(f"  Losses: {loss:4d} ({loss/args.episodes:.1%})")
    print(f"  Draws: {draw:4d} ({draw/args.episodes:.1%})")
    print(f"")
    print(f"Win Rate: {win_rate:.3%}")
    
    # Save results
    results = {
        "game": args.game,
        "episodes": args.episodes,
        "qwen_model": args.qwen_path,
        "cfr_strategy": args.cfr_strategy,
        "qwen_always_first": args.qwen_first,
        "results": {
            "wins": win,
            "losses": loss,
            "draws": draw,
            "win_rate": win_rate
        },
        "config": {
            "max_context_length": args.max_context_length,
            "max_turns": args.max_turns,
            "temperature": args.temperature
        },
        "time_elapsed": elapsed
    }
    
    results_file = f"battle_results_{args.game}_{args.episodes}ep.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")
    
    # Generate plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        categories = ['Win', 'Loss', 'Draw']
        counts = [win, loss, draw]
        colors = ['green', 'red', 'gray']
        
        plt.figure(figsize=(12, 5))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(categories, counts, color=colors)
        plt.title(f'Qwen vs CFR+ Results\n{args.game.upper()} ({args.episodes} episodes)')
        plt.ylabel('Number of Episodes')
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Win/Loss Distribution\nWin Rate: {win_rate:.1%}')
        
        plt.tight_layout()
        
        plot_file = f"battle_plot_{args.game}_{args.episodes}ep.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")


if __name__ == "__main__":
    main()