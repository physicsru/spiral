# Modular CFR+ Implementation

This module provides a modular CFR+ (Counterfactual Regret Minimization Plus) implementation that can be easily extended to support different card games while maintaining the core CFR+ algorithm.

## Architecture

The implementation is structured into several key components:

### Core Components

- **`CFRPlusBase`**: The core CFR+ algorithm implementation that works with any game interface
- **`GameInterface`**: Abstract base class defining the interface that games must implement
- **`InfoSet`**: Information set node with CFR+ regret updates and strategy averaging
- **`CFRPlusTrainer`**: High-level training interface for different games
- **`CFRPlusAgent`**: Base agent class for playing with trained strategies

### Game Implementations

- **`KuhnPokerGame`**: Complete implementation of Kuhn Poker rules for CFR+ training
- **`LeducHoldemGame`**: Basic implementation of Leduc Hold'em (simplified version)
- **`KuhnPokerCFRAgent`**: TextArena-compatible agent for Kuhn Poker
- **`LeducHoldemCFRAgent`**: TextArena-compatible agent for Leduc Hold'em

## Usage

### Training a Strategy

```python
from spiral.cfr.trainer import CFRPlusTrainer

# Train Kuhn Poker strategy
trainer = CFRPlusTrainer("kuhn_poker", delay=0)
results = trainer.train(
    iterations=100000,
    seed=42,
    save_path="kuhn_strategy.pkl"
)

# Train Leduc Hold'em strategy
trainer = CFRPlusTrainer("leduc_holdem", delay=0)
results = trainer.train(
    iterations=100000,
    seed=42,
    save_path="leduc_strategy.pkl"
)
```

### Command Line Training

```bash
# Train Kuhn Poker
python -m spiral.cfr.trainer kuhn_poker -n 100000 --save kuhn_strategy.pkl

# Train Leduc Hold'em
python -m spiral.cfr.trainer leduc_holdem -n 100000 --save leduc_strategy.pkl

# Use the standalone training script
python train_cfr_modular.py kuhn_poker -n 100000 --save-with-alpha-0
```

### Using Trained Agents

```python
from spiral.cfr.agent import KuhnPokerCFRAgent

# Load and use a trained agent
agent = KuhnPokerCFRAgent(strategy_path="kuhn_strategy.pkl")

# Use in TextArena environment
observation = "Your card is: J\nYour available actions: [Check], [Bet]"
action = agent(observation)  # Returns "[Check]" or "[Bet]"
```

### Integration with Existing Code

The modular implementation is designed to be backward-compatible with existing code:

```python
# In train_spiral_gto.py, the CFRPlusAgent is aliased
from spiral.cfr.agent import KuhnPokerCFRAgent
CFRPlusAgent = KuhnPokerCFRAgent  # Drop-in replacement

# In battle scripts
cfr_agent = KuhnPokerCFRAgent(strategy_path="strategy.pkl")
```

## Adding New Games

To add support for a new card game, implement the `GameInterface`:

```python
from spiral.cfr.base import GameInterface

class MyCardGame(GameInterface):
    def get_num_actions(self) -> int:
        # Return number of possible actions
        return 3
    
    def is_terminal(self, history: str) -> bool:
        # Check if game state is terminal
        pass
    
    def get_payoff(self, cards: List[int], history: str, player: int) -> float:
        # Calculate payoff for the player
        pass
    
    def get_valid_actions(self, history: str) -> List[int]:
        # Return valid action indices
        pass
    
    def get_infoset_key(self, player_card: int, history: str) -> str:
        # Generate unique information set identifier
        pass
    
    def get_all_possible_deals(self) -> List[Tuple[int, ...]]:
        # Return all possible card combinations
        pass
    
    def action_to_history_char(self, action: int) -> str:
        # Convert action index to history character
        pass
```

Then register it in `CFRPlusTrainer.SUPPORTED_GAMES`:

```python
CFRPlusTrainer.SUPPORTED_GAMES["my_game"] = MyCardGame
```

## Key Features

### CFR+ Algorithm Implementation

- **Positive regret matching**: Only positive regrets contribute to strategy updates
- **Linear weighted averaging**: Strategy averaging with configurable delay
- **External sampling**: Trains separate strategies for each player position
- **Chance sampling**: Iterates over all possible card deals for robust training

### Modular Design

- **Game-agnostic core**: CFR+ algorithm works with any game implementing the interface
- **Easy extensibility**: Add new games by implementing the `GameInterface`
- **Backward compatibility**: Drop-in replacement for existing CFR+ agents
- **TextArena integration**: Agents work seamlessly with TextArena environments

### Training Features

- **Progress logging**: Configurable logging frequency during training
- **Strategy persistence**: Save and load trained strategies
- **Manual overrides**: Apply expert knowledge or theoretical solutions
- **Reproducible training**: Seed-based deterministic training

## File Structure

```
spiral/cfr/
├── __init__.py          # Module exports
├── base.py              # Core CFR+ algorithm and interfaces
├── games.py             # Game implementations (Kuhn Poker, Leduc Hold'em)
├── agent.py             # Agent classes for playing with strategies
├── trainer.py           # High-level training interface
└── README.md           # This documentation
```

## Migration from Original Implementation

The original `train_cfr_.py` can be replaced with the modular system:

**Old way:**
```bash
python path/train_cfr_.py -n 100000
```

**New way:**
```bash
python train_cfr_modular.py kuhn_poker -n 100000 --save-with-alpha-0
```

The new implementation provides:
- Support for multiple games
- Cleaner, more maintainable code
- Better separation of concerns
- Easier testing and validation
- More flexible agent interfaces

## Examples

See the provided scripts for complete examples:
- `train_cfr_modular.py`: Training CFR+ strategies
- `battle_qwen_vs_cfrplus_modular.py`: Battling LLMs against CFR+ strategies
- Integration in `train_spiral_gto.py` for GTO training against LLMs