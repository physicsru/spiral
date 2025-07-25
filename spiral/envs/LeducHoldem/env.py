# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta

# For reference:
#  - J (Jack): numeric rank 0
#  - Q (Queen): numeric rank 1
#  - K (King): numeric rank 2


class LeducHoldemEnv(ta.Env):
    """
    Leduc Hold'em environment - a simplified poker variant.
    
    Rules:
    - Deck: 6 cards total (2 Jacks, 2 Queens, 2 Kings)
    - 2 players, 2 rounds of betting
    - Round 1: Each player gets 1 private card, posts an ante of 1
    - Round 2: 1 public card is revealed, betting continues
    - Win condition: Pair with public card wins, otherwise higher card wins
    - Betting: Check, Bet, Call, Raise, Fold
    - Bet amounts: 2 in round 1, 4 in round 2
    - Max 2 bets per round (bet + raise)
    """

    def __init__(self, max_rounds: int = 1):
        super().__init__()
        self.max_rounds = max_rounds
        # Deck: 6 cards - 2 each of J, Q, K (represented as 0, 1, 2)
        self.deck = [0, 0, 1, 1, 2, 2]  # Two pairs of each rank
        
        # Betting amounts per round
        self.bet_amounts = {1: 2, 2: 4}  # Round 1: 2, Round 2: 4
        
        # MODIFIED: Changed from blinds to ante system to match the document
        self.ante = 1
        
        # Action pattern for parsing
        self.action_pattern = re.compile(
            r"\[(Check|Bet|Call|Raise|Fold)\]", re.IGNORECASE
        )

    def get_observation(self):
        # Check if a round just ended and we need to start a new one
        if self.state.game_state.get("round_ended", False):
            self.state.game_state["round_ended"] = False
            self._init_round()

        return self.state.current_player_id, self.state.get_current_player_observation()

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the environment"""
        self.state = ta.State(
            num_players=num_players,
            min_players=2,
            max_players=2,
            max_turns=self.max_rounds,
            check_truncated=False,
        )

        game_state = {
            "pot": 0,
            "player_chips": {0: 0, 1: 0},  # Track total chips won/lost
            "current_round": 0,
            "starting_player": 0,  # Who acts first in the first round
        }
        
        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
            seed=seed,
        )

        # Initialize the first round
        self._init_round()

    def _init_round(self):
        """Initialize a new round of Leduc Hold'em"""
        # Check if game is complete
        if self.state.game_state["current_round"] >= self.state.max_turns:
            self._determine_final_winner()
            return

        # Shuffle the deck
        deck = self.deck.copy()
        random.shuffle(deck)
        
        # Deal cards: 2 private + 1 public
        self.state.game_state["player_cards"] = {0: deck[0], 1: deck[1]}
        self.state.game_state["public_card"] = deck[2]
        
        # MODIFIED: Changed from blinds to ante system
        # Determine who acts first and alternate each round
        first_to_act_player = self.state.game_state["starting_player"]
        
        # Calculate initial pot from antes from both players
        initial_pot = self.ante * self.state.num_players

        # Initialize betting state
        self.state.game_state.update({
            "current_round": self.state.game_state["current_round"] + 1,
            "betting_round": 1,  # 1 or 2
            "pot": initial_pot,
            "first_to_act_player": first_to_act_player,  # Store who acts first post-ante
            "current_bets": {0: 0, 1: 0},  # Bets in current betting round (start at 0)
            "total_invested": {0: self.ante, 1: self.ante},  # Total invested is the ante
            "bet_count": 0,  # Number of bets in current betting round (max 2)
            "public_card_revealed": False,
        })
        
        # Set the first player to act
        self.state.current_player_id = first_to_act_player
        
        # Update starting player for the next round
        self.state.game_state["starting_player"] = 1 - first_to_act_player
        
        # Send initial game state
        self._send_round_start_message()

    def _send_round_start_message(self):
        """Send round start information to players"""
        betting_round = self.state.game_state["betting_round"]
        
        # MODIFIED: Changed message from blinds to ante
        if betting_round == 1:
            message = (
                f"=== Leduc Hold'em Round {self.state.game_state['current_round']} - Betting Round 1 ===\n"
                f"Each player posts an ante of {self.ante}.\n"
                f"Pot: {self.state.game_state['pot']}"
            )
        else:
            public_card = self._card_to_str(self.state.game_state["public_card"])
            message = (
                f"=== Betting Round 2 ===\n"
                f"Public card: {public_card}\n"
                f"Pot: {self.state.game_state['pot']}"
            )
        
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
        
        # Show available actions to current player
        self._show_available_actions()

    def _show_available_actions(self):
        """Show available actions to the current player"""
        player_id = self.state.current_player_id
        opponent_id = 1 - player_id
        
        current_bet = self.state.game_state["current_bets"][player_id]
        opponent_bet = self.state.game_state["current_bets"][opponent_id]
        bet_count = self.state.game_state["bet_count"]
        
        actions = []
        
        if current_bet == opponent_bet:
            # No bet to call
            actions.append("[Check]")
            if bet_count < 2:  # Can still bet/raise
                actions.append("[Bet]")
        else:
            # There's a bet to call
            actions.append("[Call]")
            actions.append("[Fold]")
            if bet_count < 2:  # Can still raise
                actions.append("[Raise]")
        
        actions_str = ", ".join(actions)
        message = f"Player {player_id}, your available actions: {actions_str}"
        self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate the initial prompt for a player"""
        # MODIFIED: Updated rule description
        prompt = (
            f"You are Player {player_id} in Leduc Hold'em.\n\n"
            f"Game Rules:\n"
            f"- 6-card deck: 2 Jacks, 2 Queens, 2 Kings (J < Q < K)\n"
            f"- Each player posts an ante of 1 at the start of a hand.\n"
            f"- 2 betting rounds per hand\n"
            f"- Round 1: You get 1 private card, betting with amounts of 2\n"
            f"- Round 2: 1 public card revealed, betting with amounts of 4\n"
            f"- Max 2 bets per round (bet + raise)\n\n"
            f"Winning:\n"
            f"- If your card matches the public card rank (pair), you win\n"
            f"- If neither player has a pair, higher card wins\n"
            f"- If both have pairs, higher pair wins\n\n"
            f"Actions:\n"
            f"- [Check]: Pass (only if no bet to call)\n"
            f"- [Bet]: Make a bet (only if no bet on table)\n"
            f"- [Call]: Match opponent's bet\n"
            f"- [Raise]: Increase the bet (only if < 2 bets this round)\n"
            f"- [Fold]: Give up and lose invested chips\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Process the player's move"""
        if hasattr(self.state, "done") and self.state.done:
            return True, self.state.info

        player_id = self.state.current_player_id

        # Log the raw action
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        # Parse the action
        match = self.action_pattern.search(action.strip())
        if not match:
            self.state.set_invalid_move(
                player_id=player_id,
                reason="Action must be [Check], [Bet], [Call], [Raise], or [Fold].",
            )
            return self.state.step()

        move = match.group(1).lower()
        
        # Validate and execute the move
        if not self._is_valid_action(player_id, move):
            valid_actions = self._get_valid_actions(player_id)
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid action. Valid actions: {', '.join(valid_actions)}"
            )
            return self.state.step()

        # Execute the move
        self._execute_action(player_id, move)
        
        # Check if betting round is complete
        if self._is_betting_round_complete():
            self._complete_betting_round()
            if not self.state.done:
                return False, self.state.info
        
        return self.state.done, self.state.info

    def _is_valid_action(self, player_id: int, action: str) -> bool:
        """Check if an action is valid for the current player"""
        valid_actions = self._get_valid_actions(player_id)
        return action in [a.lower() for a in valid_actions]

    def _get_valid_actions(self, player_id: int) -> list:
        """Get list of valid actions for the current player"""
        opponent_id = 1 - player_id
        current_bet = self.state.game_state["current_bets"][player_id]
        opponent_bet = self.state.game_state["current_bets"][opponent_id]
        bet_count = self.state.game_state["bet_count"]
        
        actions = []
        
        if current_bet == opponent_bet:
            # No bet to call
            actions.append("Check")
            if bet_count < 2:
                actions.append("Bet")
        else:
            # There's a bet to call
            actions.append("Call")
            actions.append("Fold")
            if bet_count < 2:
                actions.append("Raise")
        
        return actions

    def _execute_action(self, player_id: int, action: str):
        """Execute the given action"""
        betting_round = self.state.game_state["betting_round"]
        bet_amount = self.bet_amounts[betting_round]
        
        if action == "check":
            self.state.add_observation(
                from_id=ta.GAME_ID, to_id=-1, 
                message=f"Player {player_id} checks."
            )
            
        elif action == "bet":
            self._place_bet(player_id, bet_amount)
            self.state.add_observation(
                from_id=ta.GAME_ID, to_id=-1,
                message=f"Player {player_id} bets {bet_amount}."
            )
            
        elif action == "call":
            opponent_id = 1 - player_id
            call_amount = self.state.game_state["current_bets"][opponent_id] - \
                         self.state.game_state["current_bets"][player_id]
            self._place_bet(player_id, call_amount)
            self.state.add_observation(
                from_id=ta.GAME_ID, to_id=-1,
                message=f"Player {player_id} calls {call_amount}."
            )
            
        elif action == "raise":
            opponent_id = 1 - player_id
            # First call, then raise
            call_amount = self.state.game_state["current_bets"][opponent_id] - \
                         self.state.game_state["current_bets"][player_id]
            total_bet = call_amount + bet_amount
            self._place_bet(player_id, total_bet)
            self.state.add_observation(
                from_id=ta.GAME_ID, to_id=-1,
                message=f"Player {player_id} raises by {bet_amount} (total bet: {total_bet})."
            )
            
        elif action == "fold":
            self.state.add_observation(
                from_id=ta.GAME_ID, to_id=-1,
                message=f"Player {player_id} folds."
            )
            self._handle_fold(player_id)
            return

        # Move to next player if betting continues
        if not self._is_betting_round_complete():
            self.state.current_player_id = 1 - player_id
            self._show_available_actions()

    def _place_bet(self, player_id: int, amount: int):
        """Place a bet for the player"""
        self.state.game_state["current_bets"][player_id] += amount
        self.state.game_state["total_invested"][player_id] += amount
        self.state.game_state["pot"] += amount
        
        # If this is a bet (not a call), increment bet count
        opponent_id = 1 - player_id
        if self.state.game_state["current_bets"][player_id] > \
           self.state.game_state["current_bets"][opponent_id]:
            self.state.game_state["bet_count"] += 1

    def _is_betting_round_complete(self) -> bool:
        """Check if the current betting round is complete"""
        current_bets = self.state.game_state["current_bets"]
        # Betting round is complete when both players have equal bets
        return current_bets[0] == current_bets[1]

    def _complete_betting_round(self):
        """Complete the current betting round and move to next phase"""
        betting_round = self.state.game_state["betting_round"]
        
        if betting_round == 1:
            # Move to round 2
            self.state.game_state["betting_round"] = 2
            self.state.game_state["bet_count"] = 0
            self.state.game_state["current_bets"] = {0: 0, 1: 0}
            self.state.game_state["public_card_revealed"] = True
            
            # MODIFIED: Use the stored 'first_to_act_player' to determine who acts first in round 2
            first_to_act_player = self.state.game_state["first_to_act_player"]
            self.state.current_player_id = first_to_act_player
            
            self._send_round_start_message()
        else:
            # Both betting rounds complete - showdown
            self._handle_showdown()

    def _handle_fold(self, folding_player_id: int):
        """Handle when a player folds"""
        winner_id = 1 - folding_player_id
        pot = self.state.game_state["pot"]
        
        self.state.game_state["player_chips"][winner_id] += pot
        
        reason = f"Player {folding_player_id} folded. Player {winner_id} wins {pot} chips."
        self._set_round_winner(winner_id, reason)

    def _handle_showdown(self):
        """Handle showdown - determine winner based on hand strength"""
        player_cards = self.state.game_state["player_cards"]
        public_card = self.state.game_state["public_card"]
        
        # Show all cards
        card_p0 = self._card_to_str(player_cards[0])
        card_p1 = self._card_to_str(player_cards[1])
        public_card_str = self._card_to_str(public_card)
        
        cards_msg = (
            f"Showdown! Cards revealed:\n"
            f"Player 0: {card_p0}\n"
            f"Player 1: {card_p1}\n"
            f"Public card: {public_card_str}"
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=cards_msg)
        
        # Determine hand strengths
        strength_p0 = self._get_hand_strength(player_cards[0], public_card)
        strength_p1 = self._get_hand_strength(player_cards[1], public_card)
        
        # Determine winner
        if strength_p0 > strength_p1:
            winner = 0
        elif strength_p1 > strength_p0:
            winner = 1
        else:
            # Tie - split pot (rare in Leduc Hold'em)
            self._handle_tie()
            return
        
        pot = self.state.game_state["pot"]
        self.state.game_state["player_chips"][winner] += pot
        
        # Create reason message
        hand_p0 = self._describe_hand(player_cards[0], public_card)
        hand_p1 = self._describe_hand(player_cards[1], public_card)
        
        reason = (
            f"Player 0: {hand_p0}, Player 1: {hand_p1}. "
            f"Player {winner} wins {pot} chips."
        )
        
        self._set_round_winner(winner, reason)

    def _get_hand_strength(self, private_card: int, public_card: int) -> int:
        """Get hand strength (higher is better)"""
        if private_card == public_card:
            # Pair - strength is 100 + card rank
            return 100 + private_card
        else:
            # High card - just the card rank
            return private_card

    def _describe_hand(self, private_card: int, public_card: int) -> str:
        """Describe a hand in human-readable format"""
        private_str = self._card_to_str(private_card)
        public_str = self._card_to_str(public_card)
        
        if private_card == public_card:
            return f"Pair of {private_str}s"
        else:
            return f"{private_str} high"

    def _handle_tie(self):
        """Handle a tie (very rare)"""
        pot = self.state.game_state["pot"]
        # Split pot evenly
        each_share = pot // 2
        self.state.game_state["player_chips"][0] += each_share
        self.state.game_state["player_chips"][1] += each_share
        
        self.state.set_draw(reason="Tie - pot split evenly.")

    def _set_round_winner(self, player_id: int, reason: str):
        """Set the winner of the current round"""
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=reason)
        
        if self.state.game_state["current_round"] >= self.state.max_turns:
            # Game is over
            self._determine_final_winner()
        else:
            # Mark round as ended for next round
            self.state.game_state["round_ended"] = True

    def _determine_final_winner(self):
        """Determine the final winner based on total chips"""
        chips_p0 = self.state.game_state["player_chips"][0]
        chips_p1 = self.state.game_state["player_chips"][1]
        
        if chips_p0 > chips_p1:
            winner = 0
        elif chips_p1 > chips_p0:
            winner = 1
        else:
            self.state.set_draw(
                reason=f"Game ends in a tie. Both players have {chips_p0} chips."
            )
            return
        
        self.state.set_winners(
            player_ids=[winner],
            reason=f"Player {winner} wins with {self.state.game_state['player_chips'][winner]} chips "
                   f"vs {self.state.game_state['player_chips'][1-winner]} chips."
        )

    def _card_to_str(self, card: int) -> str:
        """Convert card rank to string"""
        return {0: "J", 1: "Q", 2: "K"}.get(card, "?")