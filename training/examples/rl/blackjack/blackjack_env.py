"""Pure-Python Blackjack environment for tool-call rollouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from training.examples.rl.blackjack.blackjack_schema import BLACKJACK_ACTIONS

# Infinite deck: Ace=1, 2-9, and four 10-value cards (10, J, Q, K)
_DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def _draw_card(rng: random.Random) -> int:
    return rng.choice(_DECK)


def _sum_hand(cards: List[int]) -> Tuple[int, bool]:
    """Return (total, usable_ace). Aces start as 11; flip one to 1 if bust."""
    total = 0
    aces = 0
    for c in cards:
        if c == 1:
            aces += 1
            total += 11
        else:
            total += c
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total, aces > 0


def _is_natural(cards: List[int]) -> bool:
    """True if a two-card hand totals 21 (Ace + 10-value card)."""
    return len(cards) == 2 and _sum_hand(cards)[0] == 21


def _format_card(card: int) -> str:
    return "Ace" if card == 1 else str(card)


def _format_hand(cards: List[int]) -> str:
    return ", ".join(_format_card(c) for c in cards)


@dataclass(frozen=True)
class BlackjackStepResult:
    player_sum: int
    dealer_card: int
    usable_ace: bool
    player_cards: Tuple[int, ...]
    dealer_cards: Tuple[int, ...]  # fully revealed only after terminal step
    reward: float
    terminated: bool
    truncated: bool
    action: str
    step_index: int
    observation: str

    def as_tool_result(self) -> Dict[str, Any]:
        return {
            "observation": self.observation,
            "player_sum": self.player_sum,
            "dealer_card": self.dealer_card,
            "usable_ace": self.usable_ace,
            "player_cards": list(self.player_cards),
            "dealer_cards": list(self.dealer_cards) if self.terminated else [self.dealer_cards[0]],
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "action": self.action,
            "step_index": self.step_index,
        }


class BlackjackToolEnv:
    """Stateful Blackjack environment with infinite deck and tool-call friendly API."""

    def __init__(
        self,
        player_cards: List[int],
        dealer_cards: List[int],
        *,
        natural: bool = False,
        sab: bool = False,
        max_steps: int = 20,
    ):
        if len(player_cards) < 2:
            raise ValueError("Player must start with at least 2 cards")
        if len(dealer_cards) < 2:
            raise ValueError("Dealer must start with at least 2 cards")

        self._initial_player_cards = list(player_cards)
        self._initial_dealer_cards = list(dealer_cards)
        self._natural = natural
        self._sab = sab
        self._max_steps = max_steps

        self._player_cards: List[int] = []
        self._dealer_cards: List[int] = []
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        # seeded rng for any additional draws during the episode
        self._rng = random.Random()

    def reset(self) -> Dict[str, Any]:
        self._player_cards = list(self._initial_player_cards)
        self._dealer_cards = list(self._initial_dealer_cards)
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._rng = random.Random()
        return self._build_result(action="RESET", reward=0.0).as_tool_result()

    def step(self, action: str) -> Dict[str, Any]:
        normalized = str(action).strip().lower()
        if normalized not in BLACKJACK_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Expected one of {BLACKJACK_ACTIONS}")

        if self._terminated or self._truncated:
            return self._build_result(action=normalized, reward=0.0).as_tool_result()

        self._step_count += 1

        if normalized == "hit":
            self._player_cards.append(_draw_card(self._rng))
            player_sum, _ = _sum_hand(self._player_cards)
            if player_sum > 21:
                self._terminated = True
                return self._build_result(action=normalized, reward=-1.0).as_tool_result()
            self._truncated = self._step_count >= self._max_steps and not self._terminated
            return self._build_result(action=normalized, reward=0.0).as_tool_result()

        # action == "stick": dealer plays out
        dealer_sum, _ = _sum_hand(self._dealer_cards)
        while dealer_sum < 17:
            self._dealer_cards.append(_draw_card(self._rng))
            dealer_sum, _ = _sum_hand(self._dealer_cards)

        player_sum, _ = _sum_hand(self._player_cards)
        self._terminated = True

        player_natural = _is_natural(self._player_cards)
        dealer_natural = _is_natural(self._dealer_cards[:2])

        # Step 1: raw outcome (mirrors gymnasium's cmp logic)
        if dealer_sum > 21 or player_sum > dealer_sum:
            reward = 1.0
        elif player_sum == dealer_sum:
            reward = 0.0
        else:
            reward = -1.0

        # Step 2: apply rule variants on top of raw outcome
        if self._sab and player_natural and not dealer_natural:
            # Sutton-Barto: player natural always beats dealer non-natural
            reward = 1.0
        elif self._natural and not self._sab and player_natural and reward == 1.0:
            # Natural bonus applies to any win (including dealer bust)
            reward = 1.5

        return self._build_result(action=normalized, reward=reward).as_tool_result()

    def _build_result(self, action: str, reward: float) -> BlackjackStepResult:
        player_sum, usable_ace = _sum_hand(self._player_cards)
        dealer_visible = self._dealer_cards[0]
        observation = _format_observation(
            player_cards=self._player_cards,
            player_sum=player_sum,
            usable_ace=usable_ace,
            dealer_card=dealer_visible,
            terminated=self._terminated,
            dealer_cards=self._dealer_cards if self._terminated else None,
        )
        return BlackjackStepResult(
            player_sum=player_sum,
            dealer_card=dealer_visible,
            usable_ace=usable_ace,
            player_cards=tuple(self._player_cards),
            dealer_cards=tuple(self._dealer_cards),
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            action=action,
            step_index=self._step_count,
            observation=observation,
        )


def _format_observation(
    *,
    player_cards: List[int],
    player_sum: int,
    usable_ace: bool,
    dealer_card: int,
    terminated: bool,
    dealer_cards: Optional[List[int]],
) -> str:
    ace_note = " (usable Ace)" if usable_ace else ""
    lines = [
        f"Your hand: {_format_hand(player_cards)} = {player_sum}{ace_note}",
        f"Dealer shows: {_format_card(dealer_card)}",
    ]
    if terminated and dealer_cards is not None:
        dealer_sum, _ = _sum_hand(dealer_cards)
        lines.append(f"Dealer hand: {_format_hand(dealer_cards)} = {dealer_sum}")
    return "\n".join(lines)


def build_blackjack_tool_env(
    environment_context: Dict[str, Any] | None,
    *,
    natural: bool = False,
    sab: bool = False,
    max_steps: int = 20,
) -> BlackjackToolEnv:
    """Create a BlackjackToolEnv from a dataset ``environment_context``."""
    context = environment_context or {}

    if "player_cards" in context and "dealer_cards" in context:
        player_cards = [int(c) for c in context["player_cards"]]
        dealer_cards = [int(c) for c in context["dealer_cards"]]
    else:
        seed = context.get("seed", 0)
        rng = random.Random(seed)
        player_cards = [_draw_card(rng), _draw_card(rng)]
        dealer_cards = [_draw_card(rng), _draw_card(rng)]

    return BlackjackToolEnv(
        player_cards=player_cards,
        dealer_cards=dealer_cards,
        natural=natural,
        sab=sab,
        max_steps=max_steps,
    )


def build_blackjack_user_prompt(user_prompt_template: str | None, observation: str) -> str:
    """Render the user prompt for the current environment observation."""
    if user_prompt_template:
        if "{observation}" in user_prompt_template:
            return user_prompt_template.replace("{observation}", observation)
        return user_prompt_template
    return f"Blackjack observation:\n{observation}"


def generate_blackjack_seeds(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Generate n reproducible starting states as seed dicts."""
    rng = random.Random(seed)
    seeds = []
    for i in range(n):
        player_cards = [_draw_card(rng), _draw_card(rng)]
        dealer_cards = [_draw_card(rng), _draw_card(rng)]
        player_sum, usable_ace = _sum_hand(player_cards)
        seeds.append(
            {
                "seed": i,
                "player_cards": player_cards,
                "dealer_cards": dealer_cards,
                "player_sum": player_sum,
                "dealer_card": dealer_cards[0],
                "usable_ace": usable_ace,
            }
        )
    return seeds
