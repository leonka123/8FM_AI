"""Exact rules and scoring for 8ФМ Classic (deck 0 / version 0.2.1).

The formulas below were calibrated against all 66,610 recorded moves in the
local archive: every cumulative score matched to the point.  The archive also
confirms two legality constraints: Vanya is never played open before the player
has at least one closed card, and Marat/Detector are never played open while
the same player already has an open Alexey.

This module is deliberately dependency-free so the simulator and the RL
trainer can share it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .model import CARD_IDS


CLOSED_CARD = "__closed__"
CLOSED_POINTS = 2.0

# Total deck composition for a whole game (two rounds, six cards per player
# per round).  The counts sum to 12 * player_count for every supported size.
DECK_COUNTS: dict[int, dict[str, int]] = {
    3: {
        "alaksiej": 10, "evgen": 3, "amogus": 2, "olympiad": 6,
        "sergay": 4, "akim": 2, "marat": 2, "jaman": 4, "vanya": 2,
        "leonidia": 1, "veeeep": 0, "detector": 0, "alexandr": 0,
        "ultramax": 0,
    },
    4: {
        "alaksiej": 11, "evgen": 4, "amogus": 2, "olympiad": 9,
        "sergay": 6, "akim": 3, "marat": 2, "jaman": 6, "vanya": 3,
        "leonidia": 1, "veeeep": 1, "detector": 0, "alexandr": 0,
        "ultramax": 0,
    },
    5: {
        "alaksiej": 13, "evgen": 5, "amogus": 3, "olympiad": 10,
        "sergay": 7, "akim": 4, "marat": 3, "jaman": 7, "vanya": 3,
        "leonidia": 2, "veeeep": 2, "detector": 1, "alexandr": 0,
        "ultramax": 0,
    },
    6: {
        "alaksiej": 15, "evgen": 6, "amogus": 4, "olympiad": 12,
        "sergay": 8, "akim": 4, "marat": 3, "jaman": 8, "vanya": 4,
        "leonidia": 3, "veeeep": 3, "detector": 2, "alexandr": 0,
        "ultramax": 0,
    },
    7: {
        "alaksiej": 16, "evgen": 7, "amogus": 4, "olympiad": 13,
        "sergay": 9, "akim": 5, "marat": 4, "jaman": 8, "vanya": 4,
        "leonidia": 3, "veeeep": 3, "detector": 3, "alexandr": 3,
        "ultramax": 2,
    },
}

SUPPORTED_PLAYER_COUNTS = tuple(sorted(DECK_COUNTS))

# Alexey's cumulative total for 0..10 copies played by one player.  The
# archive confirms the non-square tail 60/75/100 after 49.
ALEXEY_TOTALS = (0, 1, 4, 9, 16, 25, 36, 49, 60, 75, 100)


@dataclass(frozen=True, order=True)
class Action:
    """A physical card chosen from the hand plus the open/closed choice."""

    card_id: str
    is_closed: bool

    @property
    def played_id(self) -> str | None:
        return None if self.is_closed else self.card_id


def deck_for_players(player_count: int) -> list[str]:
    """Return the expanded 12 * player_count card deck in a stable order."""
    if player_count not in DECK_COUNTS:
        raise ValueError(f"Unsupported player count: {player_count}")
    deck: list[str] = []
    for card_id in CARD_IDS:
        deck.extend([card_id] * DECK_COUNTS[player_count].get(card_id, 0))
    return deck


def _open_counts(plays: Iterable[tuple[str | None, bool]]) -> Counter[str]:
    return Counter(card_id for card_id, is_closed in plays if not is_closed and card_id)


def score_player(
    plays: Sequence[tuple[str | None, bool]],
    players_with_alexey: int,
) -> float:
    """Score one player's complete played-card history.

    ``plays`` contains the cards in the order they were played.  Closed cards
    use ``(None, True)``.  ``players_with_alexey`` is the global number of
    players who have at least one open Alexey on the table (including this
    player).  Detector's bonus uses that count; every other effect is local.
    """
    opens = [card_id for card_id, is_closed in plays if not is_closed and card_id]
    closed = sum(1 for _, is_closed in plays if is_closed)
    counts = Counter(opens)
    n = {card_id: counts.get(card_id, 0) for card_id in CARD_IDS}
    types = set(opens)
    if closed:
        types.add(CLOSED_CARD)

    total = CLOSED_POINTS * closed
    total += ALEXEY_TOTALS[min(n["alaksiej"], len(ALEXEY_TOTALS) - 1)]

    olympiads = n["olympiad"]
    if olympiads:
        per_olympiad = 6 if olympiads >= 6 else 4 if olympiads >= 3 else 2
        total += olympiads * per_olympiad

    total += n["evgen"] * (2 + 3 * n["amogus"] + n["detector"])
    total += n["amogus"] * (6 if n["amogus"] == 2 else 3)
    total += n["sergay"] * ((3 if closed else 1) + n["olympiad"])
    total += n["akim"] * (3 + 2 * n["sergay"])
    total += n["marat"] * ((6 if n["marat"] == 1 else 4) + n["olympiad"])

    jaman_closed = min(closed, 5)
    total += n["jaman"] * (
        (1 if n["evgen"] else 0)
        + n["evgen"]
        + n["sergay"]
        + n["akim"]
        + n["ultramax"]
        + 2 * jaman_closed
    )

    total += n["vanya"] * (5 + 3 * n["akim"])
    total += n["leonidia"] * (2 + len(types) - 2 * n["alaksiej"])
    total += n["veeeep"] * (
        (3 if n["veeeep"] == 1 else 1)
        + 3 * (n["marat"] + n["leonidia"] + n["detector"])
        - 2 * n["alaksiej"]
    )
    total += n["detector"] * (4 + players_with_alexey)

    alexandr_bonus = n["evgen"] + n["ultramax"] + n["sergay"] + n["marat"]
    alexandr_penalty = 2 if len(types) >= 6 else 0
    total += n["alexandr"] * (4 + alexandr_bonus - alexandr_penalty)

    shevchik = n["ultramax"]
    pair = min(n["evgen"], n["alexandr"])
    total += shevchik * (
        (3 if n["amogus"] == 0 else 0)
        + (2 if len(types) >= 6 else 0)
        + n["akim"]
    )
    total += 4 * pair * shevchik

    return float(total)


def score_players(
    plays_by_player: Sequence[Sequence[tuple[str | None, bool]]],
    alexey_players: Iterable[int],
) -> list[float]:
    """Score every player in a game from their own played-card histories."""
    alexey_count = len(set(alexey_players))
    return [score_player(plays, alexey_count) for plays in plays_by_player]


def has_open_alexey(plays: Iterable[tuple[str | None, bool]]) -> bool:
    return any(not is_closed and card_id == "alaksiej" for card_id, is_closed in plays)


def legal_actions(
    hand: Sequence[str],
    plays: Sequence[tuple[str | None, bool]],
) -> list[Action]:
    """Every legal (card, open/closed) action for a player's current hand.

    Playing any card closed is always legal.  Open Marat and Detector are
    blocked while the player already has an open Alexey.  Open Vanya is
    blocked until the player has at least one closed card.
    """
    open_counts = _open_counts(plays)
    own_closed = sum(1 for _, is_closed in plays if is_closed)
    has_alexey = open_counts.get("alaksiej", 0) > 0
    result: list[Action] = []
    for card_id in sorted(set(hand)):
        result.append(Action(card_id, True))
        if card_id in ("marat", "detector") and has_alexey:
            continue
        if card_id == "vanya" and own_closed == 0:
            continue
        result.append(Action(card_id, False))
    return result


def is_legal_action(
    hand: Sequence[str],
    plays: Sequence[tuple[str | None, bool]],
    action: Action,
) -> bool:
    return action.card_id in hand and action in legal_actions(hand, plays)


def score_action_delta(
    plays: Sequence[tuple[str | None, bool]],
    action: Action,
    *,
    own_has_alexey: bool,
    players_with_alexey: int,
) -> float:
    """Immediate score change from one action, with all effects recomputed."""
    before = score_player(plays, players_with_alexey)
    new_plays = list(plays) + [(action.played_id, action.is_closed)]
    new_alexey = players_with_alexey
    if not own_has_alexey and not action.is_closed and action.card_id == "alaksiej":
        new_alexey += 1
    after = score_player(new_plays, new_alexey)
    return after - before


def card_value_if_open(
    card_id: str,
    own_plays: Sequence[tuple[str | None, bool]],
    players_with_alexey: int,
) -> float:
    """Convenience helper for diagnostics: value of opening one more copy."""
    action = Action(card_id, False)
    return score_action_delta(
        own_plays,
        action,
        own_has_alexey=has_open_alexey(own_plays),
        players_with_alexey=players_with_alexey,
    )
