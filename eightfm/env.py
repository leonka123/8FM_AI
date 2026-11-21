"""Full self-play environment for 8ФМ Classic, 3–7 players.

The environment is intentionally small and dependency-free.  It implements the
whole game, not a single-turn bandit:

* one deck of 12 * N cards, split into two six-card deals per player;
* five simultaneous turns per round, then one leftover card is discarded;
* hands are passed to the neighbour after every turn (up/down per round);
* exact Classic scoring from :mod:`eightfm.rules`;
* legal-action masking (Vanya, Marat, Detector);
* per-player observations that contain only public information plus that
  player's own hand;
* dense score-difference rewards and terminal rank/score rewards.

The RL trainer uses :func:`state_features` and :func:`action_features`; the
same functions can be reused by a future live adviser.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, NamedTuple, Sequence

from .model import CARD_IDS
from .rules import (
    ALEXEY_TOTALS,
    Action,
    CLOSED_POINTS,
    DECK_COUNTS,
    SUPPORTED_PLAYER_COUNTS,
    deck_for_players,
    has_open_alexey,
    legal_actions,
    score_action_delta,
    score_players,
)


class Play(NamedTuple):
    card_id: str | None
    is_closed: bool


@dataclass(frozen=True)
class Observation:
    player: int
    player_count: int
    round_number: int
    turn_number: int
    hand: tuple[str, ...]
    own_plays: tuple[Play, ...]
    all_plays: tuple[tuple[Play, ...], ...]
    scores: tuple[float, ...]
    deck_counts: tuple[tuple[str, int], ...]
    closed_total: int
    alexey_players: frozenset[int]
    legal: tuple[Action, ...]
    pass_receiver: int
    pass_supplier: int


@dataclass
class StepResult:
    observations: list[Observation]
    rewards: list[float]
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Feature layout
#
# Two feature layouts are supported.  Version 1 is the original 115-value
# state / 25-value action encoding; version 2 is a superset that appends
# combo-progress, closed-card economics, turn context, opponent board
# pressure, Alexey pressure and success-threshold features.  Old checkpoints
# keep loading because every encoder takes an explicit version and the
# network stores the one it was trained with.

LATEST_FEATURE_VERSION = 2
SUPPORTED_FEATURE_VERSIONS = (1, 2)

# Combo completion thresholds used by :class:`eightfm.baselines.ComboPolicy`.
# The feature encoder exposes the same progress signal to the network so it
# can learn to invest in runs instead of chasing only immediate points.
COMBO_THRESHOLDS: dict[str, int] = {
    "alaksiej": 7,
    "olympiad": 6,
    "akim": 3,
    "leonidia": 2,
    "amogus": 2,
    "evgen": 4,
    "vanya": 3,
    "sergay": 4,
}

_SUCCESS_THRESHOLD = 60.0


def _state_feature_names_v1() -> list[str]:
    names = [f"players:{count}" for count in SUPPORTED_PLAYER_COUNTS]
    names.extend(f"round:{number}" for number in (1, 2))
    names.extend(f"turn:{number}" for number in (1, 2, 3, 4, 5))
    names.extend(f"hand:{card_id}" for card_id in CARD_IDS)
    names.extend(f"own_open:{card_id}" for card_id in CARD_IDS)
    names.append("own_closed")
    names.extend(f"table_open:{card_id}" for card_id in CARD_IDS)
    names.append("table_closed")
    names.extend((
        "own_score", "mean_other_score", "max_other_score", "min_other_score",
        "rank_score", "gap_leader", "gap_last", "alexey_players",
        "own_has_alexey", "global_closed",
    ))
    names.extend(f"unseen:{card_id}" for card_id in CARD_IDS)
    names.append("unseen_total")
    names.extend(f"supplier_open:{card_id}" for card_id in CARD_IDS)
    names.extend(("supplier_closed", "supplier_score"))
    names.extend(f"receiver_open:{card_id}" for card_id in CARD_IDS)
    names.extend(("receiver_closed", "receiver_score"))
    names.extend(("hand_size", "remaining_turns"))
    return names


def _state_feature_names_v2_extra() -> list[str]:
    names: list[str] = []
    for card_id in COMBO_THRESHOLDS:
        names.append(f"combo:{card_id}:progress")
        names.append(f"combo:{card_id}:potential")
        names.append(f"combo:{card_id}:reachable")
    names.extend(("own_closed_points", "table_closed_points"))
    names.extend(("is_first_turn", "is_last_turn", "is_last_round", "turns_left_in_round"))
    names.extend(f"opponents_open_rank:{rank}" for rank in range(1, 7))
    names.extend(("opponents_open_mean", "opponents_open_max"))
    names.extend(("table_alexey", "own_alexey", "max_opponent_alexey"))
    names.extend(("points_to_success", "above_success"))
    return names


def _action_feature_names_v1() -> list[str]:
    names = [f"card:{card_id}" for card_id in CARD_IDS]
    names.extend((
        "is_closed", "hand_count", "own_open_count", "table_open_count",
        "unseen_count", "immediate_delta", "is_last_copy", "own_closed",
        "opponent_delta_mean", "opponent_delta_max", "relative_delta",
    ))
    return names


def _action_feature_names_v2_extra() -> list[str]:
    return [
        "is_combo_card", "combo_progress_ratio", "combo_potential_ratio",
        "combo_reachable", "alexey_next_step", "is_last_turn", "is_last_round",
    ]


def state_feature_names(version: int = LATEST_FEATURE_VERSION) -> list[str]:
    if version == 1:
        return _state_feature_names_v1()
    if version == 2:
        return _state_feature_names_v1() + _state_feature_names_v2_extra()
    raise ValueError(f"Unsupported state feature version: {version}")


def action_feature_names(version: int = LATEST_FEATURE_VERSION) -> list[str]:
    if version == 1:
        return _action_feature_names_v1()
    if version == 2:
        return _action_feature_names_v1() + _action_feature_names_v2_extra()
    raise ValueError(f"Unsupported action feature version: {version}")


def state_feature_size(version: int = LATEST_FEATURE_VERSION) -> int:
    return len(state_feature_names(version))


def action_feature_size(version: int = LATEST_FEATURE_VERSION) -> int:
    return len(action_feature_names(version))


def _norm(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return min(max(float(value), 0.0), float(maximum)) / float(maximum)


def _count_vector(counts: Counter[str], maximum: float) -> list[float]:
    return [_norm(counts.get(card_id, 0), maximum) for card_id in CARD_IDS]


def _player_public(obs: Observation, index: int) -> tuple[Counter[str], int, float]:
    plays = obs.all_plays[index]
    counts = Counter(card_id for card_id, is_closed in plays if not is_closed and card_id)
    closed = sum(1 for _, is_closed in plays if is_closed)
    return counts, closed, float(obs.scores[index])


def _unseen_counts(obs: Observation) -> Counter[str]:
    deck = dict(obs.deck_counts)
    seen = Counter(
        card_id
        for plays in obs.all_plays
        for card_id, is_closed in plays
        if not is_closed and card_id
    )
    own_hand = Counter(obs.hand)
    return Counter({
        card_id: max(0, deck.get(card_id, 0) - seen.get(card_id, 0) - own_hand.get(card_id, 0))
        for card_id in CARD_IDS
    })


def _state_features_v1(obs: Observation) -> list[float]:
    n = obs.player_count
    hand = Counter(obs.hand)
    own_open = Counter(
        card_id for card_id, is_closed in obs.own_plays if not is_closed and card_id
    )
    own_closed = sum(1 for _, is_closed in obs.own_plays if is_closed)
    table_open: Counter[str] = Counter()
    table_closed = 0
    for index, plays in enumerate(obs.all_plays):
        if index == obs.player:
            continue
        table_open.update(
            card_id for card_id, is_closed in plays if not is_closed and card_id
        )
        table_closed += sum(1 for _, is_closed in plays if is_closed)
    table_max = max(1.0, 5.0 * (n - 1))
    own_score = float(obs.scores[obs.player])
    other_scores = [float(score) for index, score in enumerate(obs.scores) if index != obs.player]
    max_other = max(other_scores) if other_scores else own_score
    min_other = min(other_scores) if other_scores else own_score
    mean_other = sum(other_scores) / len(other_scores) if other_scores else own_score
    max_all = max(own_score, max_other)
    min_all = min(own_score, min_other)
    rank = sum(1 for score in other_scores if score > own_score)
    rank_score = 1.0 - rank / max(1, n - 1)
    unseen = _unseen_counts(obs)
    unseen_total = sum(unseen.values())
    supplier_open, supplier_closed, supplier_score = _player_public(obs, obs.pass_supplier)
    receiver_open, receiver_closed, receiver_score = _player_public(obs, obs.pass_receiver)

    result: list[float] = []
    result.extend(float(n == count) for count in SUPPORTED_PLAYER_COUNTS)
    result.extend(float(obs.round_number == number) for number in (1, 2))
    result.extend(float(obs.turn_number == number) for number in (1, 2, 3, 4, 5))
    result.extend(_count_vector(hand, 6))
    result.extend(_count_vector(own_open, 5))
    result.append(_norm(own_closed, 5))
    result.extend(_count_vector(table_open, table_max))
    result.append(_norm(table_closed, table_max))
    result.extend((
        _norm(own_score, 150),
        _norm(mean_other, 150),
        _norm(max_other, 150),
        _norm(min_other, 150),
        rank_score,
        _norm(max_all - own_score, 150),
        _norm(own_score - min_all, 150),
        _norm(len(obs.alexey_players), n),
        float(obs.player in obs.alexey_players),
        _norm(obs.closed_total, 5 * n),
    ))
    result.extend(_count_vector(unseen, max(1, 12 * n)))
    result.append(_norm(unseen_total, max(1, 12 * n)))
    result.extend(_count_vector(supplier_open, 5))
    result.extend((_norm(supplier_closed, 5), _norm(supplier_score, 150)))
    result.extend(_count_vector(receiver_open, 5))
    result.extend((_norm(receiver_closed, 5), _norm(receiver_score, 150)))
    result.extend((_norm(len(obs.hand), 6), (6 - obs.turn_number) / 5.0))
    return result


def _action_features_v1(obs: Observation, action: Action) -> list[float]:
    n = obs.player_count
    hand = Counter(obs.hand)
    own_open = Counter(
        card_id for card_id, is_closed in obs.own_plays if not is_closed and card_id
    )
    own_closed = sum(1 for _, is_closed in obs.own_plays if is_closed)
    table_open: Counter[str] = Counter()
    for index, plays in enumerate(obs.all_plays):
        if index == obs.player:
            continue
        table_open.update(
            card_id for card_id, is_closed in plays if not is_closed and card_id
        )
    unseen = _unseen_counts(obs)
    deck = dict(obs.deck_counts)
    card_id = action.card_id
    delta = score_action_delta(
        obs.own_plays,
        action,
        own_has_alexey=has_open_alexey(obs.own_plays),
        players_with_alexey=len(obs.alexey_players),
    )
    # Relative score change: only global Alexey/Detector effects can move an
    # opponent's score on this action, but the feature also lets the policy
    # learn to avoid handing the table free points.
    new_plays = list(obs.own_plays) + [Play(action.played_id, action.is_closed)]
    new_alexey = set(obs.alexey_players)
    if not action.is_closed and action.card_id == "alaksiej":
        new_alexey.add(obs.player)
    all_plays = [list(plays) for plays in obs.all_plays]
    all_plays[obs.player] = new_plays
    new_scores = score_players(all_plays, new_alexey)
    opponent_deltas = [
        new_scores[index] - float(obs.scores[index])
        for index in range(n) if index != obs.player
    ]
    mean_opponent_delta = (
        sum(opponent_deltas) / len(opponent_deltas) if opponent_deltas else 0.0
    )
    max_opponent_delta = max(opponent_deltas) if opponent_deltas else 0.0
    relative_delta = delta - mean_opponent_delta
    result = [0.0] * len(CARD_IDS)
    if card_id in CARD_IDS:
        result[CARD_IDS.index(card_id)] = 1.0
    result.extend((
        float(action.is_closed),
        _norm(hand.get(card_id, 0), 6),
        _norm(own_open.get(card_id, 0), 5),
        _norm(table_open.get(card_id, 0), max(1.0, 5.0 * (n - 1))),
        _norm(unseen.get(card_id, 0), max(1, deck.get(card_id, 1))),
        max(-1.0, min(1.0, delta / 20.0)),
        1.0 if hand.get(card_id, 0) == 1 else 0.0,
        _norm(own_closed, 5),
        max(-1.0, min(1.0, mean_opponent_delta / 20.0)),
        max(-1.0, min(1.0, max_opponent_delta / 20.0)),
        max(-1.0, min(1.0, relative_delta / 20.0)),
    ))
    return result


def _state_features_v2_extra(obs: Observation) -> list[float]:
    n = obs.player_count
    hand = Counter(obs.hand)
    own_open = Counter(
        card_id for card_id, is_closed in obs.own_plays if not is_closed and card_id
    )
    own_closed = sum(1 for _, is_closed in obs.own_plays if is_closed)
    table_open: Counter[str] = Counter()
    table_closed = 0
    opponent_open_totals: list[int] = []
    opponent_alexey: list[int] = []
    for index, plays in enumerate(obs.all_plays):
        if index == obs.player:
            continue
        opens = Counter(
            card_id for card_id, is_closed in plays if not is_closed and card_id
        )
        table_open.update(opens)
        opponent_open_totals.append(sum(opens.values()))
        opponent_alexey.append(opens.get("alaksiej", 0))
        table_closed += sum(1 for _, is_closed in plays if is_closed)
    unseen = _unseen_counts(obs)
    own_score = float(obs.scores[obs.player])

    result: list[float] = []
    for card_id, threshold in COMBO_THRESHOLDS.items():
        progress = own_open.get(card_id, 0) + hand.get(card_id, 0)
        potential = progress + unseen.get(card_id, 0)
        result.extend((
            _norm(progress, threshold),
            _norm(potential, threshold),
            1.0 if potential >= threshold else 0.0,
        ))
    result.extend((
        _norm(own_closed * CLOSED_POINTS, 20.0),
        _norm(table_closed * CLOSED_POINTS, 20.0),
    ))
    result.extend((
        1.0 if obs.turn_number == 1 else 0.0,
        1.0 if obs.turn_number >= 5 else 0.0,
        1.0 if obs.round_number == 2 else 0.0,
        _norm(5 - obs.turn_number, 4.0),
    ))
    ranked = sorted(opponent_open_totals, reverse=True)[:6]
    ranked.extend([0] * (6 - len(ranked)))
    result.extend(_norm(value, 5) for value in ranked)
    result.extend((
        _norm(sum(opponent_open_totals) / max(1, len(opponent_open_totals)), 5),
        _norm(max(opponent_open_totals, default=0), 5),
    ))
    result.extend((
        _norm(table_open.get("alaksiej", 0), 6),
        _norm(own_open.get("alaksiej", 0), 6),
        _norm(max(opponent_alexey, default=0), 6),
    ))
    result.extend((
        _norm(max(0.0, _SUCCESS_THRESHOLD - own_score), _SUCCESS_THRESHOLD),
        1.0 if own_score > _SUCCESS_THRESHOLD else 0.0,
    ))
    return result


def _action_features_v2_extra(obs: Observation, action: Action) -> list[float]:
    hand = Counter(obs.hand)
    own_open = Counter(
        card_id for card_id, is_closed in obs.own_plays if not is_closed and card_id
    )
    unseen = _unseen_counts(obs)
    card_id = action.card_id
    threshold = COMBO_THRESHOLDS.get(card_id)
    if action.is_closed or threshold is None:
        combo = (0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        progress = own_open.get(card_id, 0) + hand.get(card_id, 0)
        potential = progress + unseen.get(card_id, 0)
        combo = (
            1.0,
            _norm(progress, threshold),
            _norm(potential, threshold),
            1.0 if potential >= threshold else 0.0,
            0.0,
        )
    if card_id == "alaksiej" and not action.is_closed:
        copies = own_open.get(card_id, 0)
        current = ALEXEY_TOTALS[min(copies, len(ALEXEY_TOTALS) - 1)]
        following = ALEXEY_TOTALS[min(copies + 1, len(ALEXEY_TOTALS) - 1)]
        alexey_step = max(0.0, min(1.0, (following - current) / 20.0))
    else:
        alexey_step = 0.0
    return [
        combo[0], combo[1], combo[2], combo[3], alexey_step,
        1.0 if obs.turn_number >= 5 else 0.0,
        1.0 if obs.round_number == 2 else 0.0,
    ]


def state_features(
    obs: Observation, version: int = LATEST_FEATURE_VERSION
) -> list[float]:
    base = _state_features_v1(obs)
    if version == 1:
        return base
    if version == 2:
        return base + _state_features_v2_extra(obs)
    raise ValueError(f"Unsupported state feature version: {version}")


def action_features(
    obs: Observation, action: Action, version: int = LATEST_FEATURE_VERSION
) -> list[float]:
    base = _action_features_v1(obs, action)
    if version == 1:
        return base
    if version == 2:
        return base + _action_features_v2_extra(obs, action)
    raise ValueError(f"Unsupported action feature version: {version}")


# ---------------------------------------------------------------------------
# Environment


class ClassicEnv:
    """Two-round simultaneous-move Classic game for 3–7 players."""

    def __init__(self, player_count: int, seed: int | None = None):
        if player_count not in SUPPORTED_PLAYER_COUNTS:
            raise ValueError(f"Unsupported player count: {player_count}")
        self.player_count = int(player_count)
        self.rng = random.Random(seed)
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> list[Observation]:
        n = self.player_count
        deck = deck_for_players(n)
        self.rng.shuffle(deck)
        self.round_decks = [deck[: 6 * n], deck[6 * n:]]
        self.hands: list[list[str]] = [
            self.round_decks[0][index * 6:(index + 1) * 6] for index in range(n)
        ]
        self.played: list[list[Play]] = [[] for _ in range(n)]
        self.scores = [0.0] * n
        self.round_number = 1
        self.turn_number = 1
        self.closed_total = 0
        self.alexey_players: set[int] = set()
        self.done = False
        self.history: list[dict] = []
        return self.observations()

    @property
    def pass_offset(self) -> int:
        # Round 1 passes "up", round 2 "down"; the exact seat direction does
        # not change scoring, but it is kept fixed and observable.
        return -1 if self.round_number == 1 else 1

    def pass_receiver(self, player: int) -> int:
        return (player + self.pass_offset) % self.player_count

    def pass_supplier(self, player: int) -> int:
        return (player - self.pass_offset) % self.player_count

    # -- observations ------------------------------------------------------

    def observation(self, player: int) -> Observation:
        return Observation(
            player=player,
            player_count=self.player_count,
            round_number=self.round_number,
            turn_number=self.turn_number,
            hand=tuple(self.hands[player]),
            own_plays=tuple(self.played[player]),
            all_plays=tuple(tuple(plays) for plays in self.played),
            scores=tuple(self.scores),
            deck_counts=tuple(
                (card_id, DECK_COUNTS[self.player_count].get(card_id, 0))
                for card_id in CARD_IDS
            ),
            closed_total=self.closed_total,
            alexey_players=frozenset(self.alexey_players),
            legal=tuple(legal_actions(self.hands[player], self.played[player])),
            pass_receiver=self.pass_receiver(player),
            pass_supplier=self.pass_supplier(player),
        )

    def observations(self) -> list[Observation]:
        return [self.observation(index) for index in range(self.player_count)]

    # -- moves -------------------------------------------------------------

    def step(self, actions: Sequence[Action]) -> StepResult:
        if self.done:
            raise RuntimeError("The game is already finished")
        if len(actions) != self.player_count:
            raise ValueError("One action per player is required")
        for player, action in enumerate(actions):
            if action not in legal_actions(self.hands[player], self.played[player]):
                raise ValueError(f"Illegal action for player {player}: {action}")
            if action.card_id not in self.hands[player]:
                raise ValueError(f"Player {player} does not hold {action.card_id}")

        old_scores = list(self.scores)
        turn_actions: list[Action] = list(actions)
        for player, action in enumerate(actions):
            self.hands[player].remove(action.card_id)
            play = Play(action.played_id, action.is_closed)
            self.played[player].append(play)
            if play.is_closed:
                self.closed_total += 1
            elif play.card_id == "alaksiej":
                self.alexey_players.add(player)

        self.scores = score_players(self.played, self.alexey_players)
        deltas = [self.scores[index] - old_scores[index] for index in range(self.player_count)]
        # Dense reward: relative score gain versus the table.  The terminal
        # reward below carries the actual win/placement/success signal.
        mean_delta = sum(deltas) / len(deltas)
        rewards = [
            max(-1.0, min(1.0, 0.08 * (delta - mean_delta)))
            for delta in deltas
        ]

        record = {
            "round": self.round_number,
            "turn": self.turn_number,
            "actions": turn_actions,
            "deltas": deltas,
            "scores": list(self.scores),
            "closed_total": self.closed_total,
        }
        self.history.append(record)

        if self.turn_number < 5:
            self._pass_hands()
            self.turn_number += 1
        else:
            self._finish_round()
        return StepResult(
            observations=self.observations(),
            rewards=rewards,
            done=self.done,
            info={"record": record},
        )

    def _pass_hands(self) -> None:
        n = self.player_count
        new_hands: list[list[str] | None] = [None] * n
        for player, hand in enumerate(self.hands):
            new_hands[self.pass_receiver(player)] = hand
        if any(hand is None for hand in new_hands):
            raise RuntimeError("Passing hands lost a player")
        self.hands = [list(hand) for hand in new_hands]  # type: ignore[arg-type]

    def _finish_round(self) -> None:
        if self.round_number == 1:
            n = self.player_count
            self.round_number = 2
            self.turn_number = 1
            self.hands = [
                list(self.round_decks[1][index * 6:(index + 1) * 6])
                for index in range(n)
            ]
        else:
            self.done = True

    # -- helpers -----------------------------------------------------------

    def final_ranks(self) -> list[int]:
        """Competition ranks, 1 is best; ties share the better rank."""
        return [
            1 + sum(1 for other in self.scores if other > score)
            for score in self.scores
        ]

    def terminal_rewards(
        self,
        *,
        success_threshold: float = 60.0,
        win_weight: float = 1.5,
        placement_weight: float = 0.7,
        score_weight: float = 0.4,
        success_weight: float = 0.4,
        margin_weight: float = 0.2,
    ) -> list[float]:
        """Terminal reward per player: win, placement, score, >60, margin."""
        n = self.player_count
        ranks = self.final_ranks()
        rewards: list[float] = []
        for index, score in enumerate(self.scores):
            rank = ranks[index]
            winners = sum(1 for value in ranks if value == 1)
            win = (1.0 / winners) if rank == 1 else 0.0
            placement = (n - rank) / max(1, n - 1)
            score_norm = min(max(float(score) / 100.0, 0.0), 2.0)
            success = 1.0 if score > success_threshold else 0.0
            others = [value for i, value in enumerate(self.scores) if i != index]
            mean_other = sum(others) / len(others) if others else score
            margin = max(-1.0, min(1.0, (score - mean_other) / 100.0))
            rewards.append(
                win_weight * win
                + placement_weight * placement
                + score_weight * score_norm
                + success_weight * success
                + margin_weight * margin
            )
        return rewards


def random_policy(obs: Observation, rng: random.Random) -> Action:
    return rng.choice(obs.legal)


def greedy_score_policy(obs: Observation, rng: random.Random | None = None) -> Action:
    """Pick the legal action with the largest immediate score change."""
    own_has_alexey = has_open_alexey(obs.own_plays)
    best_delta = float("-inf")
    best: list[Action] = []
    for action in obs.legal:
        delta = score_action_delta(
            obs.own_plays,
            action,
            own_has_alexey=own_has_alexey,
            players_with_alexey=len(obs.alexey_players),
        )
        if delta > best_delta + 1e-9:
            best_delta = delta
            best = [action]
        elif abs(delta - best_delta) <= 1e-9:
            best.append(action)
    if rng is not None:
        return rng.choice(best)
    return best[0]
