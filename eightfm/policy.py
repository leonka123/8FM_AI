"""Policy model for 8ФМ Classic.

The archive records only the action a player chose, not the alternatives.
This module reconstructs the visible hand from the saved round leftovers and
the played cards, generates every available action, and trains a listwise
ranking model on moves made by successful players.

The model is an offline adviser: it never submits a move.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .agent import CARD_NAMES, Candidate, _message_context, _parse_candidates, _parse_state
from .audit import CLASSIC_DECK_ID, CLASSIC_DECK_VERSION, SUPPORTED_PLAYER_COUNTS
from .model import CARD_IDS

POLICY_FORMAT = "eightfm-policy-mlp"
POLICY_FEATURES_VERSION = 1
CLOSED_ACTION = "__closed__"
POLICY_ACTION_IDS = CARD_IDS + (CLOSED_ACTION,)


# ---------------------------------------------------------------------------
# Feature layout


def state_feature_names() -> list[str]:
    """Names in exactly the same order as ``state_features``."""
    names = [f"players:{count}" for count in (3, 4, 5, 6)]
    names.extend(f"round:{number}" for number in (1, 2))
    names.extend(f"turn:{number}" for number in (1, 2, 3, 4, 5))
    names.extend(f"own_count:{card_id}" for card_id in CARD_IDS)
    names.extend(f"table_count:{card_id}" for card_id in CARD_IDS)
    names.extend((
        "own_closed",
        "table_closed",
        "own_score",
        "mean_other_score",
        "max_other_score",
        "rank_score",
        "score_gap_leader",
        "remaining_turns",
    ))
    names.extend(f"hand_count:{card_id}" for card_id in CARD_IDS)
    names.extend((
        "unknown_hand",
        "known_hand_fraction",
        "hand_total",
        "total_open_table",
        "total_closed_table",
    ))
    return names


def action_feature_names() -> list[str]:
    names = [f"action:{card_id}" for card_id in CARD_IDS]
    names.extend(("action:closed", "hand_count", "own_count", "table_count", "is_last_copy"))
    return names


def state_feature_size() -> int:
    return len(state_feature_names())


def action_feature_size() -> int:
    return len(action_feature_names())


def policy_input_size() -> int:
    return state_feature_size() + action_feature_size()


def _norm(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return min(max(float(value), 0.0), float(maximum)) / float(maximum)


def _counts_vector(counts: dict[str, int], maximum: float) -> list[float]:
    return [_norm(counts.get(card_id, 0), maximum) for card_id in CARD_IDS]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# State and action records


@dataclass(frozen=True)
class PolicyAction:
    card_id: str | None
    is_closed: bool
    hand_count: int = 0


@dataclass(frozen=True)
class PolicyState:
    player_count: int
    round_number: int
    turn_number: int
    own_counts: dict[str, int]
    own_closed: int
    table_counts: dict[str, int]
    table_closed: int
    own_score: float
    other_scores: tuple[float, ...]
    hand_counts: dict[str, int]
    unknown_hand: int
    # Training-only context. It is deliberately not part of the feature vector.
    game_index: int = 0
    archive_game_id: int | str = 0
    player_id: str = ""
    chosen_card: str | None = None
    is_closed: bool = False
    final_score: float = 0.0
    final_rank: int = 0
    weight: float = 1.0
    success: bool = False


@dataclass
class PolicyExample:
    state: PolicyState
    actions: list[PolicyAction]
    positive_index: int


@dataclass
class PolicyDataset:
    vectors: Any
    mask: Any
    positive: Any
    weights: Any
    game_ids: Any
    archive_game_ids: Any
    success: Any
    final_scores: Any
    action_ids: Any
    max_candidates: int
    input_size: int

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])


def state_features(state: PolicyState) -> list[float]:
    """Convert the visible pre-move state into the model's input layout."""
    player_count = max(3, min(6, int(state.player_count)))
    table_max = max(1.0, 5.0 * (player_count - 1))
    all_scores = (float(state.own_score),) + tuple(float(score) for score in state.other_scores)
    max_other = max(state.other_scores) if state.other_scores else float(state.own_score)
    max_all = max(all_scores) if all_scores else float(state.own_score)
    rank = sum(1 for score in all_scores if score > float(state.own_score))
    rank_score = 1.0 - rank / max(1, player_count - 1)
    known_total = sum(state.hand_counts.values())
    total_hand = known_total + max(0, int(state.unknown_hand))
    known_fraction = known_total / total_hand if total_hand else 1.0

    result: list[float] = []
    result.extend(float(player_count == count) for count in (3, 4, 5, 6))
    result.extend(float(state.round_number == number) for number in (1, 2))
    result.extend(float(state.turn_number == number) for number in (1, 2, 3, 4, 5))
    result.extend(_counts_vector(state.own_counts, 5))
    result.extend(_counts_vector(state.table_counts, table_max))
    result.extend((
        _norm(state.own_closed, 5),
        _norm(state.table_closed, table_max),
        _norm(state.own_score, 150),
        _norm(_mean(state.other_scores), 150),
        _norm(max_other, 150),
        rank_score,
        _norm(max_all - float(state.own_score), 150),
        (6 - int(state.turn_number)) / 5.0,
    ))
    result.extend(_counts_vector(state.hand_counts, 6))
    result.extend((
        _norm(state.unknown_hand, 6),
        known_fraction,
        _norm(total_hand, 6),
        _norm(sum(state.table_counts.values()), table_max),
        _norm(state.table_closed, table_max),
    ))
    return result


def action_features(action: PolicyAction, state: PolicyState) -> list[float]:
    """Convert one candidate action into the model's action layout."""
    player_count = max(3, min(6, int(state.player_count)))
    table_max = max(1.0, 5.0 * (player_count - 1))
    known_total = sum(state.hand_counts.values())
    total_hand = known_total + max(0, int(state.unknown_hand))
    result = [0.0] * len(CARD_IDS)
    if not action.is_closed and action.card_id in CARD_IDS:
        result[CARD_IDS.index(action.card_id)] = 1.0
    result.append(float(action.is_closed))
    if action.is_closed:
        # A closed action hides the card identity. The model learns only the
        # generic decision to close one of the remaining cards.
        result.extend((_norm(total_hand, 6), 0.0, 0.0, 0.0))
    else:
        card_id = action.card_id
        hand_count = state.hand_counts.get(card_id, 0) if card_id else 0
        result.extend((
            _norm(hand_count, 6),
            _norm(state.own_counts.get(card_id, 0), 5) if card_id else 0.0,
            _norm(state.table_counts.get(card_id, 0), table_max) if card_id else 0.0,
            1.0 if hand_count == 1 else 0.0,
        ))
    return result


def policy_features(state: PolicyState, action: PolicyAction) -> list[float]:
    return state_features(state) + action_features(action, state)


# ---------------------------------------------------------------------------
# Archive reconstruction


def _card_code(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.split(".", 1)[0]


def _is_classic_game(game: dict[str, Any]) -> bool:
    metadata = game.get("metadata") or {}
    deck = metadata.get("deck") or {}
    return (
        game.get("status") == "FINISHED"
        and deck.get("id") == CLASSIC_DECK_ID
        and deck.get("version") == CLASSIC_DECK_VERSION
        and not metadata.get("modifiers")
    )


def _valid_moves(game: dict[str, Any]) -> dict[str, list[dict[str, Any]]] | None:
    players = game.get("players")
    if not isinstance(players, dict) or len(players) not in SUPPORTED_PLAYER_COUNTS:
        return None
    moves_by_player: dict[str, list[dict[str, Any]]] = {}
    for player_id, player in players.items():
        moves = player.get("moves") if isinstance(player, dict) else None
        if not isinstance(moves, list) or len(moves) != 10:
            return None
        if any(not isinstance(move, dict) for move in moves):
            return None
        if any(
            not isinstance(move.get("score"), (int, float)) or isinstance(move.get("score"), bool)
            for move in moves
        ):
            return None
        moves_by_player[player_id] = moves
    return moves_by_player


def _hand_sizes_are_consistent(
    moves_by_player: dict[str, list[dict[str, Any]]],
    seat_of: dict[str, int],
    cards: list[Any],
) -> bool:
    """Classic deals six cards per player per round; five are played."""
    for player_id, moves in moves_by_player.items():
        seat = seat_of.get(player_id)
        if seat is None:
            return False
        for round_index in (0, 1):
            round_cards = cards[round_index] if round_index < len(cards) else None
            if not isinstance(round_cards, list) or seat >= len(round_cards):
                return False
            leftover = round_cards[seat]
            if not isinstance(leftover, list):
                return False
            open_cards = 0
            closed_cards = 0
            for turn in range(5):
                move = moves[round_index * 5 + turn]
                if move.get("is_closed") is True:
                    closed_cards += 1
                elif _card_code(move.get("card_id")):
                    open_cards += 1
            if open_cards + closed_cards + len(leftover) != 6:
                return False
    return True


def iter_policy_examples(
    data: dict[str, Any],
    *,
    success_threshold: float = 60.0,
    min_score: float = 0.0,
    success_weight: float = 2.0,
    score_bonus_weight: float = 1.0,
    rank_weight: float = 1.0,
) -> Iterable[PolicyExample]:
    """Yield one example per player-turn with every visible action.

    The state uses only information visible before the turn. The hand is
    reconstructed from the five played cards and the saved round leftover;
    cards played closed are represented by the generic closed action because
    the archive hides their identity.
    """
    game_index = 0
    for game in data.get("games", []):
        if not isinstance(game, dict) or not _is_classic_game(game):
            continue
        moves_by_player = _valid_moves(game)
        if moves_by_player is None:
            continue
        arrangement = game.get("arrangement")
        cards = game.get("cards")
        if (
            not isinstance(arrangement, list)
            or not isinstance(cards, list)
            or len(cards) < 2
            or len(arrangement) != len(moves_by_player)
        ):
            continue
        seat_of = {str(player_id): index for index, player_id in enumerate(arrangement)}
        if set(seat_of) != set(moves_by_player):
            continue
        if not _hand_sizes_are_consistent(moves_by_player, seat_of, cards):
            continue

        player_count = len(moves_by_player)
        final_scores = {
            player_id: float(moves[-1]["score"])
            for player_id, moves in moves_by_player.items()
        }
        current_game_index = game_index
        game_index += 1
        for player_id, moves in moves_by_player.items():
            final_score = final_scores[player_id]
            if min_score and final_score < min_score:
                continue
            final_rank = sum(
                1
                for other_id, other_score in final_scores.items()
                if other_id != player_id and other_score > final_score
            )
            success = final_score > success_threshold
            rank_score = 1.0 - final_rank / max(1, player_count - 1)
            score_bonus = min(1.0, max(0.0, (final_score - success_threshold) / 40.0))
            weight = (
                1.0
                + success_weight * float(success)
                + score_bonus_weight * score_bonus
                + rank_weight * rank_score
            )
            seat = seat_of[player_id]

            for round_index in (0, 1):
                leftover = [
                    card_code
                    for card in cards[round_index][seat]
                    if (card_code := _card_code(card))
                ]
                for turn in range(5):
                    own_counts: Counter[str] = Counter()
                    own_closed = 0
                    for previous in range(turn):
                        move = moves[round_index * 5 + previous]
                        if move.get("is_closed") is True:
                            own_closed += 1
                        else:
                            card_id = _card_code(move.get("card_id"))
                            if card_id:
                                own_counts[card_id] += 1

                    table_counts: Counter[str] = Counter()
                    table_closed = 0
                    for other_id, other_moves in moves_by_player.items():
                        if other_id == player_id:
                            continue
                        for previous in range(turn):
                            move = other_moves[round_index * 5 + previous]
                            if move.get("is_closed") is True:
                                table_closed += 1
                            else:
                                card_id = _card_code(move.get("card_id"))
                                if card_id:
                                    table_counts[card_id] += 1

                    # Scores are cumulative across rounds, while card counts
                    # reset at the round boundary. At the first turn of round
                    # 2 the previous score is the last move of round 1.
                    score_index = round_index * 5 + turn - 1
                    own_score = float(moves[score_index]["score"]) if score_index >= 0 else 0.0
                    other_scores = tuple(
                        float(other_moves[score_index]["score"]) if score_index >= 0 else 0.0
                        for other_id, other_moves in moves_by_player.items()
                        if other_id != player_id
                    )

                    hand_counts: Counter[str] = Counter()
                    unknown_hand = 0
                    for future in range(turn, 5):
                        move = moves[round_index * 5 + future]
                        if move.get("is_closed") is True:
                            unknown_hand += 1
                        else:
                            card_id = _card_code(move.get("card_id"))
                            if card_id:
                                hand_counts[card_id] += 1
                    for card_id in leftover:
                        hand_counts[card_id] += 1

                    chosen = moves[round_index * 5 + turn]
                    is_closed = chosen.get("is_closed") is True
                    chosen_card = None if is_closed else _card_code(chosen.get("card_id"))
                    if not is_closed and (not chosen_card or chosen_card not in hand_counts):
                        continue

                    actions = [
                        PolicyAction(card_id=card_id, is_closed=False, hand_count=hand_counts[card_id])
                        for card_id in sorted(hand_counts)
                    ]
                    actions.append(PolicyAction(card_id=None, is_closed=True))
                    if is_closed:
                        positive_index = len(actions) - 1
                    else:
                        positive_index = next(
                            (
                                index
                                for index, action in enumerate(actions)
                                if not action.is_closed and action.card_id == chosen_card
                            ),
                            -1,
                        )
                    if positive_index < 0 or len(actions) < 2:
                        continue

                    state = PolicyState(
                        player_count=player_count,
                        round_number=round_index + 1,
                        turn_number=turn + 1,
                        own_counts=dict(own_counts),
                        own_closed=own_closed,
                        table_counts=dict(table_counts),
                        table_closed=table_closed,
                        own_score=own_score,
                        other_scores=other_scores,
                        hand_counts=dict(hand_counts),
                        unknown_hand=unknown_hand,
                        game_index=current_game_index,
                        archive_game_id=game.get("id", current_game_index),
                        player_id=str(player_id),
                        chosen_card=chosen_card,
                        is_closed=is_closed,
                        final_score=final_score,
                        final_rank=final_rank,
                        weight=weight,
                        success=success,
                    )
                    yield PolicyExample(state=state, actions=actions, positive_index=positive_index)


def build_policy_dataset(
    data: dict[str, Any],
    *,
    success_threshold: float = 60.0,
    min_score: float = 0.0,
    success_weight: float = 2.0,
    score_bonus_weight: float = 1.0,
    rank_weight: float = 1.0,
) -> PolicyDataset:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - dependency-free installs
        raise RuntimeError("Policy training requires NumPy; install requirements-training.txt") from error

    examples = list(
        iter_policy_examples(
            data,
            success_threshold=success_threshold,
            min_score=min_score,
            success_weight=success_weight,
            score_bonus_weight=score_bonus_weight,
            rank_weight=rank_weight,
        )
    )
    if not examples:
        raise ValueError("No valid Classic player-turn records with a reconstructable hand")

    max_candidates = max(len(example.actions) for example in examples)
    count = len(examples)
    input_size = policy_input_size()
    vectors = np.zeros((count, max_candidates, input_size), dtype=np.float32)
    mask = np.zeros((count, max_candidates), dtype=bool)
    positive = np.zeros(count, dtype=np.int64)
    weights = np.zeros(count, dtype=np.float32)
    game_ids = np.zeros(count, dtype=np.int64)
    archive_game_ids = np.empty(count, dtype=object)
    success = np.zeros(count, dtype=bool)
    final_scores = np.zeros(count, dtype=np.float32)
    action_ids = np.full((count, max_candidates), -1, dtype=np.int16)

    for index, example in enumerate(examples):
        state_vector = state_features(example.state)
        for candidate_index, action in enumerate(example.actions):
            vectors[index, candidate_index] = state_vector + action_features(action, example.state)
            mask[index, candidate_index] = True
            action_key = CLOSED_ACTION if action.is_closed else action.card_id
            if action_key not in POLICY_ACTION_IDS:
                raise ValueError(f"Unknown policy action: {action_key!r}")
            action_ids[index, candidate_index] = POLICY_ACTION_IDS.index(action_key)
        positive[index] = example.positive_index
        weights[index] = example.state.weight
        game_ids[index] = example.state.game_index
        archive_game_ids[index] = example.state.archive_game_id
        success[index] = example.state.success
        final_scores[index] = example.state.final_score

    return PolicyDataset(
        vectors=vectors,
        mask=mask,
        positive=positive,
        weights=weights,
        game_ids=game_ids,
        archive_game_ids=archive_game_ids,
        success=success,
        final_scores=final_scores,
        action_ids=action_ids,
        max_candidates=max_candidates,
        input_size=input_size,
    )


# ---------------------------------------------------------------------------
# Model


def _relu(values: Sequence[float]) -> list[float]:
    return [max(0.0, value) for value in values]


@dataclass
class PolicyModel:
    hidden_weights: list[list[float]]
    hidden_bias: list[float]
    output_weights: list[float]
    output_bias: float
    validation_top1: float
    validation_top1_success: float
    validation_mrr: float
    baseline_top1: float
    baseline_top1_success: float
    train_states: int
    validation_states: int
    train_games: int
    validation_games: int
    best_epoch: int
    success_threshold: float
    success_weight: float = 2.0
    score_bonus_weight: float = 1.0
    rank_weight: float = 1.0

    def predict_features(self, features: Sequence[float]) -> float:
        hidden = _relu(
            [
                bias + sum(weight * feature for weight, feature in zip(row, features))
                for row, bias in zip(self.hidden_weights, self.hidden_bias)
            ]
        )
        return self.output_bias + sum(
            weight * value for weight, value in zip(self.output_weights, hidden)
        )

    def predict_action(self, state: PolicyState, action: PolicyAction) -> float:
        return self.predict_features(policy_features(state, action))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": POLICY_FORMAT,
            "features_version": POLICY_FEATURES_VERSION,
            "card_ids": list(CARD_IDS),
            "state_feature_size": state_feature_size(),
            "action_feature_size": action_feature_size(),
            "input_size": policy_input_size(),
            "state_feature_names": state_feature_names(),
            "action_feature_names": action_feature_names(),
            "hidden_weights": self.hidden_weights,
            "hidden_bias": self.hidden_bias,
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
            "validation_top1": self.validation_top1,
            "validation_top1_success": self.validation_top1_success,
            "validation_mrr": self.validation_mrr,
            "baseline_top1": self.baseline_top1,
            "baseline_top1_success": self.baseline_top1_success,
            "train_states": self.train_states,
            "validation_states": self.validation_states,
            "train_games": self.train_games,
            "validation_games": self.validation_games,
            "best_epoch": self.best_epoch,
            "success_threshold": self.success_threshold,
            "success_weight": self.success_weight,
            "score_bonus_weight": self.score_bonus_weight,
            "rank_weight": self.rank_weight,
            "target": "ranking of the action chosen by the player",
            "weighting": "1 + success_weight*success + score_bonus_weight*score_bonus + rank_weight*rank_score",
            "limitations": [
                "Observational data: unchosen actions are not labelled with their counterfactual outcome.",
                "The archive hides the identity of cards played closed; the closed action is generic.",
                "The model imitates moves of successful players; it is not a game-theoretic solver.",
                "Applicable only to deck 0/version 0.2.1 without modifiers and 3-6 players.",
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyModel":
        if data.get("format") != POLICY_FORMAT:
            raise ValueError("Unsupported policy model format")
        if data.get("features_version") != POLICY_FEATURES_VERSION:
            raise ValueError("Unsupported policy feature version")
        if list(data.get("card_ids", [])) != list(CARD_IDS):
            raise ValueError("Policy model card set does not match this build")
        if int(data.get("state_feature_size", -1)) != state_feature_size():
            raise ValueError("Policy model state feature size does not match this build")
        if int(data.get("action_feature_size", -1)) != action_feature_size():
            raise ValueError("Policy model action feature size does not match this build")
        if int(data.get("input_size", -1)) != policy_input_size():
            raise ValueError("Policy model input size does not match this build")
        hidden_weights = data["hidden_weights"]
        hidden_bias = data["hidden_bias"]
        output_weights = data["output_weights"]
        if len(hidden_weights) != len(hidden_bias) or len(hidden_weights) != len(output_weights):
            raise ValueError("Policy model hidden layer shapes do not match")
        if any(len(row) != policy_input_size() for row in hidden_weights):
            raise ValueError("Policy model hidden weight width does not match")
        return cls(
            hidden_weights=[[float(value) for value in row] for row in hidden_weights],
            hidden_bias=[float(value) for value in hidden_bias],
            output_weights=[float(value) for value in output_weights],
            output_bias=float(data["output_bias"]),
            validation_top1=float(data.get("validation_top1", 0.0)),
            validation_top1_success=float(data.get("validation_top1_success", 0.0)),
            validation_mrr=float(data.get("validation_mrr", 0.0)),
            baseline_top1=float(data.get("baseline_top1", 0.0)),
            baseline_top1_success=float(data.get("baseline_top1_success", 0.0)),
            train_states=int(data.get("train_states", 0)),
            validation_states=int(data.get("validation_states", 0)),
            train_games=int(data.get("train_games", 0)),
            validation_games=int(data.get("validation_games", 0)),
            best_epoch=int(data.get("best_epoch", 0)),
            success_threshold=float(data.get("success_threshold", 60.0)),
            success_weight=float(data.get("success_weight", 2.0)),
            score_bonus_weight=float(data.get("score_bonus_weight", 1.0)),
            rank_weight=float(data.get("rank_weight", 1.0)),
        )


def load_policy(path: str | Path) -> PolicyModel:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return PolicyModel.from_dict(data)


def save_policy(model: PolicyModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(model.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


# ---------------------------------------------------------------------------
# Training


def _forward_numpy(np: Any, x: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> tuple[Any, Any, Any]:
    flat = x.reshape(-1, x.shape[-1])
    z1 = flat @ w1 + b1
    h1 = np.maximum(z1, 0.0)
    prediction = h1 @ w2 + b2
    return z1, h1, prediction.reshape(x.shape[0], x.shape[1])


def _evaluate_numpy(
    np: Any,
    vectors: Any,
    mask: Any,
    positive: Any,
    weights: Any,
    success: Any,
    final_scores: Any,
    w1: Any,
    b1: Any,
    w2: Any,
    b2: Any,
    batch_size: int = 512,
) -> dict[str, float]:
    count = int(vectors.shape[0])
    top1_hits = 0
    success_hits = 0
    success_count = 0
    reciprocal_rank = 0.0
    matched_scores: list[float] = []
    unmatched_scores: list[float] = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        xb = vectors[start:end]
        mb = mask[start:end]
        pb = positive[start:end]
        sb = success[start:end]
        fb = final_scores[start:end]
        _, _, prediction = _forward_numpy(np, xb, w1, b1, w2, b2)
        masked = np.where(mb, prediction, -1e9)
        chosen = prediction[np.arange(end - start), pb]
        ranks = 1 + np.sum(masked > chosen[:, None], axis=1)
        reciprocal_rank += float(np.sum(1.0 / ranks))
        hits = np.argmax(masked, axis=1) == pb
        top1_hits += int(np.sum(hits))
        success_hits += int(np.sum(hits & sb))
        success_count += int(np.sum(sb))
        matched_scores.extend(fb[hits].tolist())
        unmatched_scores.extend(fb[~hits].tolist())
    return {
        "top1": top1_hits / max(1, count),
        "top1_success": success_hits / max(1, success_count),
        "mrr": reciprocal_rank / max(1, count),
        "matched_final_score": _mean(matched_scores),
        "unmatched_final_score": _mean(unmatched_scores),
        "matched_count": float(len(matched_scores)),
        "unmatched_count": float(len(unmatched_scores)),
    }


def _frequency_baseline(
    np: Any,
    action_ids: Any,
    mask: Any,
    positive: Any,
    train_mask: Any,
    validation_mask: Any,
    success: Any,
) -> tuple[float, float]:
    train_positive = action_ids[train_mask, :][np.arange(int(train_mask.sum())), positive[train_mask]]
    counts: Counter[int] = Counter(int(value) for value in train_positive)
    fallback = counts.most_common(1)[0][0] if counts else -1
    val_ids = action_ids[validation_mask]
    val_mask = mask[validation_mask]
    val_positive = positive[validation_mask]
    val_success = success[validation_mask]
    hits = 0
    success_hits = 0
    success_count = int(val_success.sum())
    for row in range(val_ids.shape[0]):
        candidates = [
            (counts.get(int(val_ids[row, column]), 0), -column, int(val_ids[row, column]))
            for column in range(val_ids.shape[1])
            if val_mask[row, column]
        ]
        if not candidates:
            continue
        best_count = max(candidate[0] for candidate in candidates)
        best_actions = {candidate[2] for candidate in candidates if candidate[0] == best_count}
        if not best_actions:
            best_actions = {fallback}
        positive_action = int(val_ids[row, val_positive[row]])
        hit = positive_action in best_actions
        hits += int(hit)
        success_hits += int(hit and val_success[row])
    count = int(val_ids.shape[0])
    return hits / max(1, count), success_hits / max(1, success_count)


def train_policy_model(
    data: dict[str, Any],
    *,
    seed: int = 41,
    hidden_size: int = 64,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.003,
    l2: float = 1e-4,
    success_threshold: float = 60.0,
    min_score: float = 0.0,
    success_weight: float = 2.0,
    score_bonus_weight: float = 1.0,
    rank_weight: float = 1.0,
    patience: int = 6,
) -> PolicyModel:
    """Train a listwise ranking model, splitting by game to avoid leakage."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - dependency-free installs
        raise RuntimeError("Policy training requires NumPy; install requirements-training.txt") from error

    dataset = build_policy_dataset(
        data,
        success_threshold=success_threshold,
        min_score=min_score,
        success_weight=success_weight,
        score_bonus_weight=score_bonus_weight,
        rank_weight=rank_weight,
    )
    unique_games = sorted(int(value) for value in np.unique(dataset.game_ids))
    if len(unique_games) < 2:
        raise ValueError("At least two valid games are required for a policy split")
    validation_start = max(1, int(len(unique_games) * 0.8))
    validation_games = set(unique_games[validation_start:])
    validation_mask = np.asarray(
        [int(game_id) in validation_games for game_id in dataset.game_ids],
        dtype=bool,
    )
    if validation_mask.sum() == 0 or validation_mask.sum() == len(validation_mask):
        raise ValueError("Could not form a game-level policy train/validation split")
    train_mask = ~validation_mask

    vectors = dataset.vectors
    mask = dataset.mask
    positive = dataset.positive
    weights = dataset.weights
    action_ids = dataset.action_ids
    success = dataset.success
    final_scores = dataset.final_scores
    input_size = dataset.input_size
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, math.sqrt(2.0 / input_size), (input_size, hidden_size)).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    w2 = rng.normal(0.0, math.sqrt(1.0 / hidden_size), hidden_size).astype(np.float32)
    b2 = np.zeros((), dtype=np.float32)
    parameters = [w1, b1, w2, b2]
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    rng_py = random.Random(seed)
    step = 0
    best_metrics: dict[str, float] | None = None
    best_parameters: list[Any] | None = None
    best_epoch = 0
    stale_epochs = 0
    train_indices = np.flatnonzero(train_mask)

    for epoch in range(1, epochs + 1):
        # random.Random.shuffle needs a Python list; preserve the permutation.
        shuffled = train_indices.tolist()
        rng_py.shuffle(shuffled)
        order = np.asarray(shuffled, dtype=np.int64)

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb = vectors[indices]
            mb = mask[indices]
            pb = positive[indices]
            wb = weights[indices]
            batch_count = len(indices)
            z1, h1, prediction = _forward_numpy(np, xb, w1, b1, w2, b2)
            masked = np.where(mb, prediction, -1e9)
            max_prediction = np.max(masked, axis=1, keepdims=True)
            exp_prediction = np.exp(masked - max_prediction) * mb
            sum_exp = np.sum(exp_prediction, axis=1, keepdims=True)
            log_sum_exp = max_prediction[:, 0] + np.log(np.maximum(sum_exp[:, 0], 1e-12))
            chosen = prediction[np.arange(batch_count), pb]
            softmax = exp_prediction / np.maximum(sum_exp, 1e-12)
            d_prediction = softmax
            d_prediction[np.arange(batch_count), pb] -= 1.0
            d_prediction *= (wb / batch_count)[:, None]
            d_prediction_flat = d_prediction.reshape(-1)

            grad_w2 = h1.T @ d_prediction_flat
            grad_b2 = d_prediction_flat.sum()
            d_hidden = d_prediction_flat[:, None] * w2[None, :]
            d_z1 = d_hidden * (z1 > 0.0)
            grad_w1 = xb.reshape(-1, input_size).T @ d_z1
            grad_b1 = d_z1.sum(axis=0)
            grad_w1 += l2 * w1
            grad_w2 += l2 * w2
            gradients = [grad_w1, grad_b1, grad_w2, grad_b2]
            step += 1
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                first_moments[index] = 0.9 * first_moments[index] + 0.1 * gradient
                second_moments[index] = 0.999 * second_moments[index] + 0.001 * (gradient * gradient)
                m_hat = first_moments[index] / (1.0 - 0.9 ** step)
                v_hat = second_moments[index] / (1.0 - 0.999 ** step)
                parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        metrics = _evaluate_numpy(
            np,
            vectors[validation_mask],
            mask[validation_mask],
            positive[validation_mask],
            weights[validation_mask],
            success[validation_mask],
            final_scores[validation_mask],
            w1,
            b1,
            w2,
            b2,
        )
        if best_metrics is None or metrics["top1"] > best_metrics["top1"]:
            best_metrics = metrics
            best_parameters = [parameter.copy() for parameter in parameters]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_metrics is None or best_parameters is None:
        raise RuntimeError("Policy training did not produce a model")
    w1, b1, w2, b2 = best_parameters
    baseline_top1, baseline_top1_success = _frequency_baseline(
        np,
        action_ids,
        mask,
        positive,
        train_mask,
        validation_mask,
        success,
    )
    train_game_ids = {int(value) for value in dataset.game_ids[train_mask]}
    validation_game_ids = {int(value) for value in dataset.game_ids[validation_mask]}
    return PolicyModel(
        hidden_weights=w1.T.tolist(),
        hidden_bias=b1.tolist(),
        output_weights=w2.tolist(),
        output_bias=float(b2),
        validation_top1=best_metrics["top1"],
        validation_top1_success=best_metrics["top1_success"],
        validation_mrr=best_metrics["mrr"],
        baseline_top1=baseline_top1,
        baseline_top1_success=baseline_top1_success,
        train_states=int(train_mask.sum()),
        validation_states=int(validation_mask.sum()),
        train_games=len(train_game_ids),
        validation_games=len(validation_game_ids),
        best_epoch=best_epoch,
        success_threshold=success_threshold,
        success_weight=success_weight,
        score_bonus_weight=score_bonus_weight,
        rank_weight=rank_weight,
    )


# ---------------------------------------------------------------------------
# Live inference


@dataclass
class PolicyEstimate:
    card_id: str | None
    name: str
    play: str
    policy_score: float
    probability: float
    score_delta: float | None = None
    hand_count: int = 0
    feature_vector: list[float] | None = None

    def to_dict(self, *, include_features: bool = True) -> dict[str, Any]:
        result = {
            "card_id": self.card_id,
            "name": self.name,
            "play": self.play,
            "policy_score": self.policy_score,
            "probability": self.probability,
            "score_delta": self.score_delta,
            "hand_count": self.hand_count,
        }
        if include_features:
            result["feature_vector"] = self.feature_vector
        return result


@dataclass
class PolicyRecommendation:
    round_number: int | None
    turn_number: int | None
    player_name: str | None
    candidates: list[PolicyEstimate]
    warnings: list[str]


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    maximum = max(scores)
    values = [math.exp(score - maximum) for score in scores]
    total = sum(values)
    return [value / total for value in values]


def _card_name(card_id: str | None) -> str:
    if card_id is None:
        return "Карта в закрытую"
    for name in CARD_NAMES.get(card_id, ()):
        return name
    return card_id


def state_from_message(
    message: str,
    *,
    player_name: str | None,
) -> tuple[PolicyState, list[Candidate], str | None]:
    """Build the policy state and candidate list from a pasted message."""
    candidates = _parse_candidates(message)
    context = _message_context(message, player_name)
    _, _, rows, resolved_player_name = _parse_state(message, player_name)
    own = next((row for row in rows if row["name"] == resolved_player_name), None)
    if own is None:
        raise ValueError(
            f"Игрок '{player_name or resolved_player_name}' не найден среди строк сообщения"
        )
    other_scores = tuple(
        float(row["score"])
        for row in rows
        if row["name"] != resolved_player_name
    )
    hand_counts = Counter(
        candidate.card_id
        for candidate in candidates
        if candidate.card_id in CARD_IDS
    )
    if not hand_counts:
        raise ValueError("В сообщении не найдено ни одной известной карты-кандидата")
    turn_number = context.turn_number or 1
    # Classic deals six cards and five are played: at turn 1 the hand has six
    # cards, at turn 5 it has two. If the pasted message shows fewer known
    # cards, keep the missing ones as the generic unknown part of the hand.
    expected_hand = max(1, 7 - turn_number)
    known_total = sum(hand_counts.values())
    unknown_hand = max(0, expected_hand - known_total)
    state = PolicyState(
        player_count=context.player_count,
        round_number=context.round_number or 1,
        turn_number=turn_number,
        own_counts=context.own_counts,
        own_closed=context.own_closed,
        table_counts=context.other_table_counts,
        table_closed=context.other_table_closed,
        own_score=context.own_score,
        other_scores=other_scores,
        hand_counts=dict(hand_counts),
        unknown_hand=unknown_hand,
        player_id=resolved_player_name or "",
    )
    return state, candidates, resolved_player_name


def recommend_policy(
    message: str,
    *,
    player_name: str | None = None,
    model: PolicyModel | None = None,
    model_path: str | Path = "models/classic-policy.json",
    score_model: Any | None = None,
) -> PolicyRecommendation:
    """Rank every card in the visible hand plus the generic closed action."""
    if model is None:
        model = load_policy(model_path)
    state, candidates, resolved_player_name = state_from_message(
        message,
        player_name=player_name,
    )
    names = {
        candidate.card_id: candidate.name
        for candidate in candidates
        if candidate.card_id in CARD_IDS
    }
    actions = [
        PolicyAction(card_id=card_id, is_closed=False, hand_count=state.hand_counts[card_id])
        for card_id in sorted(state.hand_counts)
    ]
    actions.append(PolicyAction(card_id=None, is_closed=True))
    features = [policy_features(state, action) for action in actions]
    scores = [model.predict_features(vector) for vector in features]
    probabilities = _softmax(scores)
    estimates: list[PolicyEstimate] = []
    for action, vector, score, probability in zip(actions, features, scores, probabilities):
        score_delta = None
        if score_model is not None:
            from .model import make_features

            score_delta = score_model.predict_features(
                make_features(
                    card_id=None if action.is_closed else action.card_id,
                    is_closed=action.is_closed,
                    players=state.player_count,
                    round_number=state.round_number,
                    turn_number=state.turn_number,
                    own_counts=state.own_counts,
                    table_counts=state.table_counts,
                    own_closed=state.own_closed,
                    table_closed=state.table_closed,
                    own_score=state.own_score,
                    mean_other_score=_mean(state.other_scores),
                )
            )
        estimates.append(
            PolicyEstimate(
                card_id=None if action.is_closed else action.card_id,
                name="Карта в закрытую" if action.is_closed else names.get(action.card_id, _card_name(action.card_id)),
                play="закрыто" if action.is_closed else "открыто",
                policy_score=round(float(score), 4),
                probability=round(float(probability), 4),
                score_delta=round(float(score_delta), 3) if score_delta is not None else None,
                hand_count=0 if action.is_closed else int(state.hand_counts.get(action.card_id, 0)),
                feature_vector=vector,
            )
        )
    estimates.sort(key=lambda item: (-item.policy_score, item.name))
    warnings = [
        f"Политика обучена на ходах успешных игроков (порог {model.success_threshold:g} очков); "
        f"top-1 на отложенных партиях {model.validation_top1:.1%}, на успешных {model.validation_top1_success:.1%}.",
        "Это имитация ходов сильных игроков, а не доказанное оптимальное решение: в архиве нет исходов невыбранных карт.",
        "Закрытая карта в архиве обезличена, поэтому вариант «закрыто» общий; конкретную карту для закрытия выбираете вы.",
        "Вероятность в выводе — доля softmax между кандидатами, а не измеренная вероятность победы.",
    ]
    unknown_candidates = [
        candidate.name
        for candidate in candidates
        if candidate.card_id not in CARD_IDS
    ]
    if unknown_candidates:
        warnings.append(
            "Не распознаны карты-кандидаты, они не участвуют в рейтинге: "
            + ", ".join(unknown_candidates)
        )
    if state.unknown_hand:
        warnings.append(
            f"В сообщении видно меньше карт, чем ожидается для хода {state.turn_number}: "
            f"{state.unknown_hand} карт(ы) считаются неизвестными."
        )
    return PolicyRecommendation(
        round_number=state.round_number,
        turn_number=state.turn_number,
        player_name=resolved_player_name,
        candidates=estimates,
        warnings=warnings,
    )
