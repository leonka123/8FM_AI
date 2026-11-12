"""A small NumPy MLP trained only on completed Classic score transitions.

The model predicts the next recorded score delta for an observed action and
context. It is an auxiliary value estimator, not an oracle for unchosen cards:
the archive does not contain the alternatives players could have selected.
Inference is implemented with the standard library; NumPy is needed to train.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .audit import CLASSIC_DECK_ID, CLASSIC_DECK_VERSION, SUPPORTED_PLAYER_COUNTS


CARD_IDS = (
    "alaksiej", "olympiad", "evgen", "sergay", "jaman", "vanya", "amogus",
    "marat", "akim", "leonidia", "veeeep", "detector", "alexandr", "ultramax",
)
FEATURES_VERSION = 1


def feature_size() -> int:
    # action one-hot, close flag, player/round/turn one-hots, own and table
    # card counts, own/table closed counts, and current-score context.
    return len(CARD_IDS) + 1 + 1 + 4 + 2 + 5 + len(CARD_IDS) * 2 + 4


def feature_names() -> list[str]:
    """Human-readable names in exactly the same order as make_features()."""
    names = [f"action:{card_id}" for card_id in CARD_IDS]
    names.extend(("action:unknown", "is_closed"))
    names.extend(f"players:{count}" for count in (3, 4, 5, 6))
    names.extend(f"round:{number}" for number in (1, 2))
    names.extend(f"turn:{number}" for number in (1, 2, 3, 4, 5))
    names.extend(f"own_count:{card_id}" for card_id in CARD_IDS)
    names.extend(f"table_count:{card_id}" for card_id in CARD_IDS)
    names.extend(("own_closed", "table_closed", "own_score", "mean_other_score"))
    return names


def make_features(
    *,
    card_id: str | None,
    is_closed: bool,
    players: int,
    round_number: int,
    turn_number: int,
    own_counts: dict[str, int] | None = None,
    table_counts: dict[str, int] | None = None,
    own_closed: int = 0,
    table_closed: int = 0,
    own_score: float = 0.0,
    mean_other_score: float = 0.0,
) -> list[float]:
    own_counts = own_counts or {}
    table_counts = table_counts or {}
    known_ids = list(CARD_IDS)
    result = [0.0] * len(known_ids)
    action_index = known_ids.index(card_id) if card_id in known_ids else None
    if action_index is not None:
        result[action_index] = 1.0
    result.append(float(action_index is None))
    result.append(float(is_closed))
    result.extend(float(players == count) for count in (3, 4, 5, 6))
    result.extend(float(round_number == value) for value in (1, 2))
    result.extend(float(turn_number == value) for value in (1, 2, 3, 4, 5))
    result.extend(min(own_counts.get(card, 0), 10) / 10.0 for card in known_ids)
    result.extend(min(table_counts.get(card, 0), 30) / 30.0 for card in known_ids)
    result.extend((min(own_closed, 10) / 10.0, min(table_closed, 30) / 30.0))
    result.extend((min(max(own_score, 0.0), 150.0) / 150.0,
                   min(max(mean_other_score, 0.0), 150.0) / 150.0))
    return result


def _relu(values: Sequence[float]) -> list[float]:
    return [max(0.0, value) for value in values]


@dataclass
class ScoreModel:
    hidden_weights: list[list[float]]
    hidden_bias: list[float]
    output_weights: list[float]
    output_bias: float
    target_mean: float
    target_scale: float
    validation_rmse: float
    validation_baseline_rmse: float
    train_games: int
    validation_games: int
    training_records: int

    def predict_features(self, features: Sequence[float]) -> float:
        hidden = _relu(
            [
                bias + sum(weight * feature for weight, feature in zip(row, features))
                for row, bias in zip(self.hidden_weights, self.hidden_bias)
            ]
        )
        normalized = self.output_bias + sum(weight * value for weight, value in zip(self.output_weights, hidden))
        return normalized * self.target_scale + self.target_mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "eightfm-score-mlp",
            "features_version": FEATURES_VERSION,
            "card_ids": list(CARD_IDS),
            "hidden_weights": self.hidden_weights,
            "hidden_bias": self.hidden_bias,
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "validation_rmse": self.validation_rmse,
            "validation_baseline_rmse": self.validation_baseline_rmse,
            "train_games": self.train_games,
            "validation_games": self.validation_games,
            "training_records": self.training_records,
            "target": "recorded score delta for the played action",
            "limitations": [
                "Observational data: only chosen actions are present; alternatives are not labeled.",
                "Predictions estimate score changes, not probability of winning.",
                "Applicable only to deck 0/version 0.2.1 without modifiers and 3-6 players.",
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreModel":
        if data.get("format") != "eightfm-score-mlp" or data.get("features_version") != FEATURES_VERSION:
            raise ValueError("Unsupported or incompatible score model format")
        return cls(
            hidden_weights=data["hidden_weights"],
            hidden_bias=data["hidden_bias"],
            output_weights=data["output_weights"],
            output_bias=float(data["output_bias"]),
            target_mean=float(data["target_mean"]),
            target_scale=float(data["target_scale"]),
            validation_rmse=float(data["validation_rmse"]),
            validation_baseline_rmse=float(data["validation_baseline_rmse"]),
            train_games=int(data["train_games"]),
            validation_games=int(data["validation_games"]),
            training_records=int(data["training_records"]),
        )


def load_model(path: str | Path) -> ScoreModel:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return ScoreModel.from_dict(data)


def _archive_examples(data: dict[str, Any]) -> tuple[list[list[float]], list[float], list[int]]:
    """Build examples using only information available before each turn."""
    examples: list[list[float]] = []
    targets: list[float] = []
    game_ids: list[int] = []
    eligible_game_index = 0

    for game in data.get("games", []):
        if not isinstance(game, dict) or game.get("status") != "FINISHED":
            continue
        metadata = game.get("metadata") or {}
        deck = metadata.get("deck") or {}
        players = game.get("players")
        if (
            deck.get("id") != CLASSIC_DECK_ID
            or deck.get("version") != CLASSIC_DECK_VERSION
            or metadata.get("modifiers")
            or not isinstance(players, dict)
            or len(players) not in SUPPORTED_PLAYER_COUNTS
        ):
            continue
        moves_by_player: dict[str, list[dict[str, Any]]] = {}
        valid = True
        for player_id, player in players.items():
            moves = player.get("moves") if isinstance(player, dict) else None
            if not isinstance(moves, list) or len(moves) != 10 or any(not isinstance(m, dict) for m in moves):
                valid = False
                break
            if any(not isinstance(move.get("score"), (int, float)) or isinstance(move.get("score"), bool) for move in moves):
                valid = False
                break
            moves_by_player[player_id] = moves
        if not valid:
            continue

        previous_scores = {player_id: 0.0 for player_id in moves_by_player}
        game_rows: list[tuple[list[float], float]] = []
        for move_index in range(10):
            round_number = move_index // 5 + 1
            turn_number = move_index % 5 + 1
            table_counts: dict[str, int] = {}
            own_history: dict[str, dict[str, int]] = {}
            scores_before_turn = dict(previous_scores)
            for player_id, moves in moves_by_player.items():
                own_counts: dict[str, int] = {}
                own_closed = 0
                for previous_index in range(move_index):
                    # Card-count effects reset at a new round. Within the
                    # current round, exclude simultaneous same-turn actions.
                    if previous_index // 5 != move_index // 5 or previous_index % 5 >= move_index % 5:
                        continue
                    prior = moves[previous_index]
                    if prior.get("is_closed") is True:
                        own_closed += 1
                    card = prior.get("card_id")
                    # The public state does not reveal the identity of a
                    # closed card. Do not let archive-only details leak into
                    # features that the live adviser cannot reproduce.
                    if prior.get("is_closed") is not True and isinstance(card, str):
                        own_counts[card.split(".", 1)[0]] = own_counts.get(card.split(".", 1)[0], 0) + 1
                own_history[player_id] = {"closed": own_closed, **own_counts}
                for card, count in own_counts.items():
                    table_counts[card] = table_counts.get(card, 0) + count
            table_closed = sum(int(history.get("closed", 0)) for history in own_history.values())

            # Features for this player's action use a leave-one-player-out view
            # of the board-wide action counts.
            for player_id, moves in moves_by_player.items():
                move = moves[move_index]
                own = own_history[player_id]
                own_counts = {key: value for key, value in own.items() if key != "closed"}
                other_counts = dict(table_counts)
                for card, count in own_counts.items():
                    other_counts[card] = max(0, other_counts.get(card, 0) - count)
                is_closed = move.get("is_closed") is True
                card_id = move.get("card_id")
                card_id = card_id.split(".", 1)[0] if isinstance(card_id, str) and not is_closed else None
                score = float(move["score"])
                delta = score - previous_scores[player_id]
                previous_scores[player_id] = score
                other_scores = [scores_before_turn[other] for other in scores_before_turn if other != player_id]
                vector = make_features(
                    card_id=card_id,
                    is_closed=is_closed,
                    players=len(players),
                    round_number=round_number,
                    turn_number=turn_number,
                    own_counts=own_counts,
                    table_counts=other_counts,
                    own_closed=int(own.get("closed", 0)),
                    table_closed=max(0, table_closed - int(own.get("closed", 0))),
                    own_score=max(0.0, scores_before_turn[player_id]),
                    mean_other_score=sum(other_scores) / max(1, len(other_scores)),
                )
                game_rows.append((vector, delta))
        for vector, target in game_rows:
            examples.append(vector)
            targets.append(target)
            game_ids.append(eligible_game_index)
        eligible_game_index += 1

    return examples, targets, game_ids


def train_model(data: dict[str, Any], *, seed: int = 41, hidden_size: int = 48, epochs: int = 35) -> ScoreModel:
    """Train a tiny feed-forward regressor, splitting by game to avoid leakage."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised in dependency-free installs
        raise RuntimeError("Model training requires NumPy; install requirements.txt") from error

    features, targets, game_ids = _archive_examples(data)
    if len(features) < 100:
        raise ValueError("Not enough valid completed Classic move records to train the model")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    # Hold out the latest 20% of games. This is stricter than a shuffled
    # row-level split and better approximates the model's future use.
    unique_games = sorted(set(game_ids))
    validation_start = max(1, int(len(unique_games) * 0.8))
    validation_games = set(unique_games[validation_start:])
    validation_mask = np.asarray([game_id in validation_games for game_id in game_ids], dtype=bool)
    if validation_mask.sum() == 0 or validation_mask.sum() == len(y):
        raise ValueError("Could not form a game-level train/validation split")
    train_mask = ~validation_mask
    x_train, x_val = x[train_mask], x[validation_mask]
    y_train_raw, y_val = y[train_mask], y[validation_mask]
    target_mean = float(y_train_raw.mean())
    target_scale = max(float(y_train_raw.std()), 1.0)
    y_train = (y_train_raw - target_mean) / target_scale

    rng = np.random.default_rng(seed)
    input_size = x.shape[1]
    w1 = rng.normal(0.0, math.sqrt(2.0 / input_size), (input_size, hidden_size)).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    w2 = rng.normal(0.0, math.sqrt(1.0 / hidden_size), hidden_size).astype(np.float32)
    b2 = np.zeros((), dtype=np.float32)
    parameters = [w1, b1, w2, b2]
    first_moments = [np.zeros_like(param) for param in parameters]
    second_moments = [np.zeros_like(param) for param in parameters]
    rng_py = random.Random(seed)
    learning_rate = 0.003
    batch_size = 512
    step = 0

    for _epoch in range(epochs):
        order = list(range(len(x_train)))
        rng_py.shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = np.asarray(order[start:start + batch_size], dtype=np.int64)
            xb = x_train[indices]
            yb = y_train[indices]
            z1 = xb @ w1 + b1
            h1 = np.maximum(z1, 0.0)
            pred = h1 @ w2 + b2
            error = pred - yb
            d_pred = (2.0 / len(indices)) * error
            grad_w2 = h1.T @ d_pred
            grad_b2 = d_pred.sum()
            d_hidden = d_pred[:, None] * w2[None, :]
            d_z1 = d_hidden * (z1 > 0.0)
            grad_w1 = xb.T @ d_z1
            grad_b1 = d_z1.sum(axis=0)
            gradients = [grad_w1, grad_b1, grad_w2, grad_b2]
            step += 1
            for index, (param, grad) in enumerate(zip(parameters, gradients)):
                first_moments[index] = 0.9 * first_moments[index] + 0.1 * grad
                second_moments[index] = 0.999 * second_moments[index] + 0.001 * (grad * grad)
                m_hat = first_moments[index] / (1.0 - 0.9 ** step)
                v_hat = second_moments[index] / (1.0 - 0.999 ** step)
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

    z1_val = x_val @ w1 + b1
    prediction = (np.maximum(z1_val, 0.0) @ w2 + b2) * target_scale + target_mean
    validation_rmse = float(np.sqrt(np.mean((prediction - y_val) ** 2)))
    # A simple, honest point of comparison: predict the training-set average.
    baseline_rmse = float(np.sqrt(np.mean((target_mean - y_val) ** 2)))
    train_game_ids = {game_id for game_id, is_val in zip(game_ids, validation_mask) if not is_val}
    validation_game_ids = {game_id for game_id, is_val in zip(game_ids, validation_mask) if is_val}

    return ScoreModel(
        hidden_weights=w1.T.tolist(),
        hidden_bias=b1.tolist(),
        output_weights=w2.tolist(),
        output_bias=float(b2),
        target_mean=target_mean,
        target_scale=target_scale,
        validation_rmse=validation_rmse,
        validation_baseline_rmse=baseline_rmse,
        train_games=len(train_game_ids),
        validation_games=len(validation_game_ids),
        training_records=len(y_train_raw),
    )


def save_model(model: ScoreModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(model.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
