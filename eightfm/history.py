"""Descriptive statistics for completed, unmodified Classic move histories.

Recorded score deltas are outcomes observed in context, not intrinsic card
values and not causal estimates of the value of choosing that card.
"""

from __future__ import annotations

import collections
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audit import CLASSIC_DECK_ID, CLASSIC_DECK_VERSION, SUPPORTED_PLAYER_COUNTS


@dataclass
class ActionStat:
    players: int
    round: int
    turn: int
    action: str
    count: int
    mean_recorded_score_delta: float
    median_recorded_score_delta: float


@dataclass
class HistoryReport:
    source: str
    eligible_games: int
    games_used: int
    move_records_used: int
    action_stats: list[ActionStat]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_eligible(game: Any) -> bool:
    if not isinstance(game, dict) or game.get("status") != "FINISHED":
        return False
    metadata = game.get("metadata") or {}
    deck = metadata.get("deck") or {}
    players = game.get("players")
    return (
        deck.get("id") == CLASSIC_DECK_ID
        and deck.get("version") == CLASSIC_DECK_VERSION
        and not metadata.get("modifiers")
        and isinstance(players, dict)
        and len(players) in SUPPORTED_PLAYER_COUNTS
    )


def analyze_data(data: dict[str, Any], source: str = "<memory>") -> HistoryReport:
    games = data.get("games")
    if not isinstance(games, list):
        raise ValueError("Root field 'games' must be an array")

    eligible = [game for game in games if _is_eligible(game)]
    grouped: dict[tuple[int, int, int, str], list[float]] = collections.defaultdict(list)
    games_used = 0
    move_records_used = 0

    for game in eligible:
        players = game["players"]
        player_count = len(players)
        game_valid = True
        parsed_players: list[list[tuple[str, float]]] = []

        for player in players.values():
            moves = player.get("moves") if isinstance(player, dict) else None
            if not isinstance(moves, list) or len(moves) != 10:
                game_valid = False
                break
            parsed_moves: list[tuple[str, float]] = []
            previous_score = 0.0
            for move in moves:
                if not isinstance(move, dict) or move.get("submitted") is True:
                    game_valid = False
                    break
                score = move.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    game_valid = False
                    break
                is_closed = move.get("is_closed") is True
                card_id = move.get("card_id")
                if is_closed:
                    action = "closed"
                elif isinstance(card_id, str):
                    action = card_id
                else:
                    game_valid = False
                    break
                delta = float(score) - previous_score
                previous_score = float(score)
                parsed_moves.append((action, delta))
            if not game_valid:
                break
            parsed_players.append(parsed_moves)

        if not game_valid:
            continue

        games_used += 1
        for moves in parsed_players:
            for move_index, (action, delta) in enumerate(moves):
                round_index = move_index // 5
                turn_index = move_index % 5
                grouped[(player_count, round_index + 1, turn_index + 1, action)].append(delta)
                move_records_used += 1

    stats = [
        ActionStat(
            players=key[0],
            round=key[1],
            turn=key[2],
            action=key[3],
            count=len(values),
            mean_recorded_score_delta=round(statistics.fmean(values), 4),
            median_recorded_score_delta=round(statistics.median(values), 4),
        )
        for key, values in sorted(grouped.items())
    ]
    return HistoryReport(
        source=source,
        eligible_games=len(eligible),
        games_used=games_used,
        move_records_used=move_records_used,
        action_stats=stats,
        notes=[
            "Descriptive archive statistics only; they do not identify the optimal move.",
            "Recorded score deltas depend on the complete game state and may include interactions with previous cards.",
            "Only finished deck 0/version 0.2.1 games without modifiers, with 3-6 players and complete 10-move histories are used.",
        ],
    )


def analyze_file(path: str | Path) -> HistoryReport:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Root JSON value must be an object")
    return analyze_data(data, str(source))
