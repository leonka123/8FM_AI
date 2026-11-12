"""Read-only audit of the public 8 ФИЗМАТОВ JSON export.

This module deliberately does not infer hidden cards or treat the public
database as a live game feed. It reports whether archived records appear
complete enough for later analysis or replay.
"""

from __future__ import annotations

import collections
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CLASSIC_DECK_ID = 0
CLASSIC_DECK_VERSION = "0.2.1"
SUPPORTED_PLAYER_COUNTS = frozenset({3, 4, 5, 6})
CARDS_PER_ROUND = 6
ROUNDS_PER_GAME = 2


@dataclass
class AuditReport:
    source: str
    schema_version: int | None
    export_timestamp: int | float | None
    games_total: int
    users_total: int
    status_counts: dict[str, int]
    deck_counts: dict[str, int]
    classic_candidate_finished: int
    classic_candidate_with_supported_player_count: int
    classic_candidate_with_clean_move_history: int
    classic_candidate_with_reconstructable_hands: int
    classic_candidate_without_modifiers: int
    classic_candidate_with_modifiers: int
    classic_player_counts: dict[str, int]
    card_id_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deck_key(game: dict[str, Any]) -> str:
    metadata = game.get("metadata") or {}
    deck = metadata.get("deck") or {}
    return f"id={deck.get('id', '<missing>')} version={deck.get('version', '<missing>')}"


def _player_map(game: dict[str, Any]) -> dict[str, Any]:
    players = game.get("players")
    return players if isinstance(players, dict) else {}


def _has_clean_moves(game: dict[str, Any], player_count: int) -> bool:
    players = _player_map(game)
    if len(players) != player_count:
        return False
    return all(
        isinstance(player, dict)
        and isinstance(player.get("moves"), list)
        and len(player["moves"]) == ROUNDS_PER_GAME * 5
        and all(isinstance(move, dict) and move.get("submitted") is not True for move in player["moves"])
        for player in players.values()
    )


def _has_reconstructable_hands(game: dict[str, Any], player_count: int) -> bool:
    """Check that saved leftovers plus played moves rebuild six-card hands.

    In the archive ``cards[round][seat]`` is the card left in hand after the
    round, not the initial deal. Five moves plus that leftover must therefore
    add up to the six cards dealt for the round.
    """
    cards = game.get("cards")
    players = _player_map(game)
    arrangement = game.get("arrangement")
    if not isinstance(cards, list) or len(cards) != ROUNDS_PER_GAME:
        return False
    if not isinstance(arrangement, list) or len(arrangement) != player_count:
        return False
    if not _has_clean_moves(game, player_count):
        return False
    seat_of = {str(player_id): index for index, player_id in enumerate(arrangement)}
    if set(seat_of) != set(map(str, players)):
        return False
    for player_id, player in players.items():
        moves = player["moves"]
        seat = seat_of[str(player_id)]
        for round_index in range(ROUNDS_PER_GAME):
            round_hands = cards[round_index]
            if not isinstance(round_hands, list) or len(round_hands) != player_count:
                return False
            hand = round_hands[seat]
            if not isinstance(hand, list):
                return False
            if any(card is not None and not isinstance(card, str) for card in hand):
                return False
            opened = sum(
                1
                for turn in range(5)
                if moves[round_index * 5 + turn].get("is_closed") is not True
                and moves[round_index * 5 + turn].get("card_id")
            )
            closed = sum(
                1
                for turn in range(5)
                if moves[round_index * 5 + turn].get("is_closed") is True
            )
            if opened + closed + len(hand) != CARDS_PER_ROUND:
                return False
    return True


def audit_data(data: dict[str, Any], source: str = "<memory>") -> AuditReport:
    games = data.get("games")
    if not isinstance(games, list):
        raise ValueError("Корневое поле 'games' должно быть массивом")

    statuses: collections.Counter[str] = collections.Counter()
    decks: collections.Counter[str] = collections.Counter()
    classic_players: collections.Counter[str] = collections.Counter()
    card_ids: collections.Counter[str] = collections.Counter()
    warnings: list[str] = []
    candidates: list[tuple[dict[str, Any], int]] = []

    for game in games:
        if not isinstance(game, dict):
            warnings.append("В games найден элемент, который не является объектом")
            continue
        statuses[str(game.get("status", "<missing>"))] += 1
        decks[_deck_key(game)] += 1

        metadata = game.get("metadata") or {}
        deck = metadata.get("deck") or {}
        if (
            game.get("status") == "FINISHED"
            and deck.get("id") == CLASSIC_DECK_ID
            and deck.get("version") == CLASSIC_DECK_VERSION
        ):
            players = _player_map(game)
            player_count = len(players)
            candidates.append((game, player_count))
            classic_players[str(player_count)] += 1
            if metadata.get("modifiers"):
                pass
            for round_hands in game.get("cards", []) if isinstance(game.get("cards"), list) else []:
                if isinstance(round_hands, list):
                    for hand in round_hands:
                        if isinstance(hand, list):
                            for card in hand:
                                if isinstance(card, str):
                                    card_ids[card] += 1

    supported = [(game, count) for game, count in candidates if count in SUPPORTED_PLAYER_COUNTS]
    clean_moves = [(game, count) for game, count in supported if _has_clean_moves(game, count)]
    reconstructable = [
        (game, count) for game, count in supported if _has_reconstructable_hands(game, count)
    ]
    modifier_games = sum(bool((game.get("metadata") or {}).get("modifiers")) for game, _ in candidates)

    if not candidates:
        warnings.append(
            f"Не найдены завершённые игры с deck.id={CLASSIC_DECK_ID} "
            f"и deck.version={CLASSIC_DECK_VERSION}"
        )
    if supported and not reconstructable:
        warnings.append(
            "No candidate games have reconstructable six-card hands: five "
            "recorded moves plus cards[round][seat] must add up to six. "
            "Check cards[] semantics before building a replay simulator."
        )
    if modifier_games:
        warnings.append(
            f"{modifier_games} Classic candidates use modifiers; treat them "
            "separately from unmodified games."
        )
    if data.get("schema_version") != 2:
        warnings.append(
            f"Unexpected schema_version={data.get('schema_version')!r}; "
            "field handling may require an adapter."
        )

    return AuditReport(
        source=source,
        schema_version=data.get("schema_version"),
        export_timestamp=data.get("timestamp"),
        games_total=len(games),
        users_total=len(data.get("users", {})) if isinstance(data.get("users"), dict) else 0,
        status_counts=dict(sorted(statuses.items())),
        deck_counts=dict(sorted(decks.items())),
        classic_candidate_finished=len(candidates),
        classic_candidate_with_supported_player_count=len(supported),
        classic_candidate_with_clean_move_history=len(clean_moves),
        classic_candidate_with_reconstructable_hands=len(reconstructable),
        classic_candidate_without_modifiers=len(candidates) - modifier_games,
        classic_candidate_with_modifiers=modifier_games,
        classic_player_counts=dict(sorted(classic_players.items(), key=lambda pair: int(pair[0]))),
        card_id_counts=dict(sorted(card_ids.items())),
        warnings=warnings,
    )


def audit_file(path: str | Path) -> AuditReport:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Корневое значение JSON должно быть объектом")
    return audit_data(data, str(source))
