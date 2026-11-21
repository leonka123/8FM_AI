"""Round-robin tournament for 8ФМ Classic policies.

Pits ``luna``, ``sol``, ``supervised``, ``greedy`` and ``rl_classic`` against each other
across 3-7 player games.  Every policy plays the same number of games (500) and
every player count is covered equally (100 games each).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .baselines import GreedyPolicy, SupervisedPolicy
from .env import ClassicEnv
from .policy import load_policy
from .rl import RLGreedyPolicy, load_rl

DEFAULT_PARTICIPANTS: tuple[str, ...] = (
    "luna",
    "sol",
    "astra",
    "supervised",
    "greedy",
    "rl_classic",
)
DEFAULT_MODELS: dict[str, str] = {
    "luna": "models/lu_luna.json",
    "sol": "models/lu_sol.json",
    "astra": "models/lu_astra.json",
    "supervised": "models/classic-policy.json",
    "greedy": "",
    "rl_classic": "models/classic-rl.json",
}


@dataclass
class Participant:
    name: str
    policy: object


@dataclass
class SeatRecord:
    name: str
    player_count: int
    seat: int
    rank: int
    score: float
    margin: float


def build_participants(
    *,
    names: Sequence[str],
    models: dict[str, str],
    seed: int = 0,
) -> list[Participant]:
    participants: list[Participant] = []
    for index, name in enumerate(names):
        policy: object
        if name == "greedy":
            policy = GreedyPolicy(seed=seed + index)
        elif name == "supervised":
            path = Path(models[name])
            policy = SupervisedPolicy(load_policy(path), seed=seed + index)
        else:
            path = Path(models[name])
            if not path.exists():
                raise FileNotFoundError(
                    f"Model for '{name}' not found: {path}. "
                    "Put the checkpoint in models/ or pass --luna/--sol/--rl-classic."
                )
            policy = RLGreedyPolicy(load_rl(path))
        participants.append(Participant(name=name, policy=policy))
    return participants


def build_seat_plan(
    player_count: int,
    game_index: int,
    names: Sequence[str] = DEFAULT_PARTICIPANTS,
) -> list[str]:
    cycle = list(names)
    start = game_index % len(cycle)
    seated = [cycle[(start + offset) % len(cycle)] for offset in range(player_count)]
    shift = (game_index * 2 + game_index // len(cycle)) % player_count
    return seated[shift:] + seated[:shift]


def run_tournament(
    *,
    games_per_player_count: int = 100,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    names: Sequence[str] = DEFAULT_PARTICIPANTS,
    models: dict[str, str] | None = None,
    seed_base: int = 120_000,
    seed: int = 0,
) -> dict[str, Any]:
    models = dict(DEFAULT_MODELS if models is None else models)
    participants = build_participants(names=names, models=models, seed=seed)
    by_name = {participant.name: participant.policy for participant in participants}

    records: list[SeatRecord] = []
    for player_count in player_counts:
        for game_index in range(games_per_player_count):
            seat_plan = build_seat_plan(player_count, game_index, names)
            policies = [by_name[name] for name in seat_plan]
            env = ClassicEnv(player_count, seed=seed_base + player_count * 1_000_000 + game_index)
            observations = env.observations()
            while not env.done:
                actions = [policies[index](observations[index]) for index in range(player_count)]
                observations = env.step(actions).observations
            ranks = env.final_ranks()
            scores = env.scores
            for seat, name in enumerate(seat_plan):
                others = [float(scores[i]) for i in range(player_count) if i != seat]
                margin = float(scores[seat]) - (mean(others) if others else float(scores[seat]))
                records.append(
                    SeatRecord(
                        name=name,
                        player_count=player_count,
                        seat=seat,
                        rank=int(ranks[seat]),
                        score=float(scores[seat]),
                        margin=margin,
                    )
                )

    return summarize_tournament(
        records,
        names=names,
        player_counts=player_counts,
        games_per_player_count=games_per_player_count,
        seed_base=seed_base,
    )


def summarize_tournament(
    records: Sequence[SeatRecord],
    *,
    names: Sequence[str] = DEFAULT_PARTICIPANTS,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    games_per_player_count: int = 100,
    seed_base: int = 120_000,
) -> dict[str, Any]:
    by_name: dict[str, list[SeatRecord]] = {name: [] for name in names}
    for record in records:
        by_name[record.name].append(record)

    participants_report: dict[str, dict[str, Any]] = {}
    by_player_count: dict[str, dict[str, dict[str, Any]]] = {}
    for player_count in player_counts:
        entry: dict[str, dict[str, Any]] = {}
        for name in names:
            subset = [r for r in by_name[name] if r.player_count == player_count]
            entry[name] = _aggregate(subset)
        by_player_count[str(player_count)] = entry

    for name in names:
        participants_report[name] = _aggregate(by_name[name])

    leaderboard = [
        {"name": name, "games": participants_report[name]["games"], **{
            key: value for key, value in participants_report[name].items() if key != "games"
        }}
        for name in names
    ]
    leaderboard.sort(key=lambda row: (-row["win_rate"], -row["mean_score"], row["mean_rank"]))
    for position, row in enumerate(leaderboard, start=1):
        row["place"] = position

    return {
        "format": "eightfm-tournament",
        "games": len(player_counts) * games_per_player_count,
        "games_per_player_count": games_per_player_count,
        "player_counts": list(player_counts),
        "seed_base": seed_base,
        "participants": participants_report,
        "by_player_count": by_player_count,
        "leaderboard": leaderboard,
        "chance_win_rate": mean(1.0 / count for count in player_counts),
    }


def _aggregate(records: Sequence[SeatRecord]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "games": 0,
            "win_rate": 0.0,
            "top2_rate": 0.0,
            "mean_rank": 0.0,
            "mean_score": 0.0,
            "success_rate": 0.0,
            "mean_margin": 0.0,
        }
    return {
        "games": count,
        "win_rate": sum(1 for r in records if r.rank == 1) / count,
        "top2_rate": sum(1 for r in records if r.rank <= 2) / count,
        "mean_rank": mean(r.rank for r in records),
        "mean_score": mean(r.score for r in records),
        "success_rate": sum(1 for r in records if r.score > 60.0) / count,
        "mean_margin": mean(r.margin for r in records),
    }


def format_tournament_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    player_counts = report["player_counts"]
    lines.append(
        f"Tournament: {report['games']} games, "
        f"{report['games_per_player_count']} per player count "
        f"({', '.join(str(n) for n in player_counts)} players)"
    )
    lines.append(f"Chance win rate: {report['chance_win_rate']:.3f}")
    lines.append("")

    header = (
        f"{'place':>5}  {'policy':<12} {'games':>5} {'win':>6} "
        f"{'top2':>6} {'rank':>6} {'score':>7} {'success':>6} {'margin':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["leaderboard"]:
        lines.append(
            f"{row['place']:>5}  {row['name']:<12} {row['games']:>5} "
            f"{row['win_rate']:>6.3f} {row['top2_rate']:>6.3f} "
            f"{row['mean_rank']:>6.2f} {row['mean_score']:>7.2f} "
            f"{row['success_rate']:>6.3f} {row['mean_margin']:>+7.2f}"
        )

    lines.append("")
    lines.append("Win rate by player count:")
    counts = [str(n) for n in player_counts]
    lines.append(f"{'policy':<12} {'games':>5} " + " ".join(f"{count:>6}" for count in counts))
    for name in report["participants"]:
        values = [report["by_player_count"][str(n)][name]["win_rate"] for n in counts]
        lines.append(f"{name:<12} " + " ".join(f"{v:>6.3f}" for v in values))

    return "\n".join(lines)


def save_report(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
