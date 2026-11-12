"""Benchmark the trained RL and supervised policies in the same games.

Both trained models play in every game.  The remaining seats are filled by a
baseline policy (greedy by default).  Seats rotate between games so that no
model keeps a fixed position.  The report focuses on mean score, but also
includes win rate, mean rank and the share of games above 60 points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Sequence

from .baselines import (
    ComboPolicy,
    GreedyPolicy,
    Policy,
    RandomPolicy,
    SupervisedPolicy,
)
from .env import ClassicEnv
from .policy import load_policy
from .rl import RLGreedyPolicy, load_rl
from .rules import SUPPORTED_PLAYER_COUNTS


MODEL_NAMES = ("rl", "supervised")
FILLER_KINDS = ("greedy", "combo", "random")


@dataclass
class RunningStats:
    scores: list[float] = field(default_factory=list)
    ranks: list[int] = field(default_factory=list)
    wins: int = 0
    successes: int = 0

    def add(self, score: float, rank: int) -> None:
        self.scores.append(float(score))
        self.ranks.append(int(rank))
        if rank == 1:
            self.wins += 1
        if score > 60:
            self.successes += 1

    def to_dict(self) -> dict:
        games = len(self.scores)
        if games == 0:
            return {
                "games": 0,
                "mean_score": 0.0,
                "median_score": 0.0,
                "stdev_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "win_rate": 0.0,
                "mean_rank": 0.0,
                "success_rate": 0.0,
            }
        return {
            "games": games,
            "mean_score": mean(self.scores),
            "median_score": median(self.scores),
            "stdev_score": pstdev(self.scores) if games > 1 else 0.0,
            "min_score": min(self.scores),
            "max_score": max(self.scores),
            "win_rate": self.wins / games,
            "mean_rank": mean(self.ranks),
            "success_rate": self.successes / games,
        }


def _make_filler(kind: str, seed: int) -> Policy:
    if kind == "greedy":
        return GreedyPolicy(seed=seed)
    if kind == "combo":
        return ComboPolicy(seed=seed)
    if kind == "random":
        return RandomPolicy(seed=seed)
    raise ValueError(f"Unknown filler kind: {kind}")


def play_game(
    player_count: int,
    seed: int,
    policies: Sequence[Policy],
) -> tuple[list[float], list[int]]:
    """Play one complete game with the given policy for every seat."""
    if len(policies) != player_count:
        raise ValueError("One policy per player is required")
    env = ClassicEnv(player_count, seed=seed)
    while not env.done:
        observations = env.observations()
        actions = [
            policies[index](observations[index])
            for index in range(player_count)
        ]
        env.step(actions)
    return list(env.scores), env.final_ranks()


def benchmark_models(
    *,
    games: int = 20,
    player_counts: Sequence[int] = (3, 4),
    filler: str = "greedy",
    seed_base: int = 70000,
    model_path: str = "models/classic-rl.json",
    supervised_model_path: str = "models/classic-policy.json",
) -> dict:
    """Run the trained models together and aggregate their scores."""
    if games < 1:
        raise ValueError("At least one game is required")
    if filler not in FILLER_KINDS:
        raise ValueError(
            f"Unknown filler: {filler}; use {', '.join(FILLER_KINDS)}"
        )
    if not player_counts:
        raise ValueError("At least one player count is required")
    unsupported = [n for n in player_counts if n not in SUPPORTED_PLAYER_COUNTS]
    if unsupported:
        raise ValueError(f"Unsupported player counts: {unsupported}")

    rl_net = load_rl(model_path)
    supervised_model = load_policy(supervised_model_path)

    combined = {name: RunningStats() for name in MODEL_NAMES}
    combined_filler = RunningStats()
    combined_h2h = {"rl_wins": 0, "supervised_wins": 0, "ties": 0}
    report: dict = {
        "games_per_player_count": games,
        "filler": filler,
        "seed_base": seed_base,
        "model": model_path,
        "supervised_model": supervised_model_path,
        "players": {},
    }

    for player_count in player_counts:
        stats = {name: RunningStats() for name in MODEL_NAMES}
        filler_stats = RunningStats()
        head_to_head = {"rl_wins": 0, "supervised_wins": 0, "ties": 0}

        for game_index in range(games):
            seed = seed_base + player_count * 1000 + game_index
            order = [
                (game_index + offset) % player_count
                for offset in range(player_count)
            ]
            rl_seat = order[0]
            supervised_seat = order[1]
            policies: list[Policy] = []
            for seat in range(player_count):
                if seat == rl_seat:
                    policies.append(RLGreedyPolicy(rl_net))
                elif seat == supervised_seat:
                    policies.append(
                        SupervisedPolicy(supervised_model, seed=seed + seat)
                    )
                else:
                    policies.append(_make_filler(filler, seed + seat))

            scores, ranks = play_game(player_count, seed, policies)
            rl_score = scores[rl_seat]
            supervised_score = scores[supervised_seat]
            stats["rl"].add(rl_score, ranks[rl_seat])
            stats["supervised"].add(supervised_score, ranks[supervised_seat])
            combined["rl"].add(rl_score, ranks[rl_seat])
            combined["supervised"].add(
                supervised_score, ranks[supervised_seat]
            )

            if rl_score > supervised_score:
                head_to_head["rl_wins"] += 1
                combined_h2h["rl_wins"] += 1
            elif supervised_score > rl_score:
                head_to_head["supervised_wins"] += 1
                combined_h2h["supervised_wins"] += 1
            else:
                head_to_head["ties"] += 1
                combined_h2h["ties"] += 1

            for seat in range(player_count):
                if seat in (rl_seat, supervised_seat):
                    continue
                filler_stats.add(scores[seat], ranks[seat])
                combined_filler.add(scores[seat], ranks[seat])

        report["players"][str(player_count)] = {
            "games": games,
            "models": {
                name: stats[name].to_dict()
                for name in MODEL_NAMES
            },
            "filler": filler_stats.to_dict(),
            "head_to_head": head_to_head,
        }

    report["summary"] = {
        "games": games * len(player_counts),
        "models": {
            name: combined[name].to_dict()
            for name in MODEL_NAMES
        },
        "filler": combined_filler.to_dict(),
        "head_to_head": combined_h2h,
    }
    return report


def _plural(number: int, one: str, few: str, many: str) -> str:
    value = abs(number)
    if value % 10 == 1 and value % 100 != 11:
        return one
    if 2 <= value % 10 <= 4 and not 12 <= value % 100 <= 14:
        return few
    return many


def _row(label: str, stats: dict) -> str:
    return (
        f"  {label:<12} {stats['mean_score']:>8.2f} "
        f"{stats['median_score']:>8.1f} {stats['stdev_score']:>8.2f} "
        f"{stats['win_rate']:>7.1%} {stats['mean_rank']:>9.2f} "
        f"{stats['success_rate']:>7.1%}"
    )


def format_report(report: dict) -> str:
    """Render a compact text table for the terminal."""
    per_format = report["games_per_player_count"]
    lines = [
        "Сравнение обученных моделей",
        (
            f"{per_format} {_plural(per_format, 'партия', 'партии', 'партий')} "
            f"на каждый размер стола; свободные места — {report['filler']}."
        ),
        "Места RL и supervised ротируются между играми.",
        (
            f"Модели: {report['model']} и {report['supervised_model']}; "
            f"seed base {report['seed_base']}."
        ),
    ]
    for player_count, entry in report["players"].items():
        count = int(player_count)
        lines.append("")
        lines.append(
            f"{count} "
            f"{_plural(count, 'игрок', 'игрока', 'игроков')} "
            f"({entry['games']} "
            f"{_plural(entry['games'], 'партия', 'партии', 'партий')})"
        )
        lines.append(
            "  модель       средний  медиана  ст.откл.  винрейт  ср.место    >60"
        )
        for name in MODEL_NAMES:
            label = "RL" if name == "rl" else "supervised"
            lines.append(_row(label, entry["models"][name]))
        lines.append(_row(report["filler"], entry["filler"]))
        h2h = entry["head_to_head"]
        lines.append(
            "  head-to-head: "
            f"RL {h2h['rl_wins']}, supervised {h2h['supervised_wins']}, "
            f"ничьи {h2h['ties']}"
        )

    summary = report["summary"]
    lines.append("")
    lines.append(f"Итого ({summary['games']} партий, по {report['games_per_player_count']} на размер)")
    for name in MODEL_NAMES:
        stats = summary["models"][name]
        label = "RL" if name == "rl" else "supervised"
        lines.append(
            f"  {label:<12} средний {stats['mean_score']:.2f}, "
            f"винрейт {stats['win_rate']:.1%}, "
            f"ср.место {stats['mean_rank']:.2f}, "
            f">60 {stats['success_rate']:.1%}"
        )
    h2h = summary["head_to_head"]
    lines.append(
        "  head-to-head: "
        f"RL {h2h['rl_wins']}, supervised {h2h['supervised_wins']}, "
        f"ничьи {h2h['ties']}"
    )
    return "\n".join(lines)
