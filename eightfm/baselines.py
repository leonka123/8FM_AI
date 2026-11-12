"""Baseline policies and controlled-seat evaluation for the Classic simulator.

These helpers are used both for quick sanity checks and for the RL trainer's
periodic evaluation.  A "controlled seat" evaluation puts one candidate policy
in one seat and a fixed opponent in every other seat, rotating the candidate's
seat across games so that seat order cannot bias the result.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .env import (
    ClassicEnv,
    Observation,
    _unseen_counts,
    greedy_score_policy,
    random_policy,
)
from .policy import PolicyAction, PolicyModel, PolicyState
from .rules import Action, has_open_alexey, score_action_delta


Policy = Callable[[Observation], Action]


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, obs: Observation) -> Action:
        return random_policy(obs, self.rng)


class GreedyPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, obs: Observation) -> Action:
        return greedy_score_policy(obs, self.rng)


class ComboPolicy:
    """Heuristic teacher that commits to high-value card runs.

    Greedy play is myopic: a first Alexey is worth one point, so greedy tends
    to abandon Alexey runs in the 1–6 copy range, where the archive shows a
    below-average win rate.  This policy looks at the player's open copies,
    hand and the unseen pool and commits to a run when its threshold is
    reachable: Alexey 7+, Olympiad 6+, Akim 3+, Leonidia 2+, exactly two
    Amogus, Evgen 4+, Vanya 3+ or Sergay 4+.  When no run is reachable it
    falls back to the greedy score action.

    It is used as a behaviour-cloning teacher and as a league opponent, not as
    the final adviser.
    """

    COMBOS: tuple[tuple[str, int, float], ...] = (
        ("alaksiej", 7, 1.0),
        ("olympiad", 6, 0.9),
        ("akim", 3, 0.6),
        ("leonidia", 2, 0.5),
        ("amogus", 2, 0.4),
        ("evgen", 4, 0.4),
        ("vanya", 3, 0.3),
        ("sergay", 4, 0.3),
    )

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def target(self, obs: Observation) -> str | None:
        own_open = Counter(
            card_id for card_id, is_closed in obs.own_plays if not is_closed and card_id
        )
        hand = Counter(obs.hand)
        unseen = _unseen_counts(obs)
        best: tuple[float, str] | None = None
        for card_id, threshold, weight in self.COMBOS:
            if card_id == "amogus" and own_open.get("amogus", 0) >= 2:
                continue
            potential = (
                own_open.get(card_id, 0)
                + hand.get(card_id, 0)
                + unseen.get(card_id, 0)
            )
            if potential < threshold:
                continue
            score = weight * (potential - threshold + 1) + 0.1 * own_open.get(card_id, 0)
            if best is None or score > best[0]:
                best = (score, card_id)
        return best[1] if best is not None else None

    def __call__(self, obs: Observation) -> Action:
        target = self.target(obs)
        if target is not None:
            open_actions = [
                action
                for action in obs.legal
                if action.card_id == target and not action.is_closed
            ]
            if open_actions:
                return self.rng.choice(open_actions)
        return greedy_score_policy(obs, self.rng)


class SupervisedPolicy:
    """Adapter from the archive-trained listwise policy to simulator states."""

    def __init__(self, model: PolicyModel, seed: int = 0):
        self.model = model
        self.rng = random.Random(seed)

    def _state(self, obs: Observation) -> PolicyState:
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
        return PolicyState(
            player_count=obs.player_count,
            round_number=obs.round_number,
            turn_number=obs.turn_number,
            own_counts=dict(own_open),
            own_closed=own_closed,
            table_counts=dict(table_open),
            table_closed=table_closed,
            own_score=float(obs.scores[obs.player]),
            other_scores=tuple(
                float(score) for index, score in enumerate(obs.scores) if index != obs.player
            ),
            hand_counts=dict(Counter(obs.hand)),
            unknown_hand=0,
        )

    def __call__(self, obs: Observation) -> Action:
        state = self._state(obs)
        best_score = float("-inf")
        best: list[Action] = []
        for action in obs.legal:
            policy_action = PolicyAction(
                card_id=action.card_id,
                is_closed=action.is_closed,
                hand_count=state.hand_counts.get(action.card_id, 0),
            )
            score = self.model.predict_action(state, policy_action)
            if score > best_score + 1e-9:
                best_score = score
                best = [action]
            elif abs(score - best_score) <= 1e-9:
                best.append(action)
        return self.rng.choice(best)


def play_episode(
    player_count: int,
    policies: Sequence[Policy],
    seed: int,
) -> ClassicEnv:
    env = ClassicEnv(player_count, seed=seed)
    observations = env.observations()
    while not env.done:
        actions = [policies[index](observations[index]) for index in range(player_count)]
        result = env.step(actions)
        observations = result.observations
    return env


@dataclass
class SeatMetrics:
    games: int
    win_rate: float
    top2_rate: float
    mean_rank: float
    mean_score: float
    success_rate: float
    mean_margin: float
    scores: list[float]
    ranks: list[int]

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "win_rate": self.win_rate,
            "top2_rate": self.top2_rate,
            "mean_rank": self.mean_rank,
            "mean_score": self.mean_score,
            "success_rate": self.success_rate,
            "mean_margin": self.mean_margin,
        }


def evaluate_controlled_seat(
    player_count: int,
    candidate: Policy,
    opponent: Policy,
    games: int,
    seed_base: int = 1000,
) -> SeatMetrics:
    wins = 0
    top2 = 0
    rank_sum = 0
    score_sum = 0.0
    success = 0
    margin_sum = 0.0
    scores: list[float] = []
    ranks: list[int] = []
    for game_index in range(games):
        seat = game_index % player_count
        policies: list[Policy] = [opponent] * player_count
        policies[seat] = candidate
        env = play_episode(player_count, policies, seed_base + game_index)
        rank = env.final_ranks()[seat]
        score = env.scores[seat]
        others = [value for index, value in enumerate(env.scores) if index != seat]
        margin = score - (sum(others) / len(others) if others else score)
        wins += int(rank == 1)
        top2 += int(rank <= 2)
        rank_sum += rank
        score_sum += score
        success += int(score > 60.0)
        margin_sum += margin
        scores.append(float(score))
        ranks.append(int(rank))
    count = max(1, games)
    return SeatMetrics(
        games=games,
        win_rate=wins / count,
        top2_rate=top2 / count,
        mean_rank=rank_sum / count,
        mean_score=score_sum / count,
        success_rate=success / count,
        mean_margin=margin_sum / count,
        scores=scores,
        ranks=ranks,
    )


def compare_baselines(
    supervised: PolicyModel,
    *,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    games: int = 100,
    seed_base: int = 1000,
) -> dict[str, dict[int, dict]]:
    """Evaluate the main baselines in the same controlled-seat protocol."""
    candidates: dict[str, Policy] = {
        "supervised": SupervisedPolicy(supervised, seed=11),
        "greedy": GreedyPolicy(seed=12),
        "random": RandomPolicy(seed=13),
    }
    opponents: dict[str, Policy] = {
        "supervised": SupervisedPolicy(supervised, seed=21),
        "greedy": GreedyPolicy(seed=22),
        "random": RandomPolicy(seed=23),
    }
    report: dict[str, dict[int, dict]] = {}
    for candidate_name, candidate in candidates.items():
        report[candidate_name] = {}
        for player_count in player_counts:
            # Candidate versus the strongest fixed reference (supervised) and
            # versus greedy; random is kept as a floor.
            report[candidate_name][player_count] = {}
            for opponent_name in ("supervised", "greedy"):
                metrics = evaluate_controlled_seat(
                    player_count,
                    candidate,
                    opponents[opponent_name],
                    games,
                    seed_base=seed_base + 1000 * player_count,
                )
                report[candidate_name][player_count][opponent_name] = metrics.to_dict()
    return report


def greedy_immediate_delta(obs: Observation, action: Action) -> float:
    return score_action_delta(
        obs.own_plays,
        action,
        own_has_alexey=has_open_alexey(obs.own_plays),
        players_with_alexey=len(obs.alexey_players),
    )
