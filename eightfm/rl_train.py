"""Training, evaluation and live-message inference for the RL policy."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .agent import EMOJI_CARDS, _parse_candidates, _parse_state, _strip_played_card
from .baselines import (
    ComboPolicy,
    GreedyPolicy,
    Policy,
    RandomPolicy,
    SupervisedPolicy,
    evaluate_controlled_seat,
)
from .env import (
    ClassicEnv,
    LATEST_FEATURE_VERSION,
    Observation,
    Play,
    action_feature_size,
    action_features,
    state_feature_size,
    state_features,
)
from .model import CARD_IDS
from .policy import PolicyModel
from .rules import DECK_COUNTS, Action, has_open_alexey, legal_actions, score_action_delta
from .rl import (
    Adam,
    RLGreedyPolicy,
    RLNetwork,
    _pad_observations,
    act_batch,
    backward_batch,
    behaviour_clone,
    clone_network,
    collect_bc_dataset,
    create_network,
    forward_batch,
    load_rl,
    masked_softmax,
    save_rl,
)


# ---------------------------------------------------------------------------
# Episodes


@dataclass
class EpisodeData:
    states: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    chosen: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    player_ids: np.ndarray
    turns: np.ndarray
    terminal_rewards: list[float]
    scores: list[float]
    ranks: list[int]
    player_count: int


def collect_episode(
    net: RLNetwork,
    player_count: int,
    seed: int,
    rng: np.random.Generator,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    temperature: float = 1.0,
    opponent_policies: Sequence[Policy | None] | None = None,
    rl_seats: Sequence[int] | None = None,
) -> EpisodeData:
    """Play one full game and collect decisions for the learner's seats.

    ``opponent_policies`` contains one callable per seat; ``None`` marks a
    learner-controlled seat.  When omitted, every seat is self-play.  League
    episodes therefore train only on the states the learner actually faced,
    while the value head can still bootstrap from every seat's observation.

    Advantages use GAE(lambda): ``delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)``
    and ``A_t = delta_t + gamma * lambda * A_{t+1}``.  The value target is the
    corresponding lambda-return ``A_t + V(s_t)``.
    """
    env = ClassicEnv(player_count, seed=seed)
    observations = env.observations()
    n = player_count
    if rl_seats is None:
        if opponent_policies is None:
            rl_seats = tuple(range(n))
        else:
            rl_seats = tuple(
                seat for seat, policy in enumerate(opponent_policies) if policy is None
            )
    rl_seats = tuple(int(seat) for seat in rl_seats)
    if not rl_seats:
        raise ValueError("At least one learner-controlled seat is required")
    if opponent_policies is not None and len(opponent_policies) != n:
        raise ValueError("One opponent policy (or None) per seat is required")

    decisions: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, float]] = []
    dense_rewards: list[list[float]] = []
    values_history: list[np.ndarray] = []
    while not env.done:
        turn_index = len(dense_rewards)
        states, actions, mask, legal_lists = _pad_observations(
            observations, net.feature_version
        )
        scores, values, _cache = forward_batch(net, states, actions, mask)
        probabilities = masked_softmax(scores / max(1e-6, temperature), mask)
        selected: list[Action] = []
        for player in range(n):
            opponent = (
                opponent_policies[player] if opponent_policies is not None else None
            )
            if opponent is not None:
                action = opponent(observations[player])
                if action not in observations[player].legal:
                    raise ValueError(
                        f"Opponent policy chose an illegal action for seat {player}"
                    )
                selected.append(action)
                continue
            row = probabilities[player]
            row = row / max(1e-12, float(np.sum(row)))
            chosen_index = int(rng.choice(probabilities.shape[1], p=row))
            selected.append(legal_lists[player][chosen_index])
            decisions.append((
                states[player].copy(),
                actions[player].copy(),
                mask[player].copy(),
                chosen_index,
                player,
                turn_index,
                float(np.log(max(1e-12, probabilities[player, chosen_index]))),
                float(values[player]),
            ))
        values_history.append(np.asarray(values, dtype=np.float32).copy())
        result = env.step(selected)
        dense_rewards.append(list(result.rewards))
        observations = result.observations

    terminal = env.terminal_rewards()
    turns = len(dense_rewards)
    advantages_by_turn = [[0.0] * n for _ in range(turns)]
    returns_by_turn = [[0.0] * n for _ in range(turns)]
    next_value = [0.0] * n
    next_advantage = [0.0] * n
    for turn_index in range(turns - 1, -1, -1):
        for player in range(n):
            reward = dense_rewards[turn_index][player]
            if turn_index == turns - 1:
                reward += terminal[player]
            value = float(values_history[turn_index][player])
            delta = reward + gamma * next_value[player] - value
            advantage = delta + gamma * gae_lambda * next_advantage[player]
            advantages_by_turn[turn_index][player] = advantage
            returns_by_turn[turn_index][player] = advantage + value
            next_value[player] = value
            next_advantage[player] = advantage

    max_actions = max((decision[1].shape[0] for decision in decisions), default=1)
    states_array = np.zeros((len(decisions), state_feature_size(net.feature_version)), dtype=np.float32)
    actions_array = np.zeros(
        (len(decisions), max_actions, action_feature_size(net.feature_version)), dtype=np.float32
    )
    mask_array = np.zeros((len(decisions), max_actions), dtype=bool)
    chosen_array = np.zeros(len(decisions), dtype=np.int64)
    old_log_probs_array = np.zeros(len(decisions), dtype=np.float32)
    old_values_array = np.zeros(len(decisions), dtype=np.float32)
    returns_array = np.zeros(len(decisions), dtype=np.float32)
    advantages_array = np.zeros(len(decisions), dtype=np.float32)
    player_ids = np.zeros(len(decisions), dtype=np.int64)
    turn_ids = np.zeros(len(decisions), dtype=np.int64)
    for index, (
        state, action_matrix, mask_row, chosen_index, player, turn_index, old_log_prob,
        old_value,
    ) in enumerate(decisions):
        count = action_matrix.shape[0]
        states_array[index] = state
        actions_array[index, :count] = action_matrix
        mask_array[index, :count] = mask_row
        chosen_array[index] = chosen_index
        old_log_probs_array[index] = old_log_prob
        old_values_array[index] = old_value
        player_ids[index] = player
        turn_ids[index] = turn_index
        returns_array[index] = returns_by_turn[turn_index][player]
        advantages_array[index] = advantages_by_turn[turn_index][player]
    learner_players = sorted(set(rl_seats))
    return EpisodeData(
        states=states_array,
        actions=actions_array,
        mask=mask_array,
        chosen=chosen_array,
        old_log_probs=old_log_probs_array,
        old_values=old_values_array,
        returns=returns_array,
        advantages=advantages_array,
        player_ids=player_ids,
        turns=turn_ids,
        terminal_rewards=[terminal[player] for player in learner_players],
        scores=[float(env.scores[player]) for player in learner_players],
        ranks=[int(env.final_ranks()[player]) for player in learner_players],
        player_count=player_count,
    )


def combine_episodes(episodes: Sequence[EpisodeData]) -> EpisodeData:
    action_dim = episodes[0].actions.shape[-1]
    max_actions = max(episode.actions.shape[1] for episode in episodes)
    actions = np.zeros(
        (sum(episode.actions.shape[0] for episode in episodes), max_actions, action_dim),
        dtype=np.float32,
    )
    mask = np.zeros((actions.shape[0], max_actions), dtype=bool)
    offset = 0
    for episode in episodes:
        count = episode.actions.shape[0]
        actions[offset:offset + count, :episode.actions.shape[1]] = episode.actions
        mask[offset:offset + count, :episode.mask.shape[1]] = episode.mask
        offset += count
    return EpisodeData(
        states=np.concatenate([episode.states for episode in episodes], axis=0),
        actions=actions,
        mask=mask,
        chosen=np.concatenate([episode.chosen for episode in episodes], axis=0),
        old_log_probs=np.concatenate([episode.old_log_probs for episode in episodes], axis=0),
        old_values=np.concatenate([episode.old_values for episode in episodes], axis=0),
        returns=np.concatenate([episode.returns for episode in episodes], axis=0),
        advantages=np.concatenate([episode.advantages for episode in episodes], axis=0),
        player_ids=np.concatenate([episode.player_ids for episode in episodes], axis=0),
        turns=np.concatenate([episode.turns for episode in episodes], axis=0),
        terminal_rewards=[value for episode in episodes for value in episode.terminal_rewards],
        scores=[value for episode in episodes for value in episode.scores],
        ranks=[value for episode in episodes for value in episode.ranks],
        player_count=episodes[0].player_count if len({e.player_count for e in episodes}) == 1 else 0,
    )


def _value_loss_and_gradient(
    values: np.ndarray,
    returns: np.ndarray,
    old_values: np.ndarray | None,
    *,
    huber_delta: float | None,
    value_clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample value loss and its gradient with respect to ``values``.

    The loss is the (optionally Huber-robust) squared error.  When
    ``value_clip`` is set a PPO-style clipped target is evaluated as well and
    the larger loss wins.  The clipped branch has zero gradient, so the
    gradient only flows through the unclipped branch where it dominates; this
    is what keeps the value head from chasing a single noisy return.
    """
    error = values - returns
    if huber_delta is not None and huber_delta > 0.0:
        absolute = np.abs(error)
        quadratic = absolute <= huber_delta
        loss = np.where(
            quadratic,
            error * error,
            2.0 * huber_delta * absolute - huber_delta * huber_delta,
        )
        gradient = np.where(quadratic, 2.0 * error, 2.0 * huber_delta * np.sign(error))
    else:
        loss = error * error
        gradient = 2.0 * error
    if value_clip is not None and old_values is not None and value_clip > 0.0:
        clipped = old_values + np.clip(values - old_values, -value_clip, value_clip)
        clipped_error = clipped - returns
        if huber_delta is not None and huber_delta > 0.0:
            absolute = np.abs(clipped_error)
            quadratic = absolute <= huber_delta
            clipped_loss = np.where(
                quadratic,
                clipped_error * clipped_error,
                2.0 * huber_delta * absolute - huber_delta * huber_delta,
            )
        else:
            clipped_loss = clipped_error * clipped_error
        take = loss >= clipped_loss
        loss = np.where(take, loss, clipped_loss)
        gradient = np.where(take, gradient, 0.0)
    return loss, gradient


def rl_update(
    net: RLNetwork,
    episodes: Sequence[EpisodeData],
    optimizer: Adam,
    *,
    batch_size: int = 256,
    ppo_epochs: int = 4,
    clip_ratio: float = 0.2,
    value_coef: float = 0.25,
    entropy_coef: float = 0.02,
    l2: float = 1e-4,
    seed: int = 1,
    value_clip: float | None = None,
    huber_delta: float | None = None,
    target_kl: float | None = None,
) -> dict[str, float]:
    data = combine_episodes(episodes)
    if data.states.shape[0] == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "grad_norm": 0.0,
            "decisions": 0,
            "advantage_std": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
            "value_clip_fraction": 0.0,
            "ppo_epochs_run": 0.0,
        }
    advantages = data.advantages.copy()
    advantage_std = float(np.std(advantages))
    if advantage_std > 1e-6:
        advantages = (advantages - float(np.mean(advantages))) / advantage_std
    rng = np.random.default_rng(seed)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    grad_norms: list[float] = []
    clip_fractions: list[float] = []
    approx_kls: list[float] = []
    value_clip_fractions: list[float] = []
    epochs_run = 0
    early_stopped = False
    for _epoch in range(max(1, ppo_epochs)):
        epochs_run += 1
        order = np.arange(data.states.shape[0])
        rng.shuffle(order)
        epoch_kls: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            states = data.states[indices]
            actions = data.actions[indices]
            mask = data.mask[indices]
            chosen = data.chosen[indices]
            returns = data.returns[indices]
            old_log_probs = data.old_log_probs[indices]
            old_values = data.old_values[indices]
            batch_advantages = advantages[indices]
            scores, values, cache = forward_batch(net, states, actions, mask)
            probabilities = masked_softmax(scores, mask)
            log_probs = np.log(np.maximum(probabilities, 1e-12))
            chosen_log_prob = log_probs[np.arange(len(chosen)), chosen]
            ratio = np.exp(
                np.clip(chosen_log_prob - old_log_probs, -20.0, 20.0)
            ).astype(np.float32)
            approx_kl = float(np.mean(old_log_probs - chosen_log_prob))
            unclipped = ratio * batch_advantages
            clipped = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            policy_loss = -float(np.mean(np.minimum(unclipped, clipped)))
            value_loss_values, value_gradient = _value_loss_and_gradient(
                values,
                returns,
                old_values,
                huber_delta=huber_delta,
                value_clip=value_clip,
            )
            value_loss = float(np.mean(value_loss_values))
            entropy = -np.sum(probabilities * log_probs, axis=1)
            entropy_mean = float(np.mean(entropy))
            count = max(1, len(chosen))
            # PPO surrogate gradient: active only on the unclipped branch.
            active = (unclipped <= clipped).astype(np.float32)
            d_scores = probabilities.copy()
            d_scores[np.arange(count), chosen] -= 1.0
            d_scores *= (
                batch_advantages[:, None] * ratio[:, None] * active[:, None] / count
            )
            if entropy_coef:
                d_scores += (
                    entropy_coef
                    * probabilities
                    * (log_probs + entropy[:, None])
                    / count
                )
            d_values = value_coef * value_gradient / count
            gradients = backward_batch(net, cache, d_scores, d_values, l2=l2)
            grad_norms.append(optimizer.step(gradients))
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy_mean)
            clip_fractions.append(float(np.mean(np.abs(ratio - 1.0) > clip_ratio)))
            approx_kls.append(approx_kl)
            epoch_kls.append(approx_kl)
            if value_clip is not None and value_clip > 0.0:
                value_clip_fractions.append(
                    float(np.mean(np.abs(values - old_values) > value_clip))
                )
        if target_kl is not None and epoch_kls:
            epoch_kl = float(np.mean(epoch_kls))
            if epoch_kl > target_kl:
                early_stopped = True
                break
    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "decisions": int(data.states.shape[0]),
        "advantage_std": advantage_std,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "value_clip_fraction": (
            float(np.mean(value_clip_fractions)) if value_clip_fractions else 0.0
        ),
        "ppo_epochs_run": float(epochs_run),
        "early_stopped": float(1.0 if early_stopped else 0.0),
    }


def sample_player_count(rng: random.Random, progress: float) -> int:
    if progress < 0.12:
        return rng.choice([3, 4, 5])
    if progress < 0.25:
        return rng.choice([3, 4, 5, 6])
    return rng.choice([3, 4, 5, 6, 7])


# ---------------------------------------------------------------------------
# Parallel rollout
#
# The environment is a pure-Python simulator, so the only cheap way to use
# extra Kaggle cores is to play several games at once in worker processes.
# A game is described by a picklable spec; the learner network is shipped per
# task, while the opponent pool is installed in each worker once (and only
# reinstalled when the league snapshots change).


_ROLLOUT_CONTEXT: dict[str, Any] = {}


def _rollout_worker_init(
    opponents: dict[str, Any], gamma: float, gae_lambda: float
) -> None:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _ROLLOUT_CONTEXT.clear()
    _ROLLOUT_CONTEXT.update(opponents=opponents, gamma=gamma, gae_lambda=gae_lambda)


def _rollout_worker_run(
    task: tuple[RLNetwork, list[dict[str, Any]]]
) -> list[EpisodeData]:
    net, specs = task
    opponents = _ROLLOUT_CONTEXT["opponents"]
    gamma = _ROLLOUT_CONTEXT["gamma"]
    gae_lambda = _ROLLOUT_CONTEXT["gae_lambda"]
    episodes: list[EpisodeData] = []
    for spec in specs:
        rng = np.random.default_rng(spec["rng_seed"])
        if spec["opponents"] is not None:
            policies = [
                None if name is None else opponents[name] for name in spec["opponents"]
            ]
            episode = collect_episode(
                net,
                spec["player_count"],
                spec["seed"],
                rng,
                gamma=gamma,
                gae_lambda=gae_lambda,
                opponent_policies=policies,
                rl_seats=spec["rl_seats"],
            )
        else:
            episode = collect_episode(
                net,
                spec["player_count"],
                spec["seed"],
                rng,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
        episodes.append(episode)
    return episodes


def _rollout_specs(
    rng: random.Random,
    progress: float,
    games_per_update: int,
    league_fraction: float,
    active_pool: Sequence[tuple[str, Policy]],
    active_weights: Sequence[float],
) -> tuple[list[dict[str, Any]], int]:
    """Describe this update's games without running them yet."""
    specs: list[dict[str, Any]] = []
    league_games = 0
    for _ in range(games_per_update):
        player_count = sample_player_count(rng, progress)
        seed = rng.randrange(1 << 30)
        rng_seed = rng.randrange(1 << 30)
        if league_fraction > 0 and rng.random() < league_fraction:
            # Put the learner in roughly half of the seats and fill the rest
            # with a mix of baselines and frozen past checkpoints.
            rl_count = max(1, player_count // 2)
            rl_seats = tuple(rng.sample(range(player_count), rl_count))
            names: list[str | None] = [None] * player_count
            for seat in range(player_count):
                if seat in rl_seats:
                    continue
                name, _policy = rng.choices(active_pool, weights=active_weights, k=1)[0]
                names[seat] = name
            specs.append({
                "player_count": player_count,
                "seed": seed,
                "rng_seed": rng_seed,
                "rl_seats": rl_seats,
                "opponents": names,
            })
            league_games += 1
        else:
            specs.append({
                "player_count": player_count,
                "seed": seed,
                "rng_seed": rng_seed,
                "rl_seats": None,
                "opponents": None,
            })
    return specs, league_games


def _run_serial(
    net: RLNetwork,
    specs: Sequence[dict[str, Any]],
    registry: dict[str, Policy],
    gamma: float,
    gae_lambda: float,
) -> list[EpisodeData]:
    episodes: list[EpisodeData] = []
    for spec in specs:
        rng = np.random.default_rng(spec["rng_seed"])
        if spec["opponents"] is not None:
            policies = [
                None if name is None else registry[name] for name in spec["opponents"]
            ]
            episode = collect_episode(
                net,
                spec["player_count"],
                spec["seed"],
                rng,
                gamma=gamma,
                gae_lambda=gae_lambda,
                opponent_policies=policies,
                rl_seats=spec["rl_seats"],
            )
        else:
            episode = collect_episode(
                net,
                spec["player_count"],
                spec["seed"],
                rng,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
        episodes.append(episode)
    return episodes


def _chunk_specs(
    specs: Sequence[dict[str, Any]], workers: int
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for index, spec in enumerate(specs):
        chunks[index % workers].append(spec)
    return [chunk for chunk in chunks if chunk]


def _run_parallel(
    pool: Any,
    net: RLNetwork,
    specs: Sequence[dict[str, Any]],
    workers: int,
) -> list[EpisodeData]:
    chunks = _chunk_specs(specs, workers)
    results = pool.map(_rollout_worker_run, [(net, chunk) for chunk in chunks])
    return [episode for batch in results for episode in batch]


# ---------------------------------------------------------------------------
# Evaluation


def head_to_head_report(
    net: RLNetwork,
    opponent: RLNetwork,
    *,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    games: int = 20,
    seed_base: int = 60000,
) -> dict[str, Any]:
    """Controlled-seat match of ``net`` against another RL checkpoint.

    The candidate takes one seat while the opponent fills every other seat,
    rotating the candidate's seat across games.  ``win_rate`` above ``chance``
    means the candidate is the stronger policy.
    """
    candidate = RLGreedyPolicy(net)
    rival = RLGreedyPolicy(opponent)
    wins: list[float] = []
    scores: list[float] = []
    margins: list[float] = []
    for player_count in player_counts:
        metrics = evaluate_controlled_seat(
            player_count,
            candidate,
            rival,
            games,
            seed_base=seed_base + player_count * 1000,
        )
        wins.append(metrics.win_rate)
        scores.append(metrics.mean_score)
        margins.append(metrics.mean_margin)
    chance = float(np.mean([1.0 / count for count in player_counts])) if player_counts else 0.0
    return {
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "chance": chance,
        "games": games,
    }


def evaluate_rl(
    net: RLNetwork,
    *,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    games: int = 20,
    seed_base: int = 20000,
    supervised_model: PolicyModel | None = None,
    include_combo: bool = True,
    selection_opponent: RLNetwork | None = None,
    head_to_head_games: int = 20,
) -> dict[str, Any]:
    candidate = RLGreedyPolicy(net)
    report: dict[str, Any] = {"players": {}, "summary": {}}
    greedy_win: list[float] = []
    supervised_win: list[float] = []
    combo_win: list[float] = []
    for player_count in player_counts:
        greedy_metrics = evaluate_controlled_seat(
            player_count,
            candidate,
            GreedyPolicy(seed=31),
            games,
            seed_base=seed_base + player_count * 1000,
        )
        entry: dict[str, Any] = {"greedy": greedy_metrics.to_dict()}
        greedy_win.append(greedy_metrics.win_rate)
        if include_combo:
            combo_metrics = evaluate_controlled_seat(
                player_count,
                candidate,
                ComboPolicy(seed=33),
                games,
                seed_base=seed_base + player_count * 1000,
            )
            entry["combo"] = combo_metrics.to_dict()
            combo_win.append(combo_metrics.win_rate)
        if supervised_model is not None:
            supervised_metrics = evaluate_controlled_seat(
                player_count,
                candidate,
                SupervisedPolicy(supervised_model, seed=32),
                games,
                seed_base=seed_base + player_count * 1000,
            )
            entry["supervised"] = supervised_metrics.to_dict()
            supervised_win.append(supervised_metrics.win_rate)
        report["players"][str(player_count)] = entry
    if selection_opponent is not None:
        head_to_head = head_to_head_report(
            net,
            selection_opponent,
            player_counts=player_counts,
            games=head_to_head_games,
            seed_base=seed_base + 700000,
        )
        report["head_to_head"] = head_to_head
        report["summary"]["head_to_head_win_rate"] = head_to_head["win_rate"]
        report["summary"]["head_to_head_chance"] = head_to_head["chance"]
    report["summary"]["greedy_win_rate"] = float(np.mean(greedy_win)) if greedy_win else 0.0
    if combo_win:
        report["summary"]["combo_win_rate"] = float(np.mean(combo_win))
    if supervised_win:
        report["summary"]["supervised_win_rate"] = float(np.mean(supervised_win))
    return report


def _composite_score(report: dict[str, Any]) -> float:
    summary = report.get("summary", {})
    supervised = float(summary.get("supervised_win_rate", -1.0))
    greedy = float(summary.get("greedy_win_rate", 0.0))
    combo = float(summary.get("combo_win_rate", -1.0))
    if supervised >= 0.0 and combo >= 0.0:
        # The user-facing acceptance criterion is beating the archive-trained
        # supervised policy, then the greedy baseline.  Combo skill is a bonus
        # that keeps the policy from collapsing back to myopic greedy play.
        score = 0.45 * supervised + 0.30 * greedy + 0.25 * combo
    elif supervised >= 0.0:
        score = 0.5 * supervised + 0.5 * greedy
    else:
        score = greedy
    # When a reigning champion is supplied, require the candidate to beat it
    # too: add a bonus proportional to the win-rate edge over chance.  This is
    # what makes a checkpoint "top 1" instead of merely "good against bots".
    head_to_head = summary.get("head_to_head_win_rate")
    if head_to_head is not None:
        chance = float(summary.get("head_to_head_chance", 0.0))
        if chance > 0.0:
            score += 0.25 * (float(head_to_head) - chance) / chance
    return score


def rank_checkpoints(
    candidates: Sequence[str | Path],
    *,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    games: int = 60,
    seed_base: int = 123456,
    supervised_model: PolicyModel | None = None,
    selection_opponent: RLNetwork | None = None,
    head_to_head_games: int = 60,
) -> list[dict[str, Any]]:
    """Evaluate several checkpoints on one protocol and rank them.

    This is the offline tournament used to pick the final ``lu_luna`` from a
    training run's archived checkpoints: every candidate plays the same
    controlled-seat games, and the list is sorted by :func:`_composite_score`.
    """
    results: list[dict[str, Any]] = []
    for path in candidates:
        net = load_rl(path)
        report = evaluate_rl(
            net,
            player_counts=player_counts,
            games=games,
            seed_base=seed_base,
            supervised_model=supervised_model,
            selection_opponent=selection_opponent,
            head_to_head_games=head_to_head_games,
        )
        results.append({
            "path": str(path),
            "score": _composite_score(report),
            "summary": report["summary"],
        })
    results.sort(key=lambda entry: -entry["score"])
    return results


# ---------------------------------------------------------------------------
# Training loop


def train_rl(
    *,
    updates: int = 100,
    games_per_update: int = 24,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 1,
    warm_start_games: int = 800,
    bc_epochs: int = 4,
    bc_batch_size: int = 256,
    bc_learning_rate: float = 1e-3,
    learning_rate: float = 3e-4,
    learning_rate_final: float | None = None,
    batch_size: int = 256,
    ppo_epochs: int = 4,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.02,
    entropy_coef_final: float | None = None,
    value_coef: float = 0.25,
    value_clip: float | None = None,
    huber_delta: float | None = None,
    target_kl: float | None = None,
    l2: float = 1e-4,
    league_fraction: float = 0.5,
    snapshot_every: int = 0,
    snapshot_pool_size: int = 6,
    snapshot_weight: float = 0.25,
    selection_opponent: RLNetwork | None = None,
    head_to_head_games: int = 20,
    checkpoint_every: int = 0,
    checkpoint_dir: str | Path | None = None,
    hidden1: int = 128,
    hidden2: int = 64,
    action_hidden: int = 64,
    feature_version: int = LATEST_FEATURE_VERSION,
    workers: int = 1,
    title: str = "8ФМ Classic RL training (self-play + league)",
    eval_every: int = 10,
    eval_games: int = 20,
    eval_seed_base: int = 20000,
    supervised_model: PolicyModel | None = None,
    output: str | Path = "models/classic-rl.json",
    latest_path: str | Path | None = None,
    metrics_path: str | Path = "models/classic-rl-metrics.json",
    plot_path: str | Path = "plots/classic-rl-training.png",
    resume: str | Path | None = None,
    verbose: bool = True,
) -> tuple[RLNetwork, dict[str, Any]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if resume:
        net = load_rl(resume)
        feature_version = net.feature_version
        if verbose:
            print(
                f"Resumed RL model: {resume} "
                f"(features v{net.feature_version}, update {net.metadata.get('best_update', '?')})"
            )
    else:
        net = create_network(
            seed,
            hidden1=hidden1,
            hidden2=hidden2,
            action_hidden=action_hidden,
            feature_version=feature_version,
        )
    output_path = Path(output)
    if latest_path is None:
        latest_path = output_path.with_name(
            output_path.stem + "-latest" + output_path.suffix
        )
    latest_path = Path(latest_path)
    metrics: dict[str, Any] = {
        "format": "eightfm-rl-training-metrics",
        "config": {
            "updates": updates,
            "games_per_update": games_per_update,
            "player_counts": list(player_counts),
            "seed": seed,
            "warm_start_games": warm_start_games,
            "bc_epochs": bc_epochs,
            "teacher": "combo+greedy fallback",
            "learning_rate": learning_rate,
            "learning_rate_final": learning_rate_final,
            "batch_size": batch_size,
            "ppo_epochs": ppo_epochs,
            "clip_ratio": clip_ratio,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "entropy_coef": entropy_coef,
            "entropy_coef_final": entropy_coef_final,
            "value_coef": value_coef,
            "value_clip": value_clip,
            "huber_delta": huber_delta,
            "target_kl": target_kl,
            "league_fraction": league_fraction,
            "snapshot_every": snapshot_every,
            "snapshot_pool_size": snapshot_pool_size,
            "snapshot_weight": snapshot_weight,
            "selection_opponent": None if selection_opponent is None else "rl-checkpoint",
            "head_to_head_games": head_to_head_games,
            "checkpoint_every": checkpoint_every,
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "hidden1": hidden1,
            "hidden2": hidden2,
            "action_hidden": action_hidden,
            "feature_version": feature_version,
            "workers": workers,
            "eval_games": eval_games,
            "eval_seed_base": eval_seed_base,
            "output": str(output_path),
            "latest_path": str(latest_path),
        },
        "warm_start": {},
        "updates": [],
        "evaluations": [],
    }
    if warm_start_games > 0:
        if verbose:
            print(f"Warm start: collecting {warm_start_games} games for behaviour cloning...")
        start = time.time()
        dataset = collect_bc_dataset(
            games=warm_start_games,
            player_counts=player_counts,
            seed=seed + 17,
            label_policy=ComboPolicy(seed=seed + 104),
            feature_version=net.feature_version,
        )
        bc_metrics = behaviour_clone(
            net,
            dataset,
            epochs=bc_epochs,
            batch_size=bc_batch_size,
            learning_rate=bc_learning_rate,
            seed=seed + 19,
        )
        bc_metrics["games"] = float(warm_start_games)
        bc_metrics["states"] = float(dataset.size)
        bc_metrics["seconds"] = round(time.time() - start, 2)
        metrics["warm_start"] = bc_metrics
        if verbose:
            print(
                f"Warm start done: states={dataset.size}, "
                f"accuracy={bc_metrics['bc_accuracy']:.3f}, "
                f"loss={bc_metrics['bc_loss']:.4f}, "
                f"{bc_metrics['seconds']}s"
            )
    optimizer = Adam(net.parameters, learning_rate=learning_rate, max_grad_norm=5.0)
    greedy_opponent = GreedyPolicy(seed=seed + 101)
    random_opponent = RandomPolicy(seed=seed + 102)
    combo_opponent = ComboPolicy(seed=seed + 105)
    opponent_pool: list[tuple[str, Policy]] = [
        ("combo", combo_opponent),
        ("greedy", greedy_opponent),
        ("random", random_opponent),
    ]
    opponent_weights = [0.5, 0.4, 0.1]
    if supervised_model is not None:
        opponent_pool.insert(
            2,
            ("supervised", SupervisedPolicy(supervised_model, seed=seed + 103)),
        )
        opponent_weights = [0.4, 0.35, 0.15, 0.1]
    # Self-play league: frozen copies of earlier updates join the opponent pool
    # so the learner keeps facing strategies it can no longer trivially beat.
    snapshot_pool: list[tuple[str, Policy]] = []
    best_score = float("-inf")
    best_update = 0
    league_games = 0
    start_time = time.time()

    # Parallel rollouts: the simulator is a pure-Python loop, so the cheap CPU
    # win is to run several games at once in worker processes.  The learner
    # net is shipped per task; the (slow-changing) opponent pool is installed
    # once per worker and reinstalled only when the league snapshots change.
    workers = max(1, int(workers))
    pool = None
    pool_signature: tuple[str, ...] | None = None
    if workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
        except Exception:  # pragma: no cover - defensive
            workers = 1
    for update in range(1, updates + 1):
        progress = update / max(1, updates)
        if learning_rate_final is not None:
            optimizer.learning_rate = (
                learning_rate
                + (learning_rate_final - learning_rate) * (update - 1) / max(1, updates - 1)
            )
        current_entropy = entropy_coef
        if entropy_coef_final is not None:
            current_entropy = (
                entropy_coef
                + (entropy_coef_final - entropy_coef) * (update - 1) / max(1, updates - 1)
            )
        active_pool = opponent_pool + snapshot_pool
        active_weights = opponent_weights + [snapshot_weight] * len(snapshot_pool)
        registry = {name: policy for name, policy in active_pool}
        specs, new_league_games = _rollout_specs(
            rng, progress, games_per_update, league_fraction, active_pool, active_weights
        )
        league_games += new_league_games
        episodes: list[EpisodeData]
        if workers > 1:
            signature = tuple(sorted(registry))
            if pool is None or signature != pool_signature:
                if pool is not None:
                    pool.shutdown(wait=True)
                pool = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_rollout_worker_init,
                    initargs=(registry, gamma, gae_lambda),
                )
                pool_signature = signature
            try:
                episodes = _run_parallel(pool, net, specs, workers)
            except Exception as error:  # pragma: no cover - fall back to serial
                if verbose:
                    print(f"  parallel rollout failed ({error}); falling back to one worker")
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except TypeError:  # pragma: no cover - older Python
                    pool.shutdown(wait=False)
                pool = None
                pool_signature = None
                workers = 1
                episodes = _run_serial(net, specs, registry, gamma, gae_lambda)
        else:
            episodes = _run_serial(net, specs, registry, gamma, gae_lambda)
        train_metrics = rl_update(
            net,
            episodes,
            optimizer,
            batch_size=batch_size,
            ppo_epochs=ppo_epochs,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=current_entropy,
            l2=l2,
            seed=seed + update,
            value_clip=value_clip,
            huber_delta=huber_delta,
            target_kl=target_kl,
        )
        if snapshot_every > 0 and update % snapshot_every == 0:
            snapshot_pool.append(
                (f"snapshot-{update}", RLGreedyPolicy(clone_network(net)))
            )
            if len(snapshot_pool) > snapshot_pool_size:
                del snapshot_pool[0]
        terminal = [value for episode in episodes for value in episode.terminal_rewards]
        scores = [value for episode in episodes for value in episode.scores]
        ranks = [value for episode in episodes for value in episode.ranks]
        decision_returns = [
            float(np.mean(episode.returns))
            for episode in episodes
            if episode.returns.size
        ]
        win_rate = sum(1 for rank in ranks if rank == 1) / max(1, len(ranks))
        success_rate = sum(1 for score in scores if score > 60.0) / max(1, len(scores))
        entry = {
            "update": update,
            "games": update * games_per_update,
            "league_games": league_games,
            "learning_rate": optimizer.learning_rate,
            "entropy_coef": current_entropy,
            "mean_terminal_reward": float(np.mean(terminal)) if terminal else 0.0,
            "mean_return": float(np.mean(decision_returns)) if decision_returns else 0.0,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "win_rate": win_rate,
            "success_rate": success_rate,
            **train_metrics,
        }
        metrics["updates"].append(entry)
        if verbose:
            print(
                f"update {update:4d}/{updates} | games {entry['games']:5d} | "
                f"league {league_games:4d} | "
                f"reward {entry['mean_terminal_reward']:.3f} | "
                f"ret {entry['mean_return']:+.3f} | "
                f"win {win_rate:.3f} | score {entry['mean_score']:.1f} | "
                f"loss {entry['policy_loss']:.4f} | "
                f"vloss {entry['value_loss']:.4f} | "
                f"ent {entry['entropy']:.3f} | "
                f"kl {entry['approx_kl']:+.4f}"
            )
        if checkpoint_every > 0 and checkpoint_dir is not None and update % checkpoint_every == 0:
            archive_dir = Path(checkpoint_dir)
            archive_dir.mkdir(parents=True, exist_ok=True)
            save_rl(net, archive_dir / f"update-{update:04d}.json")
        if update % eval_every == 0 or update == updates:
            eval_report = evaluate_rl(
                net,
                player_counts=player_counts,
                games=eval_games,
                seed_base=eval_seed_base,
                supervised_model=supervised_model,
                selection_opponent=selection_opponent,
                head_to_head_games=head_to_head_games,
            )
            eval_report["update"] = update
            metrics["evaluations"].append(eval_report)
            score = _composite_score(eval_report)
            if verbose:
                summary = eval_report["summary"]
                print(
                    f"  eval: greedy win {summary.get('greedy_win_rate', 0):.3f}"
                    + (
                        f", combo win {summary.get('combo_win_rate', 0):.3f}"
                        if "combo_win_rate" in summary
                        else ""
                    )
                    + (
                        f", supervised win {summary.get('supervised_win_rate', 0):.3f}"
                        if "supervised_win_rate" in summary
                        else ""
                    )
                    + (
                        f", h2h win {summary.get('head_to_head_win_rate', 0):.3f}"
                        f" (chance {summary.get('head_to_head_chance', 0):.3f})"
                        if "head_to_head_win_rate" in summary
                        else ""
                    )
                    + f" | score {score:.3f}"
                )
            if score > best_score:
                best_score = score
                best_update = update
                net.metadata["best_update"] = update
                net.metadata["best_eval"] = eval_report
                save_rl(net, output_path)
                if verbose:
                    print(f"  saved best checkpoint: {output_path}")
            save_rl(net, latest_path)
        elif update % max(1, eval_every // 2) == 0:
            save_rl(net, latest_path)
    if pool is not None:
        pool.shutdown(wait=True)
        pool = None
    metrics["best_update"] = best_update
    metrics["best_score"] = best_score
    metrics["seconds"] = round(time.time() - start_time, 2)
    training_metadata = {
        "updates": updates,
        "games_per_update": games_per_update,
        "best_update": best_update,
        "best_score": best_score,
        "latest_model": str(latest_path),
    }
    net.metadata["training_metrics"] = training_metadata
    save_rl(net, latest_path)
    if best_update:
        try:
            best_net = load_rl(output_path)
        except (OSError, ValueError):
            best_net = net
        best_net.metadata["training_metrics"] = training_metadata
        save_rl(best_net, output_path)
    else:
        best_net = net
        save_rl(best_net, output_path)
    _write_metrics(metrics, metrics_path)
    _plot_training(metrics, plot_path, title=title)
    return best_net, metrics


def _write_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return [float(value) for value in values]
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += float(value)
        if index >= window:
            running -= float(values[index - window])
        result.append(running / min(index + 1, window))
    return result


def _plot_training(
    metrics: dict[str, Any],
    path: str | Path | None,
    *,
    title: str = "8ФМ Classic RL training (self-play + league)",
) -> None:
    if path is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    updates = metrics.get("updates", [])
    evaluations = metrics.get("evaluations", [])
    if not updates:
        return
    xs = [entry["update"] for entry in updates]
    window = max(3, min(15, len(updates) // 20 or 3))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.suptitle(title, fontsize=14)

    # Loss: raw lines stay faint, moving averages show the trend.
    policy_loss = [entry["policy_loss"] for entry in updates]
    value_loss = [entry["value_loss"] for entry in updates]
    axes[0, 0].plot(xs, policy_loss, color="tab:blue", alpha=0.25)
    axes[0, 0].plot(xs, _moving_average(policy_loss, window), color="tab:blue", linewidth=2.2, label="policy loss")
    axes[0, 0].plot(xs, value_loss, color="tab:orange", alpha=0.25)
    axes[0, 0].plot(xs, _moving_average(value_loss, window), color="tab:orange", linewidth=2.2, label="value loss")
    axes[0, 0].set_title("Loss (raw + moving average)")
    axes[0, 0].set_xlabel("update")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    # Reward: terminal reward and GAE return on the left, score on the right.
    terminal = [entry["mean_terminal_reward"] for entry in updates]
    mean_return = [entry["mean_return"] for entry in updates]
    mean_score = [entry["mean_score"] for entry in updates]
    axes[0, 1].plot(xs, terminal, color="tab:green", alpha=0.25)
    axes[0, 1].plot(xs, _moving_average(terminal, window), color="tab:green", linewidth=2.2, label="terminal reward")
    axes[0, 1].plot(xs, mean_return, color="tab:red", alpha=0.25)
    axes[0, 1].plot(xs, _moving_average(mean_return, window), color="tab:red", linewidth=2.2, label="GAE return")
    axes[0, 1].set_title("Reward growth")
    axes[0, 1].set_xlabel("update")
    axes[0, 1].set_ylabel("reward")
    score_axis = axes[0, 1].twinx()
    score_axis.plot(xs, _moving_average(mean_score, window), color="tab:orange", linewidth=2.0, linestyle="--", label="mean score (right)")
    score_axis.set_ylabel("score")
    lines, labels = axes[0, 1].get_legend_handles_labels()
    lines2, labels2 = score_axis.get_legend_handles_labels()
    axes[0, 1].legend(lines + lines2, labels + labels2, loc="best")
    axes[0, 1].grid(alpha=0.25)

    entropy = [entry["entropy"] for entry in updates]
    win_rate = [entry["win_rate"] for entry in updates]
    success_rate = [entry["success_rate"] for entry in updates]
    axes[1, 0].plot(xs, _moving_average(entropy, window), color="tab:purple", linewidth=2.0, label="entropy")
    axes[1, 0].plot(xs, _moving_average(win_rate, window), color="tab:blue", linewidth=2.0, label="training win rate")
    axes[1, 0].plot(xs, _moving_average(success_rate, window), color="tab:red", linewidth=2.0, label=">60 rate")
    if any("approx_kl" in entry for entry in updates):
        approx_kl = [entry.get("approx_kl", 0.0) for entry in updates]
        axes[1, 0].plot(
            xs,
            _moving_average(approx_kl, window),
            color="tab:gray",
            linewidth=1.8,
            linestyle="--",
            label="approx KL",
        )
    axes[1, 0].set_title("Exploration and training outcomes")
    axes[1, 0].set_xlabel("update")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    if evaluations:
        eval_x = [entry["update"] for entry in evaluations]
        axes[1, 1].plot(
            eval_x,
            [entry["summary"].get("greedy_win_rate", 0) for entry in evaluations],
            marker="o",
            linewidth=2.0,
            label="vs greedy",
        )
        if any("combo_win_rate" in entry["summary"] for entry in evaluations):
            axes[1, 1].plot(
                eval_x,
                [entry["summary"].get("combo_win_rate", float("nan")) for entry in evaluations],
                marker="^",
                linewidth=2.0,
                label="vs combo",
            )
        if any("supervised_win_rate" in entry["summary"] for entry in evaluations):
            axes[1, 1].plot(
                eval_x,
                [entry["summary"].get("supervised_win_rate", float("nan")) for entry in evaluations],
                marker="s",
                linewidth=2.0,
                label="vs supervised",
            )
        player_counts = metrics.get("config", {}).get("player_counts", [3, 4, 5, 6, 7])
        chance = sum(1.0 / count for count in player_counts) / max(1, len(player_counts))
        axes[1, 1].axhline(
            chance,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label="chance (average 1/N)",
        )
        axes[1, 1].set_title("Controlled-seat win rate")
        axes[1, 1].set_xlabel("update")
        axes[1, 1].set_ylim(0.0, 1.0)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.25)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    figure.savefig(destination.with_suffix(".svg"))
    plt.close(figure)


# ---------------------------------------------------------------------------
# lu_luna: the strong configuration


LU_LUNA_OUTPUT = "models/lu_luna.json"
LU_LUNA_LATEST = "models/lu_luna-latest.json"
LU_LUNA_METRICS = "models/lu_luna-metrics.json"
LU_LUNA_PLOT = "plots/lu_luna-training.png"


def train_lu_luna(
    *,
    updates: int = 320,
    games_per_update: int = 32,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 2024,
    warm_start_games: int = 2000,
    bc_epochs: int = 8,
    learning_rate: float = 3e-4,
    learning_rate_final: float = 4e-5,
    batch_size: int = 512,
    ppo_epochs: int = 6,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.03,
    entropy_coef_final: float = 0.004,
    value_coef: float = 0.5,
    value_clip: float = 0.2,
    huber_delta: float = 1.0,
    target_kl: float = 0.03,
    league_fraction: float = 0.6,
    snapshot_every: int = 10,
    snapshot_pool_size: int = 8,
    snapshot_weight: float = 0.3,
    selection_opponent_path: str | Path | None = "models/classic-rl.json",
    head_to_head_games: int = 20,
    checkpoint_every: int = 20,
    checkpoint_dir: str | Path | None = "models/lu_luna-checkpoints",
    hidden1: int = 256,
    hidden2: int = 128,
    action_hidden: int = 128,
    feature_version: int = LATEST_FEATURE_VERSION,
    workers: int = 1,
    eval_every: int = 10,
    eval_games: int = 30,
    eval_seed_base: int = 50000,
    supervised_model: PolicyModel | None = None,
    output: str | Path = LU_LUNA_OUTPUT,
    latest_path: str | Path | None = LU_LUNA_LATEST,
    metrics_path: str | Path = LU_LUNA_METRICS,
    plot_path: str | Path = LU_LUNA_PLOT,
    resume: str | Path | None = None,
    verbose: bool = True,
) -> tuple[RLNetwork, dict[str, Any]]:
    """Train ``lu_luna``: a wider network with the strongest PPO recipe.

    Compared with the default :func:`train_rl` run, ``lu_luna`` adds:

    * a wider actor-critic (256/128 towers, 128-wide action tower);
    * a longer behaviour-cloning warm start and many more updates;
    * PPO with value clipping and a Huber value loss for noisy returns;
    * an approximate-KL early stop so a single update cannot wreck the policy;
    * linear learning-rate and entropy schedules that anneal over training;
    * a self-play league of frozen past checkpoints beside the baselines;
    * a 3 -> 7 player curriculum and a larger evaluation each cycle;
    * best-checkpoint selection that also has to beat the reigning champion
      (``models/classic-rl.json``) head-to-head, not just the bots;
    * archived checkpoints every 20 updates for later inspection.

    All defaults target the ``models/lu_luna*.json`` artefacts; pass explicit
    arguments to override any of them.
    """
    selection_opponent: RLNetwork | None = None
    if selection_opponent_path and Path(selection_opponent_path).exists():
        selection_opponent = load_rl(selection_opponent_path)
    return train_rl(
        updates=updates,
        games_per_update=games_per_update,
        player_counts=player_counts,
        seed=seed,
        warm_start_games=warm_start_games,
        bc_epochs=bc_epochs,
        learning_rate=learning_rate,
        learning_rate_final=learning_rate_final,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        entropy_coef_final=entropy_coef_final,
        value_coef=value_coef,
        value_clip=value_clip,
        huber_delta=huber_delta,
        target_kl=target_kl,
        league_fraction=league_fraction,
        snapshot_every=snapshot_every,
        snapshot_pool_size=snapshot_pool_size,
        snapshot_weight=snapshot_weight,
        selection_opponent=selection_opponent,
        head_to_head_games=head_to_head_games,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        hidden1=hidden1,
        hidden2=hidden2,
        action_hidden=action_hidden,
        feature_version=feature_version,
        workers=workers,
        title="8ФМ Classic RL training — lu_luna",
        eval_every=eval_every,
        eval_games=eval_games,
        eval_seed_base=eval_seed_base,
        supervised_model=supervised_model,
        output=output,
        latest_path=latest_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
        resume=resume,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# The "absolute" configuration: the maximal Classic policy


ABSOLUTE_OUTPUT = "models/lu_luna_absolute.json"
ABSOLUTE_LATEST = "models/lu_luna_absolute-latest.json"
ABSOLUTE_METRICS = "models/lu_luna_absolute-metrics.json"
ABSOLUTE_PLOT = "plots/absolute-training.png"

# Learning-rate and entropy endpoints for the full absolute schedule.  They
# are module constants so external drivers (for example the Kaggle harness)
# can re-derive the value at any global step and keep the anneal monotone
# across resumed sessions.
ABSOLUTE_LEARNING_RATE = 3e-4
ABSOLUTE_LEARNING_RATE_FINAL = 2e-5
ABSOLUTE_ENTROPY = 0.035
ABSOLUTE_ENTROPY_FINAL = 0.003


def train_absolute(
    *,
    updates: int = 3000,
    games_per_update: int = 48,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 8_2026,
    warm_start_games: int = 6000,
    bc_epochs: int = 16,
    learning_rate: float = ABSOLUTE_LEARNING_RATE,
    learning_rate_final: float | None = ABSOLUTE_LEARNING_RATE_FINAL,
    batch_size: int = 1024,
    ppo_epochs: int = 8,
    clip_ratio: float = 0.2,
    entropy_coef: float = ABSOLUTE_ENTROPY,
    entropy_coef_final: float | None = ABSOLUTE_ENTROPY_FINAL,
    value_coef: float = 0.5,
    value_clip: float = 0.2,
    huber_delta: float = 1.0,
    target_kl: float = 0.03,
    league_fraction: float = 0.65,
    snapshot_every: int = 25,
    snapshot_pool_size: int = 12,
    snapshot_weight: float = 0.3,
    selection_opponent_path: str | Path | None = "models/classic-rl.json",
    head_to_head_games: int = 40,
    checkpoint_every: int = 25,
    checkpoint_dir: str | Path | None = "models/lu_luna_absolute-checkpoints",
    hidden1: int = 512,
    hidden2: int = 256,
    action_hidden: int = 256,
    feature_version: int = LATEST_FEATURE_VERSION,
    workers: int = 1,
    eval_every: int = 25,
    eval_games: int = 60,
    eval_seed_base: int = 50000,
    supervised_model: PolicyModel | None = None,
    output: str | Path = ABSOLUTE_OUTPUT,
    latest_path: str | Path | None = ABSOLUTE_LATEST,
    metrics_path: str | Path = ABSOLUTE_METRICS,
    plot_path: str | Path | None = ABSOLUTE_PLOT,
    resume: str | Path | None = None,
    verbose: bool = True,
) -> tuple[RLNetwork, dict[str, Any]]:
    """Train the "absolute" Classic policy: the largest recipe in the project.

    Where :func:`train_lu_luna` is the strong-single-machine configuration,
    ``train_absolute`` is meant to be run on external compute (Kaggle) and
    turns every knob up:

    * feature layout ``v2`` -- combo progress, closed-card economics, turn and
      board pressure, Alexey pressure and success-threshold signals on top of
      the v1 state (158 state / 32 action inputs instead of 115 / 25);
    * a much wider actor-critic (512/256 state towers, 256-wide action tower);
    * a long behaviour-cloning warm start and thousands of PPO updates;
    * a bigger batch, more PPO epochs, a larger self-play league and a broader
      evaluation than lu_luna.

    All of these are *defaults*: every argument can be overridden, which is how
    the resumable Kaggle driver runs the same recipe in time-budgeted chunks
    while keeping one monotone learning-rate / entropy schedule.
    """
    selection_opponent: RLNetwork | None = None
    if selection_opponent_path and Path(selection_opponent_path).exists():
        selection_opponent = load_rl(selection_opponent_path)
    return train_rl(
        updates=updates,
        games_per_update=games_per_update,
        player_counts=player_counts,
        seed=seed,
        warm_start_games=warm_start_games,
        bc_epochs=bc_epochs,
        learning_rate=learning_rate,
        learning_rate_final=learning_rate_final,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        entropy_coef_final=entropy_coef_final,
        value_coef=value_coef,
        value_clip=value_clip,
        huber_delta=huber_delta,
        target_kl=target_kl,
        league_fraction=league_fraction,
        snapshot_every=snapshot_every,
        snapshot_pool_size=snapshot_pool_size,
        snapshot_weight=snapshot_weight,
        selection_opponent=selection_opponent,
        head_to_head_games=head_to_head_games,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        hidden1=hidden1,
        hidden2=hidden2,
        action_hidden=action_hidden,
        feature_version=feature_version,
        workers=workers,
        title="8ФМ Classic RL training — absolute",
        eval_every=eval_every,
        eval_games=eval_games,
        eval_seed_base=eval_seed_base,
        supervised_model=supervised_model,
        output=output,
        latest_path=latest_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
        resume=resume,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Live message inference


def observation_from_message(
    message: str,
    player_name: str,
) -> tuple[Observation, list[str]]:
    """Reconstruct an RL observation from a pasted Telegram message.

    The message lists the current hand and the public table.  Closed cards keep
    their hidden identity, exactly as the RL state encoder expects.
    """
    stripped = _strip_played_card(message)
    candidates = _parse_candidates(stripped)
    round_number, turn_number, rows, resolved = _parse_state(stripped, player_name)
    if not resolved:
        raise ValueError(f"Игрок '{player_name}' не найден в сообщении")
    player_count = len(rows)
    if player_count not in DECK_COUNTS:
        raise ValueError(f"Неподдерживаемое число игроков: {player_count}")
    player_index = next(index for index, row in enumerate(rows) if row["name"] == resolved)
    warnings: list[str] = []
    hand: list[str] = []
    for candidate in candidates:
        if candidate.card_id:
            hand.append(candidate.card_id)
        else:
            warnings.append(f"Не распознана карта: {candidate.name}")
    if not hand:
        raise ValueError("В сообщении нет распознанных карт")

    def parse_plays(row: dict[str, Any]) -> tuple[Play, ...]:
        # The message keeps the whole game history in each row, and the
        # simulator does the same: at round 2 the round-1 cards are still on
        # the table and still contribute to the cumulative score.  Keep every
        # visible icon, but warn when the history length is not what the
        # round/turn header promises.
        expected = (int(round_number or 1) - 1) * 5 + int(turn_number or 1) - 1
        tokens = row["tokens"]
        if len(tokens) != expected:
            warnings.append(
                f"В строке {row['name']} {len(tokens)} иконок вместо ожидаемых {expected}"
            )
        plays: list[Play] = []
        for token in tokens:
            card_id = EMOJI_CARDS.get(token)
            if card_id == "__closed__":
                plays.append(Play(None, True))
            elif card_id:
                plays.append(Play(card_id, False))
            else:
                warnings.append(f"Неизвестная иконка пропущена: {token}")
        return tuple(plays)

    all_plays = tuple(parse_plays(row) for row in rows)
    own_plays = all_plays[player_index]
    closed_total = sum(1 for plays in all_plays for _, is_closed in plays if is_closed)
    alexey_players = frozenset(
        index for index, plays in enumerate(all_plays)
        if any(not is_closed and card_id == "alaksiej" for card_id, is_closed in plays)
    )
    round_value = int(round_number or 1)
    offset = -1 if round_value == 1 else 1
    legal = tuple(legal_actions(hand, own_plays))
    if not legal:
        raise ValueError("Нет легальных действий для восстановленной руки")
    observation = Observation(
        player=player_index,
        player_count=player_count,
        round_number=round_value,
        turn_number=int(turn_number or 1),
        hand=tuple(hand),
        own_plays=own_plays,
        all_plays=all_plays,
        scores=tuple(float(row["score"]) for row in rows),
        deck_counts=tuple(
            (card_id, DECK_COUNTS[player_count].get(card_id, 0)) for card_id in CARD_IDS
        ),
        closed_total=closed_total,
        alexey_players=alexey_players,
        legal=legal,
        pass_receiver=(player_index + offset) % player_count,
        pass_supplier=(player_index - offset) % player_count,
    )
    return observation, sorted(set(warnings))


def rank_message_actions(
    net: RLNetwork,
    message: str,
    player_name: str,
) -> dict[str, Any]:
    observation, warnings = observation_from_message(message, player_name)
    states, actions, mask, legal_lists = _pad_observations(
        [observation], net.feature_version
    )
    scores, values, _ = forward_batch(net, states, actions, mask)
    probabilities = masked_softmax(scores, mask)[0]
    own_has_alexey = has_open_alexey(observation.own_plays)
    ranked: list[dict[str, Any]] = []
    for index, action in enumerate(legal_lists[0]):
        delta = score_action_delta(
            observation.own_plays,
            action,
            own_has_alexey=own_has_alexey,
            players_with_alexey=len(observation.alexey_players),
        )
        ranked.append({
            "card_id": action.card_id,
            "is_closed": action.is_closed,
            "play": "закрыто" if action.is_closed else "открыто",
            "policy_score": float(scores[0, index]),
            "probability": float(probabilities[index]),
            "immediate_delta": float(delta),
        })
    ranked.sort(key=lambda item: -item["policy_score"])
    return {
        "round_number": observation.round_number,
        "turn_number": observation.turn_number,
        "player_name": player_name,
        "value": float(values[0]),
        "candidates": ranked,
        "warnings": warnings,
    }
