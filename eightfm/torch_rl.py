"""GPU/vectorised training for the ``lu_sol`` 8ФМ Classic policy.

The NumPy trainer (:mod:`eightfm.rl_train`) spends almost all of its time in
the pure-Python simulator.  This module replaces both the simulator and the
network with PyTorch, so thousands of games are played in lockstep on a GPU
and PPO runs with autograd.  The weights are exported to the same JSON format
the NumPy app loads, so nothing about deployment changes.

Only ``lu_sol`` uses this path; the reference NumPy implementation stays in
place for the desktop/Telegram apps and for evaluation.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import ComboPolicy
from .env import LATEST_FEATURE_VERSION, action_feature_size, state_feature_size
from .model import CARD_IDS
from .rl import RLNetwork, load_rl, save_rl
from .rl_train import (
    _composite_score,
    _goal_composite_score,
    evaluate_rl,
    sample_player_count,
)
from .torch_env import (
    ACTION_SIZE,
    ALEXEY,
    AMOGUS,
    CARD_INDEX,
    BatchedClassicEnv,
    BatchedObservation,
)


# Reserved league id for the reigning champion (``lu_sol`` by default).  It is
# deliberately far above the snapshot ids (4, 5, ...) so it can never clash.
CHAMPION_OPPONENT_ID = 1000


# ---------------------------------------------------------------------------
# Network


class TorchActorCritic(torch.nn.Module):
    """Same architecture as :class:`eightfm.rl.RLNetwork`, in PyTorch.

    Parameters keep the exact NumPy shapes (``W1`` is ``(state, hidden1)`` and
    so on), so :func:`to_rl_network` is a straight copy and the exported JSON
    is bit-for-bit loadable by the NumPy app.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden1: int = 512,
        hidden2: int = 256,
        action_hidden: int = 256,
        feature_version: int = LATEST_FEATURE_VERSION,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.action_hidden = int(action_hidden)
        self.feature_version = int(feature_version)

        self.W1 = torch.nn.Parameter(torch.empty(state_dim, hidden1))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden1))
        self.W2 = torch.nn.Parameter(torch.empty(hidden1, hidden2))
        self.b2 = torch.nn.Parameter(torch.zeros(hidden2))
        self.Wa = torch.nn.Parameter(torch.empty(action_dim, action_hidden))
        self.ba = torch.nn.Parameter(torch.zeros(action_hidden))
        self.Wp = torch.nn.Parameter(torch.empty(hidden2, action_hidden))
        self.wv = torch.nn.Parameter(torch.zeros(hidden2, 1))
        self.bp = torch.nn.Parameter(torch.zeros(1))
        self.bv = torch.nn.Parameter(torch.zeros(1))
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int = 1) -> None:
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.W1.normal_(0.0, math.sqrt(2.0 / self.state_dim), generator=generator)
            self.W2.normal_(0.0, math.sqrt(2.0 / self.hidden1), generator=generator)
            self.Wa.normal_(0.0, math.sqrt(2.0 / self.action_dim), generator=generator)
            self.Wp.normal_(0.0, math.sqrt(1.0 / self.hidden2), generator=generator)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = states @ self.W1 + self.b1
        h1 = F.relu(z1)
        z2 = h1 @ self.W2 + self.b2
        hs = F.relu(z2)
        values = (hs @ self.wv + self.bv).squeeze(-1)
        za = actions @ self.Wa + self.ba
        ha = F.relu(za)
        u = hs @ self.Wp
        scores = (u.unsqueeze(-2) * ha).sum(dim=-1) + self.bp
        return scores, values

    # -- conversion --------------------------------------------------------

    @classmethod
    def from_rl_network(
        cls, network: RLNetwork, *, seed: int = 1
    ) -> "TorchActorCritic":
        model = cls(
            network.W1.shape[0],
            network.Wa.shape[0],
            hidden1=network.W1.shape[1],
            hidden2=network.W2.shape[1],
            action_hidden=network.Wa.shape[1],
            feature_version=network.feature_version,
            seed=seed,
        )
        with torch.no_grad():
            model.W1.copy_(torch.from_numpy(network.W1))
            model.b1.copy_(torch.from_numpy(network.b1))
            model.W2.copy_(torch.from_numpy(network.W2))
            model.b2.copy_(torch.from_numpy(network.b2))
            model.Wa.copy_(torch.from_numpy(network.Wa))
            model.ba.copy_(torch.from_numpy(network.ba))
            model.Wp.copy_(torch.from_numpy(network.Wp))
            model.wv.copy_(torch.from_numpy(network.wv))
            model.bp.copy_(torch.from_numpy(network.bp))
            model.bv.copy_(torch.from_numpy(network.bv))
        return model


def to_rl_network(
    model: TorchActorCritic, *, metadata: dict[str, Any] | None = None
) -> RLNetwork:
    parameters = {name: value.detach().cpu().numpy() for name, value in model.named_parameters()}
    return RLNetwork(
        W1=parameters["W1"],
        b1=parameters["b1"],
        W2=parameters["W2"],
        b2=parameters["b2"],
        Wa=parameters["Wa"],
        ba=parameters["ba"],
        Wp=parameters["Wp"],
        bp=parameters["bp"],
        wv=parameters["wv"],
        bv=parameters["bv"],
        feature_version=model.feature_version,
        metadata=dict(metadata or {}),
    )


def resolve_device(device: str | None) -> torch.device:
    if device in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def masked_softmax_torch(
    scores: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    scaled = scores / max(1e-6, float(temperature))
    masked = scaled.masked_fill(~mask, -1e9)
    maximum = masked.max(dim=-1, keepdim=True).values
    exp = torch.exp(masked - maximum) * mask
    return exp / exp.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def _sample_actions(
    probabilities: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    chosen = torch.multinomial(flat, 1, generator=generator)
    return chosen.reshape(probabilities.shape[:-1])


# ---------------------------------------------------------------------------
# Batched opponent policies


class BatchedGreedyPolicy:
    """Vectorised :class:`eightfm.baselines.GreedyPolicy`."""

    def act(
        self, observation: BatchedObservation, generator: torch.Generator
    ) -> torch.Tensor:
        scores = observation.delta.masked_fill(~observation.legal, -1e9)
        noise = torch.rand(
            scores.shape,
            device=scores.device,
            generator=generator,
            dtype=scores.dtype,
        )
        return (scores + noise * 1e-9).argmax(dim=-1)


class BatchedComboPolicy:
    """Vectorised :class:`eightfm.baselines.ComboPolicy` with a greedy fallback."""

    COMBOS = ComboPolicy.COMBOS

    def __init__(self) -> None:
        self.greedy = BatchedGreedyPolicy()

    def act(
        self, observation: BatchedObservation, generator: torch.Generator
    ) -> torch.Tensor:
        device = observation.legal.device
        dtype = observation.delta.dtype
        batch, players = observation.open_counts.shape[:2]
        best_score = torch.full((batch, players), float("-inf"), device=device, dtype=dtype)
        best_card = torch.zeros((batch, players), dtype=torch.long, device=device)
        for card_id, threshold, weight in self.COMBOS:
            index = CARD_INDEX[card_id]
            own = observation.open_counts[..., index]
            potential = own + observation.hand[..., index] + observation.unseen[..., index]
            valid = potential >= threshold
            if card_id == "amogus":
                valid = valid & (own < 2)
            score = weight * (potential - threshold + 1) + 0.1 * own
            score = torch.where(valid, score, torch.full_like(score, -1e9))
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            best_card = torch.where(better, torch.full_like(best_card, index), best_card)
        target_open = best_card * 2
        has_target = best_score > -1e8
        target_legal = observation.legal.gather(-1, target_open.unsqueeze(-1)).squeeze(-1)
        use_target = has_target & target_legal
        greedy = self.greedy.act(observation, generator)
        return torch.where(use_target, target_open, greedy)


class BatchedRandomPolicy:
    def act(
        self, observation: BatchedObservation, generator: torch.Generator
    ) -> torch.Tensor:
        probabilities = observation.legal.to(observation.delta.dtype)
        return _sample_actions(probabilities, generator)


class BatchedNetPolicy:
    """A frozen torch checkpoint used as a league opponent (greedy by default).

    Feature layouts are prefix-nested (v1 ⊂ v2 ⊂ v3 ⊂ v4), so a checkpoint
    trained on an older layout can consume the first ``N`` columns of a richer
    observation.  That is what lets the v2 champion ``lu_sol`` play in the
    league of a v4 learner instead of being restricted to the selection eval.
    """

    def __init__(self, model: TorchActorCritic, *, greedy: bool = True) -> None:
        self.model = model
        self.greedy = greedy
        self.state_dim = state_feature_size(model.feature_version)
        self.action_dim = action_feature_size(model.feature_version)

    def _slice(
        self, observation: BatchedObservation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = observation.state_features
        actions = observation.action_features
        if state.shape[-1] < self.state_dim or actions.shape[-1] < self.action_dim:
            raise ValueError(
                "Opponent feature layout is richer than the learner's; only "
                "prefix-nested layouts (v1 ⊂ v2 ⊂ v3 ⊂ v4) can be mixed."
            )
        if state.shape[-1] > self.state_dim:
            state = state[..., : self.state_dim]
        if actions.shape[-1] > self.action_dim:
            actions = actions[..., : self.action_dim]
        return state, actions

    def act(
        self, observation: BatchedObservation, generator: torch.Generator
    ) -> torch.Tensor:
        state, actions = self._slice(observation)
        with torch.no_grad():
            scores, _values = self.model(state, actions)
        if self.greedy:
            return scores.masked_fill(~observation.legal, -1e9).argmax(dim=-1)
        probabilities = masked_softmax_torch(scores, observation.legal)
        return _sample_actions(probabilities, generator)


# ---------------------------------------------------------------------------
# Rollouts


@dataclass
class TorchRollout:
    states: torch.Tensor
    actions: torch.Tensor
    mask: torch.Tensor
    chosen: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    player_ids: torch.Tensor
    turn_ids: torch.Tensor
    terminal_rewards: torch.Tensor
    scores: torch.Tensor
    ranks: torch.Tensor
    player_count: int
    num_games: int

    @property
    def decisions(self) -> int:
        return int(self.states.shape[0])


def collect_torch_rollout(
    env: BatchedClassicEnv,
    model: TorchActorCritic,
    seat_types: torch.Tensor,
    opponents: dict[int, Any],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    feature_version: int | None = None,
    success_threshold: float = 60.0,
    win_weight: float = 1.5,
    placement_weight: float = 0.7,
    score_weight: float = 0.4,
    success_weight: float = 0.4,
    margin_weight: float = 0.2,
) -> TorchRollout:
    """Play one batch of games and collect learner decisions with GAE returns."""
    device = env.device
    if feature_version is None:
        feature_version = model.feature_version
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
    seat_types = seat_types.to(device=device, dtype=torch.long)
    batch, players = seat_types.shape
    learner_mask = seat_types == 0
    if not bool(learner_mask.any()):
        raise ValueError("At least one learner seat is required")

    env.reset()
    decision_states: list[torch.Tensor] = []
    decision_actions: list[torch.Tensor] = []
    decision_mask: list[torch.Tensor] = []
    decision_chosen: list[torch.Tensor] = []
    decision_log_probs: list[torch.Tensor] = []
    decision_values: list[torch.Tensor] = []
    decision_turn: list[torch.Tensor] = []
    decision_game: list[torch.Tensor] = []
    decision_player: list[torch.Tensor] = []
    dense_rewards: list[torch.Tensor] = []
    values_history: list[torch.Tensor] = []

    opponent_types = [int(value) for value in torch.unique(seat_types).tolist() if int(value) != 0]

    for turn in range(10):
        observation = env.observation(feature_version)
        with torch.no_grad():
            scores, values = model(observation.state_features, observation.action_features)
            probabilities = masked_softmax_torch(scores, observation.legal, temperature)
        learner_actions = _sample_actions(probabilities, generator)
        learner_log_probs = torch.log(
            probabilities.gather(-1, learner_actions.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)
        )
        actions = learner_actions
        for opponent_type in opponent_types:
            policy = opponents.get(opponent_type)
            if policy is None:
                raise ValueError(f"Unknown opponent type: {opponent_type}")
            opponent_action = policy.act(observation, generator)
            actions = torch.where(seat_types == opponent_type, opponent_action, actions)

        values_history.append(values)
        if bool(learner_mask.any()):
            game_index, player_index = learner_mask.nonzero(as_tuple=True)
            decision_states.append(observation.state_features[game_index, player_index])
            decision_actions.append(observation.action_features[game_index, player_index])
            decision_mask.append(observation.legal[game_index, player_index])
            decision_chosen.append(learner_actions[game_index, player_index])
            decision_log_probs.append(learner_log_probs[game_index, player_index])
            decision_values.append(values[game_index, player_index])
            decision_turn.append(torch.full_like(game_index, turn))
            decision_game.append(game_index)
            decision_player.append(player_index)

        rewards, _done = env.step(actions)
        dense_rewards.append(rewards)

    terminal = env.terminal_rewards(
        success_threshold=success_threshold,
        win_weight=win_weight,
        placement_weight=placement_weight,
        score_weight=score_weight,
        success_weight=success_weight,
        margin_weight=margin_weight,
    )
    ranks = env.final_ranks()

    turns = len(dense_rewards)
    advantages = torch.zeros(turns, batch, players, dtype=env.dtype, device=device)
    returns = torch.zeros(turns, batch, players, dtype=env.dtype, device=device)
    next_value = torch.zeros(batch, players, dtype=env.dtype, device=device)
    next_advantage = torch.zeros(batch, players, dtype=env.dtype, device=device)
    for turn in range(turns - 1, -1, -1):
        reward = dense_rewards[turn]
        if turn == turns - 1:
            reward = reward + terminal
        value = values_history[turn]
        delta = reward + gamma * next_value - value
        advantage = delta + gamma * gae_lambda * next_advantage
        advantages[turn] = advantage
        returns[turn] = advantage + value
        next_value = value
        next_advantage = advantage

    turn_ids = torch.cat(decision_turn)
    game_ids = torch.cat(decision_game)
    player_ids = torch.cat(decision_player)
    return TorchRollout(
        states=torch.cat(decision_states),
        actions=torch.cat(decision_actions),
        mask=torch.cat(decision_mask),
        chosen=torch.cat(decision_chosen),
        log_probs=torch.cat(decision_log_probs),
        values=torch.cat(decision_values),
        returns=returns[turn_ids, game_ids, player_ids],
        advantages=advantages[turn_ids, game_ids, player_ids],
        player_ids=player_ids,
        turn_ids=turn_ids,
        terminal_rewards=terminal[learner_mask],
        scores=env.scores[learner_mask],
        ranks=ranks[learner_mask],
        player_count=env.player_count,
        num_games=batch,
    )


def combine_torch_rollouts(rollouts: Sequence[TorchRollout]) -> TorchRollout:
    if not rollouts:
        raise ValueError("No rollouts to combine")
    counts = {rollout.player_count for rollout in rollouts}
    return TorchRollout(
        states=torch.cat([rollout.states for rollout in rollouts]),
        actions=torch.cat([rollout.actions for rollout in rollouts]),
        mask=torch.cat([rollout.mask for rollout in rollouts]),
        chosen=torch.cat([rollout.chosen for rollout in rollouts]),
        log_probs=torch.cat([rollout.log_probs for rollout in rollouts]),
        values=torch.cat([rollout.values for rollout in rollouts]),
        returns=torch.cat([rollout.returns for rollout in rollouts]),
        advantages=torch.cat([rollout.advantages for rollout in rollouts]),
        player_ids=torch.cat([rollout.player_ids for rollout in rollouts]),
        turn_ids=torch.cat([rollout.turn_ids for rollout in rollouts]),
        terminal_rewards=torch.cat([rollout.terminal_rewards for rollout in rollouts]),
        scores=torch.cat([rollout.scores for rollout in rollouts]),
        ranks=torch.cat([rollout.ranks for rollout in rollouts]),
        player_count=rollouts[0].player_count if len(counts) == 1 else 0,
        num_games=sum(rollout.num_games for rollout in rollouts),
    )


# ---------------------------------------------------------------------------
# Fast batched head-to-head evaluation


def head_to_head_report_torch(
    candidate: TorchActorCritic,
    opponent: RLNetwork,
    *,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    games: int = 100,
    seed_base: int = 60000,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Batched version of :func:`eightfm.rl_train.head_to_head_report`.

    The NumPy report is exact but runs one Python forward per seat per turn;
    a 500-game check costs minutes.  This version plays the same controlled-seat
    match in :class:`BatchedClassicEnv`, so the selector can afford a reliable
    head-to-head sample every evaluation without eating the training budget.

    The candidate sits in one seat per game (rotating across games) and the
    opponent fills every other seat.  ``win_rate`` above ``chance`` means the
    candidate is the stronger policy.
    """
    if device is None:
        device = next(candidate.parameters()).device
    else:
        device = torch.device(device)

    candidate.eval()
    opponent_model = TorchActorCritic.from_rl_network(opponent, seed=seed_base + 1)
    opponent_model.to(device).eval()
    opponent_state_dim = state_feature_size(opponent.feature_version)
    opponent_action_dim = action_feature_size(opponent.feature_version)
    if (
        opponent_state_dim > state_feature_size(candidate.feature_version)
        or opponent_action_dim > action_feature_size(candidate.feature_version)
    ):
        raise ValueError(
            "Opponent feature layout must be a prefix of the candidate layout "
            "(v1 ⊂ v2 ⊂ v3 ⊂ v4)."
        )

    wins: list[float] = []
    scores: list[float] = []
    margins: list[float] = []
    successes: list[float] = []
    for player_count in player_counts:
        batch = max(1, int(games))
        env = BatchedClassicEnv(player_count, batch, device=device, seed=seed_base + player_count * 1000)
        seats = torch.arange(batch, device=device) % player_count
        candidate_mask = torch.zeros(batch, player_count, dtype=torch.bool, device=device)
        candidate_mask[torch.arange(batch, device=device), seats] = True

        for _turn in range(10):
            observation = env.observation(candidate.feature_version)
            with torch.no_grad():
                candidate_scores, _ = candidate(
                    observation.state_features, observation.action_features
                )
                opponent_scores, _ = opponent_model(
                    observation.state_features[..., :opponent_state_dim],
                    observation.action_features[..., :opponent_action_dim],
                )
            candidate_actions = candidate_scores.masked_fill(~observation.legal, -1e9).argmax(dim=-1)
            opponent_actions = opponent_scores.masked_fill(~observation.legal, -1e9).argmax(dim=-1)
            actions = torch.where(candidate_mask, candidate_actions, opponent_actions)
            env.step(actions)

        final_scores = env.scores
        ranks = env.final_ranks()
        own_scores = final_scores[candidate_mask]
        own_ranks = ranks[candidate_mask]
        own_total = final_scores.gather(1, seats.unsqueeze(1)).squeeze(1)
        others_mean = (final_scores.sum(dim=1) - own_total) / max(1, player_count - 1)

        wins.append(float((own_ranks == 1).to(torch.float32).mean()))
        scores.append(float(own_scores.mean()))
        margins.append(float((own_scores - others_mean).mean()))
        successes.append(float((own_scores > 60.0).to(torch.float32).mean()))

    chance = float(np.mean([1.0 / count for count in player_counts])) if player_counts else 0.0
    return {
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "chance": chance,
        "games": games,
    }


# ---------------------------------------------------------------------------
# Behaviour cloning


@dataclass
class TorchBCDataset:
    states: torch.Tensor
    actions: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


def collect_bc_dataset_torch(
    *,
    games: int,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 1,
    random_fraction: float = 0.6,
    label_behaviour_fraction: float = 0.6,
    feature_version: int = LATEST_FEATURE_VERSION,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> TorchBCDataset:
    """Vectorised version of :func:`eightfm.rl.collect_bc_dataset`."""
    device = torch.device(device)
    rng = random.Random(seed)
    groups: dict[int, int] = {}
    for game_index in range(games):
        player_count = player_counts[game_index % len(player_counts)]
        groups[player_count] = groups.get(player_count, 0) + 1

    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for player_count, count in groups.items():
        for start in range(0, count, batch_size):
            size = min(batch_size, count - start)
            env = BatchedClassicEnv(player_count, size, device=device, seed=rng.randrange(1 << 30))
            generator = torch.Generator(device=device)
            generator.manual_seed(rng.randrange(1 << 30))
            teacher = BatchedComboPolicy()
            greedy = BatchedGreedyPolicy()
            random_policy = BatchedRandomPolicy()
            env.reset()
            for _turn in range(10):
                observation = env.observation(feature_version)
                teacher_action = teacher.act(observation, generator)
                greedy_action = greedy.act(observation, generator)
                random_action = random_policy.act(observation, generator)
                teacher_behaviour = (
                    torch.rand(
                        (size, player_count),
                        device=device,
                        generator=generator,
                    )
                    < label_behaviour_fraction
                )
                random_behaviour = (
                    ~teacher_behaviour
                    & (
                        torch.rand(
                            (size, player_count),
                            device=device,
                            generator=generator,
                        )
                        < random_fraction
                    )
                )
                behaviour = torch.where(
                    teacher_behaviour,
                    teacher_action,
                    torch.where(random_behaviour, random_action, greedy_action),
                )
                states.append(observation.state_features.reshape(-1, observation.state_features.shape[-1]))
                actions.append(observation.action_features.reshape(-1, ACTION_SIZE, observation.action_features.shape[-1]))
                masks.append(observation.legal.reshape(-1, ACTION_SIZE))
                labels.append(teacher_action.reshape(-1))
                env.step(behaviour)
    return TorchBCDataset(
        states=torch.cat(states),
        actions=torch.cat(actions),
        mask=torch.cat(masks),
        labels=torch.cat(labels),
    )


def behaviour_clone_torch(
    model: TorchActorCritic,
    dataset: TorchBCDataset,
    *,
    epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 1,
    entropy_coef: float = 0.0,
) -> dict[str, float]:
    if dataset.size == 0:
        return {"bc_loss": 0.0, "bc_accuracy": 0.0, "bc_epochs": 0.0}
    device = dataset.states.device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    losses: list[float] = []
    accuracies: list[float] = []
    for _epoch in range(epochs):
        order = torch.randperm(dataset.size, device=device, generator=generator)
        epoch_loss = 0.0
        epoch_hits = 0
        for start in range(0, dataset.size, batch_size):
            index = order[start:start + batch_size]
            scores, _values = model(dataset.states[index], dataset.actions[index])
            mask = dataset.mask[index]
            probabilities = masked_softmax_torch(scores, mask)
            log_probs = torch.log(probabilities.clamp(min=1e-12))
            chosen = log_probs.gather(1, dataset.labels[index].unsqueeze(1)).squeeze(1)
            loss = -chosen.mean()
            if entropy_coef:
                entropy = -(probabilities * log_probs).sum(dim=-1).mean()
                loss = loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(index)
            predicted = scores.masked_fill(~mask, -1e9).argmax(dim=-1)
            epoch_hits += int((predicted == dataset.labels[index]).sum())
        losses.append(epoch_loss / dataset.size)
        accuracies.append(epoch_hits / dataset.size)
    return {
        "bc_loss": float(losses[-1]) if losses else 0.0,
        "bc_accuracy": float(accuracies[-1]) if accuracies else 0.0,
        "bc_epochs": float(epochs),
    }


# ---------------------------------------------------------------------------
# PPO update


def _huber(error: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = error.abs()
    quadratic = absolute <= delta
    return torch.where(
        quadratic, error * error, 2.0 * delta * absolute - delta * delta
    )


def torch_rl_update(
    model: TorchActorCritic,
    rollout: TorchRollout,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 1024,
    ppo_epochs: int = 8,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.035,
    l2: float = 0.0,
    seed: int = 1,
    value_clip: float | None = 0.2,
    huber_delta: float | None = 1.0,
    target_kl: float | None = 0.03,
    max_grad_norm: float = 5.0,
) -> dict[str, float]:
    if rollout.decisions == 0:
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
            "early_stopped": 0.0,
        }
    device = rollout.states.device
    advantages = rollout.advantages
    advantage_std = float(advantages.std(unbiased=False))
    if advantage_std > 1e-6:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
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
        order = torch.randperm(rollout.decisions, device=device, generator=generator)
        epoch_kls: list[float] = []
        for start in range(0, rollout.decisions, batch_size):
            index = order[start:start + batch_size]
            states = rollout.states[index]
            actions = rollout.actions[index]
            mask = rollout.mask[index]
            chosen = rollout.chosen[index]
            returns = rollout.returns[index]
            old_log_probs = rollout.log_probs[index]
            old_values = rollout.values[index]
            batch_advantages = advantages[index]

            scores, values = model(states, actions)
            probabilities = masked_softmax_torch(scores, mask)
            log_probs = torch.log(probabilities.clamp(min=1e-12))
            chosen_log_prob = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
            ratio = torch.exp((chosen_log_prob - old_log_probs).clamp(-20.0, 20.0))
            approx_kl = float((old_log_probs - chosen_log_prob).mean().detach())
            unclipped = ratio * batch_advantages
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            error = values - returns
            if huber_delta is not None and huber_delta > 0.0:
                value_loss_values = _huber(error, huber_delta)
            else:
                value_loss_values = error * error
            if value_clip is not None and value_clip > 0.0:
                clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
                clipped_error = clipped_values - returns
                if huber_delta is not None and huber_delta > 0.0:
                    clipped_loss_values = _huber(clipped_error, huber_delta)
                else:
                    clipped_loss_values = clipped_error * clipped_error
                take = value_loss_values >= clipped_loss_values
                value_loss_values = torch.where(
                    take, value_loss_values, clipped_loss_values.detach()
                )
                value_clip_fraction = float(
                    ((values - old_values).abs() > value_clip).float().mean()
                )
            else:
                value_clip_fraction = 0.0
            value_loss = value_loss_values.mean()

            entropy = -(probabilities * log_probs).sum(dim=-1).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy.detach()))
            grad_norms.append(float(grad_norm))
            clip_fractions.append(
                float((ratio.detach() - 1.0).abs().gt(clip_ratio).float().mean())
            )
            approx_kls.append(approx_kl)
            value_clip_fractions.append(value_clip_fraction)
            epoch_kls.append(approx_kl)
        if target_kl is not None and epoch_kls:
            if float(np.mean(epoch_kls)) > target_kl:
                early_stopped = True
                break

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "decisions": rollout.decisions,
        "advantage_std": advantage_std,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "value_clip_fraction": float(np.mean(value_clip_fractions)) if value_clip_fractions else 0.0,
        "ppo_epochs_run": float(epochs_run),
        "early_stopped": float(1.0 if early_stopped else 0.0),
    }


# ---------------------------------------------------------------------------
# Training loop


def _sample_torch_specs(
    rng: random.Random,
    progress: float,
    games_per_update: int,
    league_fraction: float,
    opponent_ids: Sequence[int],
    opponent_weights: Sequence[float],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for _ in range(games_per_update):
        player_count = sample_player_count(rng, progress)
        if league_fraction > 0 and rng.random() < league_fraction:
            rl_count = max(1, player_count // 2)
            rl_seats = set(rng.sample(range(player_count), rl_count))
            seat_types = [
                0
                if seat in rl_seats
                else rng.choices(opponent_ids, weights=opponent_weights, k=1)[0]
                for seat in range(player_count)
            ]
        else:
            seat_types = [0] * player_count
        specs.append({
            "player_count": player_count,
            "seat_types": seat_types,
            "seed": rng.randrange(1 << 30),
            "generator_seed": rng.randrange(1 << 30),
        })
    return specs


def _group_specs(specs: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for spec in specs:
        groups.setdefault(int(spec["player_count"]), []).append(spec)
    return groups


def train_rl_torch(
    *,
    updates: int = 3000,
    games_per_update: int = 512,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 8_2026,
    warm_start_games: int = 6000,
    bc_epochs: int = 16,
    bc_batch_size: int = 1024,
    bc_learning_rate: float = 1e-3,
    learning_rate: float = 3e-4,
    learning_rate_final: float | None = 2e-5,
    batch_size: int = 1024,
    ppo_epochs: int = 8,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.035,
    entropy_coef_final: float | None = 0.003,
    value_coef: float = 0.5,
    value_clip: float | None = 0.2,
    huber_delta: float | None = 1.0,
    target_kl: float | None = 0.03,
    l2: float = 0.0,
    success_threshold: float = 60.0,
    win_weight: float = 1.5,
    placement_weight: float = 0.7,
    score_weight: float = 0.4,
    success_weight: float = 0.4,
    margin_weight: float = 0.2,
    goal_selection: bool = False,
    league_fraction: float = 0.65,
    snapshot_every: int = 25,
    snapshot_pool_size: int = 12,
    snapshot_weight: float = 0.3,
    champion_league_weight: float = 0.0,
    selection_opponent_path: str | Path | None = "models/classic-rl.json",
    head_to_head_games: int = 40,
    checkpoint_every: int = 25,
    checkpoint_dir: str | Path | None = "models/lu_sol-checkpoints",
    hidden1: int = 512,
    hidden2: int = 256,
    action_hidden: int = 256,
    feature_version: int = LATEST_FEATURE_VERSION,
    device: str | torch.device | None = "auto",
    eval_every: int = 50,
    eval_games: int = 60,
    eval_seed_base: int = 50000,
    supervised_model: Any | None = None,
    output: str | Path = "models/lu_sol.json",
    latest_path: str | Path | None = "models/lu_sol-latest.json",
    metrics_path: str | Path = "models/lu_sol-metrics.json",
    plot_path: str | Path | None = "plots/lu_sol-training.png",
    resume: str | Path | None = None,
    verbose: bool = True,
) -> tuple[TorchActorCritic, dict[str, Any]]:
    device = resolve_device(device if isinstance(device, str) else str(device))
    rng = random.Random(seed)
    start_time = time.time()

    if resume:
        network = load_rl(resume)
        if int(network.feature_version) != int(feature_version):
            raise ValueError(
                f"Resume checkpoint uses feature layout v{network.feature_version}, "
                f"but feature_version={feature_version} was requested. Refusing to "
                "silently continue with the wrong layout: start a fresh run "
                "(--reset or a new work dir) or pass the matching --feature-version."
            )
        feature_version = network.feature_version
        model = TorchActorCritic.from_rl_network(network, seed=seed)
        if verbose:
            print(
                f"Resumed torch model: {resume} (features v{feature_version}, "
                f"update {network.metadata.get('best_update', '?')})"
            )
    else:
        model = TorchActorCritic(
            state_feature_size(feature_version),
            action_feature_size(feature_version),
            hidden1=hidden1,
            hidden2=hidden2,
            action_hidden=action_hidden,
            feature_version=feature_version,
            seed=seed,
        )
    model.to(device)

    output_path = Path(output)
    if latest_path is None:
        latest_path = output_path.with_name(output_path.stem + "-latest" + output_path.suffix)
    latest_path = Path(latest_path)
    metrics: dict[str, Any] = {
        "format": "eightfm-rl-training-metrics",
        "backend": "torch",
        "device": str(device),
        "config": {
            "updates": updates,
            "games_per_update": games_per_update,
            "player_counts": list(player_counts),
            "seed": seed,
            "warm_start_games": warm_start_games,
            "bc_epochs": bc_epochs,
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
            "success_threshold": success_threshold,
            "win_weight": win_weight,
            "placement_weight": placement_weight,
            "score_weight": score_weight,
            "success_weight": success_weight,
            "margin_weight": margin_weight,
            "goal_selection": goal_selection,
            "league_fraction": league_fraction,
            "snapshot_every": snapshot_every,
            "snapshot_pool_size": snapshot_pool_size,
            "snapshot_weight": snapshot_weight,
            "champion_league_weight": champion_league_weight,
            "selection_opponent_path": str(selection_opponent_path),
            "hidden1": hidden1,
            "hidden2": hidden2,
            "action_hidden": action_hidden,
            "feature_version": feature_version,
            "device": str(device),
            "eval_every": eval_every,
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
        dataset = collect_bc_dataset_torch(
            games=warm_start_games,
            player_counts=player_counts,
            seed=seed + 17,
            feature_version=feature_version,
            device=device,
        )
        bc_metrics = behaviour_clone_torch(
            model,
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
                f"loss={bc_metrics['bc_loss']:.4f}, {bc_metrics['seconds']}s"
            )

    selection_opponent = None
    if selection_opponent_path and Path(selection_opponent_path).exists():
        selection_opponent = load_rl(selection_opponent_path)

    # The champion can also be a live league opponent, not just a selector.
    # Training directly against the exact policy the model must dethrone is a
    # far stronger signal than only evaluating against it every N updates.
    champion_policy = None
    if champion_league_weight > 0.0 and selection_opponent is not None:
        champion_model = TorchActorCritic.from_rl_network(
            selection_opponent, seed=seed + 101
        )
        champion_model.to(device).eval()
        champion_policy = BatchedNetPolicy(champion_model, greedy=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    snapshots: list[TorchActorCritic] = []
    best_score = float("-inf")
    best_update = 0
    league_games = 0

    for update in range(1, updates + 1):
        progress = update / max(1, updates)
        if learning_rate_final is not None:
            current_lr = learning_rate + (learning_rate_final - learning_rate) * (update - 1) / max(1, updates - 1)
        else:
            current_lr = learning_rate
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        current_entropy = entropy_coef
        if entropy_coef_final is not None:
            current_entropy = entropy_coef + (entropy_coef_final - entropy_coef) * (update - 1) / max(1, updates - 1)

        opponents: dict[int, Any] = {
            1: BatchedComboPolicy(),
            2: BatchedGreedyPolicy(),
            3: BatchedRandomPolicy(),
        }
        opponent_ids = [1, 2, 3]
        opponent_weights = [0.5, 0.4, 0.1]
        for index, snapshot in enumerate(snapshots):
            opponents[4 + index] = BatchedNetPolicy(snapshot, greedy=True)
            opponent_ids.append(4 + index)
            opponent_weights.append(snapshot_weight)
        if champion_policy is not None:
            opponents[CHAMPION_OPPONENT_ID] = champion_policy
            opponent_ids.append(CHAMPION_OPPONENT_ID)
            opponent_weights.append(champion_league_weight)

        specs = _sample_torch_specs(
            rng, progress, games_per_update, league_fraction, opponent_ids, opponent_weights
        )
        groups = _group_specs(specs)
        league_games += sum(
            1 for spec in specs if any(seat != 0 for seat in spec["seat_types"])
        )
        rollouts: list[TorchRollout] = []
        model.eval()
        for player_count, group in groups.items():
            batch = len(group)
            seat_types = torch.tensor(
                [spec["seat_types"] for spec in group], dtype=torch.long, device=device
            )
            env = BatchedClassicEnv(
                player_count, batch, device=device, seed=group[0]["seed"]
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(group[0]["generator_seed"])
            rollouts.append(
                collect_torch_rollout(
                    env,
                    model,
                    seat_types,
                    opponents,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    generator=generator,
                    feature_version=feature_version,
                    success_threshold=success_threshold,
                    win_weight=win_weight,
                    placement_weight=placement_weight,
                    score_weight=score_weight,
                    success_weight=success_weight,
                    margin_weight=margin_weight,
                )
            )
        data = combine_torch_rollouts(rollouts)
        model.train()
        train_metrics = torch_rl_update(
            model,
            data,
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
            snapshots.append(copy.deepcopy(model).eval())
            if len(snapshots) > snapshot_pool_size:
                del snapshots[0]

        terminal = data.terminal_rewards
        scores = data.scores
        ranks = data.ranks
        win_rate = float((ranks == 1).float().mean()) if ranks.numel() else 0.0
        success_rate = float((scores > 60.0).float().mean()) if scores.numel() else 0.0
        entry = {
            "update": update,
            "games": update * games_per_update,
            "league_games": league_games,
            "learning_rate": current_lr,
            "entropy_coef": current_entropy,
            "mean_terminal_reward": float(terminal.mean()) if terminal.numel() else 0.0,
            "mean_return": float(data.returns.mean()) if data.returns.numel() else 0.0,
            "mean_score": float(scores.mean()) if scores.numel() else 0.0,
            "win_rate": win_rate,
            "success_rate": success_rate,
            **train_metrics,
        }
        metrics["updates"].append(entry)
        if verbose:
            print(
                f"update {update:4d}/{updates} | games {entry['games']:6d} | "
                f"league {league_games:5d} | reward {entry['mean_terminal_reward']:.3f} | "
                f"ret {entry['mean_return']:+.3f} | win {win_rate:.3f} | "
                f"score {entry['mean_score']:.1f} | loss {entry['policy_loss']:.4f} | "
                f"vloss {entry['value_loss']:.4f} | ent {entry['entropy']:.3f} | "
                f"kl {entry['approx_kl']:+.4f}"
            )

        if checkpoint_every > 0 and checkpoint_dir is not None and update % checkpoint_every == 0:
            archive = Path(checkpoint_dir)
            archive.mkdir(parents=True, exist_ok=True)
            save_rl(to_rl_network(model, metadata={"update": update}), archive / f"update-{update:04d}.json")

        if (eval_every > 0 and update % eval_every == 0) or update == updates:
            network = to_rl_network(model, metadata={"best_update": update})
            eval_report = evaluate_rl(
                network,
                player_counts=player_counts,
                games=eval_games,
                seed_base=eval_seed_base,
                supervised_model=supervised_model,
                # The champion match is played in the batched torch simulator
                # below; the NumPy path would spend minutes per evaluation.
                selection_opponent=None,
                head_to_head_games=head_to_head_games,
            )
            if selection_opponent is not None:
                head_to_head = head_to_head_report_torch(
                    model,
                    selection_opponent,
                    player_counts=player_counts,
                    games=head_to_head_games,
                    seed_base=eval_seed_base + 700000,
                    device=device,
                )
                eval_report["head_to_head"] = head_to_head
                summary = eval_report["summary"]
                summary["head_to_head_win_rate"] = head_to_head["win_rate"]
                summary["head_to_head_chance"] = head_to_head["chance"]
                summary["head_to_head_edge"] = (
                    (head_to_head["win_rate"] - head_to_head["chance"])
                    / head_to_head["chance"]
                    if head_to_head["chance"] > 0.0
                    else 0.0
                )
                summary["head_to_head_gate_passed"] = bool(
                    head_to_head["win_rate"] > head_to_head["chance"]
                )
            eval_report["update"] = update
            metrics["evaluations"].append(eval_report)
            score = (
                _goal_composite_score(eval_report)
                if goal_selection
                else _composite_score(eval_report)
            )
            eval_report["selection_score"] = score
            if verbose:
                summary = eval_report["summary"]
                print(
                    f"  eval: greedy win {summary.get('greedy_win_rate', 0):.3f}"
                    + (f", combo win {summary.get('combo_win_rate', 0):.3f}" if "combo_win_rate" in summary else "")
                    + (f", supervised win {summary.get('supervised_win_rate', 0):.3f}" if "supervised_win_rate" in summary else "")
                    + (f", h2h win {summary.get('head_to_head_win_rate', 0):.3f}" if "head_to_head_win_rate" in summary else "")
                    + f" | score {score:.3f}"
                )
            if score > best_score:
                best_score = score
                best_update = update
                best_network = to_rl_network(
                    model,
                    metadata={"best_update": update, "best_eval": eval_report},
                )
                save_rl(best_network, output_path)
                if verbose:
                    print(f"  saved best checkpoint: {output_path}")
            save_rl(to_rl_network(model, metadata={"best_update": best_update}), latest_path)
        elif update % max(1, eval_every // 2) == 0:
            save_rl(to_rl_network(model, metadata={"best_update": best_update}), latest_path)

    metrics["best_update"] = best_update
    metrics["best_score"] = best_score
    metrics["seconds"] = round(time.time() - start_time, 2)
    training_metadata = {
        "updates": updates,
        "games_per_update": games_per_update,
        "best_update": best_update,
        "best_score": best_score,
        "latest_model": str(latest_path),
        "backend": "torch",
    }
    save_rl(to_rl_network(model, metadata={"training_metrics": training_metadata}), latest_path)
    if best_update:
        try:
            best_network = load_rl(output_path)
        except (OSError, ValueError):
            best_network = to_rl_network(model)
        best_network.metadata["training_metrics"] = training_metadata
        save_rl(best_network, output_path)
    else:
        best_network = to_rl_network(model, metadata={"training_metrics": training_metadata})
        save_rl(best_network, output_path)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if plot_path is not None:
        from .rl_train import _plot_training

        _plot_training(metrics, plot_path, title="8ФМ Classic RL training — lu_sol (torch)")
    return model, metrics


def train_lu_sol_torch(
    *,
    updates: int = 3000,
    games_per_update: int = 512,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 8_2026,
    warm_start_games: int = 6000,
    bc_epochs: int = 16,
    learning_rate: float = 3e-4,
    learning_rate_final: float | None = 2e-5,
    batch_size: int = 1024,
    ppo_epochs: int = 8,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.035,
    entropy_coef_final: float | None = 0.003,
    value_coef: float = 0.5,
    value_clip: float | None = 0.2,
    huber_delta: float | None = 1.0,
    target_kl: float | None = 0.03,
    league_fraction: float = 0.65,
    snapshot_every: int = 25,
    snapshot_pool_size: int = 12,
    snapshot_weight: float = 0.3,
    selection_opponent_path: str | Path | None = "models/classic-rl.json",
    head_to_head_games: int = 40,
    checkpoint_every: int = 25,
    checkpoint_dir: str | Path | None = "models/lu_sol-checkpoints",
    hidden1: int = 512,
    hidden2: int = 256,
    action_hidden: int = 256,
    feature_version: int = 2,
    device: str | torch.device | None = "auto",
    eval_every: int = 50,
    eval_games: int = 60,
    eval_seed_base: int = 50000,
    supervised_model: Any | None = None,
    output: str | Path = "models/lu_sol.json",
    latest_path: str | Path | None = "models/lu_sol-latest.json",
    metrics_path: str | Path = "models/lu_sol-metrics.json",
    plot_path: str | Path | None = "plots/lu_sol-training.png",
    resume: str | Path | None = None,
    verbose: bool = True,
) -> tuple[TorchActorCritic, dict[str, Any]]:
    """Train ``lu_sol`` on the vectorised torch backend.

    Defaults mirror :func:`eightfm.rl_train.train_lu_sol` except
    ``games_per_update`` (512 instead of 48): the batch is the point of the GPU
    path.  The exported ``models/lu_sol.json`` is the same format the NumPy app
    loads.
    """
    return train_rl_torch(
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
        selection_opponent_path=selection_opponent_path,
        head_to_head_games=head_to_head_games,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        hidden1=hidden1,
        hidden2=hidden2,
        action_hidden=action_hidden,
        feature_version=feature_version,
        device=device,
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
# lu_astra — the maximal preset
#
# ``lu_astra`` is not a new architecture: it is the proven :class:`TorchActorCritic`
# (and therefore the exact ``RLNetwork`` JSON the apps load) pushed as far as the
# 12-hour Kaggle GPU budget allows.  The v3 uplift over ``lu_sol`` is:
#
# * feature layout v4 (250 state / 42 action inputs): everything from v3 plus
#   personal hand history -- every card ever held (including passed hands),
#   every card closed, and cards seen but passed on, i.e. the card-counting
#   signal a human uses;
# * a wider net (2048/1024/1024 vs 512/256/256) for far more expressive play;
# * more games per PPO update and larger minibatches;
# * a score-heavy terminal reward, because the acceptance criterion is a 60+
#   point average rather than just winning a low-scoring table;
# * checkpoint selection by :func:`_goal_composite_score`, which makes beating
#   ``lu_sol`` a hard gate and only then optimises the candidate's own mean
#   score and >60 success rate;
# * ``lu_sol`` is also a live league opponent (``champion_league_weight``), so
#   the learner is trained directly against the champion it must dethrone.
#
# Keeping the architecture identical is deliberate: every checkpoint exports to
# the same JSON format, so the desktop/Telegram apps and the tournament load it
# with no changes at all.

LU_ASTRA_FEATURE_VERSION = 4
LU_ASTRA_LEARNING_RATE = 3e-4
LU_ASTRA_LEARNING_RATE_FINAL = 5e-6
LU_ASTRA_ENTROPY = 0.03
LU_ASTRA_ENTROPY_FINAL = 0.001
# Score first: 60 points and a >60 finish together outweigh a bare win.
LU_ASTRA_WIN_WEIGHT = 1.0
LU_ASTRA_PLACEMENT_WEIGHT = 0.3
LU_ASTRA_SCORE_WEIGHT = 1.2
LU_ASTRA_SUCCESS_WEIGHT = 1.5
LU_ASTRA_MARGIN_WEIGHT = 0.4


def train_lu_astra(
    *,
    updates: int = 15000,
    games_per_update: int = 1024,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 8_2026,
    warm_start_games: int = 20000,
    bc_epochs: int = 20,
    bc_batch_size: int = 1024,
    bc_learning_rate: float = 1e-3,
    learning_rate: float = LU_ASTRA_LEARNING_RATE,
    learning_rate_final: float | None = LU_ASTRA_LEARNING_RATE_FINAL,
    batch_size: int = 4096,
    ppo_epochs: int = 6,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = LU_ASTRA_ENTROPY,
    entropy_coef_final: float | None = LU_ASTRA_ENTROPY_FINAL,
    value_coef: float = 0.5,
    value_clip: float | None = 0.2,
    huber_delta: float | None = 1.0,
    target_kl: float | None = 0.02,
    l2: float = 0.0,
    success_threshold: float = 60.0,
    win_weight: float = LU_ASTRA_WIN_WEIGHT,
    placement_weight: float = LU_ASTRA_PLACEMENT_WEIGHT,
    score_weight: float = LU_ASTRA_SCORE_WEIGHT,
    success_weight: float = LU_ASTRA_SUCCESS_WEIGHT,
    margin_weight: float = LU_ASTRA_MARGIN_WEIGHT,
    goal_selection: bool = True,
    league_fraction: float = 0.85,
    snapshot_every: int = 15,
    snapshot_pool_size: int = 24,
    snapshot_weight: float = 0.15,
    champion_league_weight: float = 0.35,
    selection_opponent_path: str | Path | None = "models/lu_sol.json",
    head_to_head_games: int = 100,
    checkpoint_every: int = 25,
    checkpoint_dir: str | Path | None = "models/lu_astra-checkpoints",
    hidden1: int = 2048,
    hidden2: int = 1024,
    action_hidden: int = 1024,
    feature_version: int = LU_ASTRA_FEATURE_VERSION,
    device: str | torch.device | None = "auto",
    eval_every: int = 100,
    eval_games: int = 50,
    eval_seed_base: int = 50000,
    supervised_model: Any | None = None,
    output: str | Path = "models/lu_astra.json",
    latest_path: str | Path | None = "models/lu_astra-latest.json",
    metrics_path: str | Path = "models/lu_astra-metrics.json",
    plot_path: str | Path | None = "plots/lu_astra-training.png",
    resume: str | Path | None = None,
    verbose: bool = True,
) -> tuple[TorchActorCritic, dict[str, Any]]:
    """Train ``lu_astra`` on the vectorised torch backend.

    A thin preset over :func:`train_rl_torch`: the same network, simulator and
    export path that produced ``lu_sol``, only with the v4 feature layout,
    larger batches, a score-heavy reward, a live league seat for ``lu_sol`` and
    a champion-gated 60-point selection metric.  Every returned checkpoint is a
    plain :class:`eightfm.rl.RLNetwork` JSON, so it can be dropped into
    ``models/`` and used immediately.
    """
    return train_rl_torch(
        updates=updates,
        games_per_update=games_per_update,
        player_counts=player_counts,
        seed=seed,
        warm_start_games=warm_start_games,
        bc_epochs=bc_epochs,
        bc_batch_size=bc_batch_size,
        bc_learning_rate=bc_learning_rate,
        learning_rate=learning_rate,
        learning_rate_final=learning_rate_final,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        clip_ratio=clip_ratio,
        gamma=gamma,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        entropy_coef_final=entropy_coef_final,
        value_coef=value_coef,
        value_clip=value_clip,
        huber_delta=huber_delta,
        target_kl=target_kl,
        l2=l2,
        success_threshold=success_threshold,
        win_weight=win_weight,
        placement_weight=placement_weight,
        score_weight=score_weight,
        success_weight=success_weight,
        margin_weight=margin_weight,
        goal_selection=goal_selection,
        league_fraction=league_fraction,
        snapshot_every=snapshot_every,
        snapshot_pool_size=snapshot_pool_size,
        snapshot_weight=snapshot_weight,
        champion_league_weight=champion_league_weight,
        selection_opponent_path=selection_opponent_path,
        head_to_head_games=head_to_head_games,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        hidden1=hidden1,
        hidden2=hidden2,
        action_hidden=action_hidden,
        feature_version=feature_version,
        device=device,
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
