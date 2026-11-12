"""Self-play reinforcement learning for 8ФМ Classic.

The model is an actor-critic network over the full simulator state.  It is
warm-started by behaviour cloning of a combo-aware teacher, then fine-tuned
with PPO in self-play and league episodes.  The warm start is optional; the
RL objective and the final checkpoint are always produced by self-play.

Architecture
------------
state encoder:  115 -> 128 ReLU -> 64 ReLU
action encoder:  25 -> 64 ReLU
score:          bilinear(state, action) + bias
value head:     state -> 1

Training
--------
* simultaneous self-play at 3–7 players;
* GAE(lambda = 0.95) returns and PPO with a clipped surrogate;
* dense relative score reward plus terminal win/placement/score/>60 reward;
* entropy regularisation and a value baseline;
* periodic evaluation in a controlled seat against greedy, supervised and
  combo policies, with the best checkpoint selected by a composite score.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .agent import EMOJI_CARDS, _parse_candidates, _parse_state, _strip_played_card
from .baselines import (
    GreedyPolicy,
    Policy,
    SupervisedPolicy,
    evaluate_controlled_seat,
    play_episode,
)
from .env import (
    ClassicEnv,
    Observation,
    action_feature_size,
    action_features,
    state_feature_size,
    state_features,
)
from .model import CARD_IDS
from .policy import PolicyModel
from .rules import DECK_COUNTS, Action, legal_actions


RL_FORMAT = "eightfm-rl-actor-critic"
RL_FEATURES_VERSION = 1


# ---------------------------------------------------------------------------
# Network


@dataclass
class RLNetwork:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    Wa: np.ndarray
    ba: np.ndarray
    Wp: np.ndarray
    bp: np.ndarray
    wv: np.ndarray
    bv: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.W1 = np.asarray(self.W1, dtype=np.float32)
        self.b1 = np.asarray(self.b1, dtype=np.float32)
        self.W2 = np.asarray(self.W2, dtype=np.float32)
        self.b2 = np.asarray(self.b2, dtype=np.float32)
        self.Wa = np.asarray(self.Wa, dtype=np.float32)
        self.ba = np.asarray(self.ba, dtype=np.float32)
        self.Wp = np.asarray(self.Wp, dtype=np.float32)
        self.wv = np.asarray(self.wv, dtype=np.float32).reshape(-1, 1)
        self.bp = np.asarray(self.bp, dtype=np.float32).reshape(1)
        self.bv = np.asarray(self.bv, dtype=np.float32).reshape(1)

    @property
    def parameters(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.Wa, self.ba, self.Wp, self.wv, self.bp, self.bv]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": RL_FORMAT,
            "features_version": RL_FEATURES_VERSION,
            "card_ids": list(CARD_IDS),
            "state_feature_size": state_feature_size(),
            "action_feature_size": action_feature_size(),
            "state_feature_names": _state_names(),
            "action_feature_names": _action_names(),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "Wa": self.Wa.tolist(),
            "ba": self.ba.tolist(),
            "Wp": self.Wp.tolist(),
            "bp": float(self.bp[0]),
            "wv": self.wv.reshape(-1).tolist(),
            "bv": float(self.bv[0]),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RLNetwork":
        if data.get("format") != RL_FORMAT:
            raise ValueError("Unsupported RL model format")
        if int(data.get("features_version", -1)) != RL_FEATURES_VERSION:
            raise ValueError("Unsupported RL feature version")
        if list(data.get("card_ids", [])) != list(CARD_IDS):
            raise ValueError("RL model card set does not match this build")
        if int(data.get("state_feature_size", -1)) != state_feature_size():
            raise ValueError("RL model state feature size does not match this build")
        if int(data.get("action_feature_size", -1)) != action_feature_size():
            raise ValueError("RL model action feature size does not match this build")
        return cls(
            W1=np.asarray(data["W1"], dtype=np.float32),
            b1=np.asarray(data["b1"], dtype=np.float32),
            W2=np.asarray(data["W2"], dtype=np.float32),
            b2=np.asarray(data["b2"], dtype=np.float32),
            Wa=np.asarray(data["Wa"], dtype=np.float32),
            ba=np.asarray(data["ba"], dtype=np.float32),
            Wp=np.asarray(data["Wp"], dtype=np.float32),
            bp=np.asarray([data["bp"]], dtype=np.float32),
            wv=np.asarray(data["wv"], dtype=np.float32),
            bv=np.asarray([data["bv"]], dtype=np.float32),
            metadata=dict(data.get("metadata") or {}),
        )


def _state_names() -> list[str]:
    from .env import state_feature_names
    return state_feature_names()


def _action_names() -> list[str]:
    from .env import action_feature_names
    return action_feature_names()


def create_network(
    seed: int = 1,
    *,
    hidden1: int = 128,
    hidden2: int = 64,
    action_hidden: int = 64,
) -> RLNetwork:
    rng = np.random.default_rng(seed)
    state_dim = state_feature_size()
    action_dim = action_feature_size()
    W1 = rng.normal(0.0, math.sqrt(2.0 / state_dim), (state_dim, hidden1)).astype(np.float32)
    b1 = np.zeros(hidden1, dtype=np.float32)
    W2 = rng.normal(0.0, math.sqrt(2.0 / hidden1), (hidden1, hidden2)).astype(np.float32)
    b2 = np.zeros(hidden2, dtype=np.float32)
    Wa = rng.normal(0.0, math.sqrt(2.0 / action_dim), (action_dim, action_hidden)).astype(np.float32)
    ba = np.zeros(action_hidden, dtype=np.float32)
    Wp = rng.normal(0.0, math.sqrt(1.0 / hidden2), (hidden2, action_hidden)).astype(np.float32)
    wv = np.zeros((hidden2, 1), dtype=np.float32)
    return RLNetwork(
        W1=W1, b1=b1, W2=W2, b2=b2, Wa=Wa, ba=ba, Wp=Wp,
        bp=np.zeros(1, dtype=np.float32), wv=wv, bv=np.zeros(1, dtype=np.float32),
        metadata={
            "hidden1": hidden1,
            "hidden2": hidden2,
            "action_hidden": action_hidden,
            "target": "self-play actor-critic",
        },
    )


def save_rl(model: RLNetwork, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(model.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_rl(path: str | Path) -> RLNetwork:
    with Path(path).open("r", encoding="utf-8") as stream:
        return RLNetwork.from_dict(json.load(stream))


def clone_network(net: RLNetwork) -> RLNetwork:
    """Return a frozen, independent copy used as a self-play league opponent."""
    return RLNetwork(
        W1=net.W1.copy(),
        b1=net.b1.copy(),
        W2=net.W2.copy(),
        b2=net.b2.copy(),
        Wa=net.Wa.copy(),
        ba=net.ba.copy(),
        Wp=net.Wp.copy(),
        bp=net.bp.copy(),
        wv=net.wv.copy(),
        bv=net.bv.copy(),
        metadata=dict(net.metadata),
    )


# ---------------------------------------------------------------------------
# Forward / backward


@dataclass
class ForwardCache:
    states: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    z1: np.ndarray
    h1: np.ndarray
    z2: np.ndarray
    hs: np.ndarray
    za: np.ndarray
    ha: np.ndarray
    u: np.ndarray


def _relu(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, 0.0)


def forward_batch(
    net: RLNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, ForwardCache]:
    """Score every padded action and produce the state value.

    ``states``: (B, S); ``actions``: (B, A, F); ``mask``: (B, A) bool.
    """
    z1 = states @ net.W1 + net.b1
    h1 = _relu(z1)
    z2 = h1 @ net.W2 + net.b2
    hs = _relu(z2)
    values = (hs @ net.wv + net.bv).reshape(-1)

    flat_actions = actions.reshape(-1, actions.shape[-1])
    za = flat_actions @ net.Wa + net.ba
    ha = _relu(za).reshape(actions.shape[0], actions.shape[1], -1)
    u = hs @ net.Wp
    scores = np.einsum("bh,bah->ba", u, ha) + net.bp.reshape(())
    scores = np.where(mask, scores, -1e9)
    cache = ForwardCache(
        states=states, actions=actions, mask=mask,
        z1=z1, h1=h1, z2=z2, hs=hs, za=za, ha=ha, u=u,
    )
    return scores, values, cache


def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, scores, -1e9)
    maximum = np.max(masked, axis=1, keepdims=True)
    exp = np.exp(np.where(mask, scores - maximum, -1e9)) * mask
    denominator = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denominator, 1e-12)


def _one_hot(labels: np.ndarray, width: int) -> np.ndarray:
    result = np.zeros((labels.shape[0], width), dtype=np.float32)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def backward_batch(
    net: RLNetwork,
    cache: ForwardCache,
    d_scores: np.ndarray,
    d_values: np.ndarray | None = None,
    *,
    l2: float = 0.0,
) -> list[np.ndarray]:
    """Backpropagate policy and optional value gradients through both towers."""
    d_scores = np.asarray(d_scores, dtype=np.float32)
    d_u = np.einsum("ba,bah->bh", d_scores, cache.ha)
    d_Wp = cache.hs.T @ d_u
    d_bp = np.asarray([np.sum(d_scores)], dtype=np.float32)
    d_ha = d_scores[:, :, None] * cache.u[:, None, :]
    d_hs = d_u @ net.Wp.T

    if d_values is not None:
        d_values = np.asarray(d_values, dtype=np.float32).reshape(-1)
        d_hs = d_hs + d_values[:, None] * net.wv.reshape(1, -1)
        d_wv = cache.hs.T @ d_values[:, None]
        d_bv = np.asarray([np.sum(d_values)], dtype=np.float32)
    else:
        d_wv = np.zeros_like(net.wv)
        d_bv = np.zeros_like(net.bv)

    d_z2 = d_hs * (cache.z2 > 0.0)
    d_W2 = cache.h1.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0)
    d_h1 = d_z2 @ net.W2.T
    d_z1 = d_h1 * (cache.z1 > 0.0)
    d_W1 = cache.states.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0)

    d_za = d_ha.reshape(-1, d_ha.shape[-1]) * (cache.za > 0.0)
    flat_actions = cache.actions.reshape(-1, cache.actions.shape[-1])
    d_Wa = flat_actions.T @ d_za
    d_ba = np.sum(d_za, axis=0)

    gradients = [d_W1, d_b1, d_W2, d_b2, d_Wa, d_ba, d_Wp, d_wv, d_bp, d_bv]
    if l2:
        for index in (0, 2, 4, 6):
            gradients[index] = gradients[index] + l2 * net.parameters[index]
    return gradients


class Adam:
    def __init__(
        self,
        parameters: Sequence[np.ndarray],
        *,
        learning_rate: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_grad_norm: float = 5.0,
    ):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.first = [np.zeros_like(parameter) for parameter in self.parameters]
        self.second = [np.zeros_like(parameter) for parameter in self.parameters]
        self.step_count = 0

    def step(self, gradients: Sequence[np.ndarray]) -> float:
        norm_sq = sum(float(np.sum(gradient * gradient)) for gradient in gradients)
        norm = math.sqrt(norm_sq)
        scale = min(1.0, self.max_grad_norm / max(norm, 1e-12))
        self.step_count += 1
        for index, (parameter, gradient) in enumerate(zip(self.parameters, gradients)):
            gradient = gradient * scale
            self.first[index] = self.beta1 * self.first[index] + (1.0 - self.beta1) * gradient
            self.second[index] = self.beta2 * self.second[index] + (1.0 - self.beta2) * (gradient * gradient)
            first_hat = self.first[index] / (1.0 - self.beta1 ** self.step_count)
            second_hat = self.second[index] / (1.0 - self.beta2 ** self.step_count)
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)
        return norm


# ---------------------------------------------------------------------------
# Feature batching and action selection


def _pad_observations(observations: Sequence[Observation]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[Action]]]:
    batch = len(observations)
    max_actions = max(len(obs.legal) for obs in observations)
    states = np.zeros((batch, state_feature_size()), dtype=np.float32)
    actions = np.zeros((batch, max_actions, action_feature_size()), dtype=np.float32)
    mask = np.zeros((batch, max_actions), dtype=bool)
    legal_lists: list[list[Action]] = []
    for row, obs in enumerate(observations):
        states[row] = np.asarray(state_features(obs), dtype=np.float32)
        legal_lists.append(list(obs.legal))
        for column, action in enumerate(obs.legal):
            actions[row, column] = np.asarray(action_features(obs, action), dtype=np.float32)
            mask[row, column] = True
    return states, actions, mask, legal_lists


def act_batch(
    net: RLNetwork,
    observations: Sequence[Observation],
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    greedy: bool = False,
) -> tuple[list[Action], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose one action per observation and return policy statistics.

    Returns ``(actions, chosen, log_probs, entropies, values, states, action_tensor, mask)``.
    """
    states, actions, mask, legal_lists = _pad_observations(observations)
    scores, values, _ = forward_batch(net, states, actions, mask)
    probabilities = masked_softmax(scores / max(1e-6, temperature), mask)
    if greedy:
        chosen = np.argmax(np.where(mask, scores, -1e9), axis=1)
    else:
        chosen = np.asarray([
            rng.choice(probabilities.shape[1], p=probabilities[row] / max(1e-12, probabilities[row].sum()))
            for row in range(probabilities.shape[0])
        ], dtype=np.int64)
    log_probs = np.log(np.maximum(probabilities[np.arange(len(chosen)), chosen], 1e-12))
    entropies = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    selected = [legal_lists[row][int(chosen[row])] for row in range(len(observations))]
    return selected, chosen, log_probs, entropies, values, states, actions, mask


def greedy_action(net: RLNetwork, obs: Observation) -> Action:
    actions, _, _, _, _, _, _, _ = act_batch(net, [obs], np.random.default_rng(0), greedy=True)
    return actions[0]


class RLGreedyPolicy:
    def __init__(self, net: RLNetwork):
        self.net = net

    def __call__(self, obs: Observation) -> Action:
        return greedy_action(self.net, obs)


# ---------------------------------------------------------------------------
# Behaviour cloning warm start


@dataclass
class BCDataset:
    states: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    labels: np.ndarray

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


def collect_bc_dataset(
    *,
    games: int = 600,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 1,
    random_fraction: float = 0.6,
    label_policy: Callable[[Observation], Action] | None = None,
    label_behaviour_fraction: float = 0.6,
) -> BCDataset:
    """Generate diverse simulator states and label them with a teacher policy.

    When ``label_policy`` is omitted the labels are greedy actions.  Passing a
    combo-aware teacher both labels the states and drives a fraction of the
    trajectories, so the dataset contains the early "investment" states that a
    purely greedy behaviour policy would never visit.
    """
    rng = random.Random(seed)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    for game_index in range(games):
        player_count = player_counts[game_index % len(player_counts)]
        env = ClassicEnv(player_count, seed=rng.randrange(1 << 30))
        observations = env.observations()
        while not env.done:
            chosen_actions: list[Action] = []
            for obs in observations:
                # Behaviour policy: the teacher drives part of the trajectory
                # to cover combo-building states; random/greedy keeps diversity.
                if label_policy is not None and rng.random() < label_behaviour_fraction:
                    action = label_policy(obs)
                    label_action = action
                else:
                    if rng.random() < random_fraction:
                        action = rng.choice(obs.legal)
                    else:
                        from .env import greedy_score_policy
                        action = greedy_score_policy(obs, rng)
                    label_action = (
                        label_policy(obs) if label_policy is not None else _greedy_label(obs)
                    )
                chosen_actions.append(action)

                legal = list(obs.legal)
                if label_action not in legal:
                    continue
                label = legal.index(label_action)
                max_actions = len(legal)
                state_vector = np.asarray(state_features(obs), dtype=np.float32)
                action_matrix = np.zeros((max_actions, action_feature_size()), dtype=np.float32)
                for index, candidate in enumerate(legal):
                    action_matrix[index] = np.asarray(action_features(obs, candidate), dtype=np.float32)
                states.append(state_vector)
                actions.append(action_matrix)
                masks.append(np.ones(max_actions, dtype=bool))
                labels.append(label)
            result = env.step(chosen_actions)
            observations = result.observations

    max_actions = max((matrix.shape[0] for matrix in actions), default=1)
    action_tensor = np.zeros((len(actions), max_actions, action_feature_size()), dtype=np.float32)
    mask_tensor = np.zeros((len(actions), max_actions), dtype=bool)
    for row, (matrix, mask) in enumerate(zip(actions, masks)):
        action_tensor[row, :matrix.shape[0]] = matrix
        mask_tensor[row, :mask.shape[0]] = mask
    return BCDataset(
        states=np.stack(states) if states else np.zeros((0, state_feature_size()), dtype=np.float32),
        actions=action_tensor,
        mask=mask_tensor,
        labels=np.asarray(labels, dtype=np.int64),
    )


def _greedy_label(obs: Observation) -> Action:
    from .env import greedy_score_policy
    return greedy_score_policy(obs, None)


def behaviour_clone(
    net: RLNetwork,
    dataset: BCDataset,
    *,
    epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 1,
    entropy_coef: float = 0.0,
) -> dict[str, float]:
    if dataset.size == 0:
        return {"bc_loss": 0.0, "bc_accuracy": 0.0, "bc_epochs": 0.0}
    optimizer = Adam(net.parameters, learning_rate=learning_rate, max_grad_norm=5.0)
    rng = np.random.default_rng(seed)
    order = np.arange(dataset.size)
    losses: list[float] = []
    accuracies: list[float] = []
    for _epoch in range(epochs):
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_hits = 0
        for start in range(0, dataset.size, batch_size):
            indices = order[start:start + batch_size]
            states = dataset.states[indices]
            actions = dataset.actions[indices]
            mask = dataset.mask[indices]
            labels = dataset.labels[indices]
            scores, _values, cache = forward_batch(net, states, actions, mask)
            probabilities = masked_softmax(scores, mask)
            chosen_probability = probabilities[np.arange(len(labels)), labels]
            loss = -float(np.mean(np.log(np.maximum(chosen_probability, 1e-12))))
            d_scores = probabilities - _one_hot(labels, probabilities.shape[1])
            d_scores /= max(1, len(labels))
            if entropy_coef:
                log_probs = np.log(np.maximum(probabilities, 1e-12))
                entropies = -np.sum(probabilities * log_probs, axis=1)
                d_scores += entropy_coef * probabilities * (log_probs + entropies[:, None]) / max(1, len(labels))
            gradients = backward_batch(net, cache, d_scores)
            optimizer.step(gradients)
            epoch_loss += loss * len(labels)
            epoch_hits += int(np.sum(np.argmax(np.where(mask, scores, -1e9), axis=1) == labels))
        losses.append(epoch_loss / dataset.size)
        accuracies.append(epoch_hits / dataset.size)
    return {
        "bc_loss": float(losses[-1]) if losses else 0.0,
        "bc_accuracy": float(accuracies[-1]) if accuracies else 0.0,
        "bc_epochs": float(epochs),
    }
