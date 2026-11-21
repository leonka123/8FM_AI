"""Weighted ensemble of trained 8ФМ policies.

The ensemble blends the per-action preferences of several members — RL
checkpoints and the supervised policy — into one soft-voting distribution over
the legal actions.  Each member turns its raw action scores into a softmax
distribution scaled by its temperature, and the members are mixed with fixed
weights:

    P(action) = Σ_i w_i · softmax(score_i / T_i) / Σ_i w_i

The action with the largest blended probability is played.  Because the members
see exactly the same simulator observation, the blend cannot leak hidden cards;
it only arbitrates between their opinions.

An ensemble is described declaratively (member kind, checkpoint path, weight,
temperature), so it can be saved to JSON, reloaded, rebuilt from the CLI, or
dropped into the interactive game as an opponent.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .baselines import SupervisedPolicy
from .env import Observation
from .policy import PolicyAction
from .rules import Action

ENSEMBLE_FORMAT = "eightfm-ensemble"
MEMBER_KINDS = ("rl", "supervised")

# Named shorthands used by the CLI and the interactive game.
MEMBER_ALIASES: dict[str, tuple[str, str]] = {
    "lu_luna": ("rl", "models/lu_luna.json"),
    "lu-luna": ("rl", "models/lu_luna.json"),
    "classic-rl": ("rl", "models/classic-rl.json"),
    "classic_rl": ("rl", "models/classic-rl.json"),
    "supervised": ("supervised", "models/classic-policy.json"),
    "policy": ("supervised", "models/classic-policy.json"),
}

# Default blend: the strongest RL checkpoint carries half of the vote, the
# previous champion and the archive-trained supervised policy split the rest.
DEFAULT_MEMBERS: tuple[tuple[str, str, str, float, float], ...] = (
    ("lu_luna", "rl", "models/lu_luna.json", 0.5, 1.0),
    ("classic-rl", "rl", "models/classic-rl.json", 0.3, 1.0),
    ("supervised", "supervised", "models/classic-policy.json", 0.2, 1.0),
)

DEFAULT_ENSEMBLE_PATH = "models/lu_nsmbl.json"


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0:
        return np.full(values.shape, 1.0 / values.size)
    return exp / total


@dataclass
class EnsembleMember:
    """One voting member: which checkpoint, how loud, how confident."""

    name: str
    kind: str
    path: str
    weight: float = 1.0
    temperature: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "weight": self.weight,
            "temperature": self.temperature,
        }


class _RLMember:
    kind = "rl"

    def __init__(self, path: str):
        from .rl import load_rl

        self.net = load_rl(path)

    def raw_scores(self, obs: Observation) -> np.ndarray:
        from .rl import _pad_observations, forward_batch

        states, actions, mask, _ = _pad_observations([obs], self.net.feature_version)
        scores, _, _ = forward_batch(self.net, states, actions, mask)
        return np.asarray(scores[0, : len(obs.legal)], dtype=np.float64)


class _SupervisedMember:
    kind = "supervised"

    def __init__(self, path: str):
        from .policy import load_policy

        self.model = load_policy(path)
        self._adapter = SupervisedPolicy(self.model)

    def raw_scores(self, obs: Observation) -> np.ndarray:
        state = self._adapter._state(obs)
        return np.asarray(
            [
                self.model.predict_action(
                    state,
                    PolicyAction(
                        card_id=action.card_id,
                        is_closed=action.is_closed,
                        hand_count=state.hand_counts.get(action.card_id, 0),
                    ),
                )
                for action in obs.legal
            ],
            dtype=np.float64,
        )


def build_member(member: EnsembleMember):
    if member.kind == "rl":
        return _RLMember(member.path)
    if member.kind == "supervised":
        return _SupervisedMember(member.path)
    raise ValueError(f"Неизвестный тип участника ансамбля: {member.kind!r}")


def resolve_member(token: str) -> tuple[str, str]:
    """Turn ``lu_luna`` / ``classic-rl`` / ``rl:path`` into ``(kind, path)``."""
    key = token.strip()
    lowered = key.lower()
    if lowered in MEMBER_ALIASES:
        return MEMBER_ALIASES[lowered]
    if ":" in key:
        kind, _, path = key.partition(":")
        kind = kind.strip().lower()
        path = path.strip()
        if kind not in MEMBER_KINDS:
            raise ValueError(
                f"Неизвестный тип участника: {kind!r}; доступны {', '.join(MEMBER_KINDS)}"
            )
        if not path:
            raise ValueError(f"Не указан checkpoint для участника {token!r}")
        return kind, path
    if lowered in MEMBER_KINDS:
        # A bare kind needs an explicit checkpoint, except supervised has a
        # conventional default.
        if lowered == "supervised":
            return lowered, "models/classic-policy.json"
    raise ValueError(
        f"Неизвестный участник: {token!r}; доступны псевдонимы "
        f"{', '.join(sorted(MEMBER_ALIASES))} или kind:path"
    )


def members_from_tokens(
    tokens: Sequence[str],
    weights: Sequence[float] | None = None,
    temperatures: Sequence[float] | None = None,
) -> list[EnsembleMember]:
    if weights is not None and len(weights) != len(tokens):
        raise ValueError("Число весов должно совпадать с числом участников")
    if temperatures is not None and len(temperatures) != len(tokens):
        raise ValueError("Число температур должно совпадать с числом участников")
    default_weights = {name: weight for name, _, _, weight, _ in DEFAULT_MEMBERS}
    default_temps = {name: temp for name, _, _, _, temp in DEFAULT_MEMBERS}
    members: list[EnsembleMember] = []
    for index, token in enumerate(tokens):
        kind, path = resolve_member(token)
        name = token.strip()
        weight = (
            default_weights.get(name, 1.0) if weights is None else float(weights[index])
        )
        temperature = (
            default_temps.get(name, 1.0)
            if temperatures is None
            else float(temperatures[index])
        )
        if weight < 0.0:
            raise ValueError("Вес участника не может быть отрицательным")
        if temperature <= 0.0:
            raise ValueError("Температура участника должна быть положительной")
        members.append(
            EnsembleMember(
                name=name,
                kind=kind,
                path=path,
                weight=weight,
                temperature=temperature,
            )
        )
    if not members:
        raise ValueError("Нужен хотя бы один участник ансамбля")
    return members


def default_members() -> list[EnsembleMember]:
    return [
        EnsembleMember(name=name, kind=kind, path=path, weight=weight, temperature=temp)
        for name, kind, path, weight, temp in DEFAULT_MEMBERS
    ]


class EnsemblePolicy:
    """Callable policy that plays the highest-probability blended action."""

    def __init__(
        self,
        members: Sequence[EnsembleMember],
        *,
        seed: int = 0,
        name: str = "ensemble",
        scorers: Sequence[object] | None = None,
    ):
        if not members:
            raise ValueError("Ансамбль без участников")
        self.members = list(members)
        self.name = name
        self.seed = seed
        self.rng = random.Random(seed)
        self._scorers = list(scorers) if scorers is not None else None

    @property
    def scorers(self) -> list[object]:
        if self._scorers is None:
            self._scorers = [build_member(member) for member in self.members]
        return self._scorers

    def probabilities(self, obs: Observation) -> tuple[list[Action], np.ndarray]:
        legal = list(obs.legal)
        blended = np.zeros(len(legal), dtype=np.float64)
        weight_sum = 0.0
        for member, scorer in zip(self.members, self.scorers):
            raw = scorer.raw_scores(obs)
            probability = _softmax(raw / max(1e-6, member.temperature))
            blended += member.weight * probability
            weight_sum += member.weight
        if weight_sum > 0.0:
            blended /= weight_sum
        return legal, blended

    def member_probabilities(
        self, obs: Observation
    ) -> list[tuple[EnsembleMember, np.ndarray]]:
        result: list[tuple[EnsembleMember, np.ndarray]] = []
        for member, scorer in zip(self.members, self.scorers):
            raw = scorer.raw_scores(obs)
            result.append(
                (member, _softmax(raw / max(1e-6, member.temperature)))
            )
        return result

    def __call__(self, obs: Observation) -> Action:
        legal, blended = self.probabilities(obs)
        best = float(np.max(blended))
        winners = [index for index, value in enumerate(blended) if value >= best - 1e-9]
        return legal[self.rng.choice(winners)]

    def to_dict(self) -> dict:
        return {
            "format": ENSEMBLE_FORMAT,
            "name": self.name,
            "seed": self.seed,
            "members": [member.to_dict() for member in self.members],
            "description": "Weighted soft-vote blend of trained 8FM policies.",
            "combining_rule": "P(a) = sum_i w_i * softmax(score_i / T_i) / sum_i w_i",
        }


def save_ensemble(path: str | Path, policy: EnsemblePolicy) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(policy.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_ensemble(path: str | Path) -> EnsemblePolicy:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("format") != ENSEMBLE_FORMAT:
        raise ValueError(f"Не ансамблевый файл: {path}")
    members = [
        EnsembleMember(
            name=entry["name"],
            kind=entry["kind"],
            path=entry["path"],
            weight=float(entry.get("weight", 1.0)),
            temperature=float(entry.get("temperature", 1.0)),
        )
        for entry in data.get("members", [])
    ]
    return EnsemblePolicy(
        members,
        seed=int(data.get("seed", 0)),
        name=str(data.get("name", "ensemble")),
    )


def default_ensemble(seed: int = 0, *, name: str = "ensemble") -> EnsemblePolicy:
    return EnsemblePolicy(default_members(), seed=seed, name=name)
