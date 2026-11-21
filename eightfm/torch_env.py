"""Batched tensor simulator for 8ФМ Classic (GPU training backend).

This is a vectorised twin of :mod:`eightfm.env`.  It plays many games at once
with PyTorch tensors and reproduces the NumPy simulator's rules, scoring and
feature layout.  The NumPy implementation stays the reference (the desktop and
Telegram apps use it); this module exists so ``lu_sol`` can be trained on
thousands of simultaneous games on a GPU.

Design
------
* one batch is one player count (3–7); games of different sizes are grouped;
* every game in a batch is in lockstep (round 1 turns 1–5, round 2 turns
  1–5), so round/turn are plain integers;
* a hand is a ``(B, N, 14)`` count tensor, a play history is an open-count
  tensor plus a closed-count vector;
* an action is ``card_index * 2 + closed_flag`` (28 actions for 14 cards);
* scoring is a closed-form function of the open/closed counts, exactly the
  formulas from :func:`eightfm.rules.score_player`.

``tests/test_torch_env.py`` checks feature and dynamics parity against
:class:`eightfm.env.ClassicEnv` on random positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .env import COMBO_THRESHOLDS, LATEST_FEATURE_VERSION, SUPPORTED_FEATURE_VERSIONS
from .model import CARD_IDS
from .rules import ALEXEY_TOTALS, CLOSED_POINTS, DECK_COUNTS, deck_for_players


CARD_INDEX = {card_id: index for index, card_id in enumerate(CARD_IDS)}
NUM_CARDS = len(CARD_IDS)
ACTION_SIZE = NUM_CARDS * 2
SUPPORTED_PLAYER_COUNTS = tuple(sorted(DECK_COUNTS))

ALEXEY = CARD_INDEX["alaksiej"]
OLYMPIAD = CARD_INDEX["olympiad"]
EVGEN = CARD_INDEX["evgen"]
SERGAY = CARD_INDEX["sergay"]
JAMAN = CARD_INDEX["jaman"]
VANYA = CARD_INDEX["vanya"]
AMOGUS = CARD_INDEX["amogus"]
MARAT = CARD_INDEX["marat"]
AKIM = CARD_INDEX["akim"]
LEONIDIA = CARD_INDEX["leonidia"]
VEEEEP = CARD_INDEX["veeeep"]
DETECTOR = CARD_INDEX["detector"]
ALEXANDR = CARD_INDEX["alexandr"]
ULTRAMAX = CARD_INDEX["ultramax"]

_ALEXEY_TABLE_CACHE: dict[torch.device, torch.Tensor] = {}


def _alexey_table(device: torch.device) -> torch.Tensor:
    key = torch.device(device)
    table = _ALEXEY_TABLE_CACHE.get(key)
    if table is None:
        table = torch.tensor(ALEXEY_TOTALS, dtype=torch.float32, device=key)
        _ALEXEY_TABLE_CACHE[key] = table
    return table


def _norm(value: torch.Tensor, maximum: float | torch.Tensor) -> torch.Tensor:
    """``min(max(value, 0), maximum) / maximum`` (the NumPy ``_norm``)."""
    if isinstance(maximum, torch.Tensor):
        maximum = maximum.to(value.dtype)
        return torch.minimum(value.clamp(min=0.0), maximum) / maximum
    maximum = float(maximum)
    if maximum <= 0.0:
        return torch.zeros_like(value)
    return value.clamp(min=0.0, max=maximum) / maximum


def score_batched(
    open_counts: torch.Tensor,
    closed_counts: torch.Tensor,
    players_with_alexey: torch.Tensor,
) -> torch.Tensor:
    """Vectorised :func:`eightfm.rules.score_player`.

    ``open_counts`` is ``(..., C)``, ``closed_counts`` broadcasts against its
    leading dimensions and ``players_with_alexey`` is the global number of
    players with an open Alexey.
    """
    dtype = open_counts.dtype
    device = open_counts.device
    closed = closed_counts.to(dtype)
    pwa = players_with_alexey.to(dtype)

    def n(card: int) -> torch.Tensor:
        return open_counts[..., card]

    alexey = n(ALEXEY)
    table = _alexey_table(device).to(dtype)
    alexey_points = table[alexey.clamp(max=len(ALEXEY_TOTALS) - 1).long()]
    total = CLOSED_POINTS * closed + alexey_points

    olympiad = n(OLYMPIAD)
    per_olympiad = torch.where(
        olympiad >= 6,
        torch.full_like(olympiad, 6.0),
        torch.where(
            olympiad >= 3, torch.full_like(olympiad, 4.0), torch.full_like(olympiad, 2.0)
        ),
    )
    total = total + olympiad * per_olympiad

    evgen = n(EVGEN)
    amogus = n(AMOGUS)
    sergay = n(SERGAY)
    akim = n(AKIM)
    marat = n(MARAT)
    jaman = n(JAMAN)
    vanya = n(VANYA)
    leonidia = n(LEONIDIA)
    veeep = n(VEEEEP)
    detector = n(DETECTOR)
    alexandr = n(ALEXANDR)
    ultramax = n(ULTRAMAX)

    total = total + evgen * (2.0 + 3.0 * amogus + detector)
    amogus_points = torch.where(
        amogus == 2, torch.full_like(amogus, 6.0), torch.full_like(amogus, 3.0)
    )
    total = total + amogus * amogus_points
    total = total + sergay * (torch.where(closed > 0, 3.0, 1.0) + olympiad)
    total = total + akim * (3.0 + 2.0 * sergay)
    marat_points = torch.where(
        marat == 1, torch.full_like(marat, 6.0), torch.full_like(marat, 4.0)
    )
    total = total + marat * (marat_points + olympiad)

    jaman_closed = closed.clamp(max=5.0)
    total = total + jaman * (
        (evgen > 0).to(dtype)
        + evgen
        + sergay
        + akim
        + ultramax
        + 2.0 * jaman_closed
    )
    total = total + vanya * (5.0 + 3.0 * akim)

    types = (open_counts > 0).to(dtype).sum(-1) + (closed > 0).to(dtype)
    total = total + leonidia * (2.0 + types - 2.0 * alexey)
    veeep_points = torch.where(
        veeep == 1, torch.full_like(veeep, 3.0), torch.full_like(veeep, 1.0)
    )
    total = total + veeep * (
        veeep_points + 3.0 * (marat + leonidia + detector) - 2.0 * alexey
    )

    while pwa.dim() < detector.dim():
        pwa = pwa.unsqueeze(-1)
    total = total + detector * (4.0 + pwa)

    alexandr_bonus = evgen + ultramax + sergay + marat
    alexandr_penalty = torch.where(
        types >= 6, torch.full_like(types, 2.0), torch.zeros_like(types)
    )
    total = total + alexandr * (4.0 + alexandr_bonus - alexandr_penalty)

    pair = torch.minimum(evgen, alexandr)
    ultramax_points = (
        torch.where(amogus == 0, torch.full_like(amogus, 3.0), torch.zeros_like(amogus))
        + torch.where(types >= 6, torch.full_like(types, 2.0), torch.zeros_like(types))
        + akim
    )
    total = total + ultramax * ultramax_points
    total = total + 4.0 * pair * ultramax
    return total


@dataclass
class BatchedObservation:
    state_features: torch.Tensor
    action_features: torch.Tensor
    legal: torch.Tensor
    open_counts: torch.Tensor
    closed_counts: torch.Tensor
    hand: torch.Tensor
    scores: torch.Tensor
    unseen: torch.Tensor
    table_open: torch.Tensor
    delta: torch.Tensor
    add_alex: torch.Tensor
    player_count: int
    round_number: int
    turn_number: int

    @property
    def batch_size(self) -> int:
        return int(self.legal.shape[0])


class BatchedClassicEnv:
    """Vectorised two-round Classic game for ``batch_size`` simultaneous games."""

    def __init__(
        self,
        player_count: int,
        batch_size: int,
        *,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if player_count not in DECK_COUNTS:
            raise ValueError(f"Unsupported player count: {player_count}")
        self.player_count = int(player_count)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.deck = torch.tensor(
            [CARD_INDEX[card] for card in deck_for_players(self.player_count)],
            dtype=torch.long,
            device=self.device,
        )
        self.deck_counts = torch.tensor(
            [DECK_COUNTS[self.player_count].get(card, 0) for card in CARD_IDS],
            dtype=self.dtype,
            device=self.device,
        )
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def _counts(self, cards: torch.Tensor) -> torch.Tensor:
        return F.one_hot(cards, NUM_CARDS).to(self.dtype).sum(dim=2)

    def reset(self) -> BatchedObservation:
        b = self.batch_size
        n = self.player_count
        deck = self.deck.unsqueeze(0).expand(b, -1)
        noise = torch.rand(
            b, deck.shape[1], device=self.device, generator=self.generator, dtype=self.dtype
        )
        order = torch.argsort(noise, dim=1)
        shuffled = deck.gather(1, order)
        half = 6 * n
        first = shuffled[:, :half].reshape(b, n, 6)
        second = shuffled[:, half:].reshape(b, n, 6)
        self.hand = self._counts(first)
        self.round2_hand = self._counts(second)
        self.open = torch.zeros(b, n, NUM_CARDS, dtype=self.dtype, device=self.device)
        self.closed = torch.zeros(b, n, dtype=self.dtype, device=self.device)
        self.scores = torch.zeros(b, n, dtype=self.dtype, device=self.device)
        self.closed_total = torch.zeros(b, dtype=self.dtype, device=self.device)
        self.round_number = 1
        self.turn_number = 1
        self.done = False
        return self.observation()

    # -- moves -------------------------------------------------------------

    def step(self, action_indices: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if self.done:
            raise RuntimeError("The game is already finished")
        b, n = self.batch_size, self.player_count
        actions = action_indices.to(device=self.device, dtype=torch.long)
        if tuple(actions.shape) != (b, n):
            raise ValueError("One action per player per game is required")
        card = actions // 2
        is_closed = (actions % 2) == 1
        old_scores = self.scores
        minus_one = torch.full((b, n, 1), -1.0, dtype=self.dtype, device=self.device)
        self.hand.scatter_add_(-1, card.unsqueeze(-1), minus_one)
        open_add = (~is_closed).to(self.dtype).unsqueeze(-1)
        self.open.scatter_add_(-1, card.unsqueeze(-1), open_add)
        self.closed = self.closed + is_closed.to(self.dtype)
        self.closed_total = self.closed_total + is_closed.to(self.dtype).sum(dim=1)
        players_with_alexey = (self.open[..., ALEXEY] > 0).sum(dim=1)
        self.scores = score_batched(self.open, self.closed, players_with_alexey)
        deltas = self.scores - old_scores
        mean_delta = deltas.mean(dim=1, keepdim=True)
        rewards = (0.08 * (deltas - mean_delta)).clamp(-1.0, 1.0)

        if self.turn_number < 5:
            offset = -1 if self.round_number == 1 else 1
            self.hand = torch.roll(self.hand, shifts=offset, dims=1)
            self.turn_number += 1
        elif self.round_number == 1:
            self.round_number = 2
            self.turn_number = 1
            self.hand = self.round2_hand.clone()
        else:
            self.done = True
        return rewards, self.done

    # -- scoring helpers ---------------------------------------------------

    def final_ranks(self) -> torch.Tensor:
        scores = self.scores
        greater = scores.unsqueeze(-2) > scores.unsqueeze(-1)
        return 1 + greater.sum(dim=-1)

    def terminal_rewards(
        self,
        *,
        success_threshold: float = 60.0,
        win_weight: float = 1.5,
        placement_weight: float = 0.7,
        score_weight: float = 0.4,
        success_weight: float = 0.4,
        margin_weight: float = 0.2,
    ) -> torch.Tensor:
        n = self.player_count
        scores = self.scores
        ranks = self.final_ranks()
        winners = (ranks == 1).sum(dim=1, keepdim=True)
        win = torch.where(
            ranks == 1, 1.0 / winners.to(self.dtype), torch.zeros_like(scores)
        )
        placement = (n - ranks).to(self.dtype) / max(1, n - 1)
        score_norm = (scores / 100.0).clamp(0.0, 2.0)
        success = (scores > success_threshold).to(self.dtype)
        mean_other = (scores.sum(dim=1, keepdim=True) - scores) / (n - 1)
        margin = ((scores - mean_other) / 100.0).clamp(-1.0, 1.0)
        return (
            win_weight * win
            + placement_weight * placement
            + score_weight * score_norm
            + success_weight * success
            + margin_weight * margin
        )

    # -- observations ------------------------------------------------------

    def _base(self) -> dict[str, Any]:
        b, n, c = self.open.shape
        device = self.device
        dtype = self.dtype
        open_counts = self.open
        closed = self.closed
        hand = self.hand
        scores = self.scores

        table_open = open_counts.sum(dim=1, keepdim=True) - open_counts
        table_closed = closed.sum(dim=1, keepdim=True) - closed
        seen = open_counts.sum(dim=1, keepdim=True)
        unseen = (self.deck_counts.view(1, 1, c) - seen - hand).clamp(min=0.0)
        alexey_players = open_counts[..., ALEXEY] > 0
        players_with_alexey = alexey_players.sum(dim=1)

        eye_n = torch.eye(n, dtype=torch.bool, device=device)
        other_sum = scores.sum(dim=1, keepdim=True) - scores
        mean_other = other_sum / (n - 1)
        scores_exp = scores.unsqueeze(1).expand(b, n, n)
        max_other = scores_exp.masked_fill(eye_n, -1.0).max(dim=-1).values
        min_other = scores_exp.masked_fill(eye_n, 1e9).min(dim=-1).values
        rank = (scores.unsqueeze(-2) > scores.unsqueeze(-1)).sum(dim=-1).to(dtype)
        rank_score = 1.0 - rank / (n - 1)
        max_all = torch.maximum(scores, max_other)
        min_all = torch.minimum(scores, min_other)

        offset = -1 if self.round_number == 1 else 1
        index = torch.arange(n, device=device)
        supplier = (index - offset) % n
        receiver = (index + offset) % n

        own_alexey = open_counts[..., ALEXEY] > 0
        add_alex = (~own_alexey).to(dtype)
        pwa_col = players_with_alexey.to(dtype).view(b, 1)
        after_closed = score_batched(open_counts, closed + 1.0, pwa_col)
        eye_c = torch.eye(c, dtype=dtype, device=device)
        open_variants = open_counts.unsqueeze(-2) + eye_c
        alex_mask = (torch.arange(c, device=device) == ALEXEY).to(dtype)
        pwa_variants = pwa_col.unsqueeze(-1) + add_alex.unsqueeze(-1) * alex_mask.view(1, 1, c)
        after_open = score_batched(
            open_variants, closed.unsqueeze(-1), pwa_variants
        )
        delta_open = after_open - scores.unsqueeze(-1)
        delta_closed = after_closed - scores
        delta = torch.empty(b, n, ACTION_SIZE, dtype=dtype, device=device)
        delta[..., 0::2] = delta_open
        delta[..., 1::2] = delta_closed.unsqueeze(-1)

        card_of_action = torch.arange(ACTION_SIZE, device=device) // 2
        is_open_action = torch.arange(ACTION_SIZE, device=device) % 2 == 0
        is_alex_open = is_open_action & (card_of_action == ALEXEY)
        add_alex_all = add_alex.unsqueeze(-1) * is_alex_open.to(dtype).view(1, 1, ACTION_SIZE)

        has = hand > 0
        blocked = torch.zeros(b, n, c, dtype=torch.bool, device=device)
        blocked[..., MARAT] = own_alexey
        blocked[..., DETECTOR] = own_alexey
        blocked[..., VANYA] = ~(closed > 0)
        open_legal = has & ~blocked
        legal = torch.zeros(b, n, ACTION_SIZE, dtype=torch.bool, device=device)
        legal[..., 0::2] = open_legal
        legal[..., 1::2] = has

        return {
            "b": b,
            "n": n,
            "c": c,
            "open_counts": open_counts,
            "closed": closed,
            "hand": hand,
            "scores": scores,
            "table_open": table_open,
            "table_closed": table_closed,
            "unseen": unseen,
            "alexey_players": alexey_players,
            "players_with_alexey": players_with_alexey,
            "mean_other": mean_other,
            "max_other": max_other,
            "min_other": min_other,
            "rank_score": rank_score,
            "max_all": max_all,
            "min_all": min_all,
            "supplier": supplier,
            "receiver": receiver,
            "delta": delta,
            "add_alex": add_alex,
            "add_alex_all": add_alex_all,
            "card_of_action": card_of_action,
            "is_open_action": is_open_action,
            "legal": legal,
            "eye_n": eye_n,
        }

    def _state_features(self, base: dict[str, Any], version: int) -> torch.Tensor:
        b, n, c = base["b"], base["n"], base["c"]
        device = self.device
        dtype = self.dtype
        open_counts = base["open_counts"]
        closed = base["closed"]
        hand = base["hand"]
        scores = base["scores"]
        table_open = base["table_open"]
        table_closed = base["table_closed"]
        unseen = base["unseen"]
        table_max = max(1.0, 5.0 * (n - 1))

        players = torch.tensor(
            [float(n == count) for count in SUPPORTED_PLAYER_COUNTS],
            dtype=dtype,
            device=device,
        ).view(1, 1, -1).expand(b, n, -1)
        round_number = torch.tensor(
            [float(self.round_number == number) for number in (1, 2)],
            dtype=dtype,
            device=device,
        ).view(1, 1, -1).expand(b, n, -1)
        turn = torch.tensor(
            [float(self.turn_number == number) for number in (1, 2, 3, 4, 5)],
            dtype=dtype,
            device=device,
        ).view(1, 1, -1).expand(b, n, -1)

        supplier = base["supplier"]
        receiver = base["receiver"]
        supplier_open = open_counts[:, supplier, :]
        supplier_closed = closed[:, supplier]
        supplier_score = scores[:, supplier]
        receiver_open = open_counts[:, receiver, :]
        receiver_closed = closed[:, receiver]
        receiver_score = scores[:, receiver]
        closed_total = self.closed_total.view(b, 1).expand(b, n)
        pwa = base["players_with_alexey"].to(dtype).view(b, 1).expand(b, n)

        score_block = torch.stack(
            (
                _norm(scores, 150.0),
                _norm(base["mean_other"], 150.0),
                _norm(base["max_other"], 150.0),
                _norm(base["min_other"], 150.0),
                base["rank_score"],
                _norm(base["max_all"] - scores, 150.0),
                _norm(scores - base["min_all"], 150.0),
                _norm(pwa, n),
                base["alexey_players"].to(dtype),
                _norm(closed_total, 5.0 * n),
            ),
            dim=-1,
        )

        parts = [
            players,
            round_number,
            turn,
            _norm(hand, 6.0),
            _norm(open_counts, 5.0),
            _norm(closed, 5.0).unsqueeze(-1),
            _norm(table_open, table_max),
            _norm(table_closed, table_max).unsqueeze(-1),
            score_block,
            _norm(unseen, float(max(1, 12 * n))),
            _norm(unseen.sum(dim=-1), float(max(1, 12 * n))).unsqueeze(-1),
            _norm(supplier_open, 5.0),
            _norm(supplier_closed, 5.0).unsqueeze(-1),
            _norm(supplier_score, 150.0).unsqueeze(-1),
            _norm(receiver_open, 5.0),
            _norm(receiver_closed, 5.0).unsqueeze(-1),
            _norm(receiver_score, 150.0).unsqueeze(-1),
            _norm(hand.sum(dim=-1), 6.0).unsqueeze(-1),
            torch.full(
                (b, n, 1),
                (6 - self.turn_number) / 5.0,
                dtype=dtype,
                device=device,
            ),
        ]

        if version == 1:
            return torch.cat(parts, dim=-1)
        if version != 2:
            raise ValueError(f"Unsupported state feature version: {version}")

        extra: list[torch.Tensor] = []
        for card_id, threshold in COMBO_THRESHOLDS.items():
            index = CARD_INDEX[card_id]
            progress = open_counts[..., index] + hand[..., index]
            potential = progress + unseen[..., index]
            extra.append(_norm(progress, threshold).unsqueeze(-1))
            extra.append(_norm(potential, threshold).unsqueeze(-1))
            extra.append((potential >= threshold).to(dtype).unsqueeze(-1))
        extra.append(_norm(closed * CLOSED_POINTS, 20.0).unsqueeze(-1))
        extra.append(_norm(table_closed * CLOSED_POINTS, 20.0).unsqueeze(-1))
        extra.append(torch.full((b, n, 1), 1.0 if self.turn_number == 1 else 0.0, dtype=dtype, device=device))
        extra.append(torch.full((b, n, 1), 1.0 if self.turn_number >= 5 else 0.0, dtype=dtype, device=device))
        extra.append(torch.full((b, n, 1), 1.0 if self.round_number == 2 else 0.0, dtype=dtype, device=device))
        extra.append(torch.full((b, n, 1), (5 - self.turn_number) / 4.0, dtype=dtype, device=device))

        totals = open_counts.sum(dim=-1)
        totals_exp = totals.unsqueeze(1).expand(b, n, n)
        totals_masked = totals_exp.masked_fill(base["eye_n"], -1.0)
        sorted_totals = torch.sort(totals_masked, dim=-1, descending=True).values
        ranked = torch.zeros(b, n, 6, dtype=dtype, device=device)
        count = min(6, n)
        ranked[..., :count] = sorted_totals[..., :count]
        extra.append(_norm(ranked.clamp(min=0.0), 5.0))
        opponent_mean = (totals.sum(dim=1, keepdim=True) - totals) / (n - 1)
        opponent_max = totals_masked.max(dim=-1).values.clamp(min=0.0)
        extra.append(_norm(opponent_mean, 5.0).unsqueeze(-1))
        extra.append(_norm(opponent_max, 5.0).unsqueeze(-1))

        alexey_counts = open_counts[..., ALEXEY]
        table_alexey = alexey_counts.sum(dim=1, keepdim=True) - alexey_counts
        alexey_exp = alexey_counts.unsqueeze(1).expand(b, n, n)
        max_opponent_alexey = alexey_exp.masked_fill(base["eye_n"], -1.0).max(dim=-1).values.clamp(min=0.0)
        extra.append(_norm(table_alexey, 6.0).unsqueeze(-1))
        extra.append(_norm(alexey_counts, 6.0).unsqueeze(-1))
        extra.append(_norm(max_opponent_alexey, 6.0).unsqueeze(-1))
        extra.append(_norm((60.0 - scores).clamp(min=0.0), 60.0).unsqueeze(-1))
        extra.append((scores > 60.0).to(dtype).unsqueeze(-1))
        parts.append(torch.cat(extra, dim=-1))
        return torch.cat(parts, dim=-1)

    def _action_features(self, base: dict[str, Any], version: int) -> torch.Tensor:
        b, n, c = base["b"], base["n"], base["c"]
        device = self.device
        dtype = self.dtype
        open_counts = base["open_counts"]
        closed = base["closed"]
        hand = base["hand"]
        table_open = base["table_open"]
        unseen = base["unseen"]
        table_max = max(1.0, 5.0 * (n - 1))
        card_of_action = base["card_of_action"]
        is_open_action = base["is_open_action"].to(dtype)

        hand_card = hand[..., card_of_action]
        own_open_card = open_counts[..., card_of_action]
        table_open_card = table_open[..., card_of_action]
        unseen_card = unseen[..., card_of_action]
        deck_card = self.deck_counts[card_of_action].clamp(min=1.0).view(1, 1, ACTION_SIZE)

        one_hot = torch.zeros(ACTION_SIZE, c, dtype=dtype, device=device)
        one_hot[torch.arange(ACTION_SIZE, device=device), card_of_action] = 1.0
        one_hot = one_hot.view(1, 1, ACTION_SIZE, c).expand(b, n, ACTION_SIZE, c)

        delta = base["delta"]
        opponent_detector = open_counts[..., DETECTOR]
        detector_exp = opponent_detector.unsqueeze(1).expand(b, n, n)
        detector_masked = detector_exp.masked_fill(base["eye_n"], 0.0)
        mean_opponent_detector = (
            opponent_detector.sum(dim=1, keepdim=True) - opponent_detector
        ) / (n - 1)
        max_opponent_detector = detector_masked.max(dim=-1).values
        mean_opponent_delta = base["add_alex_all"] * mean_opponent_detector.unsqueeze(-1)
        max_opponent_delta = base["add_alex_all"] * max_opponent_detector.unsqueeze(-1)
        relative_delta = delta - mean_opponent_delta
        own_closed = _norm(closed, 5.0).unsqueeze(-1).expand(b, n, ACTION_SIZE)

        parts = [
            one_hot,
            (1.0 - is_open_action).view(1, 1, ACTION_SIZE, 1).expand(b, n, ACTION_SIZE, 1),
            _norm(hand_card, 6.0).unsqueeze(-1),
            _norm(own_open_card, 5.0).unsqueeze(-1),
            _norm(table_open_card, table_max).unsqueeze(-1),
            _norm(unseen_card, deck_card).unsqueeze(-1),
            (delta / 20.0).clamp(-1.0, 1.0).unsqueeze(-1),
            (hand_card == 1).to(dtype).unsqueeze(-1),
            own_closed.unsqueeze(-1),
            (mean_opponent_delta / 20.0).clamp(-1.0, 1.0).unsqueeze(-1),
            (max_opponent_delta / 20.0).clamp(-1.0, 1.0).unsqueeze(-1),
            (relative_delta / 20.0).clamp(-1.0, 1.0).unsqueeze(-1),
        ]

        if version == 1:
            return torch.cat(parts, dim=-1)
        if version != 2:
            raise ValueError(f"Unsupported action feature version: {version}")

        thresholds = torch.ones(c, dtype=dtype, device=device)
        has_threshold = torch.zeros(c, dtype=torch.bool, device=device)
        for card_id, threshold in COMBO_THRESHOLDS.items():
            index = CARD_INDEX[card_id]
            thresholds[index] = float(threshold)
            has_threshold[index] = True
        progress_card = open_counts + hand
        potential_card = progress_card + unseen
        progress_ratio = _norm(progress_card, thresholds)[..., card_of_action]
        potential_ratio = _norm(potential_card, thresholds)[..., card_of_action]
        reachable = (potential_card >= thresholds).to(dtype)[..., card_of_action]
        combo_open = (is_open_action * has_threshold[card_of_action].to(dtype)).view(1, 1, ACTION_SIZE)

        copies = open_counts[..., ALEXEY]
        table = _alexey_table(device).to(dtype)
        current = table[copies.clamp(max=len(ALEXEY_TOTALS) - 1).long()]
        following = table[(copies + 1).clamp(max=len(ALEXEY_TOTALS) - 1).long()]
        alexey_step = ((following - current) / 20.0).clamp(0.0, 1.0)
        is_alex_open = (base["is_open_action"] & (card_of_action == ALEXEY)).to(dtype)
        alexey_step = alexey_step.unsqueeze(-1) * is_alex_open.view(1, 1, ACTION_SIZE)

        parts.extend(
            (
                (has_threshold[card_of_action].to(dtype) * is_open_action).view(1, 1, ACTION_SIZE, 1).expand(b, n, ACTION_SIZE, 1),
                (progress_ratio * combo_open).unsqueeze(-1),
                (potential_ratio * combo_open).unsqueeze(-1),
                (reachable * combo_open).unsqueeze(-1),
                alexey_step.unsqueeze(-1),
                torch.full((b, n, 1, 1), 1.0 if self.turn_number >= 5 else 0.0, dtype=dtype, device=device).expand(b, n, ACTION_SIZE, 1),
                torch.full((b, n, 1, 1), 1.0 if self.round_number == 2 else 0.0, dtype=dtype, device=device).expand(b, n, ACTION_SIZE, 1),
            )
        )
        return torch.cat(parts, dim=-1)

    def observation(self, version: int = LATEST_FEATURE_VERSION) -> BatchedObservation:
        if version not in SUPPORTED_FEATURE_VERSIONS:
            raise ValueError(f"Unsupported feature version: {version}")
        base = self._base()
        return BatchedObservation(
            state_features=self._state_features(base, version),
            action_features=self._action_features(base, version),
            legal=base["legal"],
            open_counts=base["open_counts"],
            closed_counts=base["closed"],
            hand=base["hand"],
            scores=base["scores"],
            unseen=base["unseen"],
            table_open=base["table_open"],
            delta=base["delta"],
            add_alex=base["add_alex"],
            player_count=self.player_count,
            round_number=self.round_number,
            turn_number=self.turn_number,
        )

    # -- test / tooling helpers -------------------------------------------

    def load_from_classic(self, env: Any) -> None:
        """Copy a single :class:`eightfm.env.ClassicEnv` position into batch 0."""
        if self.batch_size != 1:
            raise ValueError("load_from_classic needs batch_size=1")
        n = self.player_count
        if env.player_count != n:
            raise ValueError("Player count mismatch")
        open_counts = torch.zeros(1, n, NUM_CARDS, dtype=self.dtype)
        closed_counts = torch.zeros(1, n, dtype=self.dtype)
        hand_counts = torch.zeros(1, n, NUM_CARDS, dtype=self.dtype)
        for player in range(n):
            for play in env.played[player]:
                if play.is_closed:
                    closed_counts[0, player] += 1.0
                else:
                    open_counts[0, player, CARD_INDEX[play.card_id]] += 1.0
            for card_id in env.hands[player]:
                hand_counts[0, player, CARD_INDEX[card_id]] += 1.0
        self.open = open_counts.to(self.device)
        self.closed = closed_counts.to(self.device)
        self.hand = hand_counts.to(self.device)
        self.scores = torch.tensor([env.scores], dtype=self.dtype, device=self.device)
        self.closed_total = torch.tensor(
            [float(env.closed_total)], dtype=self.dtype, device=self.device
        )
        self.round_number = int(env.round_number)
        self.turn_number = int(env.turn_number)
        self.done = bool(env.done)
        second = env.round_decks[1]
        round2 = torch.zeros(1, n, NUM_CARDS, dtype=self.dtype)
        for player in range(n):
            for card_id in second[player * 6:(player + 1) * 6]:
                round2[0, player, CARD_INDEX[card_id]] += 1.0
        self.round2_hand = round2.to(self.device)
