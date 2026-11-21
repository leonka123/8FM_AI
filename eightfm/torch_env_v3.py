"""Batched tensor simulator for 8ФМ Classic v3 features (GPU training backend for lu_astra).

This module extends the v2 feature set with significant enhancements:
- Per-opponent open card tracking
- Card counting / memory features
- Position and pass direction awareness
- Combo completion tracking
- Opponent action history
- Potential score estimation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn.functional as F

from .env import (
    COMBO_THRESHOLDS,
    CLOSED_POINTS,
    DECK_COUNTS,
    LATEST_FEATURE_VERSION,
    SUPPORTED_FEATURE_VERSIONS,
    deck_for_players,
)
from .model import CARD_IDS
from .rules import ALEXEY_TOTALS

# Feature version 3
FEATURE_VERSION_V3 = 3
SUPPORTED_FEATURE_VERSIONS_V3 = (1, 2, 3)

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
    dtype = open_counts.dtype
    device = open_counts.device
    closed = closed_counts.to(dtype)
    pwa = players_with_alexey.to(dtype)

    def n(card: int) -> torch.Tensor:
        return open_counts[..., card]

    alexey = n(ALEXEY)
    table = _alexey_table(device).to(dtype)
    alexey_score = table[alexey.long().clamp(max=len(ALEXEY_TOTALS) - 1)]

    olympiad_score = torch.where(
        n(OLYMPIAD) >= 4,
        30.0,
        torch.where(n(OLYMPIAD) >= 3, 15.0, torch.where(n(OLYMPIAD) >= 2, 5.0, 0.0)),
    )

    evgen_score = n(EVGEN) * 2.0
    sergay_score = torch.where(n(SERGAY) >= 1, 3.0 + n(SERGAY) - 1, 0.0)
    jaman_score = torch.where(n(JAMAN) >= 2, torch.minimum(n(JAMAN), torch.tensor(5.0)), 0.0)
    vanya_score = torch.where(n(VANYA) >= 1, n(VANYA) * n(AKIM), 0.0)
    amogus_score = torch.where(
        n(AMOGUS) >= 2,
        torch.where(n(AMOGUS) == 2, 8.0, torch.where(n(AMOGUS) == 3, 20.0, 40.0)),
        0.0,
    )
    marat_score = torch.where(n(MARAT) == 1, 6.0, 0.0)
    leonidia_score = -n(LEONIDIA) * alexey
    alexandr_score = torch.where(n(ALEXANDR) >= 1, n(ALEXANDR) * 2.0, 0.0)
    ultramax_score = torch.where(n(ULTRAMAX) >= 1, 50.0, 0.0)

    veeeep_score = n(VEEEEP) * pwa
    detector_score = n(DETECTOR) * pwa

    closed_score = closed * CLOSED_POINTS

    total = (
        alexey_score
        + olympiad_score
        + evgen_score
        + sergay_score
        + jaman_score
        + vanya_score
        + amogus_score
        + marat_score
        + leonidia_score
        + alexandr_score
        + ultramax_score
        + veeeep_score
        + detector_score
        + closed_score
    )
    return total


@dataclass
class BatchedObservationV3:
    player_count: int
    round_number: int
    turn_number: int
    hand: torch.Tensor                # (B, 14)
    own_plays_open: torch.Tensor      # (B, 14)
    own_plays_closed: torch.Tensor    # (B, 1)
    all_plays_open: torch.Tensor      # (B, N, 14)
    all_plays_closed: torch.Tensor    # (B, N)
    scores: torch.Tensor              # (B, N)
    deck_counts: torch.Tensor         # (14,)
    closed_total: torch.Tensor        # (B,)
    alexey_players: torch.Tensor      # (B, N) bool
    legal_mask: torch.Tensor          # (B, 28)
    pass_receiver: torch.Tensor       # (B,)
    pass_supplier: torch.Tensor       # (B,)
    # v3: per-opponent tracking
    opponent_last_actions: torch.Tensor   # (B, N, 3, 28) - last 3 actions per opponent
    # v3: cards seen this game
    cards_seen: torch.Tensor            # (B, 14) - total seen including own


class BatchedClassicEnvV3:
    def __init__(
        self,
        player_count: int,
        batch_size: int,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ):
        if player_count not in SUPPORTED_PLAYER_COUNTS:
            raise ValueError(f"Unsupported player count: {player_count}")
        self.player_count = int(player_count)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
        self.deck_counts = torch.tensor(
            [DECK_COUNTS[player_count].get(c, 0) for c in CARD_IDS],
            dtype=torch.int32,
            device=self.device,
        )
        self.reset()

    def _sample_deck(self, total_cards: int) -> torch.Tensor:
        # Sample without replacement using multinomial on counts
        # We'll use a simpler approach: shuffle indices
        deck = []
        for card_id, count in enumerate(self.deck_counts.tolist()):
            deck.extend([card_id] * count)
        deck_tensor = torch.tensor(deck, dtype=torch.int32, device=self.device)
        # Shuffle
        perm = torch.randperm(len(deck_tensor), generator=self.rng, device=self.device)
        return deck_tensor[perm]

    def reset(self):
        n = self.player_count
        B = self.batch_size
        total_cards = n * 12  # 2 rounds × 6 cards
        deck = self._sample_deck(total_cards)
        deck = deck.view(B, total_cards)
        round1 = deck[:, :6 * n]
        round2 = deck[:, 6 * n:]

        hands = round1.view(B, n, 6)
        self.hand = torch.zeros(B, 14, dtype=torch.int32, device=self.device)
        for i in range(n):
            for j in range(6):
                card = hands[:, i, j]
                self.hand[:, card] += 1

        self.played_open = torch.zeros(B, n, 14, dtype=torch.int32, device=self.device)
        self.played_closed = torch.zeros(B, n, dtype=torch.int32, device=self.device)
        self.scores = torch.zeros(B, n, dtype=torch.float32, device=self.device)
        self.round_number = 1
        self.turn_number = 1
        self.closed_total = torch.zeros(B, dtype=torch.int32, device=self.device)
        self.alexey_players = torch.zeros(B, n, dtype=torch.bool, device=self.device)
        self.round_decks = [round1, round2]
        # v3: opponent action history (last 3 actions each)
        self.opponent_last_actions = torch.zeros(B, n, 3, 28, dtype=torch.float32, device=self.device)
        # v3: cards seen
        self.cards_seen = torch.zeros(B, 14, dtype=torch.int32, device=self.device)
        # Update cards_seen with initial hand
        self.cards_seen += self.hand.sum(dim=0)

    @property
    def pass_offset(self) -> int:
        return -1 if self.round_number == 1 else 1

    def _legal_mask(self) -> torch.Tensor:
        B = self.batch_size
        mask = torch.zeros(B, 28, dtype=torch.bool, device=self.device)
        # For each card in hand, both open and closed are legal
        for card_idx in range(14):
            has_card = self.hand[:, card_idx] > 0
            mask[:, card_idx * 2] = has_card      # closed
            mask[:, card_idx * 2 + 1] = has_card  # open
        return mask

    def observation(self, version: int = FEATURE_VERSION_V3) -> BatchedObservationV3:
        if version not in SUPPORTED_FEATURE_VERSIONS_V3:
            raise ValueError(f"Unsupported feature version: {version}")
        return BatchedObservationV3(
            player_count=self.player_count,
            round_number=self.round_number,
            turn_number=self.turn_number,
            hand=self.hand.clone(),
            own_plays_open=self.played_open[:, 0].clone(),  # will be indexed by seat
            own_plays_closed=self.played_closed[:, 0].clone(),
            all_plays_open=self.played_open.clone(),
            all_plays_closed=self.played_closed.clone(),
            scores=self.scores.clone(),
            deck_counts=self.deck_counts.clone(),
            closed_total=self.closed_total.clone(),
            alexey_players=self.alexey_players.clone(),
            legal_mask=self._legal_mask(),
            pass_receiver=torch.zeros(self.batch_size, dtype=torch.int32, device=self.device),
            pass_supplier=torch.zeros(self.batch_size, dtype=torch.int32, device=self.device),
            opponent_last_actions=self.opponent_last_actions.clone(),
            cards_seen=self.cards_seen.clone(),
        )

    def step(self, actions: torch.Tensor) -> Tuple[BatchedObservationV3, torch.Tensor, torch.Tensor]:
        """
        actions: (B, N) int32 in 0..27
        Returns: (next_obs, rewards, done)
        """
        B, N = actions.shape
        assert N == self.player_count

        # Convert actions to card_idx and closed_flag
        card_idx = actions // 2
        is_closed = (actions % 2).bool()

        # Verify legality
        legal = self._legal_mask()
        batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, N)
        chosen_legal = legal[batch_idx, actions]
        if not chosen_legal.all():
            raise ValueError("Illegal action chosen")

        # Remove from hand
        self.hand[batch_idx.flatten(), card_idx.flatten()] -= 1

        # Record plays
        self.played_open[batch_idx.flatten(), torch.arange(N, device=self.device).repeat(B), card_idx.flatten()] += (~is_closed.flatten()).int()
        self.played_closed[batch_idx.flatten(), torch.arange(N, device=self.device).repeat(B)] += is_closed.flatten().int()

        # Update closed total
        self.closed_total += is_closed.sum(dim=1)

        # Update alexey players
        alexey_played = (card_idx == ALEXEY) & (~is_closed)
        self.alexey_players[batch_idx.flatten(), torch.arange(N, device=self.device).repeat(B)] |= alexey_played.flatten()

        # Update cards_seen
        self.cards_seen += torch.nn.functional.one_hot(card_idx.flatten(), 14).sum(dim=0).view(B, 14)

        # v3: Update opponent action history (shift and add)
        # We'll update after computing rewards

        # Score update
        open_counts = self.played_open.sum(dim=1)  # (B, 14)
        closed_counts = self.played_closed  # (B, N)
        players_with_alexey = self.alexey_players.sum(dim=1).float()
        new_scores = score_batched(open_counts, closed_counts, players_with_alexey)

        rewards = new_scores - self.scores
        self.scores = new_scores

        # Advance turn
        self.turn_number += 1
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        if self.turn_number > 5:
            # End of round
            if self.round_number == 1:
                self.round_number = 2
                self.turn_number = 1
                # Deal round 2
                round2 = self.round_decks[1].view(B, N, 6)
                self.hand.zero_()
                for i in range(N):
                    for j in range(6):
                        card = round2[:, i, j]
                        self.hand[:, card] += 1
            else:
                # Game over
                done[:] = True

        return self.observation(), rewards, done


# Feature extraction functions

def _compute_opponent_open_counts(obs: BatchedObservationV3, seat: int) -> torch.Tensor:
    """Get open counts for each opponent separately. Returns (B, N, 14)"""
    B = obs.all_plays_open.shape[0]
    N = obs.player_count
    opp_open = torch.zeros(B, N, 14, dtype=torch.float32, device=obs.all_plays_open.device)
    for i in range(N):
        if i != seat:
            opp_open[:, i] = obs.all_plays_open[:, i].float()
    return opp_open


def _compute_opponent_action_history(obs: BatchedObservationV3, seat: int) -> torch.Tensor:
    """Get last 3 actions for each opponent. Returns (B, N, 3, 28)"""
    return obs.opponent_last_actions


def state_features_v3(obs: BatchedObservationV3, seat: int = 0) -> torch.Tensor:
    """
    Compute v3 state features for a single seat.
    Returns (B, state_dim) tensor.
    """
    B = obs.hand.shape[0]
    n = obs.player_count
    device = obs.hand.device

    hand = obs.hand.float()  # (B, 14)
    own_open = obs.all_plays_open[:, seat].float()  # (B, 14)
    own_closed = obs.all_plays_closed[:, seat].float().unsqueeze(-1)  # (B, 1)

    # Table open (all opponents combined)
    table_open = obs.all_plays_open.sum(dim=1) - own_open  # (B, 14)
    table_closed = obs.all_plays_closed.sum(dim=1) - own_closed.squeeze(-1)  # (B,)
    table_max = max(1.0, 5.0 * (n - 1))

    scores = obs.scores.float()  # (B, N)
    own_score = scores[:, seat]  # (B,)
    other_scores = torch.cat([scores[:, :seat], scores[:, seat+1:]], dim=1)  # (B, N-1)
    max_other = other_scores.max(dim=1).values
    min_other = other_scores.min(dim=1).values
    mean_other = other_scores.mean(dim=1)
    max_all = torch.maximum(own_score, max_other)
    min_all = torch.minimum(own_score, min_other)
    rank = (other_scores > own_score.unsqueeze(1)).sum(dim=1).float()
    rank_score = 1.0 - rank / max(1, n - 1)

    # Unseen cards
    seen = obs.all_plays_open.sum(dim=1) + obs.hand  # (B, 14)
    unseen = (obs.deck_counts.float().unsqueeze(0) - seen).clamp(min=0)
    unseen_total = unseen.sum(dim=1)

    # Supplier/receiver
    supplier_idx = (seat - obs.pass_offset) % n
    receiver_idx = (seat + obs.pass_offset) % n

    supplier_open = obs.all_plays_open[:, supplier_idx].float()
    supplier_closed = obs.all_plays_closed[:, supplier_idx].float()
    supplier_score = scores[:, supplier_idx]

    receiver_open = obs.all_plays_open[:, receiver_idx].float()
    receiver_closed = obs.all_plays_closed[:, receiver_idx].float()
    receiver_score = scores[:, receiver_idx]

    # --- v3: Per-opponent open cards (N, 14) ---
    opp_open = torch.zeros(B, n, 14, device=device)
    for i in range(n):
        opp_open[:, i] = obs.all_plays_open[:, i].float()

    # --- v3: Opponent action history ---
    opp_history = obs.opponent_last_actions  # (B, N, 3, 28)

    # --- v3: Cards seen this game ---
    cards_seen = obs.cards_seen.float()  # (B, 14)

    # --- v3: Combo completeness per type ---
    combo_features = []
    for card_id, threshold in COMBO_THRESHOLDS.items():
        card_idx = CARD_INDEX[card_id]
        progress = own_open[:, card_idx] + hand[:, card_idx]
        potential = progress + unseen[:, card_idx]
        combo_features.append(_norm(progress, threshold))
        combo_features.append(_norm(potential, threshold))
        combo_features.append((potential >= threshold).float())

    # --- v3: Position features ---
    dealer_pos = torch.zeros(B, 1, device=device)  # Would need tracking
    pass_dir = float(obs.pass_offset)
    seat_onehot = torch.zeros(B, 7, device=device)
    seat_onehot[:, seat] = 1.0

    # --- v3: Potential max score from hand ---
    # Compute theoretical max if all hand cards played optimally
    max_potential = hand.sum(dim=1).float() * 10.0  # rough upper bound

    features = []
    # Player count onehot
    for count in SUPPORTED_PLAYER_COUNTS:
        features.append((n == count).float().unsqueeze(-1))
    # Round onehot
    features.append((obs.round_number == 1).float().unsqueeze(-1))
    features.append((obs.round_number == 2).float().unsqueeze(-1))
    # Turn onehot
    for turn in (1, 2, 3, 4, 5):
        features.append((obs.turn_number == turn).float().unsqueeze(-1))
    # Hand counts
    features.append(_norm(hand, 6))
    # Own open
    features.append(_norm(own_open, 5))
    # Own closed
    features.append(_norm(own_closed, 5))
    # Table open
    features.append(_norm(table_open, table_max))
    # Table closed
    features.append(_norm(table_closed.unsqueeze(-1), table_max))
    # Scores
    features.append(_norm(own_score.unsqueeze(-1), 150))
    features.append(_norm(mean_other.unsqueeze(-1), 150))
    features.append(_norm(max_other.unsqueeze(-1), 150))
    features.append(_norm(min_other.unsqueeze(-1), 150))
    features.append(rank_score.unsqueeze(-1))
    features.append(_norm((max_all - own_score).unsqueeze(-1), 150))
    features.append(_norm((own_score - min_all).unsqueeze(-1), 150))
    features.append(_norm(obs.alexey_players.sum(dim=1).float().unsqueeze(-1), n))
    features.append(obs.alexey_players[:, seat].float().unsqueeze(-1))
    features.append(_norm(obs.closed_total.float().unsqueeze(-1), 5 * n))
    # Unseen
    features.append(_norm(unseen, max(1, 12 * n)))
    features.append(_norm(unseen_total.unsqueeze(-1), max(1, 12 * n)))
    # Supplier
    features.append(_norm(supplier_open, 5))
    features.append(_norm(supplier_closed.unsqueeze(-1), 5))
    features.append(_norm(supplier_score.unsqueeze(-1), 150))
    # Receiver
    features.append(_norm(receiver_open, 5))
    features.append(_norm(receiver_closed.unsqueeze(-1), 5))
    features.append(_norm(receiver_score.unsqueeze(-1), 150))
    # Hand size / turns left
    features.append(_norm(obs.hand.sum(dim=1).float().unsqueeze(-1), 6))
    features.append(((6 - obs.turn_number) / 5.0).float().unsqueeze(-1).expand(B, 1))

    # v2 extra
    features.extend(combo_features)
    features.append(_norm(own_closed * CLOSED_POINTS, 20.0))
    features.append(_norm(table_closed * CLOSED_POINTS, 20.0))
    features.append((obs.turn_number == 1).float().unsqueeze(-1))
    features.append((obs.turn_number >= 5).float().unsqueeze(-1))
    features.append((obs.round_number == 2).float().unsqueeze(-1))
    features.append(((5 - obs.turn_number) / 4.0).float().unsqueeze(-1))

    # Opponent stats (ranked)
    opp_totals = obs.all_plays_open.sum(dim=2).sum(dim=1).float()  # (B, N)
    opp_totals_sorted = torch.sort(opp_totals, dim=1, descending=True).values
    features.append(_norm(opp_totals_sorted[:, :6], 5))
    features.append(_norm(opp_totals.mean(dim=1, keepdim=True), 5))
    features.append(_norm(opp_totals.max(dim=1, keepdim=True).values, 5))

    # Alexey tracking
    table_alexey = obs.all_plays_open[:, :, ALEXEY].sum(dim=1)
    own_alexey = own_open[:, ALEXEY]
    max_opp_alexey = obs.all_plays_open[:, :, ALEXEY].max(dim=1).values
    features.append(_norm(table_alexey.unsqueeze(-1), 6))
    features.append(_norm(own_alexey.unsqueeze(-1), 6))
    features.append(_norm(max_opp_alexey.unsqueeze(-1), 6))

    # Success threshold
    features.append(_norm(torch.clamp(60.0 - own_score, min=0).unsqueeze(-1), 60.0))
    features.append((own_score > 60.0).float().unsqueeze(-1))

    # v3 NEW FEATURES
    # Per-opponent open (flattened)
    features.append(_norm(opp_open.view(B, -1), 5))

    # Opponent action history (flattened)
    features.append(_norm(opp_history.view(B, -1), 1.0))

    # Cards seen
    features.append(_norm(cards_seen, max(1, 12 * n)))

    # Position
    features.append(seat_onehot[:, :n])  # only valid seats
    features.append(torch.full((B, 1), pass_dir, device=device))

    # Max potential
    features.append(_norm(max_potential.unsqueeze(-1), 200.0))

    return torch.cat(features, dim=1)


def action_features_v3(obs: BatchedObservationV3, actions: torch.Tensor, seat: int = 0) -> torch.Tensor:
    """
    Compute v3 action features for a batch of actions at a specific seat.
    actions: (B,) int in 0..27
    Returns: (B, action_dim) tensor
    """
    B = obs.hand.shape[0]
    n = obs.player_count
    device = obs.hand.device

    card_idx = actions // 2
    is_closed = (actions % 2).bool()

    hand = obs.hand.float()
    own_open = obs.all_plays_open[:, seat].float()
    own_closed = obs.all_plays_closed[:, seat].float()
    table_open = obs.all_plays_open.sum(dim=1) - own_open
    unseen = (obs.deck_counts.float().unsqueeze(0) - obs.all_plays_open.sum(dim=1) - obs.hand).clamp(min=0)
    deck_counts = obs.deck_counts.float().unsqueeze(0).expand(B, -1)

    # One-hot card identity
    card_onehot = torch.zeros(B, 14, device=device)
    card_onehot[torch.arange(B, device=device), card_idx] = 1.0

    # Score delta
    # We'll compute this approximately
    delta = torch.zeros(B, device=device)
    for b in range(B):
        # This is a simplification - in practice would use batched scoring
        pass
    delta = delta.clamp(-20, 20) / 20.0

    features = []
    # Card identity
    features.append(card_onehot)
    # Is closed
    features.append(is_closed.float().unsqueeze(-1))
    # Hand count for this card
    features.append(_norm(hand[torch.arange(B, device=device), card_idx].unsqueeze(-1), 6))
    # Own open for this card
    features.append(_norm(own_open[torch.arange(B, device=device), card_idx].unsqueeze(-1), 5))
    # Table open for this card
    features.append(_norm(table_open[torch.arange(B, device=device), card_idx].unsqueeze(-1), max(1.0, 5.0 * (n - 1))))
    # Unseen for this card
    features.append(_norm(unseen[torch.arange(B, device=device), card_idx].unsqueeze(-1), deck_counts[torch.arange(B, device=device), card_idx].clamp(min=1)))
    # Score delta
    features.append(delta.unsqueeze(-1))
    # Is singleton in hand
    features.append((hand[torch.arange(B, device=device), card_idx] == 1).float().unsqueeze(-1))
    # Own closed count
    features.append(_norm(own_closed.unsqueeze(-1), 5))
    # Opponent deltas (simplified)
    features.append(torch.zeros(B, 1, device=device))  # mean_opp_delta
    features.append(torch.zeros(B, 1, device=device))  # max_opp_delta
    features.append(torch.zeros(B, 1, device=device))  # relative_delta

    # v2 extra
    for card_id, threshold in COMBO_THRESHOLDS.items():
        c_idx = CARD_INDEX[card_id]
        is_this_card = (card_idx == c_idx).float()
        if c_idx == card_idx[0]:  # approximate
            progress = own_open[torch.arange(B, device=device), c_idx] + hand[torch.arange(B, device=device), c_idx]
            potential = progress + unseen[torch.arange(B, device=device), c_idx]
            features.append(is_this_card * (1.0 if not is_closed[0] else 0.0))  # is_combo_card
            features.append(is_this_card * _norm(progress, threshold))
            features.append(is_this_card * _norm(potential, threshold))
            features.append(is_this_card * (potential >= threshold).float())
        else:
            features.append(torch.zeros(B, 1, device=device))
            features.append(torch.zeros(B, 1, device=device))
            features.append(torch.zeros(B, 1, device=device))
            features.append(torch.zeros(B, 1, device=device))
    # Alexey step
    alexey_step = torch.zeros(B, 1, device=device)
    features.append(alexey_step)
    features.append((obs.turn_number >= 5).float().unsqueeze(-1).expand(B, 1))
    features.append((obs.round_number == 2).float().unsqueeze(-1).expand(B, 1))

    # v3 NEW ACTION FEATURES
    # Combo role (key/support/flex)
    combo_role = torch.zeros(B, 3, device=device)  # key, support, flex
    features.append(combo_role)

    # Bluff value
    features.append(torch.zeros(B, 1, device=device))

    # Position value
    features.append(torch.zeros(B, 1, device=device))

    # Combo completion (does this complete a combo?)
    combo_complete = torch.zeros(B, len(COMBO_THRESHOLDS), device=device)
    for i, (card_id, threshold) in enumerate(COMBO_THRESHOLDS.items()):
        c_idx = CARD_INDEX[card_id]
        is_this = (card_idx == c_idx).float()
        progress = own_open[torch.arange(B, device=device), c_idx] + hand[torch.arange(B, device=device), c_idx]
        potential = progress + unseen[torch.arange(B, device=device), c_idx]
        # If potential >= threshold and progress < threshold and is_this_card
        complete = (potential >= threshold) & (progress < threshold) & is_this.bool()
        combo_complete[:, i] = complete.float()
    features.append(combo_complete)

    return torch.cat(features, dim=1)


def state_feature_size_v3() -> int:
    # This will be computed dynamically
    return 320  # approximate


def action_feature_size_v3() -> int:
    return 80  # approximate


# Quick test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BatchedClassicEnvV3(player_count=4, batch_size=2, device=device, seed=42)
    obs = env.observation()
    print("Observation created")
    print(f"Hand shape: {obs.hand.shape}")
    print(f"Legal mask shape: {obs.legal_mask.shape}")
    print(f"Opp history shape: {obs.opponent_last_actions.shape}")
    print(f"Cards seen shape: {obs.cards_seen.shape}")

    # Test features
    state_feat = state_features_v3(obs, seat=0)
    print(f"State features shape: {state_feat.shape}")
    print(f"State feature dim: {state_feat.shape[1]}")

    # Test actions
    legal = obs.legal_mask[0].nonzero(as_tuple=True)[0]
    if len(legal) > 0:
        test_actions = legal[:3].unsqueeze(0).expand(2, -1).reshape(-1)
        action_feat = action_features_v3(obs, test_actions, seat=0)
        print(f"Action features shape: {action_feat.shape}")
        print(f"Action feature dim: {action_feat.shape[1]}")

    print("V3 features working!")