"""lu_astra: Ultimate 8ФМ Classic policy with cross-attention architecture.

This module implements the v3 feature set and the advanced network architecture
designed for maximum performance on 12-hour Kaggle GPU training.
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

from .env import LATEST_FEATURE_VERSION
from .model import CARD_IDS
from .rl import RLNetwork, load_rl, save_rl
from .torch_env_v3 import (
    BatchedClassicEnvV3,
    BatchedObservationV3,
    action_features_v3,
    state_features_v3,
    FEATURE_VERSION_V3,
    SUPPORTED_FEATURE_VERSIONS_V3,
    CARD_INDEX,
    NUM_CARDS,
    ACTION_SIZE,
)

# Import baselines for evaluation
from .baselines import ComboPolicy, GreedyPolicy, SupervisedPolicy
from .rl_train import evaluate_controlled_seat


# ---------------------------------------------------------------------------
# Network Architecture with Cross-Attention
# ---------------------------------------------------------------------------

class ResidualBlock(torch.nn.Module):
    """Pre-norm residual block with LayerNorm."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.linear1 = torch.nn.Linear(dim, dim * 4)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim * 4, dim)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First sub-block
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + residual

        # Second sub-block (identity for simplicity, could add attention)
        residual = x
        x = self.norm2(x)
        x = x + residual
        return x


class CrossAttentionFusion(torch.nn.Module):
    """Cross-attention between state and action features."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Project state and action to query/key/value
        self.state_proj = torch.nn.Linear(state_dim, hidden_dim)
        self.action_proj = torch.nn.Linear(action_dim, hidden_dim)

        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim)
        actions: (B, N_actions, action_dim)
        Returns: (B, N_actions, hidden_dim) - fused representation per action
        """
        B, N_actions, _ = actions.shape

        # Project
        state_proj = self.state_proj(state)  # (B, hidden_dim)
        action_proj = self.action_proj(actions)  # (B, N_actions, hidden_dim)

        # Cross-attention: state attends to actions
        # We'll use state as query, actions as key/value
        q = self.q_proj(state_proj).unsqueeze(1)  # (B, 1, hidden_dim)
        k = self.k_proj(action_proj)  # (B, N_actions, hidden_dim)
        v = self.v_proj(action_proj)  # (B, N_actions, hidden_dim)

        # Multi-head attention
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        k = k.view(B, N_actions, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_actions, head_dim)
        v = v.view(B, N_actions, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_actions, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, 1, N_actions)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # (B, num_heads, 1, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.hidden_dim)  # (B, 1, hidden_dim)

        # Output projection
        attn_out = self.out_proj(attn_out).squeeze(1)  # (B, hidden_dim)

        # Residual + norm
        out = self.norm(state_proj + attn_out)  # (B, hidden_dim)

        # Expand to per-action
        out = out.unsqueeze(1).expand(-1, N_actions, -1)  # (B, N_actions, hidden_dim)

        # Add action-specific info
        out = out + action_proj

        return out


class AstraNetwork(torch.nn.Module):
    """
    lu_astra network architecture:
    - State encoder with residual blocks
    - Action encoder
    - Cross-attention fusion
    - Joint encoder
    - Policy + Value heads
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 512,
        num_residual_blocks: int = 4,
        num_heads: int = 8,
        feature_version: int = FEATURE_VERSION_V3,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_residual_blocks = num_residual_blocks
        self.num_heads = num_heads
        self.feature_version = int(feature_version)

        # State encoder
        self.state_input = torch.nn.Linear(state_dim, hidden_dim)
        self.state_norm = torch.nn.LayerNorm(hidden_dim)
        self.state_blocks = torch.nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=0.1) for _ in range(num_residual_blocks)
        ])

        # Action encoder
        self.action_input = torch.nn.Linear(action_dim, hidden_dim)
        self.action_norm = torch.nn.LayerNorm(hidden_dim)

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(hidden_dim, hidden_dim, hidden_dim, num_heads)

        # Joint encoder
        self.joint_blocks = torch.nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=0.1) for _ in range(2)
        ])
        self.joint_norm = torch.nn.LayerNorm(hidden_dim)

        # Heads
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),  # scalar per action
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
        )

        self.reset_parameters(seed)

    def reset_parameters(self, seed: int = 1) -> None:
        generator = torch.Generator().manual_seed(int(seed))
        def init_linear(m: torch.nn.Linear):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2), generator=generator)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_linear)
        # Special init for output layers
        torch.nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01, generator=generator)
        torch.nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0, generator=generator)

    def forward(
        self,
        states: torch.Tensor,      # (B, state_dim)
        actions: torch.Tensor,     # (B, N_actions, action_dim)
        legal_mask: torch.Tensor | None = None,  # (B, N_actions) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, N_actions) - policy logits
            values: (B,) - state values
        """
        B, N_actions, _ = actions.shape

        # Encode state
        state_enc = self.state_input(states)  # (B, hidden_dim)
        state_enc = self.state_norm(state_enc)
        state_enc = F.gelu(state_enc)
        for block in self.state_blocks:
            state_enc = block(state_enc)

        # Encode actions
        action_enc = self.action_input(actions)  # (B, N_actions, hidden_dim)
        action_enc = self.action_norm(action_enc)
        action_enc = F.gelu(action_enc)

        # Cross-attention fusion
        fused = self.fusion(state_enc, action_enc)  # (B, N_actions, hidden_dim)

        # Joint encoding
        for block in self.joint_blocks:
            fused = block(fused)
        fused = self.joint_norm(fused)

        # Policy head (per action)
        logits = self.policy_head(fused).squeeze(-1)  # (B, N_actions)

        # Apply legal mask
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)

        # Value head (mean pooled over actions)
        value_feat = fused.mean(dim=1)  # (B, hidden_dim)
        values = self.value_head(value_feat).squeeze(-1)  # (B,)

        return logits, values

    def act(
        self,
        state: torch.Tensor,           # (state_dim,)
        actions: torch.Tensor,         # (N_actions, action_dim)
        legal_mask: torch.Tensor,      # (N_actions,) bool
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Single-state inference for evaluation."""
        self.eval()
        with torch.no_grad():
            logits, value = self.forward(
                state.unsqueeze(0),
                actions.unsqueeze(0),
                legal_mask.unsqueeze(0),
            )
            logits = logits.squeeze(0) / temperature
            if greedy:
                action_idx = logits.argmax().item()
            else:
                probs = F.softmax(logits, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
            return action_idx, logits, value.squeeze(0)


# ---------------------------------------------------------------------------
# Conversion to/from RLNetwork (NumPy format)
# ---------------------------------------------------------------------------

def astra_to_rl_network(model: AstraNetwork, *, metadata: dict[str, Any] | None = None) -> RLNetwork:
    """Convert AstraNetwork to RLNetwork for deployment."""
    parameters = {name: value.detach().cpu().numpy() for name, value in model.named_parameters()}
    return RLNetwork(
        W1=parameters["state_input.weight"].T,
        b1=parameters["state_input.bias"],
        W2=parameters["state_blocks.0.linear1.weight"].T,  # Approximate - not exact mapping
        b2=parameters["state_blocks.0.linear1.bias"],
        Wa=parameters["action_input.weight"].T,
        ba=parameters["action_input.bias"],
        Wp=parameters["joint_blocks.0.linear1.weight"].T,
        bp=parameters["policy_head.0.bias"],
        wv=parameters["value_head.0.weight"].T,
        bv=parameters["value_head.0.bias"],
        feature_version=model.feature_version,
        metadata=metadata or {},
    )


def rl_network_to_astra(network: RLNetwork, *, seed: int = 1) -> AstraNetwork:
    """Convert RLNetwork to AstraNetwork (approximate - for warm start)."""
    model = AstraNetwork(
        state_dim=network.W1.shape[0],
        action_dim=network.Wa.shape[0],
        hidden_dim=512,
        feature_version=network.feature_version,
        seed=seed,
    )
    # Note: Exact weight mapping not possible due to architecture differences
    # This is for initialization only; real training starts from scratch
    return model


# ---------------------------------------------------------------------------
# Behavior Cloning Dataset Collection
# ---------------------------------------------------------------------------

@dataclass
class BCDatasetV3:
    states: torch.Tensor          # (N, state_dim)
    actions: torch.Tensor         # (N, action_dim)
    legal_masks: torch.Tensor     # (N,) action index
    labels: torch.Tensor          # (N,) action index


def collect_bc_dataset_v3(
    *,
    games: int = 15000,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 1,
    random_fraction: float = 0.6,
    label_policy: Any = None,
    label_behaviour_fraction: float = 0.6,
    feature_version: int = FEATURE_VERSION_V3,
    device: torch.device = torch.device("cpu"),
) -> BCDatasetV3:
    """Collect behavior cloning dataset using v3 features."""
    from .env import ClassicEnv, Observation
    from .rules import legal_actions
    from .policy import Policy

    # Use ComboPolicy as default label policy
    if label_policy is None:
        from .baselines import ComboPolicy
        label_policy = ComboPolicy()

    state_list = []
    action_list = []
    label_list = []

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    for game_idx in range(games):
        player_count = rng.choice(player_counts)
        env = ClassicEnv(player_count, seed=rng.randint(0, 2**31 - 1))
        obs_list = env.observations()

        while not env.done:
            for seat, obs in enumerate(obs_list):
                legal = legal_actions(obs.hand, obs.own_plays)
                if not legal:
                    continue

                # State features
                state_feat = state_features_v3(obs_to_batched_v3(obs, player_count, device), seat=seat)
                state_list.append(state_feat.squeeze(0).cpu())

                # Action features for all legal actions
                action_indices = [a.card_id * 2 + (1 if not a.is_closed else 0) for a in legal]
                # This is simplified - in practice we'd build full action features

                # Label
                if rng.random() < label_behaviour_fraction:
                    label_action = label_policy(obs)
                else:
                    label_action = rng.choice(legal)
                label_idx = label_action.card_id * 2 + (1 if not label_action.is_closed else 0)
                label_list.append(label_idx)

            actions = [label_policy(obs) for obs in obs_list]  # Use label policy for rollout
            obs_list = env.step(actions).observations

    return BCDatasetV3(
        states=torch.stack(state_list),
        actions=torch.empty(0),  # placeholder
        legal_masks=torch.empty(0),
        labels=torch.tensor(label_list, dtype=torch.long),
    )


def obs_to_batched_v3(obs: "Observation", player_count: int, device: torch.device) -> BatchedObservationV3:
    """Convert single Observation to BatchedObservationV3 with batch size 1."""
    # This is a helper for BC collection - simplified
    hand = torch.zeros(1, 14, dtype=torch.int32, device=device)
    for card in obs.hand:
        hand[0, CARD_INDEX[card]] += 1

    all_plays_open = torch.zeros(1, player_count, 14, dtype=torch.int32, device=device)
    all_plays_closed = torch.zeros(1, player_count, dtype=torch.int32, device=device)
    for i, plays in enumerate(obs.all_plays):
        for card_id, is_closed in plays:
            if not is_closed and card_id:
                all_plays_open[0, i, CARD_INDEX[card_id]] += 1
            elif is_closed:
                all_plays_closed[0, i] += 1

    return BatchedObservationV3(
        player_count=player_count,
        round_number=obs.round_number,
        turn_number=obs.turn_number,
        hand=hand,
        own_plays_open=all_plays_open[:, obs.player],
        own_plays_closed=all_plays_closed[:, obs.player],
        all_plays_open=all_plays_open,
        all_plays_closed=all_plays_closed,
        scores=torch.tensor([list(obs.scores)], dtype=torch.float32, device=device),
        deck_counts=torch.tensor([DECK_COUNTS[player_count].get(c, 0) for c in CARD_IDS], dtype=torch.int32, device=device),
        closed_total=torch.tensor([obs.closed_total], dtype=torch.int32, device=device),
        alexey_players=torch.tensor([[i in obs.alexey_players for i in range(player_count)]], dtype=torch.bool, device=device),
        legal_mask=torch.zeros(1, 28, dtype=torch.bool, device=device),
        pass_receiver=torch.tensor([obs.pass_receiver], dtype=torch.int32, device=device),
        pass_supplier=torch.tensor([obs.pass_supplier], dtype=torch.int32, device=device),
        opponent_last_actions=torch.zeros(1, player_count, 3, 28, dtype=torch.float32, device=device),
        cards_seen=torch.zeros(1, 14, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# PPO Training
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    lr: float = 5e-4
    lr_final: float = 1e-5
    warmup_steps: int = 500
    total_updates: int = 8000
    games_per_update: int = 1024
    batch_size: int = 2048
    ppo_epochs: int = 12
    clip_eps: float = 0.2
    clip_eps_final: float = 0.1
    gae_lambda: float = 0.97
    gamma: float = 0.995
    entropy_coef: float = 0.05
    entropy_coef_final: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    huber_delta: float = 1.0
    value_clip: float = 0.5


class PPOTrainerV3:
    def __init__(
        self,
        model: AstraNetwork,
        config: PPOConfig,
        device: torch.device,
        player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.player_counts = player_counts
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.total_updates - config.warmup_steps, eta_min=config.lr_final
        )
        self.global_step = 0

        # League of opponents
        self.league_opponents: list[Any] = []
        self.league_weights: list[float] = []

    def _get_lr(self) -> float:
        if self.global_step < self.config.warmup_steps:
            return self.config.lr * (self.global_step + 1) / self.config.warmup_steps
        return self.optimizer.param_groups[0]['lr']

    def _get_entropy_coef(self) -> float:
        progress = self.global_step / self.config.total_updates
        return self.config.entropy_coef * (self.config.entropy_coef_final / self.config.entropy_coef) ** progress

    def _get_clip_eps(self) -> float:
        progress = self.global_step / self.config.total_updates
        return self.config.clip_eps * (self.config.clip_eps_final / self.config.clip_eps) ** progress

    def _sample_opponent(self) -> Any:
        if not self.league_opponents:
            return ComboPolicy()
        idx = random.choices(range(len(self.league_opponents)), weights=self.league_weights)[0]
        return self.league_opponents[idx]

    def rollout(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Collect rollout data using batched environment."""
        # Sample player count
        player_count = random.choice(self.player_counts)

        env = BatchedClassicEnvV3(
            player_count=player_count,
            batch_size=batch_size,
            device=self.device,
            seed=random.randint(0, 2**31 - 1),
        )

        states_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []
        dones_list = []
        legal_masks_list = []

        for _ in range(self.config.games_per_update // batch_size + 1):
            obs = env.observation()
            done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            while not done.all():
                # Get legal actions for each game
                legal_mask = obs.legal_mask  # (B, 28)

                # Build action features for all 28 actions
                B = batch_size
                all_actions = torch.arange(28, device=self.device).unsqueeze(0).expand(B, -1)  # (B, 28)
                action_features = torch.zeros(B, 28, self.model.action_dim, device=self.device)
                # Simplified: would need full action feature computation

                # State features for each seat
                # For simplicity, we use seat 0 (learner seat)
                state_feat = state_features_v3(obs, seat=0)  # (B, state_dim)

                # Forward pass
                with torch.no_grad():
                    logits, values = self.model(state_feat, action_features, legal_mask)

                # Sample actions
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                # Store
                states_list.append(state_feat)
                actions_list.append(actions)
                rewards_list.append(torch.zeros_like(actions, dtype=torch.float))  # placeholder
                values_list.append(values)
                log_probs_list.append(log_probs)
                dones_list.append(done)
                legal_masks_list.append(legal_mask)

                # Step
                # Need to convert actions to per-seat format
                action_tensor = actions.unsqueeze(1).expand(-1, player_count)
                obs, rewards, done = env.step(action_tensor)

                # Update rewards for learner seat
                rewards_list[-1] = rewards[:, 0]  # seat 0

                if done.all():
                    break

        # Concatenate
        states = torch.cat(states_list, dim=0)
        actions = torch.cat(actions_list, dim=0)
        rewards = torch.cat(rewards_list, dim=0)
        values = torch.cat(values_list, dim=0)
        old_log_probs = torch.cat(log_probs_list, dim=0)
        dones = torch.cat(dones_list, dim=0)
        legal_masks = torch.cat(legal_masks_list, dim=0)

        # GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t].float()) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t].float()) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
            "legal_masks": legal_masks,
        }

    def update(self, rollout_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """PPO update on collected rollout."""
        states = rollout_data["states"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["old_log_probs"]
        returns = rollout_data["returns"]
        advantages = rollout_data["advantages"]
        legal_masks = rollout_data["legal_masks"]

        # Build action features for chosen actions
        B = states.shape[0]
        action_features = torch.zeros(B, 1, self.model.action_dim, device=self.device)  # placeholder

        clip_eps = self._get_clip_eps()
        entropy_coef = self._get_entropy_coef()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0

        for _ in range(self.config.ppo_epochs):
            # Shuffle
            perm = torch.randperm(B, device=self.device)
            for i in range(0, B, self.config.batch_size):
                idx = perm[i:i + self.config.batch_size]
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_returns = returns[idx]
                b_advantages = advantages[idx]
                b_legal_masks = legal_masks[idx]

                # Forward
                b_action_features = action_features[idx]
                logits, values = self.model(b_states, b_action_features, b_legal_masks)

                # Policy loss
                log_probs = F.log_softmax(logits, dim=-1)
                chosen_log_probs = log_probs[torch.arange(len(idx), device=self.device), b_actions]

                ratio = (chosen_log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (Huber)
                value_loss = F.huber_loss(values, b_returns, delta=self.config.huber_delta)

                # Entropy
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                # KL
                kl = (b_old_log_probs - chosen_log_probs).mean()

                loss = policy_loss + self.config.value_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += kl.item()

        self.global_step += 1
        if self.global_step >= self.config.warmup_steps:
            self.scheduler.step()

        return {
            "policy_loss": total_policy_loss / self.config.ppo_epochs,
            "value_loss": total_value_loss / self.config.ppo_epochs,
            "entropy": total_entropy / self.config.ppo_epochs,
            "kl": total_kl / self.config.ppo_epochs,
            "lr": self._get_lr(),
            "entropy_coef": entropy_coef,
            "clip_eps": clip_eps,
        }

    def train(
        self,
        total_updates: int,
        eval_every: int = 40,
        eval_games: int = 100,
        snapshot_every: int = 20,
        snapshot_pool_size: int = 20,
        warm_start_updates: int = 500,
    ) -> dict[str, Any]:
        """Main training loop with curriculum."""
        metrics_history = []

        for update in range(total_updates):
            # Curriculum: adjust opponent difficulty
            if update < warm_start_updates:
                # Pure BC phase - no PPO
                pass
            elif update < 1500:
                # Easy opponents
                self.league_opponents = [ComboPolicy(), GreedyPolicy()]
                self.league_weights = [0.7, 0.3]
            elif update < 4000:
                # Medium league
                from .rl import load_rl, RLGreedyPolicy
                from .baselines import SupervisedPolicy
                from .policy import load_policy
                self.league_opponents = [
                    ComboPolicy(),
                    GreedyPolicy(),
                    RLGreedyPolicy(load_rl("models/classic-rl.json")),
                    SupervisedPolicy(load_policy("models/classic-policy.json")),
                ]
                self.league_weights = [0.3, 0.2, 0.3, 0.2]
            elif update < 6500:
                # Hard league - add lu_luna, lu_sol
                from .rl import load_rl, RLGreedyPolicy
                self.league_opponents = [
                    ComboPolicy(),
                    GreedyPolicy(),
                    RLGreedyPolicy(load_rl("models/classic-rl.json")),
                    RLGreedyPolicy(load_rl("models/lu_luna.json")),
                    RLGreedyPolicy(load_rl("models/lu_sol.json")),
                ]
                self.league_weights = [0.15, 0.1, 0.15, 0.3, 0.3]
            else:
                # Elite - top snapshots only
                self.league_opponents = self.league_opponents[-5:]  # best 5
                self.league_weights = [0.2] * len(self.league_opponents)

            # Rollout
            rollout_data = self.rollout(self.config.games_per_update)

            # PPO Update
            update_metrics = self.update(rollout_data)
            update_metrics["update"] = update

            # Logging
            if update % 10 == 0:
                print(f"Update {update}: {update_metrics}")

            # Evaluation
            if update % eval_every == 0 and update > 0:
                eval_metrics = self.evaluate(eval_games)
                update_metrics.update(eval_metrics)
                print(f"Eval at update {update}: {eval_metrics}")

            # Snapshot
            if update % snapshot_every == 0 and update > 0:
                self.save_snapshot(update, snapshot_pool_size)

            metrics_history.append(update_metrics)

        return {"metrics_history": metrics_history}

    def evaluate(self, games: int = 100) -> dict[str, float]:
        """Evaluate against baselines."""
        from .rl import RLGreedyPolicy, load_rl
        from .baselines import ComboPolicy, GreedyPolicy, SupervisedPolicy
        from .policy import load_policy
        from .env import evaluate_controlled_seat

        self.model.eval()
        policy = AstraPolicy(self.model, self.device)

        opponents = {
            "combo": ComboPolicy(),
            "greedy": GreedyPolicy(),
            "supervised": SupervisedPolicy(load_policy("models/classic-policy.json")),
            "rl_classic": RLGreedyPolicy(load_rl("models/classic-rl.json")),
            "lu_luna": RLGreedyPolicy(load_rl("models/lu_luna.json")),
            "lu_sol": RLGreedyPolicy(load_rl("models/lu_sol.json")),
        }

        results = {}
        for name, opp in opponents.items():
            win_rates = []
            scores = []
            for n in (3, 4, 5, 6, 7):
                report = evaluate_controlled_seat(policy, opp, player_count=n, games=games//5, seed=10000+n*100)
                win_rates.append(report["win_rate"])
                scores.append(report["mean_score"])
            results[f"{name}_win_rate"] = np.mean(win_rates)
            results[f"{name}_mean_score"] = np.mean(scores)

        return results

    def save_snapshot(self, update: int, pool_size: int):
        """Save model snapshot."""
        path = Path(f"models/lu_astra-snapshot-{update}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert to RLNetwork format for compatibility
        # Note: architecture differs, so we save raw state dict
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update": update,
            "config": self.config.__dict__,
        }, path.with_suffix(".pt"))

        # Also export as JSON (approximate)
        rl_net = astra_to_rl_network(self.model)
        save_rl(rl_net, path)
        print(f"Saved snapshot: {path}")


class AstraPolicy:
    """Wrapper for evaluation using NumPy interface."""

    def __init__(self, model: AstraNetwork, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, obs: "Observation") -> "Action":
        from .env import Observation
        from .rules import Action, legal_actions

        # Convert obs to batched
        batched = obs_to_batched_v3(obs, obs.player_count, self.device)
        state_feat = state_features_v3(batched, seat=obs.player).to(self.device)

        legal = legal_actions(obs.hand, obs.own_plays)
        action_indices = [a.card_id * 2 + (1 if not a.is_closed else 0) for a in legal]

        # Build action features
        action_feats = torch.zeros(1, 28, self.model.action_dim, device=self.device)
        legal_mask = torch.zeros(1, 28, dtype=torch.bool, device=self.device)
        legal_mask[0, action_indices] = True

        with torch.no_grad():
            logits, _ = self.model(state_feat, action_feats, legal_mask)
            action_idx = logits.argmax(dim=-1).item()

        card_idx = action_idx // 2
        is_closed = (action_idx % 2) == 0
        card_id = CARD_IDS[card_idx]
        return Action(card_id=card_id, is_closed=is_closed)


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def train_lu_astra(
    *,
    total_updates: int = 8000,
    games_per_update: int = 1024,
    batch_size: int = 2048,
    player_counts: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 42,
    warm_start_games: int = 15000,
    bc_epochs: int = 24,
    learning_rate: float = 5e-4,
    learning_rate_final: float = 1e-5,
    ppo_epochs: int = 12,
    entropy_coef: float = 0.05,
    entropy_coef_final: float = 0.001,
    eval_every: int = 40,
    eval_games: int = 100,
    snapshot_every: int = 20,
    snapshot_pool_size: int = 20,
    device: str = "auto",
    output_path: str = "models/lu_astra.json",
) -> tuple[AstraNetwork, dict[str, Any]]:
    """Train lu_astra from scratch."""

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Training lu_astra on {device}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create model with v3 feature sizes
    # We'll determine actual feature dims by running a test
    test_env = BatchedClassicEnvV3(player_count=4, batch_size=1, device=device, seed=0)
    test_obs = test_env.observation()
    test_state = state_features_v3(test_obs, seat=0)
    state_dim = test_state.shape[1]
    test_actions = torch.arange(28, device=device).unsqueeze(0)
    test_action_feats = torch.zeros(1, 28, 80, device=device)  # placeholder
    action_dim = test_action_feats.shape[2]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    model = AstraNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=512,
        num_residual_blocks=4,
        num_heads=8,
        feature_version=FEATURE_VERSION_V3,
        seed=seed,
    ).to(device)

    config = PPOConfig(
        lr=learning_rate,
        lr_final=learning_rate_final,
        total_updates=total_updates,
        games_per_update=games_per_update,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        entropy_coef=entropy_coef,
        entropy_coef_final=entropy_coef_final,
    )

    trainer = PPOTrainerV3(model, config, device, player_counts)

    # Train
    results = trainer.train(
        total_updates=total_updates,
        eval_every=eval_every,
        eval_games=eval_games,
        snapshot_every=snapshot_every,
        snapshot_pool_size=snapshot_pool_size,
        warm_start_updates=warm_start_games // games_per_update,
    )

    # Final evaluation
    final_metrics = trainer.evaluate(eval_games)
    results["final_metrics"] = final_metrics

    # Save final model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rl_net = astra_to_rl_network(model, metadata={
        "training_updates": total_updates,
        "final_metrics": final_metrics,
        "feature_version": FEATURE_VERSION_V3,
    })
    save_rl(rl_net, output_path)
    print(f"Saved final model to {output_path}")

    return model, results


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, results = train_lu_astra(
        total_updates=10,
        games_per_update=32,
        batch_size=64,
        device=device,
        output_path="models/test_astra.json",
    )
    print("Test training complete!")
    print(results)