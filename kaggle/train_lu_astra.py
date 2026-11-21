#!/usr/bin/env python3
"""Kaggle training driver for lu_astra - the ultimate 8ФМ Classic policy.

Usage:
    python train_lu_astra.py \
        --updates 8000 \
        --games-per-update 1024 \
        --batch-size 2048 \
        --device cuda \
        --output models/lu_astra.json \
        --resume models/lu_astra-latest.json

This script is designed for 12-hour Kaggle GPU sessions with automatic
checkpointing and resumability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eightfm.torch_astra import (
    AstraNetwork,
    PPOTrainerV3,
    PPOConfig,
    train_lu_astra,
    FEATURE_VERSION_V3,
    state_features_v3,
    BatchedClassicEnvV3,
)


@dataclass
class TrainingMetrics:
    update: int
    timestamp: float
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    lr: float
    entropy_coef: float
    clip_eps: float
    # Evaluation metrics
    combo_win_rate: float = 0.0
    greedy_win_rate: float = 0.0
    supervised_win_rate: float = 0.0
    rl_classic_win_rate: float = 0.0
    lu_luna_win_rate: float = 0.0
    lu_sol_win_rate: float = 0.0
    mean_score: float = 0.0
    success_rate: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lu_astra on Kaggle GPU")
    parser.add_argument("--updates", type=int, default=8000, help="Total PPO updates")
    parser.add_argument("--games-per-update", type=int, default=1024, help="Games per update")
    parser.add_argument("--batch-size", type=int, default=2048, help="Minibatch size")
    parser.add_argument("--ppo-epochs", type=int, default=12, help="PPO epochs per update")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--learning-rate-final", type=float, default=1e-5, help="Final learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Initial entropy coefficient")
    parser.add_argument("--entropy-coef-final", type=float, default=0.001, help="Final entropy coefficient")
    parser.add_argument("--gae-lambda", type=float, default=0.97, help="GAE lambda")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="Initial PPO clip epsilon")
    parser.add_argument("--clip-eps-final", type=float, default=0.1, help="Final PPO clip epsilon")
    parser.add_argument("--target-kl", type=float, default=0.02, help="Target KL divergence")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--value-clip", type=float, default=0.5, help="Value function clip")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber loss delta")
    parser.add_argument("--eval-every", type=int, default=40, help="Evaluate every N updates")
    parser.add_argument("--eval-games", type=int, default=100, help="Games per opponent per eval")
    parser.add_argument("--snapshot-every", type=int, default=20, help="Save snapshot every N updates")
    parser.add_argument("--snapshot-pool-size", type=int, default=20, help="Snapshot pool size")
    parser.add_argument("--player-counts", type=str, default="3,4,5,6,7", help="Player counts to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--output", type=str, default="models/lu_astra.json", help="Output model path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="8fm-astra", help="W&B project name")
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: AstraNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: list[TrainingMetrics],
    update: int,
    config: PPOConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metrics": [asdict(m) for m in metrics],
        "update": update,
        "config": asdict(config),
        "feature_version": FEATURE_VERSION_V3,
        "timestamp": time.time(),
    }, path)


def load_checkpoint(path: Path, model: AstraNetwork, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler) -> tuple[int, list[TrainingMetrics]]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    metrics = [TrainingMetrics(**m) for m in checkpoint.get("metrics", [])]
    update = checkpoint.get("update", 0)
    print(f"Resumed from update {update} ({len(metrics)} metrics entries)")
    return update, metrics


def evaluate_model(
    model: AstraNetwork,
    device: torch.device,
    games: int = 100,
    player_counts: tuple[int, ...] = (3, 4, 5, 6, 7),
) -> dict[str, float]:
    """Evaluate model against baselines."""
    from eightfm.baselines import ComboPolicy, GreedyPolicy, SupervisedPolicy
    from eightfm.rl import RLGreedyPolicy, load_rl
    from eightfm.policy import load_policy
    from eightfm.rl_train import evaluate_controlled_seat
    from eightfm.torch_astra import AstraPolicy

    model.eval()
    policy = AstraPolicy(model, device)

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
        for n in player_counts:
            report = evaluate_controlled_seat(policy, opp, player_count=n, games=games // len(player_counts), seed=10000 + n * 100)
            win_rates.append(report["win_rate"])
            scores.append(report["mean_score"])
        results[f"{name}_win_rate"] = float(np.mean(win_rates))
        results[f"{name}_mean_score"] = float(np.mean(scores))

    # Composite metrics
    all_win_rates = [results.get(f"{k}_win_rate", 0) for k in opponents]
    all_scores = [results.get(f"{k}_mean_score", 0) for k in opponents]
    results["avg_win_rate"] = float(np.mean(all_win_rates))
    results["avg_score"] = float(np.mean(all_scores))
    results["vs_lu_sol_win_rate"] = results.get("lu_sol_win_rate", 0)

    return results


def main() -> int:
    args = parse_args()
    device = setup_device(args.device)
    set_seeds(args.seed)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Parse player counts
    player_counts = tuple(int(x) for x in args.player_counts.split(","))

    # Create config
    config = PPOConfig(
        lr=args.learning_rate,
        lr_final=args.learning_rate_final,
        warmup_steps=args.warmup_steps,
        total_updates=args.updates,
        games_per_update=args.games_per_update,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        clip_eps_final=args.clip_eps_final,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        entropy_coef_final=args.entropy_coef_final,
        value_coef=0.5,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        huber_delta=args.huber_delta,
        value_clip=args.value_clip,
    )

    # Initialize model
    test_env = BatchedClassicEnvV3(player_count=4, batch_size=1, device=device, seed=0)
    test_obs = test_env.observation()
    test_state = state_features_v3(test_obs, seat=0)
    state_dim = test_state.shape[1]
    action_dim = 80  # v3 action features

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    model = AstraNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=512,
        num_residual_blocks=4,
        num_heads=8,
        feature_version=FEATURE_VERSION_V3,
        seed=args.seed,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.total_updates - config.warmup_steps, eta_min=config.lr_final
    )

    trainer = PPOTrainerV3(model, config, device, player_counts)

    # Resume if requested
    start_update = 0
    metrics_history = []
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            if resume_path.suffix == ".pt":
                start_update, metrics_history = load_checkpoint(resume_path, model, optimizer, scheduler)
            else:
                print(f"Warning: {resume_path} is not a .pt checkpoint, starting fresh")
        else:
            print(f"Warning: Resume path {resume_path} not found, starting fresh")

    # Training loop
    print(f"Starting training from update {start_update} to {args.updates}")
    print(f"Config: {config}")

    best_score = -1.0
    best_update = 0

    try:
        for update in range(start_update, args.updates):
            # Curriculum
            if update < 500:
                trainer.league_opponents = []
            elif update < 1500:
                from eightfm.baselines import ComboPolicy, GreedyPolicy
                trainer.league_opponents = [ComboPolicy(), GreedyPolicy()]
                trainer.league_weights = [0.7, 0.3]
            elif update < 4000:
                from eightfm.baselines import ComboPolicy, GreedyPolicy, SupervisedPolicy
                from eightfm.rl import RLGreedyPolicy, load_rl
                from eightfm.policy import load_policy
                trainer.league_opponents = [
                    ComboPolicy(),
                    GreedyPolicy(),
                    RLGreedyPolicy(load_rl("models/classic-rl.json")),
                    SupervisedPolicy(load_policy("models/classic-policy.json")),
                ]
                trainer.league_weights = [0.3, 0.2, 0.3, 0.2]
            elif update < 6500:
                from eightfm.baselines import ComboPolicy, GreedyPolicy
                from eightfm.rl import RLGreedyPolicy, load_rl
                trainer.league_opponents = [
                    ComboPolicy(),
                    GreedyPolicy(),
                    RLGreedyPolicy(load_rl("models/classic-rl.json")),
                    RLGreedyPolicy(load_rl("models/lu_luna.json")),
                    RLGreedyPolicy(load_rl("models/lu_sol.json")),
                ]
                trainer.league_weights = [0.15, 0.1, 0.15, 0.3, 0.3]
            else:
                trainer.league_opponents = trainer.league_opponents[-5:]
                trainer.league_weights = [0.2] * len(trainer.league_opponents)

            # Rollout
            rollout_data = trainer.rollout(config.games_per_update)

            # PPO Update
            update_metrics = trainer.update(rollout_data)
            update_metrics["update"] = update

            # Logging
            if update % 10 == 0:
                print(f"Update {update}: policy_loss={update_metrics['policy_loss']:.4f}, "
                      f"value_loss={update_metrics['value_loss']:.4f}, "
                      f"entropy={update_metrics['entropy']:.4f}, kl={update_metrics['kl']:.4f}, "
                      f"lr={update_metrics['lr']:.2e}")

            # Evaluation
            if update % args.eval_every == 0 and update > 0:
                eval_metrics = evaluate_model(model, device, args.eval_games, player_counts)
                update_metrics.update(eval_metrics)

                composite = (eval_metrics["vs_lu_sol_win_rate"] * 0.4 +
                           eval_metrics["avg_score"] / 100.0 * 0.3 +
                           eval_metrics.get("success_rate", 0) * 0.3)

                print(f"  Eval: vs_lu_sol={eval_metrics['vs_lu_sol_win_rate']:.3f}, "
                      f"avg_score={eval_metrics['avg_score']:.2f}, "
                      f"composite={composite:.3f}")

                # Save best
                if composite > best_score:
                    best_score = composite
                    best_update = update
                    # Save as latest
                    from eightfm.rl import save_rl
                    from eightfm.torch_astra import astra_to_rl_network
                    rl_net = astra_to_rl_network(model, metadata={
                        "training_updates": update,
                        "eval_metrics": eval_metrics,
                        "composite_score": composite,
                        "feature_version": FEATURE_VERSION_V3,
                    })
                    save_rl(rl_net, args.output.replace(".json", "-latest.json"))
                    print(f"  New best! Composite: {composite:.3f}")

                # Save periodic checkpoint
                if update % args.snapshot_every == 0:
                    ckpt_path = Path(args.output).parent / f"lu_astra-ckpt-{update}.pt"
                    save_checkpoint(model, optimizer, scheduler, metrics_history, update, config, ckpt_path)

            metrics_history.append(TrainingMetrics(
                update=update,
                timestamp=time.time(),
                policy_loss=update_metrics["policy_loss"],
                value_loss=update_metrics["value_loss"],
                entropy=update_metrics["entropy"],
                kl=update_metrics["kl"],
                lr=update_metrics["lr"],
                entropy_coef=update_metrics["entropy_coef"],
                clip_eps=update_metrics["clip_eps"],
            ))

            # Save metrics JSON periodically
            if update % 50 == 0:
                metrics_path = Path(args.output).parent / "lu_astra-metrics.json"
                metrics_path.write_text(json.dumps([asdict(m) for m in metrics_history], indent=2))

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final checkpoint
        ckpt_path = Path(args.output).parent / "lu_astra-final.pt"
        save_checkpoint(model, optimizer, scheduler, metrics_history, args.updates, config, ckpt_path)

        # Save final model
        from eightfm.rl import save_rl
        from eightfm.torch_astra import astra_to_rl_network
        rl_net = astra_to_rl_network(model, metadata={
            "training_updates": args.updates,
            "best_update": best_update,
            "best_composite_score": best_score,
            "feature_version": FEATURE_VERSION_V3,
        })
        save_rl(rl_net, args.output)

        # Save metrics
        metrics_path = Path(args.output).parent / "lu_astra-metrics.json"
        metrics_path.write_text(json.dumps([asdict(m) for m in metrics_history], indent=2))

        print(f"\nTraining complete!")
        print(f"Best update: {best_update}, Best composite: {best_score:.3f}")
        print(f"Model saved to: {args.output}")
        print(f"Metrics saved to: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())