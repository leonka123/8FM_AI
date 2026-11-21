"""Command-line entry point: python -m eightfm audit database.json"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .audit import audit_file
from .agent import CARD_NAMES, predict_message
from .history import analyze_file
from .model import FEATURES_VERSION, feature_names, feature_size, load_model, save_model, train_model
from .policy import load_policy, recommend_policy, save_policy, train_policy_model


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_floats(value: str) -> list[float]:
    numbers = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not numbers:
        raise ValueError("Нужен хотя бы один числовой параметр")
    return numbers


def _parse_player_counts(value: str) -> tuple[int, ...]:
    counts = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not counts:
        raise ValueError("At least one player count is required")
    unsupported = [count for count in counts if count not in (3, 4, 5, 6, 7)]
    if unsupported:
        raise ValueError(f"Unsupported player counts: {unsupported}; use 3-7")
    return counts


def main() -> int:
    # Windows consoles may default to a legacy code page; keep JSON and
    # diagnostic output readable for UTF-8 project data.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="eightfm", description="Локальный read-only помощник 8ФМ")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Проверить JSON-экспорт и пригодность архивов")
    audit_parser.add_argument("database", help="Путь к database.json")
    audit_parser.add_argument("--json", action="store_true", help="Вывести полный отчёт в JSON")
    history_parser = subparsers.add_parser("history", help="Описательная статистика завершённых ходов Classic")
    history_parser.add_argument("database", help="Path to database.json")
    history_parser.add_argument("--json", action="store_true", help="Print full report as JSON")
    train_parser = subparsers.add_parser("train", help="Train a game-level-validated score estimator")
    train_parser.add_argument("database", help="Path to database.json")
    train_parser.add_argument("--output", default="models/classic-score.json", help="Where to save the model JSON")
    train_parser.add_argument("--epochs", type=int, default=35)
    train_parser.add_argument("--seed", type=int, default=41)
    policy_parser = subparsers.add_parser(
        "train-policy",
        help="Train a listwise move policy on successful Classic players",
    )
    policy_parser.add_argument("database", help="Path to database.json")
    policy_parser.add_argument("--output", default="models/classic-policy.json", help="Where to save the policy JSON")
    policy_parser.add_argument("--epochs", type=int, default=30)
    policy_parser.add_argument("--hidden", type=int, default=64)
    policy_parser.add_argument("--batch-size", type=int, default=256)
    policy_parser.add_argument("--patience", type=int, default=6)
    policy_parser.add_argument("--success-threshold", type=float, default=60.0)
    policy_parser.add_argument("--min-score", type=float, default=0.0)
    policy_parser.add_argument("--success-weight", type=float, default=2.0)
    policy_parser.add_argument("--score-bonus-weight", type=float, default=1.0)
    policy_parser.add_argument("--rank-weight", type=float, default=1.0)
    policy_parser.add_argument("--seed", type=int, default=41)
    predict_parser = subparsers.add_parser("predict", help="Show score-model feature vectors and raw score predictions")
    predict_parser.add_argument("--player", help="Your exact name as displayed in the game message")
    predict_parser.add_argument("--model", default="models/classic-score.json", help="Trained model JSON")
    predict_parser.add_argument("--message-file", help="Read the copied Telegram message from a UTF-8 text file; defaults to stdin")
    predict_parser.add_argument("--json", action="store_true", help="Print vectors and predictions as JSON")
    play_parser = subparsers.add_parser(
        "play",
        help="Rank visible cards with the trained policy; never submits a move",
    )
    play_parser.add_argument("--player", default="Леончик", help="Your exact name as displayed in the game message")
    play_parser.add_argument("--model", default="models/classic-policy.json", help="Trained policy JSON")
    play_parser.add_argument("--score-model", default="models/classic-score.json", help="Optional auxiliary score model JSON")
    play_parser.add_argument("--message-file", help="Read the copied Telegram message from a UTF-8 text file; defaults to stdin")
    play_parser.add_argument("--json", action="store_true", help="Print the policy ranking as JSON")
    play_parser.add_argument("--features", action="store_true", help="Include the 85-value feature vector in JSON output")
    rl_train_parser = subparsers.add_parser(
        "train-rl",
        help="Train the self-play RL policy on the full Classic simulator",
    )
    rl_train_parser.add_argument("--updates", type=int, default=100)
    rl_train_parser.add_argument("--games-per-update", type=int, default=24)
    rl_train_parser.add_argument("--players", default="3,4,5,6,7", help="Comma-separated player counts")
    rl_train_parser.add_argument("--seed", type=int, default=1)
    rl_train_parser.add_argument("--warm-start-games", type=int, default=800)
    rl_train_parser.add_argument("--bc-epochs", type=int, default=4)
    rl_train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    rl_train_parser.add_argument("--ppo-epochs", type=int, default=4)
    rl_train_parser.add_argument("--clip-ratio", type=float, default=0.2)
    rl_train_parser.add_argument("--entropy-coef", type=float, default=0.02)
    rl_train_parser.add_argument("--league-fraction", type=float, default=0.5)
    rl_train_parser.add_argument("--eval-every", type=int, default=10)
    rl_train_parser.add_argument("--eval-games", type=int, default=20)
    rl_train_parser.add_argument("--eval-seed-base", type=int, default=40000)
    rl_train_parser.add_argument("--feature-version", type=int, default=2, choices=(1, 2), help="State/action feature layout (2 = richer)")
    rl_train_parser.add_argument("--workers", type=int, default=1, help="Parallel rollout processes (Kaggle/Linux recommended)")
    rl_train_parser.add_argument("--supervised-model", default="models/classic-policy.json")
    rl_train_parser.add_argument("--output", default="models/classic-rl.json")
    rl_train_parser.add_argument("--latest", default="models/classic-rl-latest.json")
    rl_train_parser.add_argument("--metrics", default="models/classic-rl-metrics.json")
    rl_train_parser.add_argument("--plot", default="plots/classic-rl-training.png")
    rl_train_parser.add_argument("--resume", help="Continue from a checkpoint; disables the BC warm start")
    lu_luna_parser = subparsers.add_parser(
        "train-lu-luna",
        help="Train lu_luna, the strong wide-network RL policy",
    )
    lu_luna_parser.add_argument("--updates", type=int, default=320)
    lu_luna_parser.add_argument("--games-per-update", type=int, default=32)
    lu_luna_parser.add_argument("--players", default="3,4,5,6,7", help="Comma-separated player counts")
    lu_luna_parser.add_argument("--seed", type=int, default=2024)
    lu_luna_parser.add_argument("--warm-start-games", type=int, default=2000)
    lu_luna_parser.add_argument("--bc-epochs", type=int, default=8)
    lu_luna_parser.add_argument("--learning-rate", type=float, default=3e-4)
    lu_luna_parser.add_argument("--learning-rate-final", type=float, default=4e-5)
    lu_luna_parser.add_argument("--batch-size", type=int, default=512)
    lu_luna_parser.add_argument("--ppo-epochs", type=int, default=6)
    lu_luna_parser.add_argument("--clip-ratio", type=float, default=0.2)
    lu_luna_parser.add_argument("--entropy-coef", type=float, default=0.03)
    lu_luna_parser.add_argument("--entropy-coef-final", type=float, default=0.004)
    lu_luna_parser.add_argument("--value-clip", type=float, default=0.2)
    lu_luna_parser.add_argument("--huber-delta", type=float, default=1.0)
    lu_luna_parser.add_argument("--target-kl", type=float, default=0.03)
    lu_luna_parser.add_argument("--league-fraction", type=float, default=0.6)
    lu_luna_parser.add_argument("--snapshot-every", type=int, default=10)
    lu_luna_parser.add_argument("--snapshot-pool-size", type=int, default=8)
    lu_luna_parser.add_argument(
        "--selection-opponent",
        default="models/classic-rl.json",
        help="Reigning champion the best checkpoint must also beat head-to-head",
    )
    lu_luna_parser.add_argument("--head-to-head-games", type=int, default=20)
    lu_luna_parser.add_argument("--checkpoint-every", type=int, default=20)
    lu_luna_parser.add_argument("--checkpoint-dir", default="models/lu_luna-checkpoints")
    lu_luna_parser.add_argument("--hidden1", type=int, default=256)
    lu_luna_parser.add_argument("--hidden2", type=int, default=128)
    lu_luna_parser.add_argument("--action-hidden", type=int, default=128)
    lu_luna_parser.add_argument("--eval-every", type=int, default=10)
    lu_luna_parser.add_argument("--eval-games", type=int, default=30)
    lu_luna_parser.add_argument("--eval-seed-base", type=int, default=50000)
    lu_luna_parser.add_argument("--feature-version", type=int, default=2, choices=(1, 2), help="State/action feature layout (2 = richer)")
    lu_luna_parser.add_argument("--workers", type=int, default=1, help="Parallel rollout processes (Kaggle/Linux recommended)")
    lu_luna_parser.add_argument("--supervised-model", default="models/classic-policy.json")
    lu_luna_parser.add_argument("--output", default="models/lu_luna.json")
    lu_luna_parser.add_argument("--latest", default="models/lu_luna-latest.json")
    lu_luna_parser.add_argument("--metrics", default="models/lu_luna-metrics.json")
    lu_luna_parser.add_argument("--plot", default="plots/lu_luna-training.png")
    lu_luna_parser.add_argument("--resume", help="Continue from a lu_luna checkpoint")
    lu_sol_parser = subparsers.add_parser(
        "train-lu-sol",
        aliases=["train-absolute"],
        help="Train lu_sol, the maximal policy (rich v2 features, 512/256 net); built for Kaggle",
    )
    lu_sol_parser.add_argument("--updates", type=int, default=3000)
    lu_sol_parser.add_argument("--games-per-update", type=int, default=48)
    lu_sol_parser.add_argument("--players", default="3,4,5,6,7", help="Comma-separated player counts")
    lu_sol_parser.add_argument("--seed", type=int, default=82026)
    lu_sol_parser.add_argument("--warm-start-games", type=int, default=6000)
    lu_sol_parser.add_argument("--bc-epochs", type=int, default=16)
    lu_sol_parser.add_argument("--learning-rate", type=float, default=3e-4)
    lu_sol_parser.add_argument("--learning-rate-final", type=float, default=2e-5)
    lu_sol_parser.add_argument("--batch-size", type=int, default=1024)
    lu_sol_parser.add_argument("--ppo-epochs", type=int, default=8)
    lu_sol_parser.add_argument("--clip-ratio", type=float, default=0.2)
    lu_sol_parser.add_argument("--entropy-coef", type=float, default=0.035)
    lu_sol_parser.add_argument("--entropy-coef-final", type=float, default=0.003)
    lu_sol_parser.add_argument("--value-clip", type=float, default=0.2)
    lu_sol_parser.add_argument("--huber-delta", type=float, default=1.0)
    lu_sol_parser.add_argument("--target-kl", type=float, default=0.03)
    lu_sol_parser.add_argument("--league-fraction", type=float, default=0.65)
    lu_sol_parser.add_argument("--snapshot-every", type=int, default=25)
    lu_sol_parser.add_argument("--snapshot-pool-size", type=int, default=12)
    lu_sol_parser.add_argument(
        "--selection-opponent",
        default="models/classic-rl.json",
        help="Reigning champion the best checkpoint must also beat head-to-head",
    )
    lu_sol_parser.add_argument("--head-to-head-games", type=int, default=40)
    lu_sol_parser.add_argument("--checkpoint-every", type=int, default=25)
    lu_sol_parser.add_argument("--checkpoint-dir", default="models/lu_sol-checkpoints")
    lu_sol_parser.add_argument("--hidden1", type=int, default=512)
    lu_sol_parser.add_argument("--hidden2", type=int, default=256)
    lu_sol_parser.add_argument("--action-hidden", type=int, default=256)
    lu_sol_parser.add_argument("--eval-every", type=int, default=50)
    lu_sol_parser.add_argument("--eval-games", type=int, default=60)
    lu_sol_parser.add_argument("--eval-seed-base", type=int, default=50000)
    lu_sol_parser.add_argument("--feature-version", type=int, default=2, choices=(1, 2), help="State/action feature layout (2 = richer)")
    lu_sol_parser.add_argument("--workers", type=int, default=1, help="Parallel rollout processes (Kaggle/Linux recommended)")
    lu_sol_parser.add_argument("--supervised-model", default="models/classic-policy.json")
    lu_sol_parser.add_argument("--output", default="models/lu_sol.json")
    lu_sol_parser.add_argument("--latest", default="models/lu_sol-latest.json")
    lu_sol_parser.add_argument("--metrics", default="models/lu_sol-metrics.json")
    lu_sol_parser.add_argument("--plot", default="plots/lu_sol-training.png")
    lu_sol_parser.add_argument("--resume", help="Continue from a lu_sol checkpoint (keeps its feature layout)")
    rl_play_parser = subparsers.add_parser(
        "play-rl",
        help="Rank visible cards with the self-play RL policy; never submits a move",
    )
    rl_play_parser.add_argument("--player", default="Леончик", help="Your exact name as displayed in the game message")
    rl_play_parser.add_argument("--model", default="models/classic-rl.json", help="RL checkpoint JSON")
    rl_play_parser.add_argument("--message-file", help="Read the copied Telegram message from a UTF-8 text file; defaults to stdin")
    rl_play_parser.add_argument("--top", type=int, default=0, help="Show only the first N options (0 = all)")
    rl_play_parser.add_argument("--json", action="store_true", help="Print the RL ranking as JSON")
    play_game_parser = subparsers.add_parser(
        "play-game",
        help="Play one 3-7 player Classic game in the terminal against bots",
    )
    play_game_parser.add_argument("--name", default="Леончик", help="Your name at the table")
    play_game_parser.add_argument(
        "--opponents",
        default="rl,supervised",
        help=(
            "2-6 bots, comma-separated. Each is a kind (rl, supervised, greedy, "
            "combo, random), a kind with a checkpoint (rl:models/lu_luna.json), "
            "or a shorthand (lu_luna, classic-rl)"
        ),
    )
    play_game_parser.add_argument("--seed", type=int, help="Deal seed; random if omitted")
    play_game_parser.add_argument(
        "--model",
        default="models/classic-rl.json",
        help="RL checkpoint used by rl bots",
    )
    play_game_parser.add_argument(
        "--supervised-model",
        default="models/classic-policy.json",
        help="Policy checkpoint used by supervised bots",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark-models",
        help="Play the RL and supervised models together and compare their scores",
    )
    benchmark_parser.add_argument("--games", type=int, default=20, help="Games per player count")
    benchmark_parser.add_argument("--players", default="3,4", help="Comma-separated player counts")
    benchmark_parser.add_argument(
        "--filler",
        default="greedy",
        choices=("greedy", "combo", "random"),
        help="Policy for the seats not occupied by the two models",
    )
    benchmark_parser.add_argument("--seed-base", type=int, default=70000)
    benchmark_parser.add_argument("--model", default="models/classic-rl.json")
    benchmark_parser.add_argument("--supervised-model", default="models/classic-policy.json")
    benchmark_parser.add_argument("--json", action="store_true")
    rl_eval_parser = subparsers.add_parser(
        "eval-rl",
        help="Evaluate an RL checkpoint against greedy, supervised, combo and random",
    )
    rl_eval_parser.add_argument("--model", default="models/classic-rl.json", help="RL checkpoint JSON")
    rl_eval_parser.add_argument("--supervised-model", default="models/classic-policy.json")
    rl_eval_parser.add_argument("--players", default="3,4,5,6,7", help="Comma-separated player counts")
    rl_eval_parser.add_argument("--games", type=int, default=100, help="Games per opponent per player count")
    rl_eval_parser.add_argument("--seed-base", type=int, default=90000)
    rl_eval_parser.add_argument("--json", action="store_true")
    rl_eval_parser.add_argument("--output", help="Optional path for the JSON report")
    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help="Blend lu_luna, classic-rl and supervised into one weighted policy and evaluate it",
    )
    ensemble_parser.add_argument(
        "--members",
        default="lu_luna,classic-rl,supervised",
        help="Comma-separated members: shorthands (lu_luna, classic-rl, supervised) or kind:path",
    )
    ensemble_parser.add_argument("--weights", help="Comma-separated weights, one per member")
    ensemble_parser.add_argument(
        "--temperatures", help="Comma-separated softmax temperatures, one per member"
    )
    ensemble_parser.add_argument("--seed", type=int, default=0, help="Seed for tie-breaking")
    ensemble_parser.add_argument("--name", default="ensemble", help="Name stored in the config")
    ensemble_parser.add_argument("--players", default="3,4,5,6,7", help="Comma-separated player counts")
    ensemble_parser.add_argument("--games", type=int, default=50, help="Games per opponent per player count")
    ensemble_parser.add_argument("--seed-base", type=int, default=91000)
    ensemble_parser.add_argument("--save", help="Save the ensemble config to this JSON path")
    ensemble_parser.add_argument("--no-eval", action="store_true", help="Only print and save the config")
    ensemble_parser.add_argument("--json", action="store_true")
    app_parser = subparsers.add_parser(
        "app",
        help="Desktop window + system tray assistant (8ФМ Ассистент)",
    )
    app_parser.add_argument("--minimized", action="store_true", help="Start hidden in the system tray")
    app_parser.add_argument("--config", help="Path to the settings JSON")
    tg_parser = subparsers.add_parser(
        "telegram-assistant",
        help="Answer pending 8ФМ Classic choices in Telegram with lu_luna",
    )
    tg_parser.add_argument("--config", help="Path to a JSON config file")
    tg_parser.add_argument("--api-id", type=int, help="Telegram api_id (or TELEGRAM_API_ID)")
    tg_parser.add_argument("--api-hash", help="Telegram api_hash (or TELEGRAM_API_HASH)")
    tg_parser.add_argument("--phone", help="Phone for the first login (or TELEGRAM_PHONE)")
    tg_parser.add_argument("--chat", help="Chat/username of the game bot (default @phys8math8bot)")
    tg_parser.add_argument("--player", help="Your name in the game (default Леончик)")
    tg_parser.add_argument("--model", help="RL checkpoint (default models/lu_luna.json)")
    tg_parser.add_argument("--session", help="Telethon session path (default data/telegram-assistant)")
    tg_parser.add_argument("--min-delay", type=float, help="Min delay in minutes (reserved; delay is currently disabled)")
    tg_parser.add_argument("--max-delay", type=float, help="Max delay in minutes (reserved; delay is currently disabled)")
    tg_parser.add_argument("--deadline-margin", type=float, help="Safety margin before the deadline, minutes (reserved)")
    tg_parser.add_argument("--delay-seconds", type=float, help="Seconds to wait before pressing (default 20)")
    tg_parser.add_argument("--seed", type=int, help="Seed for tie-breaking")
    tg_parser.add_argument("--pack-marker", help="Required pack marker (default 8ФМ Classic)")
    tg_parser.add_argument("--modifier", action="append", help="Extra modifier regex (repeatable)")
    tg_parser.add_argument("--dry-run", action="store_true", help="Decide and log, but never press a button")
    tg_parser.add_argument("--check", help="Offline: parse a message file and print the decision, no Telegram")
    tg_parser.add_argument("--json", action="store_true", help="JSON output for --check")
    args = parser.parse_args()

    if args.command == "predict":
        try:
            message = Path(args.message_file).read_text(encoding="utf-8") if args.message_file else sys.stdin.read()
            model = load_model(args.model)
            result = predict_message(message, player_name=args.player, model=model)
        except (OSError, ValueError) as error:
            print(f"Prediction failed: {error}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({
                "features_version": FEATURES_VERSION,
                "feature_size": feature_size(),
                "feature_names": feature_names(),
                "round_number": result.round_number,
                "turn_number": result.turn_number,
                "player_name": result.player_name,
                "candidates": [item.to_dict() for item in result.candidates],
                "warnings": result.warnings,
            }, ensure_ascii=True, indent=2))
            return 0
        print(f"Round {result.round_number or '?'} / turn {result.turn_number or '?'}; player: {result.player_name or args.player or 'not matched'}")
        print(f"Feature layout: version {FEATURES_VERSION}, {feature_size()} values")
        print("\nOptions, raw score-model prediction first:")
        for index, item in enumerate(result.candidates, start=1):
            print(
                f"  {index}. {item.name} ({item.play}) — model {item.predicted_score_delta:.2f} pts"
            )
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
        print("\nUse --json to inspect every feature vector and its names.")
        return 0

    if args.command == "play":
        try:
            message = Path(args.message_file).read_text(encoding="utf-8") if args.message_file else sys.stdin.read()
            policy = load_policy(args.model)
            score_model = load_model(args.score_model) if args.score_model and Path(args.score_model).exists() else None
            result = recommend_policy(
                message,
                player_name=args.player,
                model=policy,
                score_model=score_model,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Policy recommendation failed: {error}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({
                "round_number": result.round_number,
                "turn_number": result.turn_number,
                "player_name": result.player_name,
                "candidates": [
                    item.to_dict(include_features=args.features)
                    for item in result.candidates
                ],
                "warnings": result.warnings,
            }, ensure_ascii=True, indent=2))
            return 0
        print(f"Round {result.round_number or '?'} / turn {result.turn_number or '?'}; player: {result.player_name or args.player or 'not matched'}")
        print("\nPolicy ranking, higher is better:")
        for index, item in enumerate(result.candidates, start=1):
            score_delta = f"; score model {item.score_delta:+.2f} pts" if item.score_delta is not None else ""
            print(
                f"  {index}. {item.name} ({item.play}) — policy {item.policy_score:.3f}, "
                f"probability {item.probability:.1%}{score_delta}"
            )
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
        print("\nThis is advice only. Confirm any move yourself in the game.")
        return 0

    if args.command == "train-rl":
        from .rl_train import train_rl

        try:
            player_counts = _parse_player_counts(args.players)
            supervised_model = None
            if args.supervised_model and Path(args.supervised_model).exists():
                supervised_model = load_policy(args.supervised_model)
            warm_start_games = args.warm_start_games
            if args.resume:
                warm_start_games = 0
            net, metrics = train_rl(
                updates=args.updates,
                games_per_update=args.games_per_update,
                player_counts=player_counts,
                seed=args.seed,
                warm_start_games=warm_start_games,
                bc_epochs=args.bc_epochs,
                learning_rate=args.learning_rate,
                ppo_epochs=args.ppo_epochs,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                league_fraction=args.league_fraction,
                eval_every=args.eval_every,
                eval_games=args.eval_games,
                eval_seed_base=args.eval_seed_base,
                supervised_model=supervised_model,
                feature_version=args.feature_version,
                workers=args.workers,
                output=args.output,
                latest_path=args.latest,
                metrics_path=args.metrics,
                plot_path=args.plot,
                resume=args.resume,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"RL training failed: {error}", file=sys.stderr)
            return 2
        print(f"Best checkpoint: {args.output} (update {metrics['best_update']}, score {metrics['best_score']:.3f})")
        print(f"Latest checkpoint: {args.latest}")
        print(f"Metrics: {args.metrics}")
        print(f"Plot: {args.plot}")
        print(f"Elapsed: {metrics.get('seconds', 0):.1f}s")
        return 0

    if args.command in ("train-lu-sol", "train-absolute"):
        from .rl_train import train_lu_sol

        try:
            player_counts = _parse_player_counts(args.players)
            supervised_model = None
            if args.supervised_model and Path(args.supervised_model).exists():
                supervised_model = load_policy(args.supervised_model)
            warm_start_games = args.warm_start_games
            if args.resume:
                warm_start_games = 0
            net, metrics = train_lu_sol(
                updates=args.updates,
                games_per_update=args.games_per_update,
                player_counts=player_counts,
                seed=args.seed,
                warm_start_games=warm_start_games,
                bc_epochs=args.bc_epochs,
                learning_rate=args.learning_rate,
                learning_rate_final=args.learning_rate_final,
                batch_size=args.batch_size,
                ppo_epochs=args.ppo_epochs,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                entropy_coef_final=args.entropy_coef_final,
                value_clip=args.value_clip,
                huber_delta=args.huber_delta,
                target_kl=args.target_kl,
                league_fraction=args.league_fraction,
                snapshot_every=args.snapshot_every,
                snapshot_pool_size=args.snapshot_pool_size,
                selection_opponent_path=args.selection_opponent,
                head_to_head_games=args.head_to_head_games,
                checkpoint_every=args.checkpoint_every,
                checkpoint_dir=args.checkpoint_dir,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                action_hidden=args.action_hidden,
                eval_every=args.eval_every,
                eval_games=args.eval_games,
                eval_seed_base=args.eval_seed_base,
                supervised_model=supervised_model,
                feature_version=args.feature_version,
                workers=args.workers,
                output=args.output,
                latest_path=args.latest,
                metrics_path=args.metrics,
                plot_path=args.plot,
                resume=args.resume,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"lu_sol training failed: {error}", file=sys.stderr)
            return 2
        print(
            f"Best checkpoint: {args.output} "
            f"(update {metrics['best_update']}, score {metrics['best_score']:.3f})"
        )
        print(f"Latest checkpoint: {args.latest}")
        print(f"Metrics: {args.metrics}")
        print(f"Plot: {args.plot}")
        print(f"Elapsed: {metrics.get('seconds', 0):.1f}s")
        return 0

    if args.command == "train-lu-luna":
        from .rl_train import train_lu_luna

        try:
            player_counts = _parse_player_counts(args.players)
            supervised_model = None
            if args.supervised_model and Path(args.supervised_model).exists():
                supervised_model = load_policy(args.supervised_model)
            warm_start_games = args.warm_start_games
            if args.resume:
                warm_start_games = 0
            net, metrics = train_lu_luna(
                updates=args.updates,
                games_per_update=args.games_per_update,
                player_counts=player_counts,
                seed=args.seed,
                warm_start_games=warm_start_games,
                bc_epochs=args.bc_epochs,
                learning_rate=args.learning_rate,
                learning_rate_final=args.learning_rate_final,
                batch_size=args.batch_size,
                ppo_epochs=args.ppo_epochs,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                entropy_coef_final=args.entropy_coef_final,
                value_clip=args.value_clip,
                huber_delta=args.huber_delta,
                target_kl=args.target_kl,
                league_fraction=args.league_fraction,
                snapshot_every=args.snapshot_every,
                snapshot_pool_size=args.snapshot_pool_size,
                selection_opponent_path=args.selection_opponent,
                head_to_head_games=args.head_to_head_games,
                checkpoint_every=args.checkpoint_every,
                checkpoint_dir=args.checkpoint_dir,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                action_hidden=args.action_hidden,
                eval_every=args.eval_every,
                eval_games=args.eval_games,
                eval_seed_base=args.eval_seed_base,
                supervised_model=supervised_model,
                feature_version=args.feature_version,
                workers=args.workers,
                output=args.output,
                latest_path=args.latest,
                metrics_path=args.metrics,
                plot_path=args.plot,
                resume=args.resume,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"lu_luna training failed: {error}", file=sys.stderr)
            return 2
        print(
            f"Best checkpoint: {args.output} "
            f"(update {metrics['best_update']}, score {metrics['best_score']:.3f})"
        )
        print(f"Latest checkpoint: {args.latest}")
        print(f"Metrics: {args.metrics}")
        print(f"Plot: {args.plot}")
        print(f"Elapsed: {metrics.get('seconds', 0):.1f}s")
        return 0

    if args.command == "play-rl":
        from .rl import load_rl
        from .rl_train import rank_message_actions
        try:
            message = Path(args.message_file).read_text(encoding="utf-8") if args.message_file else sys.stdin.read()
            net = load_rl(args.model)
            result = rank_message_actions(net, message, args.player)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"RL recommendation failed: {error}", file=sys.stderr)
            return 2
        candidates = result["candidates"]
        if args.top > 0:
            candidates = candidates[: args.top]
        if args.json:
            payload = dict(result)
            payload["candidates"] = candidates
            print(json.dumps(payload, ensure_ascii=True, indent=2))
            return 0
        print(
            f"Round {result['round_number']} / turn {result['turn_number']}; "
            f"player: {result['player_name']}; state value {result['value']:+.3f}"
        )
        print("\nRL ranking, higher is better:")
        for index, item in enumerate(candidates, start=1):
            label = CARD_NAMES.get(item["card_id"], (item["card_id"],))[0]
            print(
                f"  {index}. {label} ({item['play']}) — policy {item['policy_score']:.3f}, "
                f"probability {item['probability']:.1%}, immediate {item['immediate_delta']:+.2f} pts"
            )
        if result["warnings"]:
            print("\nWarnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")
        print("\nThis is advice only. Confirm any move yourself in the game.")
        return 0

    if args.command == "play-game":
        from .play_game import parse_opponents, play_interactive_game

        try:
            opponents = parse_opponents(args.opponents)
            return play_interactive_game(
                name=args.name,
                opponents=opponents,
                seed=args.seed,
                model_path=args.model,
                supervised_model_path=args.supervised_model,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Game failed: {error}", file=sys.stderr)
            return 2

    if args.command == "benchmark-models":
        from .model_bench import benchmark_models, format_report

        try:
            player_counts = _parse_player_counts(args.players)
            report = benchmark_models(
                games=args.games,
                player_counts=player_counts,
                filler=args.filler,
                seed_base=args.seed_base,
                model_path=args.model,
                supervised_model_path=args.supervised_model,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Benchmark failed: {error}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_report(report))
        return 0

    if args.command == "eval-rl":
        from statistics import mean

        from .baselines import (
            ComboPolicy,
            GreedyPolicy,
            RandomPolicy,
            SupervisedPolicy,
            evaluate_controlled_seat,
        )
        from .rl import RLGreedyPolicy, load_rl

        try:
            player_counts = _parse_player_counts(args.players)
            net = load_rl(args.model)
            candidate = RLGreedyPolicy(net)
            supervised_model = (
                load_policy(args.supervised_model)
                if args.supervised_model and Path(args.supervised_model).exists()
                else None
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"RL evaluation failed: {error}", file=sys.stderr)
            return 2
        opponents: dict[str, object] = {
            "greedy": GreedyPolicy(seed=71),
        }
        if supervised_model is not None:
            opponents["supervised"] = SupervisedPolicy(supervised_model, seed=72)
        opponents["combo"] = ComboPolicy(seed=73)
        opponents["random"] = RandomPolicy(seed=74)
        report: dict[str, object] = {
            "model": args.model,
            "games": args.games,
            "seed_base": args.seed_base,
            "players": {},
            "summary": {},
            "chance": {str(n): 1.0 / n for n in player_counts},
        }
        for n in player_counts:
            entry: dict[str, object] = {}
            for name, opponent in opponents.items():
                metrics = evaluate_controlled_seat(
                    n,
                    candidate,
                    opponent,  # type: ignore[arg-type]
                    args.games,
                    seed_base=args.seed_base + n * 1000,
                )
                entry[name] = metrics.to_dict()
            report["players"][str(n)] = entry  # type: ignore[index]
        summary: dict[str, object] = {}
        for name in opponents:
            rows = [
                report["players"][str(n)][name]  # type: ignore[index]
                for n in player_counts
            ]
            summary[name] = {
                "win_rate": mean(row["win_rate"] for row in rows),
                "mean_score": mean(row["mean_score"] for row in rows),
                "mean_rank": mean(row["mean_rank"] for row in rows),
                "success_rate": mean(row["success_rate"] for row in rows),
            }
        report["summary"] = summary
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        print(f"Model: {args.model}")
        print(f"Games per opponent per player count: {args.games}")
        header = "players  " + "  ".join(f"{name:>10}" for name in opponents) + "     chance"
        print(header)
        for n in player_counts:
            row = report["players"][str(n)]  # type: ignore[index]
            values = "  ".join(
                f"{row[name]['win_rate']:>10.3f}" for name in opponents  # type: ignore[index]
            )
            print(f"{n:>7}  {values}  {1.0 / n:>10.3f}")
        values = "  ".join(
            f"{summary[name]['win_rate']:>10.3f}" for name in opponents  # type: ignore[index]
        )
        chance = mean(1.0 / n for n in player_counts)
        print(f"{'average':>7}  {values}  {chance:>10.3f}")
        print("\nMean score / mean rank:")
        for name in opponents:
            print(
                f"  {name:>10}: score {summary[name]['mean_score']:.1f}, "  # type: ignore[index]
                f"rank {summary[name]['mean_rank']:.2f}, "
                f">60 {summary[name]['success_rate']:.1%}"
            )
        return 0

    if args.command == "ensemble":
        from statistics import mean

        from .baselines import (
            ComboPolicy,
            GreedyPolicy,
            RandomPolicy,
            SupervisedPolicy,
            evaluate_controlled_seat,
        )
        from .ensemble import EnsemblePolicy, members_from_tokens, save_ensemble
        from .policy import load_policy
        from .rl import RLGreedyPolicy, load_rl

        try:
            tokens = [part.strip() for part in args.members.split(",") if part.strip()]
            weights = _parse_floats(args.weights) if args.weights else None
            temperatures = _parse_floats(args.temperatures) if args.temperatures else None
            members = members_from_tokens(tokens, weights, temperatures)
            policy = EnsemblePolicy(members, seed=args.seed, name=args.name)
            player_counts = _parse_player_counts(args.players)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Ensemble failed: {error}", file=sys.stderr)
            return 2

        composition = " + ".join(
            f"{member.name} {member.weight:.2f} (T={member.temperature:g})"
            for member in policy.members
        )
        print(f"Ансамбль: {composition}")

        if args.save:
            save_ensemble(args.save, policy)
            print(f"Конфигурация сохранена: {args.save}")

        if args.no_eval:
            if args.json:
                print(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2))
            return 0

        opponents: dict[str, object] = {
            "greedy": GreedyPolicy(seed=71),
            "combo": ComboPolicy(seed=73),
            "random": RandomPolicy(seed=74),
        }
        has_supervised_member = any(member.kind == "supervised" for member in policy.members)
        if not has_supervised_member and Path("models/classic-policy.json").exists():
            opponents["supervised"] = SupervisedPolicy(
                load_policy("models/classic-policy.json"), seed=72
            )
        for member in policy.members:
            key = f"member:{member.name}"
            try:
                if member.kind == "rl":
                    opponents[key] = RLGreedyPolicy(load_rl(member.path))
                elif member.kind == "supervised":
                    opponents[key] = SupervisedPolicy(load_policy(member.path), seed=75)
            except (OSError, ValueError, RuntimeError):
                continue

        report: dict[str, object] = {
            "ensemble": policy.to_dict(),
            "games": args.games,
            "seed_base": args.seed_base,
            "players": {},
            "summary": {},
            "chance": {str(n): 1.0 / n for n in player_counts},
        }
        for n in player_counts:
            entry: dict[str, object] = {}
            for name, opponent in opponents.items():
                metrics = evaluate_controlled_seat(
                    n,
                    policy,
                    opponent,  # type: ignore[arg-type]
                    args.games,
                    seed_base=args.seed_base + n * 1000,
                )
                entry[name] = metrics.to_dict()
            report["players"][str(n)] = entry  # type: ignore[index]
        summary: dict[str, object] = {}
        for name in opponents:
            rows = [
                report["players"][str(n)][name]  # type: ignore[index]
                for n in player_counts
            ]
            summary[name] = {
                "win_rate": mean(row["win_rate"] for row in rows),
                "mean_score": mean(row["mean_score"] for row in rows),
                "mean_rank": mean(row["mean_rank"] for row in rows),
                "success_rate": mean(row["success_rate"] for row in rows),
            }
        report["summary"] = summary

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        print(f"Games per opponent per player count: {args.games}")
        header = "players  " + "  ".join(f"{name:>18}" for name in opponents) + "     chance"
        print(header)
        for n in player_counts:
            row = report["players"][str(n)]  # type: ignore[index]
            values = "  ".join(
                f"{row[name]['win_rate']:>18.3f}" for name in opponents  # type: ignore[index]
            )
            print(f"{n:>7}  {values}  {1.0 / n:>10.3f}")
        values = "  ".join(
            f"{summary[name]['win_rate']:>18.3f}" for name in opponents  # type: ignore[index]
        )
        chance = mean(1.0 / n for n in player_counts)
        print(f"{'average':>7}  {values}  {chance:>10.3f}")
        print("\nMean score / mean rank:")
        for name in opponents:
            print(
                f"  {name:>18}: score {summary[name]['mean_score']:.1f}, "  # type: ignore[index]
                f"rank {summary[name]['mean_rank']:.2f}, "
                f">60 {summary[name]['success_rate']:.1%}"
            )
        return 0

    if args.command == "app":
        from .desktop_app import main as desktop_main

        return desktop_main(sys.argv[2:])

    if args.command == "telegram-assistant":
        from .rl import load_rl
        from .telegram_assistant import (
            DEFAULT_MODIFIER_PATTERNS,
            PackError,
            TelegramAssistant,
            build_config,
            choose_plan,
        )

        overrides: dict[str, object] = {
            "api_id": args.api_id if args.api_id is not None else _env_int("TELEGRAM_API_ID"),
            "api_hash": args.api_hash or os.environ.get("TELEGRAM_API_HASH"),
            "phone": args.phone or os.environ.get("TELEGRAM_PHONE"),
            "chat": args.chat,
            "player_name": args.player,
            "model_path": args.model,
            "session": args.session,
            "min_delay_minutes": args.min_delay,
            "max_delay_minutes": args.max_delay,
            "deadline_margin_minutes": args.deadline_margin,
            "reply_delay_seconds": args.delay_seconds,
            "seed": args.seed,
            "pack_marker": args.pack_marker,
            "dry_run": args.dry_run,
        }
        if args.modifier:
            overrides["modifier_patterns"] = list(DEFAULT_MODIFIER_PATTERNS) + list(args.modifier)
        try:
            config = build_config(config_path=args.config, overrides=overrides)
        except (OSError, ValueError) as error:
            print(f"Assistant config failed: {error}", file=sys.stderr)
            return 2

        if args.check:
            try:
                net = load_rl(config.model_path)
                message_text = Path(args.check).read_text(encoding="utf-8")
                plan = choose_plan(message_text, net=net, config=config, immediate=True)
            except (OSError, ValueError, RuntimeError, PackError) as error:
                print(f"Check failed: {error}", file=sys.stderr)
                return 1
            if args.json:
                print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(f"Решение: {plan.label}")
                print(f"Политика: score {plan.policy_score:+.3f}, p={plan.probability:.2f}")
                print(f"Дедлайн: {plan.deadline.isoformat() if plan.deadline else 'нет'}")
                print(f"Кнопка: row={plan.row}, col={plan.col}")
                for warning in plan.warnings:
                    print(f"  - {warning}")
            return 0

        try:
            net = load_rl(config.model_path)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Assistant failed: {error}", file=sys.stderr)
            return 2
        try:
            asyncio.run(TelegramAssistant(config, net).run())
        except (RuntimeError, OSError, ValueError) as error:
            print(f"Assistant failed: {error}", file=sys.stderr)
            return 2
        return 0

    if args.command == "train-policy":
        try:
            with Path(args.database).open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            model = train_policy_model(
                data,
                seed=args.seed,
                hidden_size=args.hidden,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                success_threshold=args.success_threshold,
                min_score=args.min_score,
                success_weight=args.success_weight,
                score_bonus_weight=args.score_bonus_weight,
                rank_weight=args.rank_weight,
            )
            save_policy(model, args.output)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            print(f"Policy training failed: {error}", file=sys.stderr)
            return 2
        print(f"Saved policy: {args.output}")
        print(
            f"States: {model.train_states} train / {model.validation_states} validation; "
            f"games: {model.train_games} / {model.validation_games}"
        )
        print(
            f"Validation top-1: {model.validation_top1:.1%} "
            f"(successful players {model.validation_top1_success:.1%}); "
            f"MRR {model.validation_mrr:.3f}"
        )
        print(
            f"Frequency baseline top-1: {model.baseline_top1:.1%} "
            f"(successful players {model.baseline_top1_success:.1%}); "
            f"best epoch {model.best_epoch}"
        )
        return 0

    if args.command == "train":
        try:
            with Path(args.database).open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            model = train_model(data, seed=args.seed, epochs=args.epochs)
            save_model(model, args.output)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            print(f"Training failed: {error}", file=sys.stderr)
            return 2
        print(f"Saved model: {args.output}")
        print(f"Training examples: {model.training_records}; train games: {model.train_games}; validation games: {model.validation_games}")
        print(f"Validation RMSE: {model.validation_rmse:.3f}; baseline RMSE: {model.validation_baseline_rmse:.3f}")
        return 0

    try:
        report = audit_file(args.database) if args.command == "audit" else analyze_file(args.database)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=True, indent=2))
        return 0

    if args.command == "history":
        print(f"File: {report.source}")
        print(f"Eligible games: {report.eligible_games}; used games: {report.games_used}")
        print(f"Move records: {report.move_records_used}; action groups: {len(report.action_stats)}")
        print("\nMost frequent observed actions by player count, round and turn:")
        for item in sorted(report.action_stats, key=lambda value: (-value.count, value.players, value.round, value.turn, value.action))[:30]:
            print(
                f"  players={item.players} round={item.round} turn={item.turn} "
                f"action={item.action} n={item.count} "
                f"mean_score_delta={item.mean_recorded_score_delta} "
                f"median_score_delta={item.median_recorded_score_delta}"
            )
        print("\nNotes:")
        for note in report.notes:
            print(f"  - {note}")
        return 0

    print(f"File: {report.source}")
    print(f"Schema: {report.schema_version}; export timestamp: {report.export_timestamp}")
    print(f"Games: {report.games_total}; profiles: {report.users_total}")
    print("Statuses:")
    for key, value in report.status_counts.items():
        print(f"  {key}: {value}")
    print("\nClassic candidates (deck 0 / 0.2.1):")
    print(f"  finished: {report.classic_candidate_finished}")
    print(f"  with 3-6 players: {report.classic_candidate_with_supported_player_count}")
    print(f"  with complete move history: {report.classic_candidate_with_clean_move_history}")
    print(f"  with reconstructable hands: {report.classic_candidate_with_reconstructable_hands}")
    print(f"  without modifiers: {report.classic_candidate_without_modifiers}")
    print(f"  with modifiers: {report.classic_candidate_with_modifiers}")
    print(f"  player counts: {report.classic_player_counts}")
    print("\nCard IDs found in saved cards:")
    for key, value in report.card_id_counts.items():
        print(f"  {key}: {value}")
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
