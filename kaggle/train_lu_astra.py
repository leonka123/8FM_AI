"""Resumable driver that trains the ``lu_astra`` 8FM policy on Kaggle GPU.

``lu_astra`` is the maximal preset of the proven torch engine
(:func:`eightfm.torch_rl.train_lu_astra`): the same ``RLNetwork`` architecture
that produced ``lu_sol``, only wider (1024/512/512) and trained for a long,
smoothly annealed run with a deeper league and a head-to-head selector against
``lu_sol`` itself.

Why a driver instead of one call
--------------------------------
A single Kaggle session is not always long enough for the whole run, so this
driver trains in time-budgeted **chunks** and keeps enough state on disk that
the next session continues exactly where the previous one stopped:

* the latest weights (``lu_astra-latest.json``) are the resume point;
* ``driver_state.json`` records the global update counter and the best score so
  the learning-rate / entropy anneal stays **monotone** across sessions;
* the globally-best checkpoint is kept separately from the latest one.

Typical Kaggle use (GPU accelerator, Internet On)::

    git clone https://github.com/leonka123/8FM_AI repo
    cd repo
    python kaggle/train_lu_astra.py --work-dir /kaggle/working \
        --total-updates 8000 --time-budget-seconds 39600

Re-running the same command in the next session resumes automatically.

Nothing here is Kaggle-specific, and on Windows the script must be import-safe,
which is why all work happens under ``main()``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk upwards until the ``eightfm`` package is found."""
    for candidate in [start, *start.parents]:
        if (candidate / "eightfm" / "__init__.py").exists():
            return candidate
    raise SystemExit(
        "Не удалось найти пакет 'eightfm'. Запустите драйвер из клона репозитория "
        "или укажите --repo с его корнем."
    )


def _lerp(start: float, end: float, fraction: float) -> float:
    fraction = min(1.0, max(0.0, fraction))
    return start + (end - start) * fraction


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"global_step": 0, "best_score": float("-inf"), "chunks": 0}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _merge_metrics(target: Path, chunk: dict, global_start: int) -> None:
    """Append one chunk's metrics to a combined file for later plotting."""
    if target.exists():
        try:
            combined = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            combined = None
    else:
        combined = None
    if not combined or combined.get("format") != "eightfm-rl-training-metrics":
        combined = {
            "format": "eightfm-rl-training-metrics",
            "config": chunk.get("config", {}),
            "chunks": [],
            "updates": [],
            "evaluations": [],
        }
    combined["config"] = chunk.get("config", combined.get("config", {}))
    combined.setdefault("chunks", []).append(
        {
            "global_start": global_start,
            "updates": chunk.get("updates", []),
            "evaluations": chunk.get("evaluations", []),
            "best_score": chunk.get("best_score"),
            "seconds": chunk.get("seconds"),
        }
    )
    combined["updates"].extend(chunk.get("updates", []))
    combined["evaluations"].extend(chunk.get("evaluations", []))
    combined["best_score"] = max(
        float(combined.get("best_score", float("-inf"))),
        float(chunk.get("best_score", float("-inf"))),
    )
    target.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable training driver for the 'lu_astra' 8FM policy."
    )
    parser.add_argument(
        "--repo", default=None,
        help="Repository root (defaults to auto-detection from this file)",
    )
    parser.add_argument(
        "--work-dir", default="/kaggle/working",
        help="Where checkpoints live; keep this on persistent storage between runs",
    )
    parser.add_argument("--total-updates", type=int, default=8000)
    parser.add_argument("--chunk-updates", type=int, default=200)
    parser.add_argument("--games-per-update", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=12)
    parser.add_argument("--warm-start-games", type=int, default=12000)
    parser.add_argument("--bc-epochs", type=int, default=24)

    # Network width — the architectural uplift over lu_sol.
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--action-hidden", type=int, default=512)

    # League / evaluation.
    parser.add_argument("--league-fraction", type=float, default=0.75)
    parser.add_argument("--snapshot-every", type=int, default=20)
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument(
        "--selection-opponent", default="models/lu_sol.json",
        help="Champion checkpoint the best model must beat ('none' to disable)",
    )
    parser.add_argument("--head-to-head-games", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=80)

    parser.add_argument(
        "--device", default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, mps",
    )
    parser.add_argument("--seed", type=int, default=82026)
    parser.add_argument(
        "--time-budget-seconds", type=int, default=39600,
        help="Stop cleanly before this many seconds so the session can save (0 = no limit)",
    )
    parser.add_argument(
        "--min-chunk-seconds", type=int, default=600,
        help="Do not start another chunk if less than this much budget remains",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Ignore any existing driver state and start from scratch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo = Path(args.repo).resolve() if args.repo else _find_repo_root(
        Path(__file__).resolve().parent
    )
    sys.path.insert(0, str(repo))

    try:
        from eightfm.torch_rl import (  # noqa: E402
            LU_ASTRA_ENTROPY,
            LU_ASTRA_ENTROPY_FINAL,
            LU_ASTRA_LEARNING_RATE,
            LU_ASTRA_LEARNING_RATE_FINAL,
            resolve_device,
            train_lu_astra,
        )
    except ImportError as error:
        raise SystemExit(
            f"Torch backend is unavailable ({error}). Install torch "
            "(see requirements-training.txt)."
        ) from error

    device = resolve_device(args.device)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    best_path = work / "lu_astra.json"
    latest_path = work / "lu_astra-latest.json"
    # The per-chunk call writes its in-chunk best here; the driver folds it
    # into the global best so a weak later chunk cannot overwrite a strong one.
    chunk_best_path = work / ".chunk-best.json"
    chunk_metrics_path = work / ".chunk-metrics.json"
    combined_metrics_path = work / "lu_astra-metrics.json"
    checkpoint_dir = work / "checkpoints"
    state_path = work / "driver_state.json"

    if args.reset:
        for stale in (state_path, latest_path, chunk_best_path, chunk_metrics_path,
                      combined_metrics_path, best_path):
            stale.unlink(missing_ok=True)
        print("[driver] --reset: starting a fresh run")

    selection_opponent = args.selection_opponent
    if not selection_opponent or selection_opponent.lower() == "none":
        selection_opponent = None
    elif not Path(selection_opponent).is_absolute() and (repo / selection_opponent).exists():
        selection_opponent = str(repo / selection_opponent)

    state = _load_state(state_path)
    global_step = int(state.get("global_step", 0))
    best_score = float(state.get("best_score", float("-inf")))

    print("=" * 68)
    print(" lu_astra — resumable training driver")
    print(f" repo         : {repo}")
    print(f" work dir     : {work}")
    print(f" resume step  : {global_step}/{args.total_updates}")
    print(
        f" best score   : {best_score:.4f}"
        if best_score > float("-inf")
        else " best score   : —"
    )
    print(f" device       : {device}")
    print(f" net          : {args.hidden1}/{args.hidden2}/{args.action_hidden}")
    print(f" games/update : {args.games_per_update}")
    print(f" time budget  : {args.time_budget_seconds or '∞'} s")
    print("=" * 68)

    if global_step >= args.total_updates:
        print("[driver] Target already reached. Nothing to do.")
        return 0

    start_time = time.time()
    first_chunk = True
    while global_step < args.total_updates:
        elapsed = time.time() - start_time
        remaining_budget = (
            args.time_budget_seconds - elapsed
            if args.time_budget_seconds
            else float("inf")
        )
        if remaining_budget < args.min_chunk_seconds:
            print(
                f"[driver] {remaining_budget:.0f}s left — below the minimum chunk "
                "time, stopping to let the session save."
            )
            break

        chunk = min(args.chunk_updates, args.total_updates - global_step)
        span = max(1, args.total_updates)
        lr_start = _lerp(LU_ASTRA_LEARNING_RATE, LU_ASTRA_LEARNING_RATE_FINAL, global_step / span)
        lr_end = _lerp(LU_ASTRA_LEARNING_RATE, LU_ASTRA_LEARNING_RATE_FINAL, (global_step + chunk) / span)
        ent_start = _lerp(LU_ASTRA_ENTROPY, LU_ASTRA_ENTROPY_FINAL, global_step / span)
        ent_end = _lerp(LU_ASTRA_ENTROPY, LU_ASTRA_ENTROPY_FINAL, (global_step + chunk) / span)

        resume = str(latest_path) if latest_path.exists() else None
        warm_start = args.warm_start_games if (first_chunk and resume is None) else 0

        print(
            f"\n[driver] chunk {state.get('chunks', 0) + 1}: "
            f"steps {global_step}→{global_step + chunk} "
            f"(lr {lr_start:.2e}→{lr_end:.2e}, ent {ent_start:.3f}→{ent_end:.3f}, "
            f"warm_start={warm_start}, resume={'yes' if resume else 'no'})"
        )

        _, metrics = train_lu_astra(
            updates=chunk,
            games_per_update=args.games_per_update,
            batch_size=args.batch_size,
            ppo_epochs=args.ppo_epochs,
            seed=args.seed,
            warm_start_games=warm_start,
            bc_epochs=args.bc_epochs,
            learning_rate=lr_start,
            learning_rate_final=lr_end,
            entropy_coef=ent_start,
            entropy_coef_final=ent_end,
            league_fraction=args.league_fraction,
            snapshot_every=args.snapshot_every,
            snapshot_pool_size=args.snapshot_pool_size,
            selection_opponent_path=selection_opponent,
            head_to_head_games=args.head_to_head_games,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            action_hidden=args.action_hidden,
            device=device,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            resume=resume,
            output=chunk_best_path,
            latest_path=latest_path,
            metrics_path=chunk_metrics_path,
            checkpoint_dir=checkpoint_dir,
            plot_path=None,
            verbose=True,
        )

        global_step += chunk
        state["global_step"] = global_step
        state["chunks"] = int(state.get("chunks", 0)) + 1

        chunk_best = float(metrics.get("best_score", float("-inf")))
        if chunk_best > best_score and chunk_best_path.exists():
            best_score = chunk_best
            shutil.copyfile(chunk_best_path, best_path)
            print(f"[driver] new global best {best_score:.4f} -> {best_path.name}")

        state["best_score"] = best_score
        _save_state(state_path, state)
        _merge_metrics(combined_metrics_path, metrics, global_step - chunk)

        print(
            f"[driver] progress {global_step}/{args.total_updates} "
            f"({100 * global_step / max(1, args.total_updates):.1f}%) | "
            f"elapsed {time.time() - start_time:.0f}s"
        )
        first_chunk = False

    print("\n" + "=" * 68)
    print("[driver] finished this session")
    print(f" latest : {latest_path}")
    print(f" best   : {best_path}  (score {best_score:.4f})")
    print(f" metrics: {combined_metrics_path}")
    print(f" step   : {global_step}/{args.total_updates}")
    print(" Re-run the same command to continue; download the JSON checkpoints")
    print(" from the notebook's Output tab to use them on your laptop.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    raise SystemExit(main())