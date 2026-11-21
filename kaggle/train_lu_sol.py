"""Resumable driver that trains the ``lu_sol`` 8FM policy on external compute.

The problem this solves
-----------------------
Training the maximal Classic policy needs many hours of CPU, which is painful
on a laptop but cheap on Kaggle (about 4 vCPU and ~12 h per session).  A single
Kaggle session is *not* long enough for the whole run, so this driver trains in
time-budgeted **chunks** and keeps enough state on disk that the next session
continues exactly where the previous one stopped:

* the latest weights (:func:`eightfm.rl_train.train_lu_sol` writes them every
  few updates) are the resume point;
* a small ``driver_state.json`` records the global update counter and the best
  score so far, so the learning-rate / entropy anneal stays **monotone** across
  sessions instead of restarting every time;
* the globally-best checkpoint is kept separately from the latest one.

Typical Kaggle use::

    git clone https://github.com/leonka123/8FM_AI repo
    cd repo
    python kaggle/train_lu_sol.py --work-dir /kaggle/working --workers 4 \
        --total-updates 6000 --time-budget-seconds 30000

Re-running the same command in the next session resumes automatically.

Nothing here is Kaggle-specific: it runs anywhere, and ``--workers 1`` gives
the serial fallback.  On Windows spawn the script must be import-safe, which is
why all work happens under ``main()``.
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
        description="Resumable training driver for the 'lu_sol' 8FM policy."
    )
    parser.add_argument(
        "--repo", default=None,
        help="Repository root (defaults to auto-detection from this file)",
    )
    parser.add_argument(
        "--work-dir", default="/kaggle/working",
        help="Where checkpoints live; keep this on persistent storage between runs",
    )
    parser.add_argument("--total-updates", type=int, default=6000)
    parser.add_argument("--chunk-updates", type=int, default=200)
    parser.add_argument("--games-per-update", type=int, default=48)
    parser.add_argument("--warm-start-games", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=82026)
    parser.add_argument(
        "--time-budget-seconds", type=int, default=30000,
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

    repo = Path(args.repo).resolve() if args.repo else _find_repo_root(Path(__file__).resolve().parent)
    sys.path.insert(0, str(repo))

    # Imported only after sys.path is fixed.
    from eightfm.rl_train import (  # noqa: E402
        LU_SOL_ENTROPY,
        LU_SOL_ENTROPY_FINAL,
        LU_SOL_LEARNING_RATE,
        LU_SOL_LEARNING_RATE_FINAL,
        train_lu_sol,
    )

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    best_path = work / "lu_sol.json"
    latest_path = work / "lu_sol-latest.json"
    # The per-chunk call writes its in-chunk best here; the driver folds it
    # into the global best so a weak later chunk cannot overwrite a strong one.
    chunk_best_path = work / ".chunk-best.json"
    chunk_metrics_path = work / ".chunk-metrics.json"
    combined_metrics_path = work / "lu_sol-metrics.json"
    checkpoint_dir = work / "checkpoints"
    state_path = work / "driver_state.json"

    if args.reset:
        for stale in (state_path,):
            stale.unlink(missing_ok=True)
        print("[driver] --reset: starting a fresh run")

    state = _load_state(state_path)
    global_step = int(state.get("global_step", 0))
    best_score = float(state.get("best_score", float("-inf")))

    print("=" * 68)
    print(" lu_sol — resumable training driver")
    print(f" repo         : {repo}")
    print(f" work dir     : {work}")
    print(f" resume step  : {global_step}/{args.total_updates}")
    print(f" best score   : {best_score:.4f}" if best_score > float("-inf") else " best score   : —")
    print(f" workers      : {args.workers}")
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
        lr_start = _lerp(LU_SOL_LEARNING_RATE, LU_SOL_LEARNING_RATE_FINAL, global_step / span)
        lr_end = _lerp(LU_SOL_LEARNING_RATE, LU_SOL_LEARNING_RATE_FINAL, (global_step + chunk) / span)
        ent_start = _lerp(LU_SOL_ENTROPY, LU_SOL_ENTROPY_FINAL, global_step / span)
        ent_end = _lerp(LU_SOL_ENTROPY, LU_SOL_ENTROPY_FINAL, (global_step + chunk) / span)

        resume = str(latest_path) if latest_path.exists() else None
        warm_start = args.warm_start_games if (first_chunk and resume is None) else 0

        print(
            f"\n[driver] chunk {state.get('chunks', 0) + 1}: "
            f"steps {global_step}→{global_step + chunk} "
            f"(lr {lr_start:.2e}→{lr_end:.2e}, ent {ent_start:.3f}→{ent_end:.3f}, "
            f"warm_start={warm_start}, resume={'yes' if resume else 'no'})"
        )

        _, metrics = train_lu_sol(
            updates=chunk,
            games_per_update=args.games_per_update,
            seed=args.seed,
            warm_start_games=warm_start,
            learning_rate=lr_start,
            learning_rate_final=lr_end,
            entropy_coef=ent_start,
            entropy_coef_final=ent_end,
            workers=args.workers,
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
