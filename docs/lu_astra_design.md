# lu_astra — the maximal 8ФМ Classic policy

**Goal**: the strongest deployable 8ФМ Classic policy, targeting a **60+ point**
average, trained on a Kaggle GPU in a single ~12-hour overnight session.

`lu_astra` is deliberately **not** a new architecture. It is the proven torch
engine that produced `lu_sol` — the same simulator, the same `TorchActorCritic`
network, the same `RLNetwork` JSON export — pushed as far as the GPU budget
allows. That choice is what makes the result actually usable: every checkpoint
loads in the desktop/Telegram apps and the tournament with no code changes.

> Earlier drafts of this document proposed v3 features, cross-attention and
> residual stacks. Those were removed: they could not be exported to the
> `RLNetwork` JSON the app loads, and the scaffold never trained. Everything
> below matches the shipped code.

---

## 1. What is reused (unchanged, parity-tested)

| Component | File | Notes |
|-----------|------|-------|
| Batched GPU simulator | `eightfm/torch_env.py` | Vectorised twin of `eightfm.env`; parity-tested |
| Features v2 | `eightfm/env.py` | 158 state + 32 action; `torch_env` mirrors them |
| Network | `TorchActorCritic` | Same shapes as `RLNetwork`; straight export |
| Trainer | `train_rl_torch` | PPO + behaviour-cloning warm start + league |
| Export | `to_rl_network` / `save_rl` | Plain `RLNetwork` JSON |

Everything that worked for `lu_sol` is untouched, so there is no new parity
surface and no deployment risk.

---

## 2. What `lu_astra` changes (the uplift)

Implemented as a preset wrapper, `eightfm.torch_rl.train_lu_astra`, over
`train_rl_torch`.

| Parameter | `lu_sol` | `lu_astra` | Why |
|-----------|----------|------------|-----|
| hidden1 / hidden2 / action_hidden | 512 / 256 / 256 | **1024 / 512 / 512** | ~4× parameters; more expressive policy |
| updates (target) | 6000 | **8000** | longer anneal |
| games per update | 512 | 512 | same GPU step cost basis |
| PPO epochs | 8 | **12** | more passes per batch |
| batch size | 1024 | 1024 | stable minibatches |
| warm-start games | 6000 | **12000** | stronger behaviour-cloning start |
| BC epochs | 16 | **24** | fit the larger net |
| learning rate | 3e-4 → 2e-5 | **4e-4 → 1e-5** | stronger early, finer late |
| entropy | 0.035 → 0.003 | **0.04 → 0.002** | explore longer, then commit |
| league fraction | 0.65 | **0.75** | more games against the league |
| snapshot pool | 12 | **16** | deeper self-play history |
| snapshot weight | 0.3 | **0.35** | favour recent strong selves |
| eval games | 60 | **80** | steadier checkpoint selection |
| head-to-head games | 40 | **60** | reliable champion comparison |
| selection opponent | classic-rl | **lu_sol** | select the model that *beats lu_sol* |

The selector is the important change: `_composite_score` already adds a bonus
for beating the selection opponent above chance, so the saved "best" checkpoint
is the run that actually out-plays `lu_sol`, not merely one that looks good
against the bots.

---

## 3. Training schedule (12-hour budget)

The driver `kaggle/train_lu_astra.py` trains in time-budgeted chunks and is
fully resumable:

* `total-updates` 8000, `chunk-updates` 200;
* default `time-budget-seconds` 39600 (11 h) so the session can save;
* the learning-rate / entropy anneal is computed from the **global** step, so
  it stays monotone across sessions;
* `lu_astra-latest.json` is the resume point, `lu_astra.json` is the global
  best, `lu_astra-metrics.json` accumulates all chunks and evaluations;
* checkpoints land in `checkpoints/update-NNNN.json`.

If the full 8000 updates do not fit, the driver simply stops at the last
complete chunk and resumes next session. Quality degrades gracefully: the best
checkpoint so far is always kept.

```bash
python kaggle/train_lu_astra.py \
    --work-dir /kaggle/working \
    --total-updates 8000 \
    --time-budget-seconds 39600
```

Re-running the same command resumes automatically.

---

## 4. Evaluation and success criteria

Every `eval-every` updates the trainer runs NumPy `evaluate_rl` against greedy,
combo, supervised and (through the selector) `lu_sol`, across 3–7 players. The
composite score weights supervised/greedy/combo wins and adds a head-to-head
edge bonus over chance.

Targets:

| Metric | Target |
|--------|--------|
| Mean score (evaluation games) | **≥ 60** |
| Win rate vs `lu_sol` | **> chance** (composite bonus) |
| Tournament win rate (6 policies, 500 games each) | **top 1–2** |
| >60-point success rate | track and maximise |

---

## 5. Deployment

The exported `models/lu_astra.json` is a normal `RLNetwork`. To use it:

```bash
python -X utf8 -m eightfm tournament \
    --participants luna,sol,astra,supervised,greedy,rl_classic \
    --astra models/lu_astra.json \
    --players 3,4,5,6,7 --games 100 \
    --output tournament_astra_report.json
```

The desktop and Telegram apps auto-select `lu_luna` / `lu_sol`; `lu_astra` can
be selected as an RL checkpoint wherever a model path is accepted.

---

## 6. Honest risks

| Risk | Mitigation |
|------|------------|
| Bigger net overfits or destabilises | behaviour-cloning warm start, entropy floor, `target_kl`, value clipping, Huber value loss |
| 12 h not enough for 8000 updates | time-budgeted chunks + resume; best checkpoint always saved |
| Selection noise | 80 eval games and 60 head-to-head games |
| Memory | 1024/512/512 is small for a T4; no gradient checkpointing needed |

---

*Built on the same engine as `lu_sol`; the uplift is capacity, compute and
champion-aware selection, not a new architecture.*