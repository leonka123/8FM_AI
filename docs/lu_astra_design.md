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
| Features v4 | `eightfm/env.py` | 250 state + 42 action (incl. hand history); `torch_env` mirrors them |
| Network | `TorchActorCritic` | Same shapes as `RLNetwork`; straight export |
| Trainer | `train_rl_torch` | PPO + behaviour-cloning warm start + league |
| Export | `to_rl_network` / `save_rl` | Plain `RLNetwork` JSON |

Everything that worked for `lu_sol` is untouched, so there is no new parity
surface and no deployment risk.

---

## 2. What `lu_astra` changes (the v4 uplift)

Implemented as a preset wrapper, `eightfm.torch_rl.train_lu_astra`, over
`train_rl_torch`.

| Parameter | `lu_sol` (v2) | `lu_astra` (v4) | Why |
|-----------|---------------|-----------------|-----|
| **Feature layout** | v2: 158 state / 32 action | **v4: 250 state / 42 action** | v3: marginal values, combo distance, opponent pressure, hand quality, synergies; v4: personal hand history (seen/closed/passed cards) |
| hidden1 / hidden2 / action_hidden | 512 / 256 / 256 | **2048 / 1024 / 1024** | ~16× parameters for v3 features |
| updates (target) | 6000 | **15000** | longer anneal, more data |
| games per update | 512 | **1024** | 2× more samples per PPO step |
| PPO epochs | 8 | **6** | fewer epochs, more games → less overfit |
| batch size | 1024 | **4096** | stable minibatches for wider net |
| warm-start games | 6000 | **20000** | strongest BC start on v3 features |
| BC epochs | 16 | **20** | fit larger net to BC data |
| learning rate | 3e-4 → 2e-5 | **3e-4 → 5e-6** | stronger early, much finer late |
| entropy | 0.035 → 0.003 | **0.03 → 0.001** | explore longer, then commit |
| league fraction | 0.65 | **0.85** | more games against the league |
| snapshot pool | 12 | **24** | deeper self-play history |
| snapshot weight | 0.3 | **0.15** | less overfit to recent selves |
| eval every | 50 | **100** | less evaluation overhead |
| eval games | 60 | **50** | adequate for selection |
| head-to-head games | 40 | **100** (per player count) | reliable champion comparison |
| selection opponent | classic-rl | **lu_sol** | select the model that *beats lu_sol* |
| champion league weight | 0 | **0.35** | train directly against `lu_sol`, not just evaluate against it |
| **Terminal reward weights** | win 1.5 / placement 0.7 / score 0.4 / success 0.4 / margin 0.2 | **win 1.0 / placement 0.3 / score 1.2 / success 1.5 / margin 0.4** | Optimise for 60+ points, not just winning |
| **Checkpoint selector** | `_composite_score` (win-rate based) | **`_goal_composite_score`** (hard gate on beating `lu_sol`, then mean_score + success_rate) | Guarantees no regression below the champion |

### v4 feature layout (250 state / 42 action)

**v3 state additions over v2 (+49):**
- `next_open_points:{card}` ×14 — exact marginal score of opening one more copy.
- `combo_remaining:{combo}` ×8 — normalised copies still needed for each combo.
- `combo_ready:{combo}` ×8 — flag: combo already reachable from hand+table.
- `opponent_combo_max:{combo}` ×8 — best opponent progress on each combo (blocking).
- `best_open_delta`, `mean_open_delta`, `combo_actions_available` — hand-quality summary.
- `potential_score`, `ceiling_score` — expected score from hand, absolute ceiling.
- `synergy_evgen_amogus`, `synergy_sergay_olympiad`, `synergy_akim_sergay`, `synergy_marat_olympiad`, `synergy_vanya_akim`, `synergy_jaman_closed` — explicit scoring interactions.

**v3 action additions over v2 (+7):**
- `completes_combo` — 1 if this open action finishes a combo.
- `combo_remaining_after` — normalised remaining copies after this action.
- `open_delta` — marginal value of opening this card.
- `delta_vs_best`, `delta_vs_mean` — relative to best/mean legal action.
- `own_score` — current player score (normalised).
- `points_to_success` — normalised distance to 60.

**v4 state additions over v3 (+43) — personal hand history:**
- `seen_counts:{card}` ×14 — copies of each card ever held in hand (initial deal plus every hand received by passing, including cards passed on unplayed). This is the card-counting signal a human keeps in their head.
- `closed_counts:{card}` ×14 — copies of each card this player has closed (flat 2 points). Closed plays lose their identity in `own_plays`, so the environment tracks this separately.
- `passed_cards:{card}` ×14 — cards seen but no longer in hand, not open, not closed: cards that passed through the hand and were given away.
- `seen_fraction` — fraction of the whole deck already seen.

**v4 action additions over v3 (+3):**
- `closed_copies_on_card` — how many copies of this card were already closed.
- `seen_copies_on_card` — how many copies of this card ever passed through the hand.
- `passed_copies_on_card` — copies seen but passed on.

Parity between the NumPy reference and the batched torch twin is covered by
`tests/test_torch_env.py` for versions 1–4.

The selector is the important change. When a head-to-head report against `lu_sol`
is present, `_goal_composite_score` applies a **hard gate**: a checkpoint that
does not beat the champion's win rate is ranked below every checkpoint that
does, no matter how many points it scored. Within the gate the candidate's own
mean score (capped at 1.5×60) and >60 success rate are the main terms, and the
size of the head-to-head edge breaks ties. Win/combo are secondary. The saved
"best" checkpoint is therefore both a 60-point player and a proven upgrade over
`lu_sol`.

`lu_sol` is also a **live league opponent** (`champion_league_weight = 0.35`).
The learner spends a meaningful share of its opponent seats playing the exact
champion it must dethrone, so the policy is optimised against it directly rather
than only measured against it every 100 updates. The v2 champion consumes the
first 158 state / 32 action columns of the v4 observation, which is valid because
the layouts are prefix-nested.

---

## 3. Training schedule (12-hour budget)

The driver `kaggle/train_lu_astra.py` trains in time-budgeted chunks and is
fully resumable:

* `total-updates` 15000, `chunk-updates` 200;
* default `time-budget-seconds` 39600 (11 h) so the session can save;
* the learning-rate / entropy anneal is computed from the **global** step, so
  it stays monotone across sessions;
* `lu_astra-latest.json` is the resume point, `lu_astra.json` is the global
  best, `lu_astra-metrics.json` accumulates all chunks and evaluations;
* checkpoints land in `checkpoints/update-NNNN.json`.

If the full 15000 updates do not fit, the driver simply stops at the last
complete chunk and resumes next session. Quality degrades gracefully: the best
checkpoint so far is always kept.

```bash
python kaggle/train_lu_astra.py \
    --work-dir /kaggle/working \
    --total-updates 15000 \
    --time-budget-seconds 39600
```

Re-running the same command resumes automatically.

---

## 4. Evaluation and success criteria

Every `eval-every` updates the trainer runs NumPy `evaluate_rl` against greedy,
combo, supervised and `lu_sol`, across 3–7 players. The goal composite score is:

1. **Hard gate:** `head_to_head_win_rate > chance` (beats `lu_sol`) → `+100`;
   otherwise the checkpoint cannot outrank any gate-passing checkpoint.
2. **Edge:** up to `+0.5` as a tie-breaker for a clearer head-to-head margin
   over chance.
3. **60-point objective:** 50% mean score (normalised to 60, capped at 1.5×),
   20% success rate (>60), 15% win rate, 10% combo win rate.

Targets:

| Metric | Target |
|--------|--------|
| Mean score (evaluation games) | **≥ 60** |
| Success rate (>60) | **≥ 0.25** |
| Win rate vs `lu_sol` | **> chance, by a clear margin** (hard gate) |
| Tournament win rate (6 policies, 500 games each) | **top 1–2** |

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
| 12 h not enough for 15000 updates | time-budgeted chunks + resume; best checkpoint always saved |
| Selection noise | 50 eval games and 100 head-to-head games per player count; the hard gate prevents a lucky low-score checkpoint from beating a real upgrade |
| Memory | 2048/1024/1024 fits comfortably on a T4/P100; no gradient checkpointing needed |
| v4 features don't help | parity tests guarantee NumPy↔torch match; smoke test trains in seconds; the champion league directly optimises the required matchup |

---

*Built on the same engine as `lu_sol`; the v4 uplift is feature richness
(including hand history), capacity, compute budget, live training against the
champion and a champion-gated 60-point selection metric.*