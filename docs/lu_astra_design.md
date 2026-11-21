# lu_astra — Ultimate 8ФМ Classic Policy Design

**Goal**: Create the strongest possible 8ФМ Classic RL policy that averages **60+ points** and dominates all existing policies (lu_sol, lu_luna, supervised, greedy, rl_classic).

**Training Budget**: 12 hours on Kaggle GPU (T4×2 or P100)

---

## 1. Feature Engineering v3 (Major Upgrade)

Current v2: 158 state + 32 action features

### v3 State Features (target: ~280-320)
| Category | Current | v3 Additions | Rationale |
|----------|---------|--------------|-----------|
| Hand cards | 36 | +18 (combo potential flags per card) | Know which cards form combos |
| Open cards | 36 | +18 (opponent-specific open cards) | Track each opponent separately |
| Round info | 4 | +8 (round phase, cards left in deck) | Endgame awareness |
| Score state | 5 | +15 (per-opponent score diffs, rank) | Relative standing matters |
| Alexey/Marat | 4 | +8 (detector count, global Alexey tracker) | Critical scoring mechanics |
| **Position** | 0 | **+12** (dealer pos, pass direction, seat) | Position determines pass strategy |
| **Memory** | 0 | **+40** (cards seen this game, per suit/rank) | Card counting = huge edge |
| **Opponent modeling** | 0 | **+24** (last 3 actions per opponent) | Predict opponent plays |
| **Combo tracking** | 0 | **+12** (combo completeness per type) | Know which combos are live |
| **Potential** | 0 | **+10** (max possible score from hand) | Upper bound estimation |

### v3 Action Features (target: ~60-80)
| Category | Current | v3 Additions |
|----------|---------|--------------|
| Card identity | 18 | +18 (combo role: key/support/flex) |
| Open/closed | 2 | +2 (opponent-specific legality) |
| Score delta | 1 | +5 (immediate + projected round score) |
| Combo completion | 0 | **+12** (does this complete a combo?) |
| Bluff value | 0 | **+6** (deceptive play value) |
| Position value | 0 | **+4** (pass direction advantage) |

**Total v3**: ~300 state + ~70 action = **~370 dimensions** (2.3× v2)

---

## 2. Network Architecture (Major Upgrade)

Current: 512 → 256 → 256 (shared trunk, policy/value heads)

### lu_astra Architecture
```
Input: 370 (state) + 70×N (action) per action
        │
        ▼
    [State Encoder] ──► 512 ──► LayerNorm ──► ReLU
        │                 │
        │                 ▼
        │            [Residual Block 1] 512→512 (×2)
        │                 │
        │                 ▼
        │            [Residual Block 2] 512→512 (×2)
        │                 │
        ▼                 ▼
[Action Encoder] ◄─── [Cross-Attention] ◄── State features
  (per action)         (8 heads, 64 dim)
        │
        ▼
    [Joint Encoder] 512 → 256 (shared)
        │
        ├──► Policy Head: 256 → 128 → N_actions (masked logits)
        │
        └──► Value Head:  256 → 128 → 1 (scalar)
```

**Key innovations**:
- **Cross-attention** between state and each action (learns action-state interaction)
- **Residual blocks** with LayerNorm (deeper without degradation)
- **Separate state/action encoders** then joint fusion (more capacity)
- **Tabular-nums output** for value head (stable regression)

**Parameters**: ~4.2M (vs ~1.8M for lu_sol) — fits easily in GPU memory

---

## 3. Training Scale (12-Hour Budget)

| Param | lu_sol | lu_astra | Rationale |
|-------|--------|----------|-----------|
| updates | 3,000 | **8,000** | 2.7× more gradient steps |
| games_per_update | 512 | **1,024** | Larger batch = better gradient estimate |
| warm_start_games | 6,000 | **15,000** | Stronger BC initialization |
| bc_epochs | 16 | **24** | More supervised pretraining |
| ppo_epochs | 8 | **12** | More PPO passes per batch |
| batch_size | 1,024 | **2,048** | Larger minibatches |
| eval_every | 50 | **40** | More frequent checkpoints |
| eval_games | 60 | **100** | More reliable evaluation |
| snapshot_pool | 12 | **20** | Deeper league |

**Estimated games**: 8,000 × 1,024 = **8.2M games** (vs 1.5M for lu_sol)
**Estimated time**: ~10-11 hours on T4×2 (leaves buffer)

---

## 4. Advanced RL Techniques

### Curriculum Learning
| Phase | Updates | Opponents | Focus |
|-------|---------|-----------|-------|
| 1 (BC) | 0-500 | — | Pure behavior cloning |
| 2 (Warm) | 500-1,500 | combo, greedy | Basic RL vs weak |
| 3 (League) | 1,500-4,000 | +rl_classic, lu_luna, snapshots | Diverse league |
| 4 (Elite) | 4,000-6,500 | +lu_sol, best snapshots | Beat the best |
| 5 (Polish) | 6,500-8,000 | Top-5 snapshots only | Final refinement |

### Reward Shaping (Intermediate Rewards)
- **+0.1** per combo card played correctly
- **+0.05** per point scored above baseline
- **-0.05** for illegal action attempt (masked anyway)
- **+0.2** for winning a round
- **Terminal**: +1 win, 0 draw, -1 loss (standard)

### Adaptive Hyperparameters
```python
lr_schedule: 5e-4 → 1e-5 (cosine with warmup 500 steps)
entropy_schedule: 0.05 → 0.001 (exponential)
gae_lambda: 0.97 (higher for credit assignment)
clip_eps: 0.2 → 0.1 (anneal)
kl_target: 0.02 (adaptive: if kl > 1.5× target, reduce lr)
value_clip: 0.5 (prevent value explosion)
huber_delta: 1.0
```

### Self-Play ELO Tracking
- Track ELO of each snapshot in pool
- Weight league sampling by ELO (stronger opponents more often)
- Auto-promote snapshots that beat current best 3× in a row

---

## 5. Evaluation Protocol (Stronger Than lu_sol)

| Metric | lu_sol | lu_astra |
|--------|--------|----------|
| eval_games | 60 | **100** |
| head_to_head | 40 | **80** |
| player_counts | 3-7 | **3-7** (equal weight) |
| vs lu_sol | — | **required** |
| vs lu_luna | — | **required** |
| success_threshold | 60 pts | **62 pts** |

**Selection criteria** (lexicographic):
1. Head-to-head win rate vs lu_sol > 52%
2. Composite score (win_rate × 0.4 + mean_score/100 × 0.3 + success_rate × 0.3)
3. 3-player win rate (most decisive format)
4. Consistency (low variance across seeds)

---

## 6. Implementation Checklist

- [ ] Feature v3 implementation in `torch_env.py` and `env.py`
- [ ] `AstraNetwork` class in `torch_rl.py` with cross-attention
- [ ] `train_lu_astra_torch` function with curriculum
- [ ] Kaggle driver `train_lu_astra.py` with resume/checkpoint
- [ ] Notebook `lu_astra-training.ipynb` with 12-hour config
- [ ] CLI command `train-lu-astra` with all hyperparameters
- [ ] Tests for v3 features and network parity
- [ ] Tournament integration (add astra to participants)

---

## 7. Success Criteria

| Metric | Target |
|--------|--------|
| Mean score vs greedy | **> 62** |
| Win rate vs lu_sol | **> 55%** |
| Tournament win rate (5 policies) | **> 45%** |
| >60 pts rate | **> 40%** |
| 3-player win rate | **> 50%** |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OOM on GPU | Gradient accumulation; batch_size=512, accum_steps=4 |
| Overfitting | Stronger entropy, dropout 0.1 in residual blocks, larger snapshot pool |
| Training instability | KL adaptive, value clipping, gradient norm clip 0.5 |
| Feature parity bugs | Exhaustive v2→v3 parity tests; value-for-value vs NumPy |
| 12hr timeout | Checkpoint every 20 updates; resume from best; early stopping if no improvement 500 updates |

---

*This design incorporates all learnings from lu_luna → lu_sol progression and pushes every dimension to its practical limit for the 12-hour Kaggle budget.*