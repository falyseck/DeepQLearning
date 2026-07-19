# DQN Atari Agent — Boxing (ALE/Boxing-v5)

Group members: Faly, Nick, Kayonga Elvis

## Setup

```bash
pip install -r requirements.txt
```

Windows users who hit `gymnasium.error.NamespaceNotFound: Namespace ALE not found` also need:
```bash
pip install ale-py autorom[accept-rom-license]
AutoROM --accept-license
```

## Environment

We used **ALE/Boxing-v5** from Gymnasium's Atari collection. Boxing is one of the
faster-learning Atari benchmarks for DQN, so training budgets were kept in the
100k–200k timestep range across experiments rather than training to full
convergence, so that differences between hyperparameter configurations would
still be visible rather than every run converging to a similarly strong score.

## Files

- `train.py` — trains a DQN agent on Boxing. Every hyperparameter is a CLI flag,
  so each member ran their own 10 experiments by re-running this script with
  different values. Each run auto-appends a row to `experiments_log.csv`.
- `play.py` — loads a saved model and plays Boxing using a greedy (deterministic)
  policy — always picks the highest Q-value action, no exploration.
- `experiments_log.csv` — full log of all 30 runs across the group (raw source
  of the table below).
- `HYPERPARAMETER_TUNING.md` — dedicated writeup explaining what each hyperparameter
  did and why, grouped by parameter (lr, gamma, batch size, epsilon schedule) rather
  than by run. Read this for presentation/Q&A prep on trade-offs and reasoning.

## Policy comparison: MLP vs CNN

| Policy | Config | Result | Notes |
|---|---|---|---|
| MlpPolicy | lr=1e-4, gamma=0.999, batch=128 (identical to CNN champion) | Plateaued at -28, never recovered | Flattening the raw pixel frames destroys spatial structure the network needs |
| CnnPolicy | lr=1e-4, gamma=0.999, batch=128 | Reward climbed from -8.2 to +4.33 (greedy eval) | Convolutional filters extract the paddle/glove/opponent spatial patterns MLP cannot |

**Conclusion:** CnnPolicy substantially outperforms MlpPolicy on Boxing under identical
hyperparameters (Nick's Exp6 vs Exp9, a controlled comparison — only the policy network
architecture changed). This confirms CNNs are necessary for Atari's raw-pixel observation
space, consistent with the DQN literature.

## Hyperparameter Experiments (30 total — 10 per member)

### Faly

| Run | lr | gamma | batch | eps_start | eps_end | eps_decay_frac | Noted Behavior |
|---|---|---|---|---|---|---|---|
| baseline | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Mixed baseline: one run around -13 (short-medium episodes), another around -31 (poor performance). |
| FalyExp2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Likely unstable due to high learning rate; rewards inconsistent, episodes terminate quickly. |
| FalyExp3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Strongly negative rewards (~-25 to -30), short episodes; agent fails to learn effective policy. |
| FalyExp4 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.1 | Moderately negative (~-15 to -19), episodes somewhat longer; lower gamma may reduce stability. |
| FalyExp5 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.1 | Slight improvement (~-17), longer episodes; high gamma stabilizes but still negative rewards. |
| FalyExp6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.1 | Poor performance (~-30), shorter episodes; small batch size hurts stability. |
| FalyExp7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.1 | Positive rewards (~+2), longer episodes; larger batch size yields stable and successful learning. |
| FalyExp8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | Negative rewards, short episodes; reduced exploration fraction limits learning. |
| FalyExp9 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.3 | Strongly negative (~-34), unstable episodes; high exploration fraction prevents convergence. |
| FalyExp10 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Similar to baseline, negative rewards, medium episodes; no significant improvement. |

**Faly's takeaway:** Batch size 128 (Exp7) was the standout — the only run that turned
positive. Both learning-rate extremes (Exp2 high, Exp3 low) and high exploration fraction
(Exp9) clearly hurt.

### Nick

| Run | lr | gamma | batch | eps_start | eps_end | eps_decay_frac | steps | Noted Behavior |
|---|---|---|---|---|---|---|---|---|
| NickExp1_baseline | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | 100k | Dipped to ~-30 during exploration then recovered steadily to -18.5; learns but slowly, curve still rising at cutoff. |
| NickExp2_lr5e-4 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | 100k | Declined monotonically to -35.7 with no recovery; even 5x higher lr destabilizes the Q-updates, so speed cannot come from lr. |
| NickExp3_lr1e-5 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | 100k | Sank to ~-30 then drifted flat at -28; updates too small to recover within 100k steps. Brackets lr: 1e-4 confirmed near-optimal. |
| NickExp4_batch64 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.1 | 100k | New best (-10.4): shallower dip (-24) and smooth near-monotonic recovery; larger batches cut gradient variance. Still rising at cutoff. |
| NickExp5_batch128 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.1 | 100k | Best so far and only positive run (+0.43): shallow dip then strictly monotonic climb, still rising steeply at cutoff. Batch settled at 128. |
| NickExp6_b128_g999 | 1e-4 | 0.999 | 128 | 1.0 | 0.05 | 0.1 | 100k | **New best (+4.33)**: gamma 0.999 stacks with batch 128 — crossed zero earlier and climbed steadily; longer credit-assignment horizon pays off in Boxing's ~1800-step episodes. |
| NickExp7_epsend001 | 1e-4 | 0.999 | 128 | 1.0 | 0.01 | 0.1 | 100k | Worse than Exp6 (-4.34 vs +4.33, only eps_end changed): cutting the exploration floor to 1% starved the replay buffer of diverse transitions and slowed learning. |
| NickExp8_expfrac02 | 1e-4 | 0.999 | 128 | 1.0 | 0.05 | 0.2 | 100k | Worse (-8.0): doubling the exploration phase delayed the learning turnaround ~15k steps and the run ran out of budget. Brackets epsilon: default 0.1/0.05 is the sweet spot. |
| NickExp9_mlp | 1e-4 | 0.999 | 128 | 1.0 | 0.05 | 0.1 | 100k | MLP with identical hyperparameters to Exp6 plateaued at -28 and never recovered; flattened pixels lose all spatial structure. Controlled proof CnnPolicy is required. |
| NickExp10_final200k | 1e-4 | 0.999 | 128 | 1.0 | 0.05 | 0.1 | 200k | Reached only +1.02 training / -3.2 greedy eval, losing to Exp6 (+4.8 greedy): exploration_fraction is relative, so doubling steps stretched the epsilon decay and reproduced Exp8's flaw. |

**Nick's takeaway:** Best model overall is **Exp6** (lr=1e-4, gamma=0.999, batch=128,
100k steps, default epsilon schedule). Batch size and gamma were the two hyperparameters
that mattered most; naively scaling up timesteps without re-tuning `exploration_fraction`
backfired (Exp10).

### Kayonga Elvis

⚠️ Several of these runs manually reduced `batch_size` from the planned 128 to cut
runtime, which confounds the variable being tested — flagged per-row below.

| Run | lr | gamma | batch | eps_start | eps_end | eps_decay_frac | steps | Noted Behavior |
|---|---|---|---|---|---|---|---|---|
| Elvis_baseline_seed7 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.1 | 150k | ⚠️ batch reduced 128→32 to cut runtime, not a clean seed-only comparison. Reward -9.8→-30.3 (best -6.9), declining. Much worse than Nick's champion, consistent with the group's batch-size trend. |
| Elvis_lr3e-4 | 3e-4 | 0.999 | 64 | 1.0 | 0.05 | 0.1 | 150k | ⚠️ batch reduced 128→64. Reward -8.2→-18.2 (best -8.1), declining. lr=3e-4 looks unstable/too high at this batch size, in line with Faly/Nick's higher-lr runs. |
| Elvis_lr5e-5 | 5e-5 | 0.999 | 64 | 1.0 | 0.05 | 0.1 | 150k | ⚠️ batch reduced 128→64. Reward -8.2→-17.1 (best -8.1), declining. Confounded by smaller batch, doesn't cleanly confirm/deny lr=5e-5 on its own. |
| Elvis_gamma95 | 1e-4 | 0.95 | 128 | 1.0 | 0.05 | 0.1 | 150k | **Reward -8.2→13.5 (best 13.5), improving** — highest raw reward of any run in the group. |
| Elvis_gamma9999 | 1e-4 | 0.9999 | 128 | 1.0 | 0.05 | 0.1 | 150k | Reward -8.2→3.9 (best 3.9), improving. |
| Elvis_batch256 | 1e-4 | 0.999 | 256 | 1.0 | 0.05 | 0.1 | 150k | Reward -8.2→6.1 (best 6.1), improving. |
| Elvis_epsstart05 | 1e-4 | 0.999 | 128 | 0.5 | 0.05 | 0.1 | 150k | Reward -8.2→6.9 (best 6.9), improving. |
| Elvis_epsend01 | 1e-4 | 0.999 | 128 | 1.0 | 0.1 | 0.1 | 150k | Reward -8.2→-0.2 (best -0.2), improving but weaker. |
| Elvis_expfrac005 | 1e-4 | 0.999 | 128 | 1.0 | 0.05 | 0.05 | 150k | Reward -8.2→1.7 (best 2.1), improving. |
| Elvis_targetupdate500 | 1e-4 | 0.999 | 128 | 1.0 | 0.05 | 0.1 | 150k | Reward -8.2→-27.0 (best -8.1), declining — faster target-network updates hurt stability. |

**Elvis's takeaway:** `Elvis_gamma95` posted the highest training-reward number
out of all of his experiments (13.5). 

## Best configuration — candidates to greedy-evaluate

| Candidate | lr | gamma | batch | eps schedule | steps | Reported reward |
|---|---|---|---|---|---|---|
| Nick Exp6 (current group champion) | 1e-4 | 0.999 | 128 | 1.0→0.05, frac 0.1 | 100k | +4.33 (greedy eval — already confirmed) |
| Elvis_gamma95 | 1e-4 | 0.95 | 128 | 1.0→0.05, frac 0.1 | 150k | 13.5 (training only — needs greedy eval) |
| Elvis_batch256 | 1e-4 | 0.999 | 256 | 1.0→0.05, frac 0.1 | 150k | 6.1 (training only — needs greedy eval) |

```bash
python play.py --model_path models/NickExp6_b128_g999/dqn_model.zip --episodes 5
python play.py --model_path models/Elvis_gamma95/dqn_model.zip --episodes 5
python play.py --model_path models/Elvis_batch256/dqn_model.zip --episodes 5
```

Whichever posts the best average greedy-eval reward becomes the final model used
for the demo clip below and the "best configuration" writeup.

**Why gamma/batch matter here:** across all three members, batch_size=128 (vs 16/32/64)
and gamma≈0.95–0.999 (vs 0.90) were the two consistent drivers of positive reward. Small
batches and low/very-high gamma extremes consistently underperformed. Learning rate
1e-4 was consistently near-optimal — both higher (3e-4, 5e-4, 1e-3) and lower (1e-5, 5e-5)
values hurt across every member's sweep.


## How to reproduce a run

```bash
python train.py --member yourname --run_name yourname_exp01 \
    --lr 1e-4 --gamma 0.999 --batch_size 128 \
    --exploration_initial_eps 1.0 --exploration_final_eps 0.05 --exploration_fraction 0.1 \
    --total_timesteps 100000
```

Then visualize training curves:
```bash
tensorboard --logdir tb_logs
```
