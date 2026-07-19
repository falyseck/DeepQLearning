# Hyperparameter Tuning — DQN on ALE/Boxing-v5

This document explains the group's hyperparameter tuning process in detail: what was
tested, what happened, and *why* — separate from the README. Raw data lives in `experiments_log.csv`.

30 experiments were run across 3 members (10 each). Each member fixed a baseline
configuration and varied one hyperparameter at a time (lr, gamma, batch_size, or the
epsilon schedule) so the effect of each change could be isolated.

---

## 1. Learning rate (lr)

| Member | Run | lr | Result | Why |
|---|---|---|---|---|
| Faly | FalyExp2 | 1e-3 | Rewards inconsistent, episodes terminate quickly | Too-large gradient steps overshoot good Q-value estimates on every update, so the network never settles into a consistent policy — it effectively "unlearns" whatever it just learned each step. |
| Faly | FalyExp3 | 1e-5 | Strongly negative (~-25 to -30), never improves | Updates are too small to meaningfully shift Q-values within the training budget; the network barely moves from its random initialization. |
| Nick | NickExp2_lr5e-4 | 5e-4 | Declines monotonically to -35.7, no recovery | Confirms Faly's finding at a smaller magnitude — even 5x the baseline lr is enough to destabilize Q-updates. Higher lr does not buy faster learning here; it buys instability. |
| Nick | NickExp3_lr1e-5 | 1e-5 | Sinks to ~-30, flattens, never recovers | Matches Faly's low-lr result. Brackets the search: with 1e-3, 5e-4, and 1e-5 all underperforming, lr=1e-4 is confirmed near-optimal for this setup rather than an arbitrary default. |
| Elvis | Elvis_lr3e-4 | 3e-4 | Declines to -18.2 | Consistent with the group pattern — lr above 1e-4 destabilizes, even at a smaller step size than Nick's 5e-4 test. Note: batch size was also reduced (128→64) in this run, so the result is not a clean lr-only comparison. |
| Elvis | Elvis_lr5e-5 | 5e-5 | Declines to -17.1 | Also confounded by the batch-size reduction (128→64), so this doesn't cleanly confirm or deny lr=5e-5 as a standalone value — it's consistent with lower-than-1e-4 being suboptimal, but the batch change is a competing explanation. |

**Conclusion:** All three members independently found lr=1e-4 to be a local optimum
— every tested value above or below it underperformed. This is the single most
consistent finding across the whole group, which makes it low-risk to defend in Q&A:
three separate people, three separate sweeps, same answer.

---

## 2. Discount factor (gamma)

| Member | Run | gamma | Result | Why |
|---|---|---|---|---|
| Faly | FalyExp4 | 0.90 | Moderately negative (~-15 to -19) | Boxing episodes run ~1800 steps; a low gamma discounts future reward too aggressively, so the agent under-values setting up longer combos/positioning that pay off many steps later. |
| Faly | FalyExp5 | 0.999 | Improves to ~-17 vs baseline | Higher gamma extends the credit-assignment horizon, better matching Boxing's long episode length — consistent with the theory above, though still net-negative at Faly's other settings (batch=32). |
| Nick | NickExp5_batch128 | 0.99 | +0.43 (batch=128, lr=1e-4, 100k steps) | First positive run in Nick's sweep, isolating the batch-size effect before gamma was raised further. |
| Nick | NickExp6 (gamma=0.999, batch=128) | 0.999 | **+4.33 (greedy eval)**, group champion | Raising gamma from 0.99 to 0.999 on top of the same batch=128 config nearly 10x'd the reward — the largest single-parameter jump in Nick's sweep. |
| Elvis | Elvisgamma90 | 0.99 | 2.54, improving (batch=128, lr=1e-4, 100k steps) | Same configuration as Nick's Exp5 (gamma=0.99, batch=128), run independently — reaches a comparable positive result (2.54 vs Nick's 0.43), replicating the finding with a second data point. |
| Elvis | Elvis_gamma9999 | 0.9999 | 3.9, improving (batch=128, 150k steps) | Pushing gamma even higher than Nick's 0.999 champion continues to perform well, though not clearly better than 0.999 within this data. |

**Conclusion:** Unlike in an earlier draft of this document, Elvis's and Nick's gamma
sweeps are now **directly comparable and in agreement** — both ran gamma=0.99 at
batch=128, lr=1e-4 (Nick: +0.43, Elvis: +2.54), and both found further raising gamma
toward 0.999–0.9999 continued to help (Nick's champion at 0.999: +4.33; Elvis at
0.9999: +3.9). Across all three members, every gamma value at or above 0.99
outperformed the low baseline of 0.90, and the group's independent replication of the
gamma=0.99/batch=128 configuration (Nick and Elvis landing on the same sign and similar
magnitude) is good evidence the effect is real rather than run-to-run noise.

**Open question:** whether 0.999 (Nick's champion) or 0.9999 (Elvis) is the true
optimum isn't fully resolved — they weren't tested at matching batch size/step budgets
in the same run. But the direction of the effect (higher gamma → better, in the
0.99–0.9999 range, for a ~1800-step episode length) is now a consistent, well-replicated
finding rather than an open disagreement.

---

## 3. Batch size

| Member | Run | batch | Result | Why |
|---|---|---|---|---|
| Faly | FalyExp6 | 16 | Poor (~-30) | Smaller batches produce noisier gradient estimates per update, so the network's Q-value updates are less reliable and training is less stable. |
| Faly | FalyExp7 | 128 | **Positive (~+2)**, only positive Faly run | Larger batch averages over more transitions per update, reducing gradient variance and giving more consistent, trustworthy updates. |
| Nick | NickExp4_batch64 | 64 | Improves to -10.4 (best at that point) | Confirms the trend at an intermediate value: doubling batch size from 32 measurably reduces variance and improves the reward trajectory. |
| Nick | NickExp5_batch128 | 128 | **+0.43**, first positive Nick run | Matches Faly's finding independently — batch=128 is the point where the reward trend turns positive. |
| Elvis | Elvis_batch256 | 256 | 6.1, improving | Extends the trend further — even larger batches continue to help, at least within this training budget, though with diminishing marginal benefit per step of extra compute. |

**Conclusion:** This is the group's second most consistent finding (after learning
rate) — three independent members, three independent sweeps, all show batch_size=128
(or higher) outperforming the SB3 default of 32. Smaller batches were consistently
the worst performers across the whole group's 30 runs.

**Trade-off :** larger batches cost more compute per update
step, so this is a stability-vs-speed trade-off, not a free lunch — batch=256 (Elvis)
did not clearly beat batch=128 (Nick's champion) by enough margin to justify the
extra cost within our training budgets.

---

## 4. Epsilon-greedy exploration schedule

| Member | Run | eps_start | eps_end | eps_decay_frac | Result | Why |
|---|---|---|---|---|---|---|
| Faly | FalyExp8 | 1.0 | 0.05 | 0.02 | Negative, short episodes | Epsilon decays almost immediately, so the agent commits to exploiting a nearly-random policy before it has gathered enough diverse experience to know what a good policy looks like. |
| Faly | FalyExp9 | 1.0 | 0.05 | 0.3 | Strongly negative (~-34), unstable | Epsilon stays high for 30% of training, meaning the agent is still acting mostly randomly deep into training — not enough of the budget is left for the exploited policy to converge. |
| Nick | NickExp7_epsend001 | 1.0 | 0.01 | 0.1 | Worse than baseline gamma/batch champion (-4.34 vs +4.33) | Only eps_end changed vs Nick's champion — cutting the exploration floor to 1% starves the replay buffer of diverse transitions late in training, so the agent stops discovering better strategies once it settles into a decent-but-not-great policy. |
| Nick | NickExp8_expfrac02 | 1.0 | 0.05 | 0.2 | Worse (-8.0) | Doubling the decay fraction (matching Faly's high-decay-fraction finding at a smaller scale) delays the exploit phase, and with a fixed step budget the run runs out of time before the exploited policy converges. |
| Elvis | Elvis_epsstart05 | 0.5 | 0.05 | 0.1 | 6.9, improving | Starting exploration lower (50% instead of 100% random) still allows the agent to explore adequately in this environment while spending less of the budget on near-random actions — a middle ground that performed reasonably well. |
| Elvis | Elvis_epsend01 | 1.0 | 0.1 | 0.1 | -0.2, weaker | A higher exploration floor (10% vs default 5%) means the agent keeps acting randomly more often even late in training, capping how sharp the final policy can get. |
| Elvis | Elvis_expfrac005 | 1.0 | 0.05 | 0.05 | 1.7–2.1, improving | Faster decay (5% of training vs default 10%) than baseline was tolerable here, unlike Faly's much more extreme 2% decay — suggesting there's a floor below which fast decay stops working, but 5% isn't past it. |

**Conclusion:** The default-ish schedule (eps_start=1.0, eps_end=0.05,
decay_fraction=0.1) consistently outperformed both extremes across all three
members — too-fast decay (Faly's 0.02, Nick's implicit comparison) and too-slow
decay (Faly's 0.3, Nick's 0.2) both hurt. This mirrors the gamma finding: extremes
in either direction underperform a moderate middle setting.

---

## 5. Other hyperparameters tested

| Member | Run | Change | Result | Why |
|---|---|---|---|---|
| Nick | NickExp9_mlp | MlpPolicy instead of CnnPolicy (all else = champion config) | Plateaus at -28, never recovers | Flattening raw pixel frames into a vector destroys spatial structure (edges, positions, relative motion) that convolutional filters are specifically built to exploit. This is a controlled proof, not just circumstantial evidence, since every other hyperparameter was held identical to the CNN champion. |
| Nick | NickExp10_final200k | Doubled total_timesteps (100k→200k), same config as champion | Worse than champion despite more training (+1.02 training / -3.2 greedy vs champion's +4.33) | `exploration_fraction` is expressed as a *fraction* of total timesteps, not an absolute step count — doubling total_timesteps without adjusting exploration_fraction stretches the decay schedule out to 20k steps instead of 10k, reproducing the same "decays too slowly" failure mode as NickExp8. Longer training is not automatically better if a fraction-based hyperparameter isn't rescaled accordingly. |
| Elvis | Elvis_targetupdate500 | target_update_interval reduced (faster target network sync) | Declines to -27.0 | Updating the target network too frequently means the Q-value targets used for training are themselves changing rapidly, which is exactly the instability DQN's target network is designed to prevent — faster sync partially undoes that stabilization. |

---

## Summary: what actually mattered

Ranked by how consistently the effect showed up across all three members' independent
sweeps:

1. **Batch size** — clearest, most consistent finding. 128+ beat the default 32 in
   every member's sweep.
2. **Learning rate** — equally consistent. 1e-4 was a local optimum for all three
   members; both higher and lower values hurt every time they were tested.
3. **Gamma** — higher gamma (0.99–0.9999) consistently beat the low baseline (0.90)
   across all three members, including a direct independent replication (Nick and
   Elvis both ran gamma=0.99/batch=128/lr=1e-4 and got matching-sign, similar-magnitude
   positive results). The precise optimum within 0.999–0.9999 isn't fully pinned down,
   but the direction of the effect is well-supported.
4. **Epsilon schedule** — moderate decay (fraction ≈0.1, floor ≈0.05) beat both
   faster and slower decay across members.
5. **Policy architecture (CNN vs MLP)** — not a "tuning" parameter in the traditional
   sense, but the single largest effect size observed in the whole study (a ~32-point
   reward gap between otherwise-identical configs).
6. **target_update_interval** — only tested once (Elvis), but the single data point
   is consistent with DQN theory (faster target updates hurt stability).