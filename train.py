"""
train.py — Train a DQN agent on an Atari environment using Stable Baselines3.

Usage examples
--------------
Run with defaults:
    python train.py

Run one of your 10 hyperparameter experiments (override anything via CLI flags):
    python train.py --lr 1e-4 --gamma 0.99 --batch_size 32 \
                     --exploration_initial_eps 1.0 --exploration_final_eps 0.05 \
                     --exploration_fraction 0.1 --policy CnnPolicy \
                     --total_timesteps 200000 --member alice --run_name lr1e-4_g99

Each run:
  - trains a DQN agent on the chosen Atari env
  - logs reward/episode-length trends to TensorBoard (./tb_logs/<run_name>)
  - saves the final model to models/<run_name>/dqn_model.zip
  - appends a row to experiments_log.csv so the group's hyperparameter table
    can be generated automatically instead of copy-pasted by hand.

Notes for the group
--------------------
- Everyone should use the SAME environment (--env_id) so results are comparable.
- Use --member <your name> and a unique --run_name for every one of your 10 runs.
- Keep --total_timesteps modest (e.g. 100k-300k) so 30 experiments across the
  group finish in reasonable time; you're comparing trends, not chasing full
  convergence on every single run.
"""

import argparse
import csv
import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import ale_py

def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN agent on an Atari environment.")

    # Identification (for the shared hyperparameter table / presentation)
    p.add_argument("--member", type=str, default="unknown",
                    help="Your name, e.g. --member alice (used in experiments_log.csv)")
    p.add_argument("--run_name", type=str, default=None,
                    help="Unique name for this run. Defaults to a timestamp.")

    # Environment & policy
    p.add_argument("--env_id", type=str, default="ALE/Boxing-v5",
                    help="Gymnasium Atari environment id (keep this the same across the group).")
    p.add_argument("--policy", type=str, default="CnnPolicy",
                    choices=["CnnPolicy", "MlpPolicy"],
                    help="CnnPolicy is standard for Atari (raw pixel input). "
                         "MlpPolicy is included only for the required MLP-vs-CNN comparison.")
    p.add_argument("--n_envs", type=int, default=4, help="Number of parallel envs.")
    p.add_argument("--frame_stack", type=int, default=4, help="Number of frames to stack.")

    # Core hyperparameters to tune (per assignment instructions)
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--batch_size", type=int, default=32, help="Minibatch size.")
    p.add_argument("--buffer_size", type=int, default=100_000, help="Replay buffer size.")
    p.add_argument("--learning_starts", type=int, default=10_000,
                    help="Steps of random exploration before learning starts.")
    p.add_argument("--train_freq", type=int, default=4, help="Update the model every N steps.")
    p.add_argument("--target_update_interval", type=int, default=1000,
                    help="Steps between target network updates.")

    # Epsilon-greedy exploration schedule (maps to epsilon_start/epsilon_end/epsilon_decay)
    p.add_argument("--exploration_initial_eps", type=float, default=1.0,
                    help="Epsilon start.")
    p.add_argument("--exploration_final_eps", type=float, default=0.05,
                    help="Epsilon end.")
    p.add_argument("--exploration_fraction", type=float, default=0.1,
                    help="Fraction of training over which epsilon decays "
                         "(this is SB3's equivalent of 'epsilon_decay').")

    # Training length / logging
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", type=str, default="tb_logs")
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--log_csv", type=str, default="experiments_log.csv")

    return p.parse_args()


class EpisodeStatsCallback(BaseCallback):
    """Prints reward/episode-length trends periodically during training."""

    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            ep_info_buffer = self.model.ep_info_buffer
            if ep_info_buffer and len(ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in ep_info_buffer]
                lengths = [ep["l"] for ep in ep_info_buffer]
                mean_r = sum(rewards) / len(rewards)
                mean_l = sum(lengths) / len(lengths)
                print(f"[step {self.num_timesteps}] "
                      f"mean_reward={mean_r:.2f}  mean_ep_length={mean_l:.1f}")
        return True


def build_env(env_id: str, n_envs: int, frame_stack: int, policy: str, seed: int):
    """
    Builds the Atari env. Uses SB3's make_atari_env helper, which applies the
    standard Atari preprocessing wrappers (frame skip, grayscale, resize, etc.).
    """
    vec_env = make_atari_env(env_id, n_envs=n_envs, seed=seed)
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    if policy == "CnnPolicy":
        # SB3's CnnPolicy expects channel-first images.
        vec_env = VecTransposeImage(vec_env)

    return vec_env


def main():
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    env = build_env(args.env_id, args.n_envs, args.frame_stack, args.policy, args.seed)

    model = DQN(
        policy=args.policy,
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    callback = EpisodeStatsCallback(check_freq=10_000)

    print(f"\n=== Starting run: {run_name} ({args.member}) ===")
    print(f"Env: {args.env_id} | Policy: {args.policy} | Timesteps: {args.total_timesteps}")
    print(f"lr={args.lr} gamma={args.gamma} batch_size={args.batch_size} "
          f"eps_start={args.exploration_initial_eps} eps_end={args.exploration_final_eps} "
          f"eps_decay_fraction={args.exploration_fraction}\n")

    model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name=run_name)

    # Save model
    model_path = os.path.join(args.model_dir, run_name)
    os.makedirs(model_path, exist_ok=True)
    save_path = os.path.join(model_path, "dqn_model.zip")
    model.save(save_path)
    print(f"\nSaved model to {save_path}")

    # Append a row to the shared experiments log (becomes the hyperparameter table)
    log_exists = os.path.isfile(args.log_csv)
    with open(args.log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                "member", "run_name", "env_id", "policy", "lr", "gamma", "batch_size",
                "exploration_initial_eps", "exploration_final_eps", "exploration_fraction",
                "total_timesteps", "model_path", "noted_behavior"
            ])
        writer.writerow([
            args.member, run_name, args.env_id, args.policy, args.lr, args.gamma,
            args.batch_size, args.exploration_initial_eps, args.exploration_final_eps,
            args.exploration_fraction, args.total_timesteps, save_path,
            "TODO: fill in observed behavior after reviewing TensorBoard logs"
        ])

    env.close()
    print(f"Logged this run to {args.log_csv}. Fill in the 'noted_behavior' column by hand "
          f"after reviewing reward/episode-length trends in TensorBoard "
          f"(run: tensorboard --logdir {args.log_dir}).")


if __name__ == "__main__":
    main()