"""
play.py — Load a trained DQN model and watch it play the Atari environment.

Usage
-----
    python play.py --model_path models/best_run/dqn_model.zip --episodes 5 --render

Notes
-----
- Must use the SAME env_id (and same frame-stack/preprocessing) that the model
  was trained with in train.py, otherwise observations won't match and the
  agent will behave randomly or error out.
- Evaluation uses a greedy policy (deterministic=True), i.e. the agent always
  picks the action with the highest Q-value — no exploration/epsilon here.
  This is the "GreedyQPolicy" behavior the assignment asks for.
- Use --record to save an mp4 of the gameplay for the README/presentation,
  in addition to (or instead of) --render for a live GUI window.
"""

import argparse
import os
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained DQN Atari agent.")
    p.add_argument("--model_path", type=str, default="models\\run_20260708_151506\\dqn_model.zip",  #"models/best_run/dqn_model.zip"
                    help="Path to the trained dqn_model.zip.")
    p.add_argument("--env_id", type=str, default="ALE/Boxing-v5",
                    help="Must match the environment used during training.")
    p.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"],
                    help="Must match the policy used during training.")
    p.add_argument("--frame_stack", type=int, default=4,
                    help="Must match the frame stack used during training.")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to play.")
    p.add_argument("--render", action="store_true", help="Open a live render window (GUI).")
    p.add_argument("--record", action="store_true", help="Save gameplay as an mp4 video.")
    p.add_argument("--video_dir", type=str, default="videos", help="Where to save the video.")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def build_eval_env(args):
    render_mode = "human" if args.render and not args.record else "rgb_array"

    env = make_atari_env(
        args.env_id,
        n_envs=1,
        seed=args.seed,
        env_kwargs={"render_mode": render_mode} if render_mode == "rgb_array" or args.render else None,
    )
    env = VecFrameStack(env, n_stack=args.frame_stack)

    if args.policy == "CnnPolicy":
        env = VecTransposeImage(env)

    if args.record:
        os.makedirs(args.video_dir, exist_ok=True)
        env = VecVideoRecorder(
            env,
            args.video_dir,
            record_video_trigger=lambda step: step == 0,
            video_length=10_000,  # long enough to capture full episodes
            name_prefix="dqn_gameplay",
        )

    return env


def main():
    args = parse_args()

    model = DQN.load(args.model_path)
    env = build_eval_env(args)

    obs = env.reset()
    episode_rewards = []
    current_reward = 0.0
    episodes_done = 0

    print(f"Playing {args.episodes} episode(s) on {args.env_id} using {args.model_path}")

    while episodes_done < args.episodes:
        # deterministic=True -> greedy action selection (highest Q-value), no exploration.
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_reward += reward[0]

        if args.render and not args.record:
            env.render()

        if done[0]:
            episodes_done += 1
            episode_rewards.append(current_reward)
            print(f"Episode {episodes_done}/{args.episodes} — reward: {current_reward:.1f}")
            current_reward = 0.0

    env.close()

    if episode_rewards:
        avg = sum(episode_rewards) / len(episode_rewards)
        print(f"\nAverage reward over {len(episode_rewards)} episodes: {avg:.2f}")

    if args.record:
        print(f"Saved gameplay video to ./{args.video_dir}/ — use this clip in the README "
              f"and the group presentation.")


if __name__ == "__main__":
    main()