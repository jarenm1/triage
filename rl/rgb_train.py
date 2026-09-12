"""Small E000 RGB/PufferLib integration smoke test."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.policy import RGBPolicy
from rl.rgb_environment import RGBEnv
from rl.vendor.torch_pufferl import PuffeRL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.horizon <= 0 or args.steps <= 0:
        parser.error("num-envs, horizon and steps must be positive")
    if args.steps % (args.num_envs * args.horizon) != 0:
        parser.error("steps must be a multiple of num-envs*horizon")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed_all(args.seed)
    with RGBEnv(
        n=args.num_envs,
        seed=args.seed,
        device=args.device,
        library=args.library,
    ) as env:
        policy = RGBPolicy(env.observation_shape).to(env.device)
        config = {
            "horizon": args.horizon,
            "total_timesteps": args.steps,
            "minibatch_size": args.horizon * min(64, args.num_envs),
            "learning_rate": 0.0003,
            "replay_ratio": 1.0,
            "beta1": 0.9,
            "eps": 1e-8,
            "prio_alpha": 0.8,
            "prio_beta0": 0.2,
            "clip_coef": 0.2,
            "vf_clip_coef": 0.2,
            "anneal_lr": True,
            "min_lr_ratio": 0.0,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
            "vf_coef": 0.5,
            "ent_coef": 0.001,
            "max_grad_norm": 1.0,
        }
        learner = PuffeRL(config, env, policy)
        while learner.global_step < args.steps:
            learner.rollouts()
            learner.train()
            print(
                json.dumps(
                    {
                        "steps": learner.global_step,
                        "losses": learner.losses,
                        "rgb_checksum": env.checksum(),
                        "completed_episodes": float(learner.episode_stats[0].item()),
                    }
                ),
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "steps": learner.global_step,
                    "parameters": sum(parameter.numel() for parameter in policy.parameters()),
                    "device": str(env.device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device)
                    if torch.cuda.is_available()
                    else None,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
