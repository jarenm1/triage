"""CUDA hover learning and checkpoint evaluation: python rl/train.py --help."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.environment import HoverEnv, TASK_CONFIG
from rl.policy import HoverPolicy
from rl.vendor.torch_pufferl import PuffeRL

PUFFERLIB_COMMIT = "42f70d6932c30ac977736f861006809c50168ba9"
CHECKPOINT_VERSION = 1


def seed_all(seed, device):
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(policy, n, seed, device, library=None, baseline=False):
    """First episodes only; reset states never enter success or return metrics."""
    with HoverEnv(
        n=n, seed=seed, max_steps=2000, device=device, library=library
    ) as env:
        active = torch.ones(n, dtype=torch.bool, device=env.device)
        safe = active.clone()
        failed = torch.zeros_like(active)
        returns = torch.zeros(n, device=env.device)
        lengths = torch.zeros(n, device=env.device)
        max_distance = torch.zeros(n, device=env.device)
        max_tilt = torch.zeros(n, device=env.device)
        zeros = torch.zeros(n, 4, device=env.device)
        # The native final observation is always the pre-autoreset state.
        for _ in range(2000):
            if baseline:
                actions = zeros
            else:
                distribution, _, _ = policy.forward_eval(env.observations)
                actions = distribution.mean.contiguous()
            env.step(actions)
            final = env.final_observations
            distance = final[:, :3].norm(dim=1)
            tilt = final[:, 14].clamp(-1, 1).acos()
            finite = torch.isfinite(final).all(dim=1)
            safe &= ~active | (finite & (distance <= 1.0) & (tilt <= 0.7))
            max_distance = torch.where(
                active, torch.maximum(max_distance, distance), max_distance
            )
            max_tilt = torch.where(active, torch.maximum(max_tilt, tilt), max_tilt)
            returns.add_(torch.where(active, env.rewards, 0))
            lengths.add_(active)
            failed |= active & (env.terminated > 0)
            active &= (env.terminated + env.truncated) == 0
        success = safe & ~failed & (lengths == 2000)
        # One result transfer, after the entire evaluation rollout.
        metrics = (
            torch.stack(
                [
                    success.float().mean(),
                    returns.mean(),
                    lengths.mean(),
                    failed.float().mean(),
                    max_distance.mean(),
                    max_tilt.mean(),
                    success.float().sum(),
                ]
            )
            .cpu()
            .tolist()
        )
        return dict(
            zip(
                [
                    "success_rate",
                    "mean_return",
                    "mean_length",
                    "failure_rate",
                    "mean_max_distance",
                    "mean_max_tilt",
                    "success_count",
                ],
                metrics,
            ),
            episodes=n,
            seed=seed,
            duration_seconds=20,
            criterion="complete 2000 controls without failure; distance <=1m and tilt <=0.7rad throughout",
            policy="zero_raw_action_hover_equilibrium"
            if baseline
            else "checkpoint_mean_action",
        )


def save_checkpoint(path, learner, args, baseline, evaluation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "pufferlib_commit": PUFFERLIB_COMMIT,
            "task": dict(TASK_CONFIG, max_steps=args.max_steps),
            "policy_config": {"hidden_size": args.hidden_size},
            "policy": learner.policy.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "train_config": learner.config,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "global_step": learner.global_step,
            "epoch": learner.epoch,
            "torch_version": str(torch.__version__),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(args.device),
            "baseline": baseline,
            "evaluation": evaluation,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--checkpoint", default="cuda/build/hover.pt")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--replay-ratio", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="updates between evaluation/checkpointing (0: final only)",
    )
    args = parser.parse_args()
    for name in (
        "num_envs",
        "horizon",
        "steps",
        "eval_envs",
        "hidden_size",
        "max_steps",
        "minibatch_size",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.replay_ratio <= 0 or not math.isfinite(args.replay_ratio):
        parser.error("--replay-ratio must be finite and positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        parser.error("--learning-rate must be finite and positive")
    if args.eval_every < 0 or args.minibatch_size % args.horizon:
        parser.error(
            "eval-every must be nonnegative; minibatch-size must be a multiple of horizon"
        )
    seed_all(args.seed, args.device)
    if args.mode == "eval":
        checkpoint = torch.load(
            args.checkpoint, map_location=f"cuda:{args.device}", weights_only=True
        )
        if (
            checkpoint["checkpoint_version"] != CHECKPOINT_VERSION
            or checkpoint["pufferlib_commit"] != PUFFERLIB_COMMIT
        ):
            raise ValueError("Unsupported checkpoint/learner version")
        expected_task = dict(TASK_CONFIG, max_steps=checkpoint["task"]["max_steps"])
        if checkpoint["task"] != expected_task:
            raise ValueError(
                "Checkpoint task contract does not match this native adapter"
            )
        args.eval_seed = (
            checkpoint["eval_seed"] if args.eval_seed is None else args.eval_seed
        )
        if args.eval_seed == checkpoint["seed"]:
            parser.error("evaluation seed must be held out from the training seed")
        policy = HoverPolicy(**checkpoint["policy_config"]).cuda(args.device)
        policy.load_state_dict(checkpoint["policy"])
        policy.eval()
        print(
            json.dumps(
                {
                    "baseline": evaluate(
                        policy,
                        args.eval_envs,
                        args.eval_seed,
                        args.device,
                        args.library,
                        True,
                    ),
                    "evaluation": evaluate(
                        policy,
                        args.eval_envs,
                        args.eval_seed,
                        args.device,
                        args.library,
                    ),
                }
            ),
            flush=True,
        )
        return
    args.eval_seed = (
        (args.seed + 1_000_003) % (2**63) if args.eval_seed is None else args.eval_seed
    )
    if args.eval_seed == args.seed:
        parser.error("evaluation seed must be held out from the training seed")
    policy = HoverPolicy(args.hidden_size).cuda(args.device)
    baseline = evaluate(
        policy, args.eval_envs, args.eval_seed, args.device, args.library, True
    )
    initial = evaluate(
        policy, args.eval_envs, args.eval_seed, args.device, args.library
    )
    print(json.dumps({"baseline": baseline, "initial_policy": initial}), flush=True)
    config = dict(
        horizon=args.horizon,
        total_timesteps=args.steps,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        replay_ratio=args.replay_ratio,
        beta1=0.9,
        eps=1e-8,
        prio_alpha=0.8,
        prio_beta0=0.2,
        clip_coef=0.2,
        vf_clip_coef=0.2,
        anneal_lr=True,
        min_lr_ratio=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0,
        vf_coef=0.5,
        ent_coef=0.001,
        max_grad_norm=1.0,
    )
    with HoverEnv(
        args.num_envs, args.seed, args.max_steps, args.device, args.library
    ) as env:
        learner = PuffeRL(config, env, policy)
        start = time.perf_counter()
        evaluation = initial
        while learner.global_step < args.steps:
            policy.train()
            learner.rollouts()
            learner.train()
            if learner.epoch % args.log_every == 0:
                count, total_return, total_length = learner.episode_stats.cpu().tolist()
                print(
                    json.dumps(
                        {
                            "epoch": learner.epoch,
                            "steps": learner.global_step,
                            "steps_per_second": learner.global_step
                            / (time.perf_counter() - start),
                            "completed_episodes": count,
                            "mean_episode_return": total_return / count
                            if count
                            else None,
                            "mean_episode_length": total_length / count
                            if count
                            else None,
                            "loss": learner.losses,
                        }
                    ),
                    flush=True,
                )
            if (
                args.eval_every and learner.epoch % args.eval_every == 0
            ) or learner.global_step >= args.steps:
                policy.eval()
                evaluation = evaluate(
                    policy, args.eval_envs, args.eval_seed, args.device, args.library
                )
                save_checkpoint(args.checkpoint, learner, args, baseline, evaluation)
                print(
                    json.dumps(
                        {
                            "steps": learner.global_step,
                            "evaluation": evaluation,
                            "checkpoint": args.checkpoint,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
