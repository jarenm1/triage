"""Checkpoint helpers for the E000 RGB learner."""

from pathlib import Path

import torch


SCHEMA = "E000/2-rgb-checkpoint-v1"


def _device_of(policy):
    return next(policy.parameters()).device


def save_checkpoint(path, *, policy, learner, env, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": SCHEMA,
        "recurrent": bool(getattr(policy, "recurrent", False)),
        "seed": int(seed),
        "observation_shape": tuple(env.observation_shape),
        "action_size": int(env.action_size),
        "config": dict(learner.config),
        "policy": policy.state_dict(),
        "optimizer": learner.optimizer.state_dict(),
        "epoch": int(learner.epoch),
        "global_step": int(learner.global_step),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, *, policy, learner=None, env=None, expected_config=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(
        path,
        map_location=_device_of(policy),
        weights_only=True,
    )
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("unsupported RGB checkpoint schema")
    if env is not None:
        if tuple(checkpoint["observation_shape"]) != tuple(env.observation_shape):
            raise ValueError("checkpoint observation shape does not match environment")
        if int(checkpoint["action_size"]) != int(env.action_size):
            raise ValueError("checkpoint action size does not match environment")
    if expected_config is not None:
        saved_config = checkpoint["config"]
        for key, value in expected_config.items():
            if key != "total_timesteps" and saved_config.get(key) != value:
                raise ValueError(f"checkpoint config mismatch for {key}")
    if bool(checkpoint.get("recurrent", False)) != bool(
        getattr(policy, "recurrent", False)
    ):
        raise ValueError("checkpoint policy kind does not match the given policy")
    policy.load_state_dict(checkpoint["policy"])
    if learner is not None:
        learner.optimizer.load_state_dict(checkpoint["optimizer"])
        learner.epoch = int(checkpoint["epoch"])
        learner.global_step = int(checkpoint["global_step"])
    return checkpoint


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_state = checkpoint.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
