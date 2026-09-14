"""PufferLib 4.0 torch learner adaptation; provenance in PROVENANCE.txt.

Preserves Muon, prioritized segment replay, V-trace correction and clipped
policy/value objectives. Only the CUDA environment boundary, trajectory indexing,
bootstrap handling, and host-synchronizing profiling are replaced.
"""

from collections import defaultdict
import math
import torch
from .muon import Muon


def sample_logits(logits, action=None):
    if isinstance(logits, torch.distributions.Normal):
        batch = logits.loc.shape[0]
        if action is None:
            # torch.normal(mean, std) validates tensor std through a host read.
            # Reparameterized arithmetic samples the same Gaussian on-device.
            action = (logits.loc + logits.scale * torch.randn_like(logits.loc)).view(
                batch, -1
            )
        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy


class PuffeRL:
    def __init__(self, config, vec, policy):
        self.config = config
        self.device = vec.device
        self._vec = vec
        self.policy = policy
        self.total_agents = n = vec.n
        self.observation_shape = tuple(getattr(vec, "observation_shape", (22,)))
        self.action_size = int(getattr(vec, "action_size", 4))
        horizon = config["horizon"]
        self.batch_size = n * horizon
        self.minibatch_segments = config["minibatch_size"] // horizon
        if self.minibatch_segments < 1 or config["minibatch_size"] % horizon:
            raise ValueError("minibatch_size must be a positive multiple of horizon")
        self.total_epochs = max(
            1, (config["total_timesteps"] + self.batch_size - 1) // self.batch_size
        )

        def buffer(*shape):
            return torch.zeros(*shape, device=self.device)

        self.observations = buffer(horizon, n, *self.observation_shape)
        self.actions = buffer(horizon, n, self.action_size)
        self.values = buffer(horizon, n)
        self.logprobs = buffer(horizon, n)
        self.rewards = buffer(horizon, n)
        self.terminated = buffer(horizon, n)
        self.truncated = buffer(horizon, n)
        self.final_values = buffer(horizon, n)
        self.bootstrap_value = buffer(n)
        self.advantages = buffer(n, horizon)
        self.ratio = torch.ones(n, horizon, device=self.device)
        self.episode_stats = buffer(3)
        self.paired = bool(config.get("paired", False)) and getattr(
            vec, "paired_observations", None
        ) is not None
        if self.paired:
            self.paired_observations = buffer(horizon, n, *self.observation_shape)
        self.recurrent = bool(getattr(policy, "recurrent", False))
        if self.recurrent:
            self.hidden_size = int(policy.hidden_size)
            self.segment_states = buffer(n, self.hidden_size)
            self.prev_actions = buffer(horizon, n, self.action_size)
            self._state = policy.initial_state(n, self.device)
            self._prev_action = torch.zeros(n, self.action_size, device=self.device)
        else:
            self._state = ()
            self._prev_action = None
        self.optimizer = Muon(
            policy.parameters(),
            lr=config["learning_rate"],
            momentum=config["beta1"],
            eps=config["eps"],
        )
        self.epoch = self.global_step = 0

    @torch.no_grad()
    def rollouts(self):
        env = self._vec
        self.episode_stats.zero_()
        for t in range(self.config["horizon"]):
            self.observations[t].copy_(env.observations)
            if self.recurrent:
                if t == 0:
                    self.segment_states.copy_(self._state)
                self.prev_actions[t].copy_(self._prev_action)
            logits, value, next_state = self.policy.forward_eval(
                self.observations[t], self._state, self._prev_action
            )
            if self.paired:
                self.paired_observations[t].copy_(env.paired_observations)
            action, logprob, _ = sample_logits(logits)
            self.actions[t].copy_(action)
            self.logprobs[t].copy_(logprob)
            self.values[t].copy_(value.flatten())
            env.step(self.actions[t])
            self.rewards[t].copy_(env.rewards)
            self.terminated[t].copy_(env.terminated)
            self.truncated[t].copy_(env.truncated)
            # Every final state is evaluated without a host-side done check.
            # A failed nonfinite state must not poison the critic through NaN*0.
            _, final_value, _ = self.policy.forward_eval(
                torch.nan_to_num(env.final_observations),
                next_state if self.recurrent else (),
                action if self.recurrent else None,
            )
            self.final_values[t].copy_(final_value.flatten())
            done = (env.terminated + env.truncated).clamp(max=1)
            if self.recurrent:
                reset = done.unsqueeze(-1)
                self._state = next_state * (1.0 - reset)
                self._prev_action.copy_(action * (1.0 - reset))
            self.episode_stats[0].add_(done.sum())
            self.episode_stats[1].add_((env.completed_returns * done).sum())
            self.episode_stats[2].add_((env.completed_lengths * done).sum())
        _, value, _ = self.policy.forward_eval(
            env.observations, self._state, self._prev_action
        )
        self.bootstrap_value.copy_(value.flatten())
        self.global_step += self.batch_size

    def train(self):
        losses = defaultdict(float)
        config = self.config
        device = self.device

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        learning_rate = config["learning_rate"]
        if config["anneal_lr"] and self.epoch > 0:
            lr_ratio = self.epoch / self.total_epochs
            lr_min = config["learning_rate"] * config["min_lr_ratio"]
            learning_rate = lr_min + 0.5 * (learning_rate - lr_min) * (
                1 + math.cos(math.pi * lr_ratio)
            )
            self.optimizer.param_groups[0]["lr"] = learning_rate

        # Transpose from [horizon, agents] (contiguous writes) to [agents, horizon] (minibatch indexing)
        obs = self.observations.transpose(0, 1).contiguous()
        act = self.actions.transpose(0, 1).contiguous()
        val = self.values.T.contiguous()
        lp = self.logprobs.T.contiguous()
        rew = self.rewards.T.contiguous().clamp(-1, 1)
        ter = self.terminated.T.contiguous()
        tru = self.truncated.T.contiguous()
        final_val = self.final_values.T.contiguous()

        num_minibatches = max(
            1, int(config["replay_ratio"] * self.batch_size / config["minibatch_size"])
        )
        for mb in range(num_minibatches):
            advantages = self.advantages
            advantages = compute_puff_advantage(
                val,
                rew,
                ter,
                tru,
                final_val,
                self.bootstrap_value,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )

            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(
                prio_probs, self.minibatch_segments, replacement=True
            )
            mb_prio = (self.total_agents * prio_probs[idx, None]) ** -anneal_beta

            mb_obs = obs[idx]
            mb_actions = act[idx]
            mb_logprobs = lp[idx]
            mb_values = val[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            if self.recurrent:
                logits, newvalue = self.policy.forward_sequence(
                    mb_obs,
                    self.segment_states[idx],
                    self.prev_actions.transpose(0, 1)[idx],
                )
            else:
                logits, newvalue = self.policy(mb_obs)
            if self.paired:
                mb_paired = self.paired_observations.transpose(0, 1)[idx]
                consistency = (
                    self.policy.features(mb_obs) - self.policy.features(mb_paired)
                ).pow(2).mean()
            else:
                consistency = torch.zeros((), device=device)
            actions, newlogprob, entropy = sample_logits(logits, action=mb_actions)

            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            # Duplicate replacement samples have identical pre-update predictions.
            self.ratio.index_copy_(0, idx, ratio.detach())

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = (
                pg_loss
                + config["vf_coef"] * v_loss
                - config["ent_coef"] * entropy_loss
                + config.get("consistency_coef", 0.0) * consistency
            )
            val.index_copy_(0, idx, newvalue.detach().float())

            losses["policy_loss"] += pg_loss
            losses["value_loss"] += v_loss
            losses["entropy"] += entropy_loss
            losses["old_approx_kl"] += old_approx_kl
            losses["approx_kl"] += approx_kl
            losses["clipfrac"] += clipfrac
            losses["importance"] += ratio.mean()
            losses["consistency"] += consistency

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), config["max_grad_norm"]
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        losses = {k: v.item() / num_minibatches for k, v in losses.items()}
        y_pred = val.flatten()
        y_true = advantages.flatten() + val.flatten()
        var_y = y_true.var()
        explained_var = torch.where(
            var_y > 0, 1 - (y_true - y_pred).var() / var_y, torch.nan
        ).item()
        losses["explained_variance"] = explained_var

        self.losses = losses
        self.epoch += 1


@torch.jit.script
def compute_puff_advantage(
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    final_values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
):
    # A row is one environment's full temporal segment, not flattened agents.
    # r[t] belongs to (s[t], a[t]); preserve all horizon actions, including the last.
    carry = torch.zeros_like(bootstrap_value)
    for t in range(values.size(1) - 1, -1, -1):
        next_value = bootstrap_value if t == values.size(1) - 1 else values[:, t + 1]
        next_value = torch.where(truncated[:, t] > 0, final_values[:, t], next_value)
        next_value = torch.where(
            terminated[:, t] > 0, torch.zeros_like(next_value), next_value
        )
        rho = ratio[:, t].clamp(max=vtrace_rho_clip)
        c = ratio[:, t].clamp(max=vtrace_c_clip)
        delta = rho * (rewards[:, t] + gamma * next_value - values[:, t])
        continuation = (terminated[:, t] + truncated[:, t]) == 0
        carry = delta + gamma * gae_lambda * c * torch.where(
            continuation, carry, torch.zeros_like(carry)
        )
        advantages[:, t].copy_(carry)
    return advantages
