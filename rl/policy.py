"""Feed-forward continuous policy; raw Gaussian actions map in the CUDA task."""

import torch
from torch import nn


class HoverPolicy(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(22, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, 4)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((4,), -1.0))
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 2**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, observations):
        features = self.encoder(observations.reshape(-1, 22))
        mean = self.actor(features)
        distribution = torch.distributions.Normal(
            mean, self.log_std.clamp(-5, 2).exp().expand_as(mean), validate_args=False
        )
        return distribution, self.critic(features)

    def forward_eval(self, observations, state=(), prev_action=None):
        distribution, value = self(observations)
        return distribution, value, ()

    def initial_state(self, batch_size, device=None):
        return ()


class RGBPolicy(nn.Module):
    """Small feed-forward RGB policy for E000's staged visual loop."""

    def __init__(self, observation_shape=(12, 64, 64), hidden_size=128):
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError("observation_shape must be channels, height, width")
        self.observation_shape = tuple(observation_shape)
        self.action_size = 2
        channels = self.observation_shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, self.action_size)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((self.action_size,), -1.0))
        for layer in self.encoder:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, 2**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, observations):
        observations = observations.reshape(-1, *self.observation_shape)
        channels_first = observations.float() / 255.0
        features = self.encoder(channels_first)
        mean = self.actor(features)
        distribution = torch.distributions.Normal(
            mean,
            self.log_std.clamp(-5, 2).exp().expand_as(mean),
            validate_args=False,
        )
        return distribution, self.critic(features)

    def forward_eval(self, observations, state=(), prev_action=None):
        distribution, value = self(observations)
        return distribution, value, ()

    def initial_state(self, batch_size, device=None):
        return ()


class RecurrentRGBPolicy(nn.Module):
    """CNN+GRU RGB policy for E001; inputs are RGB history and previous action."""

    recurrent = True

    def __init__(self, observation_shape=(12, 64, 64), hidden_size=128):
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError("observation_shape must be channels, height, width")
        self.observation_shape = tuple(observation_shape)
        self.action_size = 2
        self.hidden_size = hidden_size
        channels = self.observation_shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_size),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden_size + self.action_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, self.action_size)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((self.action_size,), -1.0))
        for layer in self.encoder:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, 2**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def _features(self, observations):
        observations = observations.reshape(-1, *self.observation_shape)
        return self.encoder(observations.float() / 255.0)

    def _heads(self, features):
        mean = self.actor(features)
        distribution = torch.distributions.Normal(
            mean,
            self.log_std.clamp(-5, 2).exp().expand_as(mean),
            validate_args=False,
        )
        return distribution, self.critic(features)


    def forward_sequence(self, observations, hidden, prev_actions):
        """obs (B,T,C,H,W), hidden (B,H), prev_actions (B,T,A) shifted by one step."""
        batch, horizon = observations.shape[:2]
        features = self._features(observations).view(batch, horizon, -1)
        output, _ = self.gru(
            torch.cat([features, prev_actions], dim=-1), hidden.unsqueeze(0)
        )
        return self._heads(output.reshape(batch * horizon, -1))

    def forward_eval(self, observations, state, prev_action):
        features = self._features(observations)
        step = torch.cat([features, prev_action], dim=-1).unsqueeze(1)
        output, next_hidden = self.gru(step, state.unsqueeze(0))
        distribution, value = self._heads(output.squeeze(1))
        return distribution, value, next_hidden.squeeze(0)

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size, device=device)
