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

    def forward_eval(self, observations, state=()):
        distribution, value = self(observations)
        return distribution, value, ()

    def initial_state(self, batch_size, device=None):
        return ()
