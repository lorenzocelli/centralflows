from dataclasses import dataclass

import numpy as np
from torch import nn

from .base import ActivationList, Architecture, make_activation

@dataclass
class MLP(Architecture):
    activation: ActivationList = "gelu"
    width: int = 256

    def create(self, input_shape, output_dim):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), self.width),
            make_activation(self.activation),
            nn.Linear(self.width, self.width),
            make_activation(self.activation),
            nn.Linear(self.width, output_dim),
        )

        return model


@dataclass
class UnbiasedMLP(Architecture):
    activation: ActivationList = "gelu"
    width: int = 256

    def create(self, input_shape, output_dim):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), self.width, bias=False),
            make_activation(self.activation),
            nn.Linear(self.width, self.width, bias=False),
            make_activation(self.activation),
            nn.Linear(self.width, output_dim, bias=False),
        )

        return model
