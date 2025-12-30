from dataclasses import dataclass

import numpy as np
from torch import nn

from .base import Architecture


@dataclass
class Linear(Architecture):
    def create(self, input_shape, output_dim):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), output_dim, bias=False),
        )
        return model
