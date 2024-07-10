import math
from typing import Optional, Set, Tuple

import torch
from jaxtyping import Float
from nerfstudio.field_components.base_field_component import FieldComponent
from torch import Tensor, nn


class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), requires_grad=True)
        )
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus(beta=100)
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max()

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(
            input, self.weight * scale.unsqueeze(1), self.bias
        )


class LipschitzMLP(FieldComponent):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = (
            set(skip_connections) if skip_connections else set()
        )
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(LipschitzLinear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert (
                        i not in self._skip_connections
                    ), "Skip connection at layer 0 doesn't make sense."
                    linear_layer = LipschitzLinear(self.in_dim, self.layer_width)
                    layers.append(linear_layer)
                elif i in self._skip_connections:
                    linear_layer = LipschitzLinear(
                        self.layer_width + self.in_dim, self.layer_width
                    )
                    layers.append(linear_layer)
                else:
                    linear_layer = LipschitzLinear(self.layer_width, self.layer_width)
                    layers.append(linear_layer)
            linear_layer = LipschitzLinear(self.layer_width, self.out_dim)
            layers.append(linear_layer)
        self.layers = nn.ModuleList(layers)

    def pytorch_fwd(
        self, in_tensor: Float[Tensor, "*bs in_dim"]
    ) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def forward(
        self, in_tensor: Float[Tensor, "*bs in_dim"]
    ) -> Float[Tensor, "*bs out_dim"]:
        return self.pytorch_fwd(in_tensor)

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        return loss_lipc
