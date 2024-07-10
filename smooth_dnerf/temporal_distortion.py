import torch
from jaxtyping import Float
from nerfstudio.field_components.encodings import Encoding, SHEncoding
from nerfstudio.field_components.mlp import MLP
from torch import Tensor, nn

from smooth_dnerf.lipschitz import LipschitzMLP


class DNeRFDistortion(nn.Module):
    """Optimizable temporal deformation using an MLP.
    Args:
        position_encoding: An encoding for the XYZ of distortion
        time_embedding: Size of the time embedding
        time_num_layers: Number of layers in temporal encoding MLP
        time_size: Size of hidden layer for the temporal encoding MLP
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
        self,
        position_encoding: Encoding = SHEncoding(levels=4),
        mlp_num_layers: int = 3,
        mlp_layer_width: int = 128,
        lipschitz=False,
        time_embedding=256,
        time_num_layers=2,
        time_size=128,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.lipschitz = lipschitz
        self.temporal_encoding = MLP(
            in_dim=1,
            out_dim=time_embedding,
            num_layers=time_num_layers,
            layer_width=time_size,
        )
        if lipschitz is True:
            self.mlp_deform = LipschitzMLP(
                in_dim=self.position_encoding.get_out_dim() + time_embedding,
                out_dim=3,
                num_layers=mlp_num_layers,
                layer_width=mlp_layer_width,
            )
        else:
            self.mlp_deform = MLP(
                in_dim=self.position_encoding.get_out_dim() + time_embedding,
                out_dim=3,
                num_layers=mlp_num_layers,
                layer_width=mlp_layer_width,
            )

    def forward(
        self, positions: Float[Tensor, "*bs 3"], times: Float[Tensor, "*bs 1"]
    ) -> Float[Tensor, "*bs 3"]:
        p = self.position_encoding(positions)
        t = self.temporal_encoding(times)
        return self.mlp_deform(torch.cat([p, t], dim=-1))
