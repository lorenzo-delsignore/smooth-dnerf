# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2024 Lorenzo Del Signore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications:
# - Add temporal distortion to Nerfplayer-Nerfacto.
# - Add option to use Lipschitz temporal distortion.

"""
Implementation of vanilla nerf.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, cast

import numpy as np
import tinycudann as tcnn
import torch
from jaxtyping import Float
from nerfplayer.temporal_grid import TemporalGridEncoder
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.utils import colormaps
from torch import Tensor
from torch.nn import Parameter

from smooth_dnerf.temporal_distortion import DNeRFDistortion


@dataclass
class NerfactoNerfplayerDnerfModelConfig(NerfactoModelConfig):
    """NerfactoNerfplayerDnerf Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoNerfplayerDnerf)
    near_plane: float = 2
    """How far along the ray to start sampling."""
    far_plane: float = 6
    """How far along the ray to stop sampling."""
    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"lipschitz": True})
    """Parameters to instantiate temporal distortion with"""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "temporal_dim": 32,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 64,
            },
            {
                "hidden_dim": 16,
                "temporal_dim": 32,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 256,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "uniform"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    temporal_tv_weight: float = 1.0
    """Temporal TV balancing weight for feature channels."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.08
    """Distortion loss multiplier."""
    lipschitz_loss_mult: float = 3e-4
    """Lipschitz loss multiplier."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""


class NerfactoNerfplayerDnerf(Model):
    """Vanilla NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: NerfactoNerfplayerDnerfModelConfig

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        appearance_embedding_dim = (
            self.config.appearance_embed_dim
            if self.config.use_appearance_embedding
            else 0
        )
        self.field_fine = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )
        if getattr(self.config, "enable_temporal_distortion", False):
            self.temporal_distortion_model = DNeRFDistortion(
                **self.config.temporal_distortion_params
            )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks: List[TemporalHashMLPDensityField] = []
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = TemporalHashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = TemporalHashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )
        self.proposal_networks = cast(
            Sequence[TemporalHashMLPDensityField],
            torch.nn.ModuleList(self.proposal_networks),
        )

        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        initial_sampler = None
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.normals_shader = NormalsShader()
        self.rgb_loss = MSELoss()
        self.step = 0
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = True
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field_fine.parameters())
        if self.temporal_distortion_model is not None:
            param_groups["temporal_distortion"] = list(
                self.temporal_distortion_model.parameters()
            )
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        assert ray_bundle.times is not None, "Time not provided."
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle,
            density_fns=[
                functools.partial(f, times=ray_bundle.times) for f in self.density_fns
            ],
        )
        if self.temporal_distortion_model is not None:
            offsets = None
            if ray_samples.times is not None:
                offsets = self.temporal_distortion_model(
                    ray_samples.frustums.get_positions(), ray_samples.times
                )
            ray_samples.frustums.set_offsets(offsets)
        field_outputs_fine = self.field_fine.forward(ray_samples)
        weights_fine = ray_samples.get_weights(
            field_outputs_fine[FieldHeadNames.DENSITY]
        )
        weights_list.append(weights_fine)
        ray_samples_list.append(ray_samples)
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        with torch.no_grad():
            depth_fine = self.renderer_depth(
                weights=weights_fine, ray_samples=ray_samples
            )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        expected_depth = self.renderer_expected_depth(
            weights=weights_fine, ray_samples=ray_samples
        )
        outputs = {
            "rgb_fine": rgb_fine,
            "accumulation_fine": accumulation_fine,
            "depth_fine": depth_fine,
            "expected_depth": expected_depth,
        }
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )
        return outputs

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        if self.training:
            loss_dict["interlevel_loss"] = (
                self.config.interlevel_loss_mult
                * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = (
                self.config.distortion_loss_mult * metrics_dict["distortion"]
            )
            loss_dict["temporal_tv_loss"] = 0
            for net in self.proposal_networks:
                loss_dict["temporal_tv_loss"] += net.encoding.get_temporal_tv_loss()
            loss_dict["temporal_tv_loss"] *= self.config.temporal_tv_weight
            if self.temporal_distortion_model.lipschitz is True:
                loss_dict["lipschitz_loss"] = (
                    self.config.lipschitz_loss_mult
                    * self.temporal_distortion_model.mlp_deform.get_lipschitz_loss()
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb_fine = outputs["rgb_fine"]
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        assert self.config.collider_params is not None
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        combined_rgb = torch.cat([image, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_fine], dim=1)
        combined_depth = torch.cat([depth_fine], dim=1)
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)
        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation_fine"],
            )
            images_dict[key] = prop_depth_i
        return metrics_dict, images_dict

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb_fine"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        return metrics_dict


class TemporalHashMLPDensityField(Field):
    """A lightweight temporal density field module.

    Args:
        aabb: Parameters of scene aabb bounds
        temporal_dim: Hashing grid parameter. A higher temporal dim means a higher temporal frequency.
        num_layers: Number of hidden layers
        hidden_dim: Dimension of hidden layers
        spatial_distortion: Spatial distortion module
        num_levels: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        max_res: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        base_res: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        log2_hashmap_size: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        features_per_level: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
    """

    def __init__(
        self,
        aabb: Tensor,
        temporal_dim: int = 64,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.encoding = TemporalGridEncoder(
            input_dim=3,
            temporal_dim=temporal_dim,
            num_levels=num_levels,
            level_dim=features_per_level,
            per_level_scale=growth_factor,
            base_resolution=base_res,
            log2_hashmap_size=log2_hashmap_size,
        )
        self.linear = tcnn.Network(
            n_input_dims=num_levels * features_per_level,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

    def density_fn(
        self,
        positions: Float[Tensor, "*bs 3"],
        times: Optional[Float[Tensor, "bs 1"]] = None,
    ) -> Float[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
            times: the time of rays
        """
        assert (
            times is not None
        ), "TemporalHashMLPDensityField requires times to be specified"
        if len(positions.shape) == 3 and len(times.shape) == 2:
            # position is [ray, sample, 3]; times is [ray, 1]
            times = times[:, None]  # RaySamples can handle the shape
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )
        positions_flat = positions.view(-1, 3)
        assert ray_samples.times is not None
        time_flat = ray_samples.times.reshape(-1, 1)
        x = self.encoding(positions_flat, time_flat).to(positions)
        density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)
        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        return density, None

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> dict:
        return {}
