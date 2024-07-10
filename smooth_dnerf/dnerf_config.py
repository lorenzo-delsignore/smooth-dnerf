"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from smooth_dnerf.nerfacto_dnerf import NerfactoDnerfModelConfig
from smooth_dnerf.nerfacto_nerfplayer_dnerf import NerfactoNerfplayerDnerfModelConfig
from smooth_dnerf.vanilla_dnerf import NeRFModel, VanillaDnerfModelConfig

nerfacto_dnerf = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto-dnerf",
        steps_per_eval_batch=1000,
        steps_per_eval_image=1000,
        steps_per_save=2000,
        max_num_iterations=30020,
        steps_per_eval_all_images=30000,
        mixed_precision=False,
        use_grad_scaler=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DNeRFDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_res_scale_factor=0.5,
            ),
            model=NerfactoDnerfModelConfig(
                average_init_density=0.01,
                enable_temporal_distortion=True,
                temporal_distortion_params={
                    "lipschitz": True,
                },
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-12),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "temporal_distortion": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-12),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
        log_gradients=True,
    ),
    description="Nerfacto with dnerf backbone",
)

nerfacto_nerfplayer_dnerf = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto-nerfplayer-dnerf",
        steps_per_eval_batch=1000,
        steps_per_eval_image=1000,
        steps_per_save=2000,
        max_num_iterations=30020,
        steps_per_eval_all_images=30000,
        mixed_precision=False,
        use_grad_scaler=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DNeRFDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_res_scale_factor=0.5,  # DNeRF train on 400x400
            ),
            model=NerfactoNerfplayerDnerfModelConfig(
                average_init_density=0.01,
                enable_temporal_distortion=True,
                temporal_distortion_params={
                    "lipschitz": True,
                },
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-12),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "temporal_distortion": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-12),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
        log_gradients=True,
    ),
    description="Nerfacto with proposals of Nerfplayer with dnerf temporal distortion",
)

vanilla_dnerf = MethodSpecification(
    config=TrainerConfig(
        method_name="vanilla-dnerf",
        steps_per_eval_batch=1000,
        steps_per_eval_image=1000,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        mixed_precision=False,
        use_grad_scaler=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DNeRFDataParserConfig(),
            ),
            model=VanillaDnerfModelConfig(
                _target=NeRFModel,
                enable_temporal_distortion=True,
                temporal_distortion_params={
                    "lipschitz": True,
                },
                eval_num_rays_per_chunk=1024,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
            "temporal_distortion": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
        log_gradients=True,
    ),
    description="Vanilla NeRF with temporal distortion",
)
