"""
Project BUILD - 物理与力场插件库
自动导入并注册所有势能插件、外部力场引擎与退火调度器。
"""
from project_build.physics.potentials.base_potential import BasePotential
from project_build.physics.potentials.valency_field import ContinuousValencyPotential
from project_build.physics.potentials.pocket_attraction import PocketSurfaceAttractionPotential
from project_build.physics.schedulers.snr_scheduler import (
    BaseLossScheduler,
    SNRSigmoidScheduler,
    SNRCosineScheduler,
    StepThresholdScheduler,
)

__all__ = [
    "BasePotential",
    "ContinuousValencyPotential",
    "PocketSurfaceAttractionPotential",
    "BaseLossScheduler",
    "SNRSigmoidScheduler",
    "SNRCosineScheduler",
    "StepThresholdScheduler",
]
