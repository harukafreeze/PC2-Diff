"""
Project BUILD 3.0 - 几何等变与局域径向图主干库
"""
from project_build.models.backbones.egnn_layer import EquivariantPhysicsGraphLayer
from project_build.models.backbones.egnn_backbone import DualSpaceEGNNBackbone
from project_build.models.backbones.target_sbdd_layer import LocalRadialSBDDLayer
from project_build.models.backbones.target_sbdd_backbone import TargetSBDDBackbone

# 兼容旧命名别名
TargetAwarePGCrossAttentionLayer = LocalRadialSBDDLayer

__all__ = [
    "EquivariantPhysicsGraphLayer",
    "DualSpaceEGNNBackbone",
    "LocalRadialSBDDLayer",
    "TargetAwarePGCrossAttentionLayer",
    "TargetSBDDBackbone",
]
