from typing import Tuple
import torch
import torch.nn as nn

class DifferentiableKabschLoss(nn.Module):
    """
    可微 Kabsch 刚体旋转配准损失 (AlphaFold / TargetDiff 3D 几何标准)。
    
    数学机制:
        利用 SVD 奇异值分解计算预测构象 pred 与真值 target 之间的最佳正交旋转矩阵 R* ∈ SO(3)，
        消除无意义的全局刚体旋转方差，精准计算三维坐标结构均方误差 (Kabsch RMSD Loss)。
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            pred: [B, N, 3] 网络预测的三维坐标
            target: [B, N, 3] 真实三维坐标
            
        返回:
            (loss, pred_aligned): 刚体配近后的 MSE 损失与对齐坐标
        """
        B, N, _ = pred.shape
        device = pred.device
        
        # 1. 质心对齐
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        target_centered = target - torch.mean(target, dim=1, keepdim=True)
        
        # 2. 计算协方差矩阵 H = pred^T * target [B, 3, 3]
        H = torch.matmul(pred_centered.transpose(1, 2), target_centered)
        
        # 3. 批处理 SVD 奇异值分解: H = U * S * V^T
        U, S, V = torch.linalg.svd(H)
        
        # 4. 计算最佳旋转矩阵 R = V * U^T [B, 3, 3]
        R = torch.matmul(V, U.transpose(1, 2))
        
        # 5. 反射镜像校正 (保证 det(R) == +1，严格位于 SO(3) 旋转群内)
        det = torch.det(R)
        reflection_mask = (det < 0).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
        
        if reflection_mask.any():
            V_clone = V.clone()
            V_clone[reflection_mask.squeeze(-1).squeeze(-1), :, -1] *= -1.0
            R = torch.matmul(V_clone, U.transpose(1, 2))
            
        # 6. 对齐坐标
        pred_aligned = torch.matmul(pred_centered, R.transpose(1, 2))
        
        # 7. 计算对齐后的真正几何坐标均方误差
        loss_msd = torch.mean((pred_aligned - target_centered) ** 2)
        return loss_msd, pred_aligned
