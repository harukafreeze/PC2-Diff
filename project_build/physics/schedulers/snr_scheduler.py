import abc
import math
from typing import Optional
import torch
import torch.nn as nn
from project_build.core.registry import SCHEDULER_REGISTRY

class BaseLossScheduler(nn.Module, abc.ABC):
    """退火调度器抽象基类"""
    def __init__(self, lambda_max: float = 1.0) -> None:
        super().__init__()
        self.lambda_max: float = float(lambda_max)

    @abc.abstractmethod
    def forward(self, alpha_bar_t: torch.Tensor, step_ratio: Optional[float] = None) -> torch.Tensor:
        """
        根据扩散系数 ᾱ_t 或当前训练步数比例计算物理损失权重因子 λ(t)。
        """
        raise NotImplementedError

@SCHEDULER_REGISTRY.register("SNRSigmoidScheduler")
class SNRSigmoidScheduler(BaseLossScheduler):
    """
    基于信噪比对数的 Sigmoid 饱和调度器 (白皮书标准推荐)。
    
    数学公式:
        SNR(t) = ᾱ_t / (1 - ᾱ_t)
        λ(t) = λ_max * Sigmoid( k * ( log(SNR(t)) - τ ) )
    """
    def __init__(
        self,
        lambda_max: float = 1.0,
        k: float = 2.0,
        tau: float = 1.0,
        eps: float = 1e-8
    ) -> None:
        super().__init__(lambda_max=lambda_max)
        self.k: float = k
        self.tau: float = tau
        self.eps: float = eps

    def forward(self, alpha_bar_t: torch.Tensor, step_ratio: Optional[float] = None) -> torch.Tensor:
        alpha_clamped = torch.clamp(alpha_bar_t, min=self.eps, max=1.0 - self.eps)
        snr = alpha_clamped / (1.0 - alpha_clamped)
        log_snr = torch.log(snr)
        
        weight = self.lambda_max * torch.sigmoid(self.k * (log_snr - self.tau))
        return weight

@SCHEDULER_REGISTRY.register("SNRCosineScheduler")
class SNRCosineScheduler(BaseLossScheduler):
    """余弦平滑退火调度器 (消融实验可选)"""
    def __init__(self, lambda_max: float = 1.0) -> None:
        super().__init__(lambda_max=lambda_max)

    def forward(self, alpha_bar_t: torch.Tensor, step_ratio: Optional[float] = None) -> torch.Tensor:
        cos_weight = 0.5 * (1.0 - torch.cos(alpha_bar_t * math.pi))
        return self.lambda_max * cos_weight

@SCHEDULER_REGISTRY.register("StepThresholdScheduler")
class StepThresholdScheduler(BaseLossScheduler):
    """阶梯阈值截断调度器 (消融实验可选，仅在信噪比高于阈值时开启)"""
    def __init__(self, lambda_max: float = 1.0, snr_threshold: float = 1.0) -> None:
        super().__init__(lambda_max=lambda_max)
        self.snr_threshold = snr_threshold

    def forward(self, alpha_bar_t: torch.Tensor, step_ratio: Optional[float] = None) -> torch.Tensor:
        snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)
        mask = (snr >= self.snr_threshold).float()
        return self.lambda_max * mask
