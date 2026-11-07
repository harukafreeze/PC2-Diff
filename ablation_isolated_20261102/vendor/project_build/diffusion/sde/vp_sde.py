import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

class DualSpaceVPSDE(nn.Module):
    """
    连续-离散双轨 VP-SDE 扩散系统 (带 Tweedie 投影与后验步进).
    """
    def __init__(
        self,
        schedule_type: str = "cosine",
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        cosine_s: float = 0.008
    ) -> None:
        super().__init__()
        self.schedule_type = schedule_type
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s

    def compute_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = torch.clamp(t, min=0.0, max=1.0)
        if self.schedule_type == "cosine":
            f_t = torch.cos(((t_clamped + self.cosine_s) / (1.0 + self.cosine_s)) * (math.pi / 2.0)) ** 2
            f_0 = math.cos((self.cosine_s / (1.0 + self.cosine_s)) * (math.pi / 2.0)) ** 2
            alpha_bar = f_t / f_0
        elif self.schedule_type == "linear":
            log_alpha_bar = -0.5 * (self.beta_max - self.beta_min) * (t_clamped ** 2) - self.beta_min * t_clamped
            alpha_bar = torch.exp(log_alpha_bar)
        else:
            raise ValueError(f"不支持的调度类型: {self.schedule_type}")
            
        return torch.clamp(alpha_bar, min=1e-4, max=0.9999)

    def sample_forward_noise(
        self,
        x_0: torch.Tensor,
        h_0: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x_0.shape
        alpha_bar_t = self.compute_alpha_bar(t)
        
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(B, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t).view(B, 1, 1)
        
        eps_x_raw = torch.randn_like(x_0)
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            eps_x_raw = eps_x_raw * mask_float
            num_nodes = torch.clamp(torch.sum(mask_float, dim=1, keepdim=True), min=1.0)
            eps_x = (eps_x_raw - torch.sum(eps_x_raw, dim=1, keepdim=True) / num_nodes) * mask_float
            eps_h = torch.randn_like(h_0) * mask.unsqueeze(-1).float()
        else:
            eps_x = eps_x_raw - torch.mean(eps_x_raw, dim=1, keepdim=True)
            eps_h = torch.randn_like(h_0)
            
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * eps_x
        h_t = sqrt_alpha_bar * h_0 + sqrt_one_minus_alpha_bar * eps_h
        
        if mask is not None:
            x_t = x_t * mask_float
            x_t = (x_t - torch.sum(x_t, dim=1, keepdim=True) / num_nodes) * mask_float
        else:
            x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)
            
        return x_t, h_t, eps_x, eps_h, alpha_bar_t

    def compute_tweedie_x0(
        self,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        eps_x_pred: torch.Tensor,
        eps_h_pred: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于 Tweedie 引理 2.1 从预测噪声反解后验洁净构象 (x̂_0, ĥ_0)。
        """
        B, N, _ = x_t.shape
        sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-4)).view(B, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-4)).view(B, 1, 1)
        
        x0_hat = (x_t - sqrt_one_minus_alpha_bar * eps_x_pred) / sqrt_alpha_bar
        h0_hat = (h_t - sqrt_one_minus_alpha_bar * eps_h_pred) / sqrt_alpha_bar
        
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            x0_hat = x0_hat * mask_float
            num_nodes = torch.clamp(torch.sum(mask_float, dim=1, keepdim=True), min=1.0)
            mean_x0 = torch.sum(x0_hat, dim=1, keepdim=True) / num_nodes
            x0_hat = (x0_hat - mean_x0) * mask_float
            h0_hat = h0_hat * mask.unsqueeze(-1).float()
        else:
            x0_hat = x0_hat - torch.mean(x0_hat, dim=1, keepdim=True)
            
        return x0_hat, h0_hat

    def q_posterior_step(
        self,
        x_0_pred: torch.Tensor,
        h_0_pred: torch.Tensor,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        t_curr: torch.Tensor,
        t_prev: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_t.shape[0]
        a_bar_c = self.compute_alpha_bar(t_curr).view(B, 1, 1)
        a_bar_p = self.compute_alpha_bar(t_prev).view(B, 1, 1)
        
        alpha_t = torch.clamp(a_bar_c / (a_bar_p + 1e-8), min=1e-4, max=1.0)
        beta_t = 1.0 - alpha_t
        
        c_x0 = torch.sqrt(a_bar_p) * beta_t / (1.0 - a_bar_c + 1e-8)
        c_xt = torch.sqrt(alpha_t) * (1.0 - a_bar_p) / (1.0 - a_bar_c + 1e-8)
        
        mu_x = c_x0 * x_0_pred + c_xt * x_t
        mu_h = c_x0 * h_0_pred + c_xt * h_t
        
        sigma_sq = beta_t * (1.0 - a_bar_p) / (1.0 - a_bar_c + 1e-8)
        sigma_t = torch.sqrt(torch.clamp(sigma_sq, min=0.0))
        
        z_x_raw = torch.randn_like(x_t)
        z_h = torch.randn_like(h_t)
        
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            num_nodes = torch.clamp(torch.sum(mask_float, dim=1, keepdim=True), min=1.0)
            z_x = (z_x_raw - torch.sum(z_x_raw * mask_float, dim=1, keepdim=True) / num_nodes) * mask_float
            x_prev = mu_x + sigma_t * z_x
            h_prev = mu_h + sigma_t * z_h * mask.unsqueeze(-1).float()
            x_prev = x_prev * mask_float
            x_prev = (x_prev - torch.sum(x_prev, dim=1, keepdim=True) / num_nodes) * mask_float
        else:
            z_x = z_x_raw - torch.mean(z_x_raw, dim=1, keepdim=True)
            x_prev = mu_x + sigma_t * z_x
            h_prev = mu_h + sigma_t * z_h
            x_prev = x_prev - torch.mean(x_prev, dim=1, keepdim=True)
            
        return x_prev, h_prev
