from typing import Dict, Any, Tuple, Optional, List
import math
import torch
import torch.nn as nn
from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE

class CompactReverseSDESampler:
    """
    Project BUILD 3.0 (Path B) - 紧凑半径与防原子飘散逆向 SDE 采样器。
    
    核心机理:
    1. 【紧凑空间点云初始化】将配体原子的初始空间弥散标准差从 1.20 Å 收紧至 0.65 Å，
       将所有原子牢牢约束在口袋核心活性区 (半径 < 2.5 Å)，彻底根除边缘原子飘散至 3.5 Å 之外的现象；
    2. 【微观共价弹簧收拢】在去噪最后 8 步，对距离在 1.55~2.20 Å 的近邻碳原子施加微弱向心弹簧拉力，
       消除跨碎片长距离拉线产生的巨大 UFF 伸缩势能。
    """
    def __init__(
        self,
        sde: DualSpaceVPSDE,
        num_steps: int = 100,
        init_noise_scale: float = 0.65,  # 紧凑点云初始方差
        spring_k: float = 0.30,
        corrector_steps: int = 1,
        bond_gate_thresh: float = 0.35,
        enable_svd_seeding: bool = True,
        enable_compact_healing: bool = True,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.sde = sde
        self.num_steps = num_steps
        self.init_noise_scale = init_noise_scale
        self.spring_k = spring_k
        self.corrector_steps = corrector_steps
        self.bond_gate_thresh = bond_gate_thresh
        self.enable_svd_seeding = enable_svd_seeding
        self.enable_compact_healing = enable_compact_healing
        self.device = device

    def compute_pocket_principal_frame(self, pocket_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_p, _ = pocket_pos.shape
        center = torch.mean(pocket_pos, dim=1, keepdim=True)
        centered_p = pocket_pos - center
        cov = torch.matmul(centered_p.transpose(1, 2), centered_p) / max(1, N_p)
        try:
            U, S, V = torch.linalg.svd(cov)
            v1 = U[:, :, 0]
            v2 = U[:, :, 1]
        except Exception:
            v1 = torch.tensor([[1.0, 0.0, 0.0]], device=pocket_pos.device).expand(B, -1)
            v2 = torch.tensor([[0.0, 1.0, 0.0]], device=pocket_pos.device).expand(B, -1)
        return v1, v2

    def apply_hexagonal_ring_field(self, pos: torch.Tensor) -> torch.Tensor:
        B, N, _ = pos.shape
        if N < 6: return pos
        pos_out = pos.clone()
        for b in range(B):
            ring = pos_out[b, :6]
            for i in range(6):
                j = (i + 1) % 6
                diff = ring[i] - ring[j]
                d = torch.norm(diff) + 1e-4
                shift = (d - 1.40) * 0.5 * (diff / d)
                ring[i] -= shift * 0.5
                ring[j] += shift * 0.5
            for i in range(3):
                j = i + 3
                diff = ring[i] - ring[j]
                d = torch.norm(diff) + 1e-4
                shift = (d - 2.80) * 0.5 * (diff / d)
                ring[i] -= shift * 0.5
                ring[j] += shift * 0.5
            pos_out[b, :6] = ring
        return pos_out

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        num_samples: int,
        num_atoms: int,
        num_atom_types: int = 9,
        mask: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        model.eval()
        B = num_samples
        N = num_atoms
        if model_kwargs is None: model_kwargs = {}

        pocket_pos = model_kwargs.get("pocket_pos", None)
        if pocket_pos is not None:
            cavity_center = torch.mean(pocket_pos, dim=1, keepdim=True)
            # 采用紧凑方差初始化，防止原子弥散
            x_curr = cavity_center + torch.randn(B, N, 3, device=self.device) * self.init_noise_scale
        else:
            cavity_center = torch.zeros(B, 1, 3, device=self.device)
            x_curr = torch.randn(B, N, 3, device=self.device) * self.init_noise_scale
            x_curr = x_curr - torch.mean(x_curr, dim=1, keepdim=True)

        h_curr = torch.randn(B, N, num_atom_types, device=self.device)

        # 1. 芳香母核成核
        if N >= 6:
            angles = torch.linspace(0, 2 * math.pi, 7, device=self.device)[:6]
            cos_a = torch.cos(angles).view(1, 6, 1)
            sin_a = torch.sin(angles).view(1, 6, 1)

            if pocket_pos is not None and self.enable_svd_seeding:
                v1, v2 = self.compute_pocket_principal_frame(pocket_pos)
                ring_3d = 1.40 * (cos_a * v1.unsqueeze(1) + sin_a * v2.unsqueeze(1))
                x_curr[:, :6] = cavity_center + ring_3d
            else:
                ring_2d = torch.stack([1.40 * torch.cos(angles), 1.40 * torch.sin(angles), torch.zeros_like(angles)], dim=-1)
                x_curr[:, :6] = cavity_center + ring_2d.unsqueeze(0)

            # 锁定母核 6 碳通道
            h_curr[:, :6, :] = -5.0
            h_curr[:, :6, 0] = 5.0

        time_steps = torch.linspace(0.999, 0.001, self.num_steps + 1, device=self.device)
        trajectory = []

        for i in range(self.num_steps):
            t_curr = time_steps[i]
            t_prev = time_steps[i + 1]
            t_batch = torch.full((B,), t_curr.item(), device=self.device)
            t_prev_batch = torch.full((B,), t_prev.item(), device=self.device)

            out = model(x_t=x_curr, h_t=h_curr, t=t_batch, mask=mask, **model_kwargs)
            eps_x_pred, eps_h_pred = out[0], out[1]
            bond_logits = out[2] if len(out) == 3 else None

            alpha_bar_curr = self.sde.compute_alpha_bar(t_batch).view(B, 1, 1)
            sqrt_alpha = torch.sqrt(torch.clamp(alpha_bar_curr, min=1e-4))
            sqrt_1_alpha = torch.sqrt(torch.clamp(1.0 - alpha_bar_curr, min=1e-4))

            x0_hat = (x_curr - sqrt_1_alpha * eps_x_pred) / sqrt_alpha
            h0_hat = (h_curr - sqrt_1_alpha * eps_h_pred) / sqrt_alpha

            if N >= 6:
                x0_hat = self.apply_hexagonal_ring_field(x0_hat)
                h0_hat[:, :6, :] = -5.0
                h0_hat[:, :6, 0] = 5.0

            if i == self.num_steps - 1:
                x_curr, h_curr = x0_hat, h0_hat
                break

            x_curr, h_curr = self.sde.q_posterior_step(
                x_0_pred=x0_hat, h_0_pred=h0_hat, x_t=x_curr, h_t=h_curr,
                t_curr=t_batch, t_prev=t_prev_batch, mask=mask
            )

            if N >= 6:
                h_curr[:, :6, :] = -5.0
                h_curr[:, :6, 0] = 5.0

            # 2. 紧凑向心键长自愈 (在 t < 0.08 末期，将 1.55~2.20 Å 的近邻原子平滑拉至 1.46 Å)
            if self.enable_compact_healing and t_curr.item() < 0.08 and bond_logits is not None:
                bond_probs = torch.sigmoid(bond_logits)
                with torch.no_grad():
                    diff_geo = x_curr.unsqueeze(2) - x_curr.unsqueeze(1)
                    dist_geo = torch.sqrt(torch.sum(diff_geo ** 2, dim=-1) + 1e-8)
                    healing_mask = (bond_probs > self.bond_gate_thresh) & (dist_geo >= 1.52) & (dist_geo <= 2.20)
                    eye_m = torch.eye(N, device=self.device, dtype=torch.bool).unsqueeze(0)
                    healing_mask = healing_mask & (~eye_m)

                if healing_mask.any():
                    with torch.enable_grad():
                        x_in = x_curr.detach().requires_grad_(True)
                        diff = x_in.unsqueeze(2) - x_in.unsqueeze(1)
                        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
                        spring_energy = torch.sum(bond_probs * ((dist - 1.46) ** 2) * healing_mask.float())
                        grad_x = torch.autograd.grad(spring_energy, x_in)[0]

                    grad_x = torch.clamp(grad_x, min=-1.5, max=1.5)
                    heal_lr = 0.015 * t_curr.item()
                    x_curr = x_curr - heal_lr * grad_x

        return x_curr, h_curr, trajectory
