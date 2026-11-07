import abc
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from project_build.core.registry import POTENTIAL_REGISTRY

class BasePotential(nn.Module, abc.ABC):
    """
    所有物理与量子化学势能插件的抽象基类。
    
    统一规范能量计算、可微受力推导（-∇_x U）、显存隔离截断机制，
    以及在特殊正交群 SO(3) 作用下的几何不变性自检接口。
    """
    def __init__(self, name: str = "BasePotential") -> None:
        super().__init__()
        self._potential_name = name

    @property
    def potential_name(self) -> str:
        return self._potential_name

    @abc.abstractmethod
    def compute_energy(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        计算分子体系的总势能 U(x)。
        
        参数:
            pos: [B, N, 3] 或 [Total_Nodes, 3] 的连续空间原子坐标矩阵 (单位: Å)
            atom_types: [B, N] 或 [Total_Nodes] 的离散原子类型张量 (标量索引)
            batch_indices: [Total_Nodes] 批处理索引映射
            
        返回:
            energy: [B] 各分子构象的总势能
        """
        raise NotImplementedError("所有势能插件必须显式实现 compute_energy 方法！")

    def forward(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        return self.compute_energy(pos=pos, atom_types=atom_types, batch_indices=batch_indices, **kwargs)

    def compute_forces(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        detach_force: bool = True,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        利用自动微分精确计算原子受力场: F(x) = -∇_x U(x)。
        """
        if not pos.requires_grad:
            pos_eval = pos.clone().detach().requires_grad_(True)
        else:
            pos_eval = pos

        energy = self.compute_energy(
            pos=pos_eval,
            atom_types=atom_types,
            batch_indices=batch_indices,
            **kwargs
        )
        
        energy_sum = torch.sum(energy)
        grad_outputs = torch.ones_like(energy_sum)
        
        grads = torch.autograd.grad(
            outputs=energy_sum,
            inputs=pos_eval,
            grad_outputs=grad_outputs,
            create_graph=not detach_force,
            retain_graph=not detach_force,
            only_inputs=True
        )[0]
        
        if grads is None:
            raise RuntimeError(f"势能插件 '{self._potential_name}' 的梯度计算返回了 None！")
            
        forces = -grads
        
        if detach_force:
            forces = forces.detach()
            
        return energy, forces

    def verify_so3_invariance(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        tolerance: float = 1e-4
    ) -> bool:
        """
        数值检验势能函数在随机 3D 旋转变换下的 SO(3) 不变性: U(R·x) == U(x)。
        """
        self.eval()
        with torch.no_grad():
            e_orig = self.compute_energy(pos, atom_types)
            
            q, _ = torch.linalg.qr(torch.randn(3, 3, device=pos.device, dtype=pos.dtype))
            if torch.det(q) < 0:
                q[:, 0] = -q[:, 0]
            rot_mat = q
            
            pos_rot = torch.matmul(pos, rot_mat)
            e_rot = self.compute_energy(pos_rot, atom_types)
            
            max_abs_diff = torch.max(torch.abs(e_orig - e_rot)).item()
            return max_abs_diff < tolerance
