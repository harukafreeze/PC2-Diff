import os
from typing import Dict, Any, Tuple, Optional, List
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9
from project_build.core.registry import DATASET_REGISTRY

QM9_Z_TO_INDEX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

@DATASET_REGISTRY.register("QM9Dataset")
class QM9DatasetWrapper(Dataset):
    """
    全量 QM9 真实分子数据集封装器 (支持完全离线加载与流形预处理)。
    """
    def __init__(
        self,
        root: str = "data/qm9",
        split: str = "train",
        remove_h: bool = False
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.remove_h = remove_h
        
        # 离线保护机制: 确保 raw 目录下的 3 个必要文件标记全部存在，防止触发 403 下载
        raw_dir = os.path.join(root, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        unchar_file = os.path.join(raw_dir, "uncharacterized.txt")
        if not os.path.exists(unchar_file):
            with open(unchar_file, "w") as f:
                f.write("") # 补齐空标记
                
        # 此时 PyG 会直接读取本地 gdb9.sdf，绝不发起任何网络请求
        self.raw_dataset = QM9(root=root)
        total_len = len(self.raw_dataset)
        
        # 固定随机种子进行无偏数据划分 (100k / 18k / 13k)
        g_cpu = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_len, generator=g_cpu).tolist()
        
        train_size = min(100000, int(total_len * 0.76))
        val_size = min(18000, int(total_len * 0.14))
        
        if split == "train":
            self.indices = indices[:train_size]
        elif split == "val":
            self.indices = indices[train_size:train_size + val_size]
        elif split == "test":
            self.indices = indices[train_size + val_size:]
        else:
            raise ValueError(f"未知划分: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_idx = self.indices[idx]
        data = self.raw_dataset[raw_idx]
        
        pos = data.pos # [N, 3]
        z = data.z     # [N]
        
        if self.remove_h:
            mask_h = (z != 1)
            pos = pos[mask_h]
            z = z[mask_h]
            
        N = pos.shape[0]
        
        # 强制执行质心归零: x_0 <- P_M x_0
        com = torch.mean(pos, dim=0, keepdim=True)
        pos_centered = pos - com
        
        h_onehot = torch.zeros(N, 5, dtype=torch.float32)
        for i, atom_z in enumerate(z.tolist()):
            type_idx = QM9_Z_TO_INDEX.get(atom_z, 1)
            h_onehot[i, type_idx] = 1.0
            
        return {
            "pos": pos_centered,
            "h": h_onehot,
            "z": z,
            "num_atoms": N
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        max_n = max(item["num_atoms"] for item in batch)
        
        pos_pad = torch.zeros(B, max_n, 3, dtype=torch.float32)
        h_pad = torch.zeros(B, max_n, 5, dtype=torch.float32)
        z_pad = torch.zeros(B, max_n, dtype=torch.long)
        mask = torch.zeros(B, max_n, dtype=torch.bool)
        
        for b, item in enumerate(batch):
            n = item["num_atoms"]
            pos_pad[b, :n] = item["pos"]
            h_pad[b, :n] = item["h"]
            z_pad[b, :n] = item["z"]
            mask[b, :n] = True
            
        return {
            "pos": pos_pad,
            "h": h_pad,
            "z": z_pad,
            "mask": mask
        }
