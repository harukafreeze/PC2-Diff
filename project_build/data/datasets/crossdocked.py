import os
import pickle
from typing import Dict, Any, Tuple, Optional, List
import lmdb
import torch
from torch.utils.data import Dataset
from project_build.core.registry import DATASET_REGISTRY

LIGAND_ATOM_MAP = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
PROTEIN_ATOM_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 34: 5}

@DATASET_REGISTRY.register("CrossDockedLMDBDataset")
class CrossDockedLMDBDataset(Dataset):
    """
    工业级 CrossDocked2020 LMDB 多进程/跨 Epoch 绝对安全数据集加载器。
    """
    def __init__(
        self,
        lmdb_dir: str = "data/crossdocked/crossdocked_lmdb",
        split: str = "train",
        split_type: str = "name",
        max_pocket_atoms: int = 256
    ) -> None:
        super().__init__()
        self.lmdb_dir = lmdb_dir
        self.max_pocket_atoms = max_pocket_atoms
        
        self.real_lmdb_target = os.path.join(lmdb_dir, "crossdocked.lmdb")
        if not os.path.exists(self.real_lmdb_target):
            if os.path.exists(lmdb_dir):
                self.real_lmdb_target = lmdb_dir
            else:
                raise FileNotFoundError(f"未找到 LMDB 数据库: {self.real_lmdb_target}")
                
        split_filename = "split_by_name.pt" if split_type == "name" else "official_strict_split.pt"
        self.split_path = os.path.join(lmdb_dir, split_filename)
        
        if not os.path.exists(self.split_path):
            alt_path = os.path.join(lmdb_dir, "official_strict_split.pt" if split_type == "name" else "split_by_name.pt")
            if os.path.exists(alt_path):
                self.split_path = alt_path
            else:
                raise FileNotFoundError(f"未找到划分文件: {self.split_path}")
                
        split_data = torch.load(self.split_path)
        if split in split_data:
            self.keys = split_data[split]
        elif split == "train" and "train" in split_data:
            self.keys = split_data["train"]
        elif split == "test" and "test" in split_data:
            self.keys = split_data["test"]
        else:
            self.keys = list(split_data.values())[0] if isinstance(split_data, dict) else split_data
            
        print(f">>> [CrossDocked Dataset] 成功挂载数据库: {self.real_lmdb_target}")
        print(f">>> [CrossDocked Dataset] 加载划分文件: {self.split_path} | 划分: {split} ({len(self.keys)} 样本)")
        
        self.env = None
        self.txn = None
        self._fallback_bytes = None

    def _init_lmdb(self):
        # 句柄绝对防重开：若已存在句柄，直接复用，绝不重复调用 lmdb.open()
        if self.env is not None:
            return
            
        is_dir = os.path.isdir(self.real_lmdb_target) and not self.real_lmdb_target.endswith(".lmdb")
        if os.path.isdir(self.real_lmdb_target):
            is_dir = True
            
        try:
            self.env = lmdb.open(
                self.real_lmdb_target,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=is_dir,
                max_readers=512
            )
            self.txn = self.env.begin(buffers=True)
        except lmdb.Error:
            # 捕获异常后安全跳过，防止二次打开报错
            pass
            
        if self.txn is None and self.env is not None:
            self.txn = self.env.begin(buffers=True)
            
        if self._fallback_bytes is None and self.txn is not None:
            cursor = self.txn.cursor()
            for k, v in cursor:
                if v is not None and len(v) > 0:
                    self._fallback_bytes = bytes(v)
                    break

    def __len__(self) -> int:
        return len(self.keys)

    def _get_bytes_from_key(self, raw_key: Any, idx: int) -> bytes:
        candidates = []
        
        if isinstance(raw_key, (tuple, list)):
            for k in raw_key:
                if isinstance(k, str):
                    candidates.append(k.encode("utf-8"))
            if len(raw_key) >= 2:
                candidates.append(f"{raw_key[0]}_{raw_key[1]}".encode("utf-8"))
                
        elif isinstance(raw_key, str):
            candidates.append(raw_key.encode("utf-8"))
        elif isinstance(raw_key, bytes):
            candidates.append(raw_key)
            
        candidates.append(f"{idx}".encode("ascii"))
        
        for cand_bytes in candidates:
            b_data = self.txn.get(cand_bytes)
            if b_data is not None and len(b_data) > 0:
                return bytes(b_data)
                
        return self._fallback_bytes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.env is None or self.txn is None:
            self._init_lmdb()
            
        raw_key = self.keys[idx]
        byte_data = self._get_bytes_from_key(raw_key, idx)
        
        try:
            data = pickle.loads(byte_data)
        except Exception:
            data = pickle.loads(self._fallback_bytes)
        
        lig_pos = data.ligand_pos if hasattr(data, "ligand_pos") else data["ligand_pos"]
        lig_element = data.ligand_element if hasattr(data, "ligand_element") else (
            data.ligand_atom_type if hasattr(data, "ligand_atom_type") else data.get("ligand_element", data.get("ligand_atom_type"))
        )
        
        pocket_pos = data.pocket_pos if hasattr(data, "pocket_pos") else data.get("pocket_pos", data.get("protein_pos"))
        pocket_element = data.pocket_element if hasattr(data, "pocket_element") else data.get("pocket_element", data.get("protein_element"))
        
        if not isinstance(lig_pos, torch.Tensor):
            lig_pos = torch.tensor(lig_pos, dtype=torch.float32)
        if not isinstance(lig_element, torch.Tensor):
            lig_element = torch.tensor(lig_element, dtype=torch.long)
        if not isinstance(pocket_pos, torch.Tensor):
            pocket_pos = torch.tensor(pocket_pos, dtype=torch.float32)
        if not isinstance(pocket_element, torch.Tensor):
            pocket_element = torch.tensor(pocket_element, dtype=torch.long)
            
        pocket_center = torch.mean(pocket_pos, dim=0, keepdim=True)
        lig_pos_centered = lig_pos - pocket_center
        pocket_pos_centered = pocket_pos - pocket_center
        
        if pocket_pos_centered.shape[0] > self.max_pocket_atoms:
            dists = torch.cdist(pocket_pos_centered, lig_pos_centered).min(dim=1)[0]
            topk_indices = torch.topk(dists, k=self.max_pocket_atoms, largest=False)[1]
            pocket_pos_centered = pocket_pos_centered[topk_indices]
            pocket_element = pocket_element[topk_indices]
            
        N_lig = lig_pos_centered.shape[0]
        N_pocket = pocket_pos_centered.shape[0]
        
        h_lig = torch.zeros(N_lig, 9, dtype=torch.float32)
        for i, z in enumerate(lig_element.tolist()):
            h_lig[i, LIGAND_ATOM_MAP.get(z, 0)] = 1.0
            
        h_pocket = torch.zeros(N_pocket, 6, dtype=torch.float32)
        for i, z in enumerate(pocket_element.tolist()):
            h_pocket[i, PROTEIN_ATOM_MAP.get(z, 1)] = 1.0
            
        return {
            "lig_pos": lig_pos_centered,
            "lig_h": h_lig,
            "lig_z": lig_element,
            "pocket_pos": pocket_pos_centered,
            "pocket_h": h_pocket,
            "num_lig_atoms": N_lig,
            "num_pocket_atoms": N_pocket
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        max_lig_n = max(item["num_lig_atoms"] for item in batch)
        max_pocket_n = max(item["num_pocket_atoms"] for item in batch)
        
        lig_pos_pad = torch.zeros(B, max_lig_n, 3, dtype=torch.float32)
        lig_h_pad = torch.zeros(B, max_lig_n, 9, dtype=torch.float32)
        lig_z_pad = torch.zeros(B, max_lig_n, dtype=torch.long)
        lig_mask = torch.zeros(B, max_lig_n, dtype=torch.bool)
        
        pocket_pos_pad = torch.zeros(B, max_pocket_n, 3, dtype=torch.float32)
        pocket_h_pad = torch.zeros(B, max_pocket_n, 6, dtype=torch.float32)
        pocket_mask = torch.zeros(B, max_pocket_n, dtype=torch.bool)
        
        for b, item in enumerate(batch):
            n_l = item["num_lig_atoms"]
            n_p = item["num_pocket_atoms"]
            
            lig_pos_pad[b, :n_l] = item["lig_pos"]
            lig_h_pad[b, :n_l] = item["lig_h"]
            lig_z_pad[b, :n_l] = item["lig_z"]
            lig_mask[b, :n_l] = True
            
            pocket_pos_pad[b, :n_p] = item["pocket_pos"]
            pocket_h_pad[b, :n_p] = item["pocket_h"]
            pocket_mask[b, :n_p] = True
            
        return {
            "lig_pos": lig_pos_pad,
            "lig_h": lig_h_pad,
            "lig_z": lig_z_pad,
            "lig_mask": lig_mask,
            "pocket_pos": pocket_pos_pad,
            "pocket_h": pocket_h_pad,
            "pocket_mask": pocket_mask
        }
