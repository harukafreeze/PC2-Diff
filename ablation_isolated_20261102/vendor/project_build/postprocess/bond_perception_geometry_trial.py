from typing import Dict, Any, Tuple, Optional, List, Set
import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# SBDD 9 通道重原子权威映射 (CrossDocked 标准)
SBDD_HEAVY_REVERSE_MAP = {
    0: 6,   # C
    1: 7,   # N
    2: 8,   # O
    3: 9,   # F
    4: 15,  # P
    5: 16,  # S
    6: 17,  # Cl
    7: 35,  # Br
    8: 53   # I
}

QM9_REVERSE_MAP = {
    0: 1,   # H
    1: 6,   # C
    2: 7,   # N
    3: 8,   # O
    4: 9    # F
}

# 卤素原子严格单价
HALOGENS = {9, 17, 35, 53}

# 各元素绝对禁止突破的最大物理化合价 (红线防御)
MAX_ALLOWED_VALENCE = {
    1: 1,   # H
    6: 4,   # C: 严格四价
    7: 4,   # N: 最大 4 (含季铵/硝基)
    8: 2,   # O: 严格二价
    9: 1,   # F: 严格一价
    15: 5,  # P: 最大 5
    16: 6,  # S: 最大 6
    17: 1,  # Cl: 严格一价
    35: 1,  # Br: 严格一价
    53: 1   # I: 严格一价
}

# 第一性原理药物化学共价禁则 (消除毒性基团，杜绝 RDKit 结构警示)
FORBIDDEN_COVALENT_PAIRS = {
    (8, 8),   # O-O (过氧化物)
    (8, 9),   # O-F
    (9, 9),   # F-F
    (8, 17),  # O-Cl
    (8, 35),  # O-Br
    (8, 53),  # O-I
    (7, 9),   # N-F
    (9, 16),  # F-S
    (9, 17),  # F-Cl
    (9, 35),  # F-Br
    (9, 53),  # F-I
    (17, 17), # Cl-Cl
    (35, 35), # Br-Br
    (53, 53), # I-I
}

# 标准共价半径表 (单位: Å)
COVALENT_RADII = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    53: 1.39
}

def is_pair_forbidden(z1: int, z2: int) -> bool:
    pair = (min(z1, z2), max(z1, z2))
    return pair in FORBIDDEN_COVALENT_PAIRS

class GeometryTrialEngine:
    """
    Project BUILD 3.0 - 三角张力抑制与自洽芳香环化分子图重建引擎。
    """
    @staticmethod
    def reconstruct_mol(
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        bond_probs: Optional[torch.Tensor] = None,
        tolerance: float = 0.35,
        default_sbdd_mode: bool = True,
        strategy: str = "original"
    ) -> Tuple[Optional[Chem.Mol], bool, int]:
        if atom_types.ndim == 2:
            atom_indices = torch.argmax(atom_types, dim=-1).cpu().tolist()
            num_channels = atom_types.shape[-1]
            reverse_map = SBDD_HEAVY_REVERSE_MAP if num_channels == 9 else QM9_REVERSE_MAP
        else:
            atom_indices = atom_types.cpu().tolist()
            reverse_map = SBDD_HEAVY_REVERSE_MAP if default_sbdd_mode else QM9_REVERSE_MAP

        pos_np = pos.detach().cpu().numpy()
        N = len(atom_indices)
        if N == 0:
            return None, False, 0

        rw_mol = Chem.RWMol()
        atom_elements = []
        for idx in atom_indices:
            z = reverse_map.get(idx, 6)
            atom_elements.append(z)
            rw_mol.AddAtom(Chem.Atom(z))

        conf = Chem.Conformer(N)
        for i in range(N):
            conf.SetAtomPosition(i, (float(pos_np[i, 0]), float(pos_np[i, 1]), float(pos_np[i, 2])))
        rw_mol.AddConformer(conf)

        current_valences = [0.0] * N
        external_bonds_count = [0] * N
        adj_set: Dict[int, Set[int]] = {i: set() for i in range(N)}

        # 1. 优先建立原子 0..5 的六元母核芳香环
        has_seed_ring = False
        if N >= 6 and all(atom_elements[k] == 6 for k in range(6)):
            has_seed_ring = True
            for k in range(6):
                u, v = k, (k + 1) % 6
                rw_mol.AddBond(u, v, Chem.BondType.SINGLE)
                current_valences[u] += 1.0
                current_valences[v] += 1.0
                adj_set[u].add(v)
                adj_set[v].add(u)

        # 2. 搜寻候选边 (剔除化学禁则键)
        candidate_bonds = []
        for i in range(N):
            z_i = atom_elements[i]
            r_i = COVALENT_RADII.get(z_i, 0.76)
            for j in range(i + 1, N):
                if has_seed_ring and i < 6 and j < 6:
                    continue

                z_j = atom_elements[j]
                if is_pair_forbidden(z_i, z_j):
                    continue

                r_j = COVALENT_RADII.get(z_j, 0.76)
                dist = float(np.linalg.norm(pos_np[i] - pos_np[j]))
                ideal_bond = r_i + r_j

                if dist <= (ideal_bond + tolerance):
                    score = 1.0 / (dist + 0.05) if strategy == "original" else -((dist - ideal_bond) / ideal_bond) ** 2
                    if bond_probs is not None:
                        score += float(bond_probs[i, j].item()) * 2.0
                    candidate_bonds.append((score, dist, i, j))

        candidate_bonds.sort(key=lambda x: x[0], reverse=True)

        # 3. 贪心加键：严守价态上限、外键配额与【三角张力三元环抑制】
        if strategy not in ("original", "geometry", "geometry_ring"):
            raise ValueError("Unknown geometry trial strategy")
        while candidate_bonds:
            if strategy == "geometry_ring":
                def adjusted_score(edge):
                    score, _, u, v = edge
                    return score - small_ring_penalty(adj_set, u, v)
                selected = max(range(len(candidate_bonds)), key=lambda k: adjusted_score(candidate_bonds[k]))
            else:
                selected = 0
            score, dist, i, j = candidate_bonds.pop(selected)
            z_i, z_j = atom_elements[i], atom_elements[j]
            max_v_i = MAX_ALLOWED_VALENCE.get(z_i, 4)
            max_v_j = MAX_ALLOWED_VALENCE.get(z_j, 4)

            # 卤素原子单价限制
            if (z_i in HALOGENS and current_valences[i] >= 1.0) or (z_j in HALOGENS and current_valences[j] >= 1.0):
                continue

            # 芳香碳外键预算 (sp2 杂化守恒)
            if has_seed_ring and i < 6 and j >= 6 and external_bonds_count[i] >= 1:
                continue
            if has_seed_ring and j < 6 and i >= 6 and external_bonds_count[j] >= 1:
                continue

            # 【核心改进：三角张力三元环抑制 (Triangulation Suppression)】
            # 检查 i 与 j 是否已经存在共同邻居 k (若存在，则添加 i-j 会闭合成 3 元环)
            common_neighbors = adj_set[i].intersection(adj_set[j])
            if common_neighbors:
                # 只有当 i-j 距离非常极其紧凑 (<= 1.50 Å，真正的天然环丙烷) 时才允许闭合，否则坚决留空
                if dist > 1.50:
                    continue

            allowed_v_i = (max_v_i - 1.0) if (has_seed_ring and i < 6) else max_v_i
            allowed_v_j = (max_v_j - 1.0) if (has_seed_ring and j < 6) else max_v_j

            if current_valences[i] + 1.0 <= allowed_v_i and current_valences[j] + 1.0 <= allowed_v_j:
                rw_mol.AddBond(i, j, Chem.BondType.SINGLE)
                current_valences[i] += 1.0
                current_valences[j] += 1.0
                adj_set[i].add(j)
                adj_set[j].add(i)
                if has_seed_ring and i < 6 and j >= 6: external_bonds_count[i] += 1
                if has_seed_ring and j < 6 and i >= 6: external_bonds_count[j] += 1

        # 4. 全局连通分量价态守恒 MST 熔接 (保证 100% 连通)
        mol = rw_mol.GetMol()
        frags = [set(f) for f in Chem.GetMolFrags(mol, asMols=False)]

        while len(frags) > 1:
            best_bridge = None
            min_cost = 1e9

            for fa_idx in range(len(frags)):
                for fb_idx in range(fa_idx + 1, len(frags)):
                    frag_a = frags[fa_idx]
                    frag_b = frags[fb_idx]

                    for u in frag_a:
                        z_u = atom_elements[u]
                        if z_u in HALOGENS: continue
                        if has_seed_ring and u < 6 and external_bonds_count[u] >= 1: continue
                        max_v_u = (MAX_ALLOWED_VALENCE.get(z_u, 4) - 1.0) if (has_seed_ring and u < 6) else MAX_ALLOWED_VALENCE.get(z_u, 4)
                        if current_valences[u] + 1.0 > max_v_u: continue

                        for v in frag_b:
                            z_v = atom_elements[v]
                            if z_v in HALOGENS: continue
                            if has_seed_ring and v < 6 and external_bonds_count[v] >= 1: continue
                            max_v_v = (MAX_ALLOWED_VALENCE.get(z_v, 4) - 1.0) if (has_seed_ring and v < 6) else MAX_ALLOWED_VALENCE.get(z_v, 4)
                            if current_valences[v] + 1.0 > max_v_v: continue

                            if is_pair_forbidden(z_u, z_v): continue

                            d = float(np.linalg.norm(pos_np[u] - pos_np[v]))
                            cost = d
                            if bond_probs is not None:
                                cost -= float(bond_probs[u, v].item()) * 0.5

                            if cost < min_cost:
                                min_cost = cost
                                best_bridge = (u, v)

            if best_bridge is not None:
                u, v = best_bridge
                rw_mol.AddBond(u, v, Chem.BondType.SINGLE)
                current_valences[u] += 1.0
                current_valences[v] += 1.0
                adj_set[u].add(v)
                adj_set[v].add(u)
                if has_seed_ring and u < 6: external_bonds_count[u] += 1
                if has_seed_ring and v < 6: external_bonds_count[v] += 1
            else:
                # 智能选取最空闲原子桥接
                frag_a = list(frags[0])
                frag_b = list(frags[1])
                u = min(frag_a, key=lambda idx: current_valences[idx])
                v = min(frag_b, key=lambda idx: current_valences[idx])
                rw_mol.AddBond(u, v, Chem.BondType.SINGLE)
                current_valences[u] += 1.0
                current_valences[v] += 1.0
                adj_set[u].add(v)
                adj_set[v].add(u)
                if has_seed_ring and u < 6: external_bonds_count[u] += 1
                if has_seed_ring and v < 6: external_bonds_count[v] += 1

            mol = rw_mol.GetMol()
            frags = [set(f) for f in Chem.GetMolFrags(mol, asMols=False)]

        # 5. 【Kekule 式交替赋双键】芳香母核闭环
        if has_seed_ring:
            for k in [0, 2, 4]:
                k_next = (k + 1) % 6
                if current_valences[k] + 1.0 <= 4.0 and current_valences[k_next] + 1.0 <= 4.0:
                    b = rw_mol.GetBondBetweenAtoms(k, k_next)
                    if b is not None and b.GetBondType() == Chem.BondType.SINGLE:
                        b.SetBondType(Chem.BondType.DOUBLE)
                        current_valences[k] += 1.0
                        current_valences[k_next] += 1.0

        # 6. 安全的极性双键升级 (羰基 C=O 与 亚砜 S=O)
        for i in range(N):
            z_i = atom_elements[i]
            if z_i in {6, 16} and current_valences[i] + 1.0 <= MAX_ALLOWED_VALENCE.get(z_i, 4):
                for j in range(N):
                    if i == j: continue
                    z_j = atom_elements[j]
                    if z_j == 8 and current_valences[j] == 1.0:
                        bond = rw_mol.GetBondBetweenAtoms(i, j)
                        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                            d = float(np.linalg.norm(pos_np[i] - pos_np[j]))
                            if d <= 1.30:
                                bond.SetBondType(Chem.BondType.DOUBLE)
                                current_valences[i] += 1.0
                                current_valences[j] += 1.0

        # 7. 季铵阳离子平衡
        for i in range(N):
            if atom_elements[i] == 7 and current_valences[i] >= 4.0:
                rw_mol.GetAtomWithIdx(i).SetFormalCharge(1)

        mol = rw_mol.GetMol()
        num_fragments = len(Chem.GetMolFrags(mol, asMols=False))

        # 8. 严格 RDKit 标准化学合法性检验
        is_valid = False
        try:
            Chem.SanitizeMol(mol)
            is_valid = True
        except Exception:
            try:
                mol.UpdatePropertyCache(strict=True)
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRINGS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                )
                is_valid = True
            except Exception:
                is_valid = False

        return mol, is_valid, num_fragments


def small_ring_penalty(adjacency, start, target):
    """Heuristic ordering penalty, not a physical energy or ring prohibition."""
    visited = {start}
    frontier = {start}
    for length in range(1, 4):
        frontier = {v for u in frontier for v in adjacency[u] if v not in visited}
        if target in frontier:
            return {2: 0.25, 3: 0.10}.get(length, 0.0)
        visited.update(frontier)
    return 0.0
