import os
import sys
import json
import urllib.request
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# 严格按照 SYCLE (ICML 2026 Table 5) 定义的 15 大 LIT-PCBA 官方靶点清单
LIT_PCBA_TARGETS = [
    {"target": "ADRB2",      "pdb_id": "4ldo", "name": "Beta-2 adrenergic receptor (ADRB2)"},
    {"target": "ALDH1",      "pdb_id": "5l2m", "name": "Aldehyde dehydrogenase 1 (ALDH1)"},
    {"target": "ESR_ago",    "pdb_id": "2p15", "name": "Estrogen receptor alpha agonist (ESR_ago)"},
    {"target": "ESR_antago", "pdb_id": "2iok", "name": "Estrogen receptor alpha antagonist (ESR_antago)"},
    {"target": "FEN1",       "pdb_id": "5fv7", "name": "Flap endonuclease 1 (FEN1)"},
    {"target": "GBA",        "pdb_id": "2v3d", "name": "Acid beta-glucocerebrosidase (GBA)"},
    {"target": "IDH1",       "pdb_id": "4umx", "name": "Isocitrate dehydrogenase 1 (IDH1)"},
    {"target": "KAT2A",      "pdb_id": "5h86", "name": "Histone acetyltransferase KAT2A"},
    {"target": "MAPK1",      "pdb_id": "4zzn", "name": "Mitogen-activated protein kinase 1 (MAPK1)"},
    {"target": "MTORC1",     "pdb_id": "4dri", "name": "FKBP51 PPIase domain / mTOR pathway"},
    {"target": "OPRK1",      "pdb_id": "6b73", "name": "Kappa opioid receptor (OPRK1)"},
    {"target": "PKM2",       "pdb_id": "4jpg", "name": "Pyruvate kinase muscle isoform (PKM2)"},
    {"target": "PPARG",      "pdb_id": "5y2t", "name": "Peroxisome proliferator-activated receptor gamma"},
    {"target": "TP53",       "pdb_id": "3zme", "name": "Cellular tumor antigen p53 (TP53)"},
    {"target": "VDR",        "pdb_id": "3a2i", "name": "Vitamin D receptor (VDR)"}
]

# 蛋白质原子映射字典 (对齐 CrossDocked2020 标准 6 通道)
PROTEIN_ATOM_MAP = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'SE': 5}
NON_LIGAND_RESIDUES = {'HOH', 'WAT', 'SO4', 'PO4', 'DMS', 'GOL', 'NAG', 'EDO', 'ACT', 'CL', 'NA', 'MG', 'ZN'}

def download_pdb(pdb_id: str, save_path: str):
    """从 RCSB PDB 官方镜像下载晶体 PDB 文件"""
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    try:
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        print(f"  ❌ 下载 {pdb_id} 失败: {e}")
        return False

def parse_and_extract_pocket(pdb_path: str, cutoff: float = 10.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """解析 PDB，自动识别小分子配体，并提取配体周围 10.0 Å 内的蛋白质口袋重原子"""
    protein_atoms = [] # (x, y, z, elem, line)
    het_atoms = {}     # res_id -> list of (x, y, z, elem)
    
    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            record = line[:6].strip()
            if record == "ATOM":
                elem = line[76:78].strip().upper()
                if not elem:
                    elem = line[12:14].strip()[0].upper()
                if elem == 'H': continue # 忽略氢
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    protein_atoms.append((x, y, z, elem, line))
                except Exception:
                    pass
            elif record == "HETATM":
                res_name = line[17:20].strip()
                if res_name in NON_LIGAND_RESIDUES:
                    continue
                chain = line[21]
                res_seq = line[22:26].strip()
                res_key = f"{res_name}_{chain}_{res_seq}"
                
                elem = line[76:78].strip().upper()
                if not elem:
                    elem = line[12:14].strip()[0].upper()
                if elem == 'H': continue
                
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if res_key not in het_atoms:
                        het_atoms[res_key] = []
                    het_atoms[res_key].append((x, y, z, elem))
                except Exception:
                    pass

    if not het_atoms or not protein_atoms:
        return None, None, None, []

    # 挑选原子数最多的 HETATM 作为主要共晶配体
    best_lig_key = max(het_atoms.keys(), key=lambda k: len(het_atoms[k]))
    lig_coords = np.array([[a[0], a[1], a[2]] for a in het_atoms[best_lig_key]])
    
    if len(lig_coords) < 6: # 过滤过小的金属离子或杂质
        return None, None, None, []

    prot_coords = np.array([[a[0], a[1], a[2]] for a in protein_atoms])
    prot_elems = [a[3] for a in protein_atoms]
    prot_lines = [a[4] for a in protein_atoms]

    # 计算蛋白质原子到配体所有原子的最小距离
    dists = np.linalg.norm(prot_coords[:, None, :] - lig_coords[None, :, :], axis=-1)
    min_dists = np.min(dists, axis=1)
    
    # 提取 10.0 Å 范围内的口袋重原子
    pocket_mask = min_dists <= cutoff
    pocket_pos = prot_coords[pocket_mask]
    pocket_elements = [prot_elems[i] for i in range(len(prot_elems)) if pocket_mask[i]]
    pocket_pdb_lines = [prot_lines[i] for i in range(len(prot_lines)) if pocket_mask[i]]

    # 截断最大 256 原子 (对齐 CrossDocked 训练标准)
    if len(pocket_pos) > 256:
        p_dists = min_dists[pocket_mask]
        top_idx = np.argsort(p_dists)[:256]
        pocket_pos = pocket_pos[top_idx]
        pocket_elements = [pocket_elements[i] for i in top_idx]
        pocket_pdb_lines = [pocket_pdb_lines[i] for i in top_idx]

    # 构建 6 通道 One-hot 特征
    pocket_h = np.zeros((len(pocket_pos), 6), dtype=np.float32)
    for i, elem in enumerate(pocket_elements):
        ch_idx = PROTEIN_ATOM_MAP.get(elem, 1) # 默认碳
        pocket_h[i, ch_idx] = 1.0

    return pocket_pos, pocket_h, lig_coords, pocket_pdb_lines

def main():
    out_base_dir = "data/lit_pcba"
    raw_pdb_dir = os.path.join(out_base_dir, "raw_pdbs")
    processed_dir = os.path.join(out_base_dir, "processed_pockets")
    
    os.makedirs(raw_pdb_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("\n" + "=" * 95)
    print(">>> 📥 [LIT-PCBA 数据集构建] 自动下载 15 大经典临床靶点并提取 10.0 Å 活性空腔")
    print("=" * 95)

    summary_list = []

    for idx, item in enumerate(LIT_PCBA_TARGETS, 1):
        target_code = item["target"]
        pdb_id = item["pdb_id"]
        target_name = item["name"]
        
        print(f"\n[{idx:02d}/15] 正在处理靶点: {target_code:<10} | PDB ID: {pdb_id.upper()} ({target_name})...")
        
        raw_pdb_path = os.path.join(raw_pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(raw_pdb_path):
            print(f"  └── 正在从 RCSB PDB 下载 {pdb_id.upper()}.pdb ...")
            success = download_pdb(pdb_id, raw_pdb_path)
            if not success:
                continue
        else:
            print(f"  └── 已存在原始文件: {raw_pdb_path}")

        # 裁剪 10.0 Å 口袋
        pocket_pos, pocket_h, lig_coords, pdb_lines = parse_and_extract_pocket(raw_pdb_path, cutoff=10.0)
        
        if pocket_pos is None or len(pocket_pos) == 0:
            print(f"  ❌ 口袋提取失败 (未找到合适小分子配体)！")
            continue

        target_out_dir = os.path.join(processed_dir, f"{target_code}_{pdb_id.upper()}")
        os.makedirs(target_out_dir, exist_ok=True)

        # 1. 保存口袋标准 PDB 文件
        pocket_pdb_path = os.path.join(target_out_dir, "pocket.pdb")
        with open(pocket_pdb_path, 'w', encoding='utf-8') as f:
            f.write(f"REMARK    LIT-PCBA TARGET POCKET: {target_code} (PDB: {pdb_id.upper()})\n")
            f.writelines(pdb_lines)
            f.write("END\n")

        # 2. 保存 PyTorch 张量字典 (.pt)，供模型无缝加载推理
        # 执行零质心平移
        p_center = np.mean(pocket_pos, axis=0, keepdims=True)
        pocket_pos_centered = pocket_pos - p_center
        lig_center_centered = np.mean(lig_coords, axis=0, keepdims=True) - p_center

        tensor_data = {
            "target": target_code,
            "pdb_id": pdb_id.upper(),
            "name": target_name,
            "pocket_pos": torch.tensor(pocket_pos_centered, dtype=torch.float32), # [N_p, 3]
            "pocket_h": torch.tensor(pocket_h, dtype=torch.float32),             # [N_p, 6]
            "original_pocket_center": torch.tensor(p_center[0], dtype=torch.float32),
            "ligand_cavity_center": torch.tensor(lig_center_centered[0], dtype=torch.float32),
            "num_pocket_atoms": len(pocket_pos)
        }
        
        pt_path = os.path.join(target_out_dir, "pocket_tensor.pt")
        torch.save(tensor_data, pt_path)

        print(f"  ✅ [成功提取] 口袋重原子数: {len(pocket_pos)} 个 | 原晶体配体尺寸: {len(lig_coords)} 原子")
        print(f"  └── 📁 衍生文件: {target_out_dir}")

        summary_list.append({
            "target": target_code,
            "pdb_id": pdb_id.upper(),
            "name": target_name,
            "num_pocket_atoms": len(pocket_pos),
            "pt_path": pt_path
        })

    summary_json_path = os.path.join(out_base_dir, "lit_pcba_targets_summary.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_list, f, indent=2)

    print("\n" + "=" * 95)
    print(f"🎉 15 大 LIT-PCBA 经典临床靶点数据集全部构建完成！总索引: {summary_json_path}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()
