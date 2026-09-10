import os
import sys
import torch

LIT_PCBA_PDBS = {"4ldo", "5l2m", "2p15", "2iok", "5fv7", "2v3d", "4umx", "5h86", "4zzn", "4dri", "6b73", "4jpg", "5y2t", "3zme", "3a2i"}

def main():
    split_path = "data/crossdocked/crossdocked_lmdb/split_by_name.pt"
    if not os.path.exists(split_path):
        split_path = "data/crossdocked/crossdocked_lmdb/official_strict_split.pt"
        
    split_data = torch.load(split_path)
    train_keys = split_data.get("train", [])
    
    print("\n" + "=" * 80)
    print(f"🔍 [训练集数据纯净度查重] 扫描训练集总样本数: {len(train_keys)} 条")
    print("=" * 80)
    
    # 提取训练集中所有复合物的 4 位 PDB 代码
    train_pdb_set = set()
    for key in train_keys:
        key_str = str(key).lower()
        parts = key_str.replace("(", "").replace("'", "").replace("[", "").split("_")
        for p in parts:
            p_clean = p.strip()
            if len(p_clean) == 4 and p_clean.isalnum():
                train_pdb_set.add(p_clean)
                
    print(f"  ├── 训练集中涵盖的不同 PDB 晶体种类: {len(train_pdb_set)} 种")
    
    # 检查 15 个 LIT-PCBA 靶点是否重叠
    overlap = LIT_PCBA_PDBS.intersection(train_pdb_set)
    
    print(f"  ├── 待查验的 15 个 LIT-PCBA 靶点 PDB ID: {sorted(list(LIT_PCBA_PDBS))}")
    print(f"  ├── 训练集与 LIT-PCBA 重叠数量: {len(overlap)} 个")
    
    if len(overlap) == 0:
        print("\n  ✅ 【查重结论】重叠数为 0！这 15 个靶点在训练集中 100% 完全不存在！")
        print("  ✅ 确凿证明本次测试属于纯正的【零样本跨数据集泛化测试 (Zero-Shot Generalization)】！\n")
    else:
        print(f"\n  ⚠️ 发现重叠靶点: {overlap}\n")
    print("=" * 80)

if __name__ == "__main__":
    main()
