import os
import sys
import argparse
import math
import copy
import yaml
from typing import Dict, Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from project_build.core.registry import BACKBONE_REGISTRY, POTENTIAL_REGISTRY, SCHEDULER_REGISTRY
from project_build.data.datasets.crossdocked import CrossDockedLMDBDataset
from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE
from project_build.postprocess.bond_perception import BondPerceptionEngine
from project_build.evaluation.docking_vina import TargetDockingEvaluator

import project_build.physics
import project_build.models.backbones

class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for p_shadow, p_model in zip(self.shadow.parameters(), model.parameters()):
                p_shadow.data.mul_(self.decay).add_(p_model.data, alpha=1.0 - self.decay)

def get_dynamic_lr(epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float, min_lr: float) -> float:
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def parse_args():
    parser = argparse.ArgumentParser(description="Project BUILD 3.0 - 局域 RBF 径向跨注意力 SBDD 体系")
    parser.add_argument("--config", type=str, default="configs/models/build_sbdd_full.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()

def train_one_epoch(
    model: torch.nn.Module,
    ema: ExponentialMovingAverage,
    sde: DualSpaceVPSDE,
    valency_potential: torch.nn.Module,
    scheduler_module: torch.nn.Module,
    bce_loss_fn: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_steps: int = -1
) -> Dict[str, float]:
    model.train()
    total_loss_sum = 0.0
    loss_x_sum = 0.0
    loss_h_sum = 0.0
    loss_bond_sum = 0.0
    num_batches = 0
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        lig_pos = batch["lig_pos"].to(device)
        lig_h = batch["lig_h"].to(device)
        lig_z = batch["lig_z"].to(device)
        lig_mask = batch["lig_mask"].to(device)
        
        pocket_pos = batch["pocket_pos"].to(device)
        pocket_h = batch["pocket_h"].to(device)
        pocket_mask = batch["pocket_mask"].to(device)
        
        B, N_l, _ = lig_pos.shape
        t = torch.rand(B, device=device)
        
        # 1. 真实共价成键标签计算
        diff_gt = lig_pos.unsqueeze(2) - lig_pos.unsqueeze(1)
        dist_gt = torch.sqrt(torch.sum(diff_gt ** 2, dim=-1) + 1e-8)
        bond_true = (dist_gt < 1.90).float()
        eye_mask = torch.eye(N_l, device=device, dtype=torch.bool).unsqueeze(0)
        bond_true = bond_true.masked_fill(eye_mask, 0.0)
        
        # 2. 保方差正向扩散加噪
        x_t, h_t, eps_x_true, eps_h_true, alpha_bar_t = sde.sample_forward_noise(lig_pos, lig_h, t, mask=lig_mask)
        
        # 3. 局域 RBF 径向主干前向预测 (8.0 Å 局域几何 + PG-Attention)
        eps_x_pred, eps_h_pred, pred_bond_logits = model(
            x_t=x_t, h_t=h_t, t=t,
            pocket_pos=pocket_pos, pocket_h=pocket_h,
            lig_mask=lig_mask, pocket_mask=pocket_mask
        )
        
        mask_3d = lig_mask.unsqueeze(-1).float()
        num_valid_atoms = torch.clamp(torch.sum(lig_mask.float()), min=1.0)
        
        loss_x = torch.sum(((eps_x_pred - eps_x_true) ** 2) * mask_3d) / (num_valid_atoms * 3.0)
        loss_h = torch.sum(((eps_h_pred - eps_h_true) ** 2) * mask_3d) / (num_valid_atoms * 9.0)
        
        pair_mask = (lig_mask.unsqueeze(2) & lig_mask.unsqueeze(1)) & (~eye_mask)
        loss_bond = bce_loss_fn(pred_bond_logits[pair_mask], bond_true[pair_mask])
        
        x0_hat, h0_hat = sde.compute_tweedie_x0(x_t, h_t, eps_x_pred, eps_h_pred, alpha_bar_t, mask=lig_mask)
        lambda_t = scheduler_module(alpha_bar_t)
        v_energy = valency_potential(pos=x0_hat, atom_types=lig_z, mask=lig_mask)
        loss_v = torch.mean(lambda_t * v_energy)
        
        loss = loss_x + loss_h + 1.0 * loss_bond + 0.05 * loss_v
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        ema.update(model)
        
        total_loss_sum += loss.item()
        loss_x_sum += loss_x.item()
        loss_h_sum += loss_h.item()
        loss_bond_sum += loss_bond.item()
        num_batches += 1
        
        if step % 25 == 0 or step == max_steps - 1:
            print(f"  [BUILD 3.0 Epoch {epoch:02d} | Step {step:04d}/{len(dataloader)}] "
                  f"总损失: {loss.item():.4f} | 坐标 MSE: {loss_x.item():.4f} | "
                  f"化学键 BCE: {loss_bond.item():.4f} | 量子价态: {loss_v.item():.4f}")
                  
        if max_steps > 0 and step >= max_steps - 1:
            break
            
    return {
        "loss_total": total_loss_sum / num_batches,
        "loss_x": loss_x_sum / num_batches,
        "loss_h": loss_h_sum / num_batches,
        "loss_bond": loss_bond_sum / num_batches
    }

def main():
    args = parse_args()
    print("\n" + "=" * 80)
    print(f">>> 🚀 [Project BUILD 3.0] 局域 RBF 径向图 SBDD 体系启动 | 设备: {args.device}")
    print("=" * 80)
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(args.device)
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    
    train_dataset = CrossDockedLMDBDataset(lmdb_dir=cfg["dataset"]["lmdb_dir"], split="train", split_type=cfg["dataset"]["split_type"])
    val_dataset = CrossDockedLMDBDataset(lmdb_dir=cfg["dataset"]["lmdb_dir"], split="test", split_type=cfg["dataset"]["split_type"])
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["dataset"]["batch_size"], shuffle=True,
        num_workers=cfg["dataset"]["num_workers"], persistent_workers=True if cfg["dataset"]["num_workers"] > 0 else False,
        collate_fn=CrossDockedLMDBDataset.collate_fn, pin_memory=True if args.device == "cuda" else False
    )
    
    model = BACKBONE_REGISTRY.build(cfg["model"]).to(device)
    ema = ExponentialMovingAverage(model, decay=0.999)
    sde = DualSpaceVPSDE(schedule_type=cfg["diffusion"]["schedule_type"]).to(device)
    valency_potential = POTENTIAL_REGISTRY.build(cfg["physics"]["valency_potential"]).to(device)
    scheduler_module = SCHEDULER_REGISTRY.build(cfg["physics"]["scheduler"]).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    base_lr = float(cfg["training"]["lr"])
    min_lr = float(cfg["training"]["min_lr"])
    warmup_epochs = cfg["training"]["warmup_epochs"]
    total_epochs = cfg["training"]["epochs"]
    
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(cfg["training"]["weight_decay"]))
    evaluator = TargetDockingEvaluator(clash_cutoff=1.5)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Project BUILD 3.0 主干网络参数总量: {num_params / 1e6:.2f} M")
    
    epochs = 1 if args.demo else total_epochs
    max_steps = 50 if args.demo else -1
    
    print("\n>>> 开始执行 BUILD 3.0 局部径向流训练循环...")
    for epoch in range(1, epochs + 1):
        curr_lr = get_dynamic_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
            
        metrics = train_one_epoch(
            model=model, ema=ema, sde=sde,
            valency_potential=valency_potential, scheduler_module=scheduler_module,
            bce_loss_fn=bce_loss_fn, dataloader=train_loader, optimizer=optimizer, device=device,
            epoch=epoch, max_steps=max_steps
        )
        
        if epoch % cfg["training"]["val_interval"] == 0 or args.demo:
            print(f"\n" + "-" * 70)
            print(f">>> [Epoch {epoch:03d} 触发 BUILD 3.0 局域径向学术体检] 正在测试蛋白质口袋生成...")
            
            val_sample = val_dataset[0]
            p_pos = val_sample["pocket_pos"].unsqueeze(0).to(device)
            p_h = val_sample["pocket_h"].unsqueeze(0).to(device)
            real_lig_pos = val_sample["lig_pos"].to(device)
            n_l = val_sample["num_lig_atoms"]
            
            eval_model = model if (epoch <= 3 or args.demo) else ema.shadow
            eval_model.eval()
            
            # 【受体结合位点精确成核】：在真实配体空腔中心 1.5 Å 范围内初始化高斯种子
            cavity_center = torch.mean(real_lig_pos, dim=0, keepdim=True)
            x_curr = cavity_center.unsqueeze(0) + torch.randn(5, n_l, 3, device=device) * 1.5
            h_curr = torch.randn(5, n_l, 9, device=device)
            
            time_steps = torch.linspace(0.999, 0.001, 101, device=device)
            with torch.no_grad():
                for i in range(100):
                    t_c = time_steps[i]
                    t_p = time_steps[i+1]
                    t_tensor = torch.full((5,), t_c.item(), device=device)
                    t_prev_tensor = torch.full((5,), t_p.item(), device=device)
                    
                    eps_x, eps_h, bond_logits = eval_model(
                        x_t=x_curr, h_t=h_curr, t=t_tensor,
                        pocket_pos=p_pos.repeat(5, 1, 1), pocket_h=p_h.repeat(5, 1, 1)
                    )
                    
                    a_c = sde.compute_alpha_bar(t_tensor).view(5, 1, 1)
                    sqrt_a = torch.sqrt(torch.clamp(a_c, min=1e-4))
                    sqrt_1_a = torch.sqrt(torch.clamp(1.0 - a_c, min=1e-4))
                    
                    # 洁净构象估计 (严格保持在受体参考系内)
                    x0_h = (x_curr - sqrt_1_a * eps_x) / sqrt_a
                    h0_h = (h_curr - sqrt_1_a * eps_h) / sqrt_a
                    
                    if i == 99:
                        x_curr = x0_h
                        h_curr = h0_h
                        break
                        
                    x_curr, h_curr = sde.q_posterior_step(
                        x_0_pred=x0_h, h_0_pred=h0_h,
                        x_t=x_curr, h_t=h_curr,
                        t_curr=t_tensor, t_prev=t_prev_tensor
                    )
                    
            pred_bond_probs = torch.sigmoid(bond_logits)
            
            total_clashes = valid_count = single_frag_count = 0
            strains, vina_scores = [], []
            
            for b_idx in range(5):
                mol_b, is_v, n_f = BondPerceptionEngine.reconstruct_mol(
                    pos=x_curr[b_idx], atom_types=h_curr[b_idx], bond_probs=pred_bond_probs[b_idx], tolerance=0.60
                )
                total_clashes += evaluator.compute_pocket_clashes(x_curr[b_idx], p_pos[0])
                if is_v:
                    valid_count += 1
                    strains.append(evaluator.compute_strain_energy(mol_b))
                if n_f == 1: single_frag_count += 1
                v_s = evaluator.compute_official_vina_score(mol_b, p_pos[0], p_h[0])
                vina_scores.append(v_s)
                
            print(f"  ├── 🎯 化学合法率 (Validity):        {(valid_count/5)*100:.1f}%")
            print(f"  ├── 🔗 分子连通率 (Connectivity):    {(single_frag_count/5)*100:.1f}% (MST 稳健连通保障)")
            print(f"  ├── 🛡️ 口袋碰撞数 (Clashes):          {total_clashes/5:.2f} 个/分子 (PG-CrossAttn 保护)")
            print(f"  ├── ⚡ 平均应变能 (Strain Energy):    {sum(strains)/len(strains) if strains else 0.0:.2f} kcal/mol")
            print(f"  └── 🏆 靶点 Vina 结合能:             {sum(vina_scores)/len(vina_scores) if vina_scores else 0.0:.2f} kcal/mol")
            print("-" * 70 + "\n")

if __name__ == "__main__":
    main()
