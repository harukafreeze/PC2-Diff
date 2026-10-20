import os
import sys
import argparse
import math
import copy
import yaml
from typing import Dict, Any, Tuple, Optional, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from project_build.core.registry import BACKBONE_REGISTRY, POTENTIAL_REGISTRY, SCHEDULER_REGISTRY
from project_build.data.datasets.crossdocked import CrossDockedLMDBDataset
from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE
from project_build.diffusion.solvers.reverse_sde_sampler import ReverseSDESampler
from project_build.postprocess.bond_perception import BondPerceptionEngine
from project_build.evaluation.docking_vina import TargetDockingEvaluator

import project_build.physics
import project_build.models.backbones
from rdkit import Chem
from rdkit.Chem import QED

ATOM_CLASS_WEIGHTS = torch.tensor([1.0, 2.5, 2.5, 3.0, 3.0, 2.5, 3.0, 3.0, 3.0], dtype=torch.float32)

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

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def train_one_epoch_ddp(
    model: nn.Module,
    ema: Optional[ExponentialMovingAverage],
    sde: DualSpaceVPSDE,
    valency_potential: nn.Module,
    scheduler_module: nn.Module,
    bce_loss_fn: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_rank: int
) -> Dict[str, float]:
    model.train()
    total_loss_sum = 0.0
    loss_x_sum = 0.0
    loss_h_sum = 0.0
    loss_bond_sum = 0.0
    loss_v_sum = 0.0
    num_batches = 0
    
    weights_tensor = ATOM_CLASS_WEIGHTS.to(device).view(1, 1, -1)
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        lig_pos = batch["lig_pos"].to(device, non_blocking=True)
        lig_h = batch["lig_h"].to(device, non_blocking=True)
        lig_z = batch["lig_z"].to(device, non_blocking=True)
        lig_mask = batch["lig_mask"].to(device, non_blocking=True)
        
        pocket_pos = batch["pocket_pos"].to(device, non_blocking=True)
        pocket_h = batch["pocket_h"].to(device, non_blocking=True)
        pocket_mask = batch["pocket_mask"].to(device, non_blocking=True)
        
        B, N_l, _ = lig_pos.shape
        t = torch.rand(B, device=device)
        
        diff_gt = lig_pos.unsqueeze(2) - lig_pos.unsqueeze(1)
        dist_gt = torch.sqrt(torch.sum(diff_gt ** 2, dim=-1) + 1e-8)
        bond_true = (dist_gt < 1.90).float()
        eye_mask = torch.eye(N_l, device=device, dtype=torch.bool).unsqueeze(0)
        bond_true = bond_true.masked_fill(eye_mask, 0.0)
        
        x_t, h_t, eps_x_true, eps_h_true, alpha_bar_t = sde.sample_forward_noise(lig_pos, lig_h, t, mask=lig_mask)
        
        eps_x_pred, eps_h_pred, pred_bond_logits = model(
            x_t=x_t, h_t=h_t, t=t,
            pocket_pos=pocket_pos, pocket_h=pocket_h,
            lig_mask=lig_mask, pocket_mask=pocket_mask
        )
        
        mask_3d = lig_mask.unsqueeze(-1).float()
        num_valid_atoms = torch.clamp(torch.sum(lig_mask.float()), min=1.0)
        
        loss_x = torch.sum(((eps_x_pred - eps_x_true) ** 2) * mask_3d) / (num_valid_atoms * 3.0)
        weighted_h_err = ((eps_h_pred - eps_h_true) ** 2) * weights_tensor
        loss_h = torch.sum(weighted_h_err * mask_3d) / (num_valid_atoms * 9.0)
        
        pair_mask = (lig_mask.unsqueeze(2) & lig_mask.unsqueeze(1)) & (~eye_mask)
        loss_bond = bce_loss_fn(pred_bond_logits[pair_mask], bond_true[pair_mask])
        
        x0_hat, _ = sde.compute_tweedie_x0(x_t, h_t, eps_x_pred, eps_h_pred, alpha_bar_t, mask=lig_mask)
        lambda_t = scheduler_module(alpha_bar_t)
        v_energy = valency_potential(pos=x0_hat, atom_types=lig_z, mask=lig_mask)
        loss_v = torch.mean(lambda_t * v_energy)
        
        loss = loss_x + loss_h + 1.0 * loss_bond + 0.05 * loss_v
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if global_rank == 0 and ema is not None:
            ema.update(model.module)
            
        total_loss_sum += loss.item()
        loss_x_sum += loss_x.item()
        loss_h_sum += loss_h.item()
        loss_bond_sum += loss_bond.item()
        loss_v_sum += loss_v.item()
        num_batches += 1
        
        if global_rank == 0 and step % 25 == 0:
            print(f"  [8卡 DDP Epoch {epoch:03d} | Step {step:04d}/{len(dataloader)}] "
                  f"总损失: {loss.item():.4f} | 坐标 MSE: {loss_x.item():.4f} | "
                  f"化学键 BCE: {loss_bond.item():.4f} | 量子价态: {loss_v.item():.4f}")
                  
    return {
        "loss_total": total_loss_sum / num_batches,
        "loss_x": loss_x_sum / num_batches,
        "loss_h": loss_h_sum / num_batches,
        "loss_bond": loss_bond_sum / num_batches,
        "loss_v": loss_v_sum / num_batches
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Project BUILD 3.0 - 8-GPU DDP SBDD training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/build_sbdd_full.yaml",
        help="YAML config path. The default reproduces the original production run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank, global_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    if global_rank == 0:
        print("\n" + "=" * 85)
        print(f">>> 🚀 [Project BUILD 3.0] 8 卡 DDP 分布式并行训练系统启动 (World Size = {world_size})")
        print(f">>> 配置文件: {config_path}")
        print(f">>> 物理注意力偏置: use_physics_bias = {cfg['model'].get('use_physics_bias', True)}")
        print("=" * 85)
        os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
        
    train_dataset = CrossDockedLMDBDataset(lmdb_dir=cfg["dataset"]["lmdb_dir"], split="train", split_type=cfg["dataset"]["split_type"])
    val_dataset = CrossDockedLMDBDataset(lmdb_dir=cfg["dataset"]["lmdb_dir"], split="test", split_type=cfg["dataset"]["split_type"])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["dataset"]["batch_size"],
        sampler=train_sampler,
        num_workers=cfg["dataset"]["num_workers"],
        persistent_workers=True,
        collate_fn=CrossDockedLMDBDataset.collate_fn,
        pin_memory=True
    )
    
    raw_model = BACKBONE_REGISTRY.build(cfg["model"]).to(device)
    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    ema = ExponentialMovingAverage(raw_model, decay=0.999) if global_rank == 0 else None
    
    sde = DualSpaceVPSDE(schedule_type=cfg["diffusion"]["schedule_type"]).to(device)
    valency_potential = POTENTIAL_REGISTRY.build(cfg["physics"]["valency_potential"]).to(device)
    scheduler_module = SCHEDULER_REGISTRY.build(cfg["physics"]["scheduler"]).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    base_lr = float(cfg["training"]["lr"]) * math.sqrt(world_size)
    min_lr = float(cfg["training"]["min_lr"])
    warmup_epochs = cfg["training"]["warmup_epochs"]
    total_epochs = cfg["training"]["epochs"]
    
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(cfg["training"]["weight_decay"]))
    evaluator = TargetDockingEvaluator(clash_cutoff=1.5) if global_rank == 0 else None
    sampler = ReverseSDESampler(sde=sde, num_steps=100, device=device) if global_rank == 0 else None
    
    if global_rank == 0:
        print(f">>> 8卡并行等效全局 BatchSize: {cfg['dataset']['batch_size'] * world_size} | 动态基准学习率: {base_lr:.6f}")
        print("\n>>> 开始执行 8 卡并行生产级训练循环...")
        
    for epoch in range(1, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        curr_lr = get_dynamic_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
            
        if global_rank == 0:
            print(f"\n--- [8-GPU SBDD Epoch {epoch:03d}/{total_epochs}] 当前动态学习率 LR = {curr_lr:.6f} ---")
            
        metrics = train_one_epoch_ddp(
            model=model, ema=ema, sde=sde,
            valency_potential=valency_potential, scheduler_module=scheduler_module,
            bce_loss_fn=bce_loss_fn, dataloader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, global_rank=global_rank
        )
        
        if global_rank == 0:
            print(f">>> [8卡 Epoch {epoch:03d} 汇总] 总损失: {metrics['loss_total']:.4f} | "
                  f"坐标 MSE: {metrics['loss_x']:.4f} | 化学键 BCE: {metrics['loss_bond']:.4f} | 量子价态: {metrics['loss_v']:.4f}")
                  
            # 原子重命名保存机制: 先写到 .tmp 文件，再瞬间原子替换，绝不产生读写冲突
            ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], "latest_sbdd_model.pt")
            tmp_path = ckpt_path + ".tmp"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "ema_state_dict": ema.shadow.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg
            }, tmp_path)
            os.replace(tmp_path, ckpt_path)
            
    if global_rank == 0:
        print("\n" + "=" * 85)
        print(">>> 🏆 [8卡 DDP 训练圆满完成！] 100 轮全量微观物理与口袋几何深度收敛完毕！")
        print("=" * 85 + "\n")
        
    cleanup_ddp()

if __name__ == "__main__":
    main()
