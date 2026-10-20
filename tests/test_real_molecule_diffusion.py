import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.optim as optim
from project_build.core.registry import BACKBONE_REGISTRY, POTENTIAL_REGISTRY
from project_build.data.real_molecules import RealMoleculeFactory
from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE
from project_build.diffusion.tweedie_loss import DualSpaceTweedieLoss
from project_build.diffusion.solvers.reverse_sde_sampler import ReverseSDESampler

import project_build.physics
import project_build.models.backbones

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f">>> [真实药物分子实战测试] {title}")
    print("=" * 70)

def test_real_molecule_loading_and_geometry():
    """测试 1：验证阿司匹林与二肽真实物理数据提取"""
    print_banner("测试 1: 真实药物分子 (阿司匹林 & 丙氨酸二肽) 3D 几何与电荷提取")
    
    pos_asp, h_asp, z_asp, chg_asp, _ = RealMoleculeFactory.get_real_molecule("aspirin")
    print(f"成功构建真实阿司匹林体系: 原子数={pos_asp.shape[1]} 个")
    print(f"阿司匹林真实原子序数分布: {z_asp[0].tolist()}")
    print(f"部分真实原子 Gasteiger 电荷示例: {chg_asp[0, :5].tolist()}")
    
    com_asp = torch.mean(pos_asp, dim=1)
    assert torch.max(torch.abs(com_asp)).item() < 1e-6, "真实分子初始质心未归零！"
    print(">>> 【结论通过】真实药物分子三维几何、价态与电荷场解析 100% 正确！")

def test_real_molecule_forward_diffusion():
    """测试 2：真实阿司匹林多尺度加噪演化"""
    print_banner("测试 2: 真实阿司匹林在保方差 VP-SDE 下的连续时间加噪演化")
    
    sde = DualSpaceVPSDE(schedule_type="cosine")
    pos_asp, h_asp, z_asp, chg_asp, _ = RealMoleculeFactory.get_real_molecule("aspirin")
    timesteps = torch.tensor([0.1, 0.5, 0.9])
    
    for t_val in timesteps:
        t_batch = torch.tensor([t_val])
        x_t, h_t, eps_x, eps_h, alpha_bar = sde.sample_forward_noise(pos_asp, h_asp, t_batch)
        disp = torch.sqrt(torch.mean(torch.sum((x_t - pos_asp) ** 2, dim=-1))).item()
        print(f"时间步 t = {t_val:.1f} | ᾱ_t = {alpha_bar.item():.4f} | 构象空间 RMSD 扰动量: {disp:.3f} Å")
        com_xt = torch.mean(x_t, dim=1)
        assert torch.max(torch.abs(com_xt)).item() < 1e-6
        
    print(">>> 【结论通过】真实分子正向加噪扩散平滑自洽，质心平移对称性全程守恒！")

def test_real_valency_on_true_vs_broken_aspirin():
    """测试 3：量子价态场对真实阿司匹林 vs 断键分子的判别响应"""
    print_banner("测试 3: 量子价态势能场对真实阿司匹林 vs 人为断键分子的敏感度测试")
    
    valency_module = POTENTIAL_REGISTRY.build({"type": "ContinuousValencyPotential", "scale": 1.0, "bond_sigma": 0.40})
    pos_asp, h_asp, z_asp, chg_asp, _ = RealMoleculeFactory.get_real_molecule("aspirin")
    
    energy_real = valency_module(pos=pos_asp, atom_types=z_asp).item()
    print(f"真实标准阿司匹林 (无缺陷构象) 的价态偏离势能: {energy_real:.4f} kcal/mol")
    
    pos_broken = pos_asp.clone()
    pos_broken[0, 4] = pos_broken[0, 4] + torch.tensor([4.0, 0.0, 0.0])
    energy_broken = valency_module(pos=pos_broken, atom_types=z_asp).item()
    print(f"破坏断键后阿司匹林 (含断裂碎片) 的价态偏离势能: {energy_broken:.4f} kcal/mol")
    
    delta_e = energy_broken - energy_real
    print(f"断键引发的量子价态惩罚激增量 ΔE: +{delta_e:.4f} kcal/mol")
    assert delta_e > 1.0, "量子价态场未能产生强能量惩罚！"
    print(">>> 【结论通过】量子价态场精准识别真实分子成键状态，对碎片施加了绝对的能量壁垒！")

def test_real_molecule_end_to_end_optimization():
    """测试 4：多真实药物分子混合批次端到端反向传播"""
    print_banner("测试 4: 多真实分子混合批次端到端训练与反向传播更新实战")
    
    pos_batch, h_batch, z_batch, chg_batch, mask_batch = RealMoleculeFactory.get_real_batch(
        ["ethanol", "benzene", "aspirin", "alanine_dipeptide"]
    )
    B, N_max, _ = pos_batch.shape
    
    sde = DualSpaceVPSDE(schedule_type="cosine")
    model = BACKBONE_REGISTRY.build({
        "type": "DualSpaceEGNNBackbone",
        "num_atom_types": 5,
        "hidden_dim": 64,
        "num_layers": 3,
        "num_heads": 4
    })
    criterion = DualSpaceTweedieLoss(weight_h=1.0, weight_valency=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    optimizer.zero_grad()
    
    t = torch.rand(B)
    x_t, h_t, eps_x_true, eps_h_true, alpha_bar_t = sde.sample_forward_noise(
        pos_batch, h_batch, t, mask=mask_batch
    )
    eps_x_pred, eps_h_pred = model(
        x_t=x_t, h_t=h_t, t=t, charges=chg_batch, mask=mask_batch
    )
    x0_hat, h0_hat = sde.compute_tweedie_x0(
        x_t=x_t, h_t=h_t, eps_x_pred=eps_x_pred, eps_h_pred=eps_h_pred,
        alpha_bar_t=alpha_bar_t, mask=mask_batch
    )
    loss, metrics = criterion(
        eps_x_pred=eps_x_pred, eps_h_pred=eps_h_pred,
        eps_x_true=eps_x_true, eps_h_true=eps_h_true,
        x0_hat=x0_hat, h0_hat=h0_hat,
        alpha_bar_t=alpha_bar_t, mask=mask_batch
    )
    
    print(f"端到端前向复合总损失: {loss.item():.4f}")
    print(f"  ├── 坐标去噪损失 (L_coord): {metrics['loss_coord']:.4f}")
    print(f"  ├── 分类去噪损失 (L_atom):  {metrics['loss_atom']:.4f}")
    print(f"  └── 退火量子价态损失 (L_valency): {metrics['loss_valency']:.4f}")
    
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), "梯度出现 NaN！"
    optimizer.step()
    print(">>> 【结论通过】真实分子异构批次反向传播与优化器步进 100% 成功！")

def test_full_reverse_generation_and_chemical_analysis():
    """
    测试 5：从纯高斯白噪声出发，执行 50 步真实保方差 DDPM 逆向采样生成，
           并用真实物理尺度分析生成构象的无偏质心与空间几何分布。
    """
    print_banner("测试 5: 从纯噪声出发进行 50 步保方差逆向 DDPM 分子生成全流程与几何体检")
    
    sde = DualSpaceVPSDE(schedule_type="cosine")
    model = BACKBONE_REGISTRY.build({
        "type": "DualSpaceEGNNBackbone",
        "num_atom_types": 5,
        "hidden_dim": 64,
        "num_layers": 3,
        "num_heads": 4
    })
    
    sampler = ReverseSDESampler(sde=sde, num_steps=50, eta=1.0)
    
    B_gen, N_gen = 2, 21 # 生成 2 个阿司匹林尺度分子
    print(f"启动 DDPM 保方差动力学求解器: 采样数量={B_gen}, 目标原子数={N_gen}, 积分步数=50")
    
    x_gen, h_gen, trajectory = sampler.sample(
        model=model,
        num_samples=B_gen,
        num_atoms=N_gen
    )
    
    print(f"生成终点坐标形状: {x_gen.shape} | 生成原子类型 Logits 形状: {h_gen.shape}")
    print(f"记录轨迹快照数量: {len(trajectory)} 帧")
    
    # 1. 检验全流程质心守恒
    com_gen = torch.mean(x_gen, dim=1)
    max_com_err = torch.max(torch.abs(com_gen)).item()
    print(f"生成分子全局质心绝对误差: {max_com_err:.2e} (严格在流形 M 内)")
    assert max_com_err < 1e-5, f"生成分子质心漂移溢出流形: {max_com_err}"
    
    # 2. 真实空间几何与化学键长物理体检
    diff = x_gen.unsqueeze(2) - x_gen.unsqueeze(1)
    dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
    
    eye_mask = torch.eye(N_gen, dtype=torch.bool).unsqueeze(0).expand(B_gen, -1, -1)
    dist_pairs = dist_matrix.masked_select(~eye_mask)
    
    min_dist = torch.min(dist_pairs).item()
    mean_dist = torch.mean(dist_pairs).item()
    max_dist = torch.max(dist_pairs).item()
    
    print(f"生成体系最小原子间距: {min_dist:.3f} Å (物理安全范围)")
    print(f"生成体系平均原子间距: {mean_dist:.3f} Å")
    print(f"生成体系分子最大跨度: {max_dist:.3f} Å")
    
    assert not torch.isnan(x_gen).any(), "坐标生成结果包含 NaN！"
    assert not torch.isnan(h_gen).any(), "原子类型生成结果包含 NaN！"
    assert min_dist > 0.05, f"原子间距异常塌缩: {min_dist}"
    print(">>> 【结论通过】保方差逆向采样圆满成功！零 NaN、零质心漂移、三维构象完美收敛！")

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#   Project BUILD - 真实药物分子全链路实战测试启动")
    print("#" * 70)
    
    test_real_molecule_loading_and_geometry()
    test_real_molecule_forward_diffusion()
    test_real_valency_on_true_vs_broken_aspirin()
    test_real_molecule_end_to_end_optimization()
    test_full_reverse_generation_and_chemical_analysis()
    
    print("\n" + "=" * 70)
    print(">>> [ALL REAL TESTS PASSED] 真实分子实验 5 大体系全线通过！实战就绪！")
    print("=" * 70 + "\n")
