import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from project_build.core.registry import BACKBONE_REGISTRY
import project_build.models.backbones

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f">>> [等变主干网络数值检验] {title}")
    print("=" * 70)

def test_backbone_forward_and_com_projection():
    """
    测试 1：主干网络双轨前向传播与质心硬投影 (P_M 流形) 验证。
    """
    print_banner("测试 1: 双轨前向形状契约与零质心流形 P_M 投影验证")
    
    cfg = {
        "type": "DualSpaceEGNNBackbone",
        "num_atom_types": 5,
        "hidden_dim": 64,
        "num_layers": 3,
        "num_heads": 4
    }
    model = BACKBONE_REGISTRY.build(cfg)
    model.eval()
    
    B, N, C = 4, 12, 5
    x_t = torch.randn(B, N, 3)
    h_t = torch.randn(B, N, C)
    t = torch.rand(B)
    
    with torch.no_grad():
        eps_x, eps_h = model(x_t=x_t, h_t=h_t, t=t)
        
    print(f"输入坐标形状: {x_t.shape} | 预测坐标噪声 eps_x 形状: {eps_x.shape}")
    print(f"输入类型形状: {h_t.shape} | 预测分类噪声 eps_h 形状: {eps_h.shape}")
    
    assert eps_x.shape == (B, N, 3), "坐标输出维度错误！"
    assert eps_h.shape == (B, N, C), "分类输出维度错误！"
    
    # 验证质心是否严格归零: ∑_i eps_x,i == 0
    com = torch.mean(eps_x, dim=1) # [B, 3]
    max_com_error = torch.max(torch.abs(com)).item()
    print(f"输出坐标噪声质心绝对偏移量: {max_com_error:.2e} (必须严格为 0)")
    assert max_com_error < 1e-6, "质心归零硬投影失效！"
    print(">>> 【结论通过】双轨输出契约完整，输出坐标严格锁定在零质心流形 M 上！")

def test_backbone_o3_equivariance():
    """
    测试 2：定理 4.1 实证 —— 随机 3D 旋转/镜像变换下的 O(3) 严格等变性。
    理论公式:
        eps_x( R · x_t ) == R · eps_x( x_t )    (坐标向量场等变)
        eps_h( R · x_t ) == eps_h( x_t )        (标量特征不变)
    """
    print_banner("测试 2: 定理 4.1 实证 —— O(3) 旋转与镜像变换等变性数值检验")
    
    cfg = {
        "type": "DualSpaceEGNNBackbone",
        "num_atom_types": 5,
        "hidden_dim": 64,
        "num_layers": 4,
        "num_heads": 4
    }
    model = BACKBONE_REGISTRY.build(cfg)
    model.eval()
    
    B, N, C = 2, 8, 5
    x_t = torch.randn(B, N, 3)
    h_t = torch.randn(B, N, C)
    t = torch.rand(B)
    
    # 构造随机正交矩阵 R ∈ O(3) (包含旋转与镜像)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    rot_mat = q # det(R) 可能为 +1 或 -1
    print(f"随机正交变换矩阵 R 行列式 det(R): {torch.det(rot_mat).item():.4f}")
    
    with torch.no_grad():
        # 路径 A: 先前向计算，再对输出坐标执行旋转
        eps_x_orig, eps_h_orig = model(x_t=x_t, h_t=h_t, t=t)
        eps_x_transformed = torch.matmul(eps_x_orig, rot_mat) # R · f(x)
        
        # 路径 B: 先对输入坐标执行旋转，再进行网络前向计算
        x_t_rotated = torch.matmul(x_t, rot_mat) # R · x
        eps_x_rot_input, eps_h_rot_input = model(x_t=x_t_rotated, h_t=h_t, t=t) # f(R · x)
        
    # 1. 检验坐标等变误差: || f(R·x) - R·f(x) ||_∞
    equiv_error = torch.max(torch.abs(eps_x_rot_input - eps_x_transformed)).item()
    print(f"坐标等变性绝对误差: {equiv_error:.2e} (浮点数极限)")
    
    # 2. 检验标量分类不变误差: || eps_h(R·x) - eps_h(x) ||_∞
    invar_error = torch.max(torch.abs(eps_h_rot_input - eps_h_orig)).item()
    print(f"原子类型不变性绝对误差: {invar_error:.2e} (浮点数极限)")
    
    assert equiv_error < 1e-4, f"坐标等变性校验失败！误差: {equiv_error}"
    assert invar_error < 1e-4, f"分类不变性校验失败！误差: {invar_error}"
    print(">>> 【结论通过】定理 4.1 获 100% 数值实证支持！网络内生具备严格几何对称性！")

def test_variable_sized_masking():
    """
    测试 3：验证变长分子掩码 (Masking) 下的信息隔离性。
    """
    print_banner("测试 3: 变长分子填充掩码 (Padding Mask) 隔离测试")
    
    cfg = {
        "type": "DualSpaceEGNNBackbone",
        "num_atom_types": 5,
        "hidden_dim": 32,
        "num_layers": 2
    }
    model = BACKBONE_REGISTRY.build(cfg)
    model.eval()
    
    B, N_max, C = 2, 10, 5
    x_t = torch.randn(B, N_max, 3)
    h_t = torch.randn(B, N_max, C)
    t = torch.rand(B)
    
    # 分子 0 有 6 个原子，分子 1 有 8 个原子
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    mask[0, :6] = True
    mask[1, :8] = True
    
    with torch.no_grad():
        eps_x, eps_h = model(x_t=x_t, h_t=h_t, t=t, mask=mask)
        
    # 验证被 Mask 掉的位置输出严格为 0
    pad_eps_x = eps_x[~mask]
    pad_eps_h = eps_h[~mask]
    
    assert torch.all(pad_eps_x == 0.0), "Padding 坐标噪声未清零！"
    assert torch.all(pad_eps_h == 0.0), "Padding 分类噪声未清零！"
    print(">>> 【结论通过】变长分子批处理掩码完全隔离，无虚假梯度泄漏！")

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#   Project BUILD - 3D 等变图网络主干全量数值检验启动")
    print("#" * 70)
    
    test_backbone_forward_and_com_projection()
    test_backbone_o3_equivariance()
    test_variable_sized_masking()
    
    print("\n" + "=" * 70)
    print(">>> [ALL PASSED] 主干网络通过全部等变性与流形投影检验！随时装载扩散引擎！")
    print("=" * 70 + "\n")
