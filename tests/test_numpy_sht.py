import numpy as np
import pytest
from kivy_gpgpu_pde.numpy_sht import NumPySHT

@pytest.fixture
def sht_instance():
    # 使用较低的分辨率以加快测试速度
    return NumPySHT(l_max=16)

def test_initialization(sht_instance):
    sht = sht_instance
    assert sht.l_max == 16
    assert sht.n_lat == int(1.5 * 16)
    assert sht.n_lon == 2 * sht.n_lat

    # 高斯求积权重和应为 2 (从 -1 到 1 的积分)
    assert np.isclose(np.sum(sht.weights), 2.0)

def test_forward_inverse_identity(sht_instance):
    """
    测试前向变换后跟逆向变换应该能恢复原数据。
    L2 误差应该很小。
    """
    sht = sht_instance

    # 构造一个测试函数: f(theta, phi) = cos(theta) + sin^2(theta) * cos(2*phi)
    lats, lons = np.meshgrid(sht.lats, sht.lons, indexing='ij')

    # 实数场数据
    data = np.cos(lats) + np.sin(lats)**2 * np.cos(2 * lons)

    # 1. 前向变换
    coeffs = sht.forward_sht(data)

    # 2. 逆向变换
    reconstructed_data = sht.inverse_sht(coeffs)

    # 3. 计算误差
    error = np.linalg.norm(data - reconstructed_data) / np.linalg.norm(data)

    # 容差设为 1e-10 (考虑到浮点精度和截断)
    assert error < 1e-10, f"前向/逆向变换误差过大: {error}"

def test_inverse_sht_grad(sht_instance):
    """
    测试带梯度的逆向变换
    """
    sht = sht_instance

    # 测试函数 f(theta, phi) = cos(theta) + sin(theta) * cos(phi)
    # \partial f / \partial \phi = -sin(theta) * sin(phi)
    # \partial f / \partial \theta = -sin(theta) + cos(theta) * cos(phi)

    lats, lons = np.meshgrid(sht.lats, sht.lons, indexing='ij')
    data = np.cos(lats) + np.sin(lats) * np.cos(lons)

    # 解析导数
    exact_df_dphi = -np.sin(lats) * np.sin(lons)
    exact_df_dtheta = -np.sin(lats) + np.cos(lats) * np.cos(lons)

    # 1. 前向变换获取系数
    coeffs = sht.forward_sht(data)

    # 2. 使用逆变换求导
    df_dphi, df_dtheta = sht.inverse_sht_grad(coeffs)

    # 3. 比较经向导数误差
    error_phi = np.linalg.norm(df_dphi - exact_df_dphi) / np.linalg.norm(exact_df_dphi)

    # 4. 比较纬向导数误差
    error_theta = np.linalg.norm(df_dtheta - exact_df_dtheta) / np.linalg.norm(exact_df_dtheta)

    assert error_phi < 1e-10, f"经向导数误差过大: {error_phi}"
    assert error_theta < 1e-10, f"纬向导数误差过大: {error_theta}"

def test_harmonic_orthogonality(sht_instance):
    """
    测试预计算的基函数是否满足正交归一化条件
    对于固定的 m，不同 l 的基函数内积：
    \int_0^\pi P_l^m(\cos\theta) P_{l'}^m(\cos\theta) \sin\theta d\theta = \delta_{ll'}

    由于我们是在高斯节点上离散化，可以转化为对权重的求和
    \sum_i w_i P_l^m(x_i) P_{l'}^m(x_i) = \delta_{ll'}
    """
    sht = sht_instance
    m = 1
    i_m = np.where(sht.m_values == m)[0][0]

    # 测试两个不同的 l
    l1 = 2
    l2 = 3

    p1 = sht.alp[i_m, l1, :]
    p2 = sht.alp[i_m, l2, :]

    # 正交性
    inner_prod = np.sum(p1 * p2 * sht.weights)
    assert np.abs(inner_prod) < 1e-12, f"正交性失败: {inner_prod}"

    # 归一化条件 (自身内积)
    norm = np.sum(p1 * p1 * sht.weights)

    # 由于我们在 _precompute_alp 里的归一化条件，它应该等于 1
    # 对于 m>0 我们不需要乘 2（因为那是在重构 f_m 的实部虚部时乘的）
    assert np.isclose(norm, 1.0 / (2 * np.pi), atol=1e-12), f"归一化失败: {norm}"
