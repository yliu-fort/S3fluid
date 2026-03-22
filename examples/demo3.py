#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维球面湍流（barotropic vorticity equation on the unit sphere）
不依赖 shtns，仅使用 NumPy + SciPy：
- 纬向：高斯-勒让德网格
- 经向：FFT
- 球谐变换：Gauss-Legendre 求积 + 显式缔合勒让德函数

这一版的重点是：
1. 运行时球谐变换 fully vectorized：
   - analysis : einsum('jml,jm->ml')
   - synthesis: einsum('jml,ml->jm')
2. 稠密规则张量存储：P[j,m,l], dP[j,m,l]
3. 预分配复频域/谱域工作区，避免 Python 层重复分配
4. 尽量减少临时数组；但 NumPy/SciPy FFT 仍会返回自己的输出缓冲区，
   因此这里做到的是“Python 层近似零额外拷贝 / 零重复分配”，
   而不是 FFT 内核层面的绝对 zero-copy。

方程（单位球面）：
    ∂ζ/∂t + J(ψ, ζ) = ν ∇²ζ
    ζ = ∇²ψ

说明：
1. 这是教学/原型版实现，分辨率适合 lmax <= 63 或 127；
   若上到 500+，纯 Python 显式球谐矩阵会明显变慢。
2. 采用复球谐系数 a[m, l]（仅存 m>=0 部分），
   实场通过厄米共轭自动恢复。
3. 使用高斯-勒让德求积，变换在该网格上是谱方法。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, rfft
from scipy.special import gammaln, lpmv


def _compute_legendre_matrices(lmax: int, mu: np.ndarray, sin_theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用稳定的递推关系计算正交归一化的缔合勒让德函数 P_l^m(mu) 及其导数。
    返回:
        P : (J, M, L)
        dP: (J, M, L), 这里 dP 是 dP_lm/dtheta
        valid: (M, L) 布尔矩阵，l >= m 为 True
    """
    J = len(mu)
    M = L = lmax + 1
    P = np.zeros((J, M, L), dtype=np.float64)
    dP = np.zeros((J, M, L), dtype=np.float64)
    
    # 1. 基础值：P_00 = sqrt(1 / 4pi)
    P[:, 0, 0] = np.sqrt(1.0 / (4.0 * np.pi))
    
    # 2. 对角线递推：l = m
    # P_mm = -sqrt((2m+1)/(2m)) * sin_theta * P_{m-1,m-1}
    for m in range(1, M):
        P[:, m, m] = -np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sin_theta * P[:, m-1, m-1]
        
    # 3. 次对角线递推：l = m + 1
    # P_{m+1,m} = mu * sqrt(2m+3) * P_mm
    for m in range(M):
        if m + 1 < L:
            P[:, m, m+1] = mu * np.sqrt(2.0 * m + 3.0) * P[:, m, m]
            
    # 4. 三对角递推：l > m + 1
    # P_lm = a_lm * mu * P_{l-1,m} - b_lm * P_{l-2,m}
    for m in range(M):
        for l in range(m + 2, L):
            a_lm = np.sqrt((4.0 * l**2 - 1.0) / (l**2 - m**2))
            b_lm = np.sqrt(((2.0 * l + 1.0) * ((l - 1)**2 - m**2)) / ((2.0 * l - 3.0) * (l**2 - m**2)))
            P[:, m, l] = a_lm * mu * P[:, m, l-1] - b_lm * P[:, m, l-2]
            
    # 5. 计算导数 dP/dtheta
    # dP/dtheta = (l*mu*P_lm - sqrt((2l+1)(l^2-m^2)/(2l-1)) * P_{l-1,m}) / sin_theta
    # 注意：sin_theta 在 __post_init__ 中已由 arccos(mu) 计算，并设置了 1e-30 的 floor
    for m in range(M):
        for l in range(m, L):
            if l > 0:
                c_lm = np.sqrt(((2.0 * l + 1.0) * (l**2 - m**2)) / (2.0 * l - 1.0))
                dP[:, m, l] = (l * mu * P[:, m, l] - c_lm * P[:, m, l-1]) / sin_theta
            else:
                dP[:, m, l] = 0.0

    # 构造 valid 掩码
    m_idx = np.arange(M)[:, None]
    l_idx = np.arange(L)[None, :]
    valid = l_idx >= m_idx
    
    return P, dP, valid


@dataclass
class SphericalHarmonicTransform:
    lmax: int
    nlat: int | None = None
    nlon: int | None = None

    def __post_init__(self) -> None:
        if self.nlat is None:
            self.nlat = self.lmax + 1
        if self.nlon is None:
            self.nlon = 2 * (self.lmax + 1)

        self.nlat = int(self.nlat)
        self.nlon = int(self.nlon)

        # 高斯-勒让德节点：mu = cos(theta), 权重 w，对 dmu 积分精确
        self.mu, self.w = np.polynomial.legendre.leggauss(self.nlat)
        self.theta = np.arccos(self.mu)
        self.sin_theta = np.sqrt(np.maximum(1.0 - self.mu**2, 1e-30))
        self.phi = 2.0 * np.pi * np.arange(self.nlon) / self.nlon

        M = L = self.lmax + 1
        J = self.nlat
        K = self.nlon // 2 + 1
        self.M = M
        self.L = L
        self.J = J
        self.K = K

        # 使用稳定递推计算 P_{lm} 和 dP_{lm}/dtheta
        P, dP, self.valid = _compute_legendre_matrices(self.lmax, self.mu, self.sin_theta)

        # 存成 C contiguous 张量，避免后续隐式复制
        self.P = np.ascontiguousarray(P, dtype=np.float64)      # (J,M,L)
        self.dP = np.ascontiguousarray(dP, dtype=np.float64)    # (J,M,L)
        self.Pw = np.ascontiguousarray(self.P * self.w[:, None, None], dtype=np.float64)

        # 拉普拉斯本征值：Δ Y_lm = -l(l+1) Y_lm
        ell = np.broadcast_to(np.arange(L, dtype=np.float64)[None, :], (M, L))
        self.lap = np.zeros((M, L), dtype=np.float64)
        self.lap[self.valid] = -ell[self.valid] * (ell[self.valid] + 1.0)
        self.inv_lap = np.zeros_like(self.lap)
        mask = self.lap != 0.0
        self.inv_lap[mask] = 1.0 / self.lap[mask]

        self.mvals = np.arange(M, dtype=np.float64)
        self.im = 1j * self.mvals[:, None]

        # 预分配工作区：复频域 / 复谱域
        self._freq = np.zeros((J, K), dtype=np.complex128)
        self._lm = np.zeros((M, L), dtype=np.complex128)

    def analysis(self, field: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """
        网格 -> 球谐系数
        返回 a[m, l]，只在 l>=m 区域有效；其它位置保持 0。
        """
        if field.shape != (self.J, self.nlon):
            raise ValueError(f"field shape must be {(self.J, self.nlon)}, got {field.shape}")

        if out is None:
            out = np.empty((self.M, self.L), dtype=np.complex128)

        F = rfft(field, axis=1)  # FFT 输出缓冲区无法避免
        np.einsum('jml,jm->ml', self.Pw, F[:, : self.M], out=out, optimize=True)
        out *= (2.0 * np.pi / self.nlon)
        out[~self.valid] = 0.0
        return out

    def synthesis(self, a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """
        球谐系数 -> 网格
        a 的形状为 (m,l)，仅 l>=m 区域有效。
        """
        if a.shape != (self.M, self.L):
            raise ValueError(f"coeff shape must be {(self.M, self.L)}, got {a.shape}")

        self._freq.fill(0.0)
        np.einsum('jml,ml->jm', self.P, a, out=self._freq[:, : self.M], optimize=True)
        self._freq[:, : self.M] *= self.nlon

        field = irfft(self._freq, n=self.nlon, axis=1)
        if out is not None:
            out[...] = field
            return out
        return field

    def dphi(self, a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """谱空间求 ∂/∂phi 后回到网格。"""
        if a.shape != (self.M, self.L):
            raise ValueError(f"coeff shape must be {(self.M, self.L)}, got {a.shape}")

        np.multiply(a, self.im, out=self._lm)
        self._lm[~self.valid] = 0.0
        return self.synthesis(self._lm, out=out)

    def dtheta(self, a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """谱空间求 ∂/∂theta 后回到网格。"""
        if a.shape != (self.M, self.L):
            raise ValueError(f"coeff shape must be {(self.M, self.L)}, got {a.shape}")

        self._freq.fill(0.0)
        np.einsum('jml,ml->jm', self.dP, a, out=self._freq[:, : self.M], optimize=True)
        self._freq[:, : self.M] *= self.nlon

        field = irfft(self._freq, n=self.nlon, axis=1)
        if out is not None:
            out[...] = field
            return out
        return field

    def apply_laplacian(self, a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if a.shape != (self.M, self.L):
            raise ValueError(f"coeff shape must be {(self.M, self.L)}, got {a.shape}")
        if out is None:
            out = np.empty_like(a)
        np.multiply(a, self.lap, out=out)
        out[~self.valid] = 0.0
        return out

    def invert_laplacian(self, a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """解 Δψ = a，对 l=0 模置 0。"""
        if a.shape != (self.M, self.L):
            raise ValueError(f"coeff shape must be {(self.M, self.L)}, got {a.shape}")
        if out is None:
            out = np.empty_like(a)
        np.multiply(a, self.inv_lap, out=out)
        out[~self.valid] = 0.0
        out[0, 0] = 0.0
        return out


@dataclass
class SphereTurbulence2D:
    sht: SphericalHarmonicTransform
    nu: float = 1.0e-7
    filter_alpha: float = 36.0
    filter_order: int = 8

    def __post_init__(self) -> None:
        l = np.arange(self.sht.L, dtype=np.float64)
        self.spec_filter = np.exp(-self.filter_alpha * (l / self.sht.lmax) ** self.filter_order)
        self.spec_filter[0] = 1.0

        self.init_slope = np.ones(self.sht.L, dtype=np.float64)
        mask = l > 0
        self.init_slope[mask] = l[mask] ** (-1.0 / 3.0)

        # 复谱工作区
        self._psi = np.zeros((self.sht.M, self.sht.L), dtype=np.complex128)
        self._diff = np.zeros((self.sht.M, self.sht.L), dtype=np.complex128)

    def filter_coeffs(self, a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty_like(a)
        np.multiply(a, self.spec_filter[None, :], out=out)
        out[~self.sht.valid] = 0.0
        return out

    def streamfunction_from_vorticity(self, zeta_lm: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        return self.sht.invert_laplacian(zeta_lm, out=out)

    def velocity_from_streamfunction(self, psi_lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        返回 (u_theta, u_phi)，采用约定：
            u_theta =  (1/sinθ) ∂ψ/∂φ
            u_phi   = -(∂ψ/∂θ)
        """
        dpsi_dphi = self.sht.dphi(psi_lm)
        dpsi_dtheta = self.sht.dtheta(psi_lm)
        u_theta = dpsi_dphi / self.sht.sin_theta[:, None]
        u_phi = -dpsi_dtheta
        return u_theta, u_phi

    def rhs(self, zeta_lm: np.ndarray) -> np.ndarray:
        psi_lm = self.streamfunction_from_vorticity(zeta_lm, out=self._psi)

        u_theta, u_phi = self.velocity_from_streamfunction(psi_lm)
        dzeta_dtheta = self.sht.dtheta(zeta_lm)
        dzeta_dphi = self.sht.dphi(zeta_lm) / self.sht.sin_theta[:, None]

        adv = u_theta * dzeta_dtheta + u_phi * dzeta_dphi
        adv_lm = self.filter_coeffs(self.sht.analysis(adv))

        self.sht.apply_laplacian(zeta_lm, out=self._diff)
        self._diff *= self.nu
        return -adv_lm + self._diff

    def random_initial_vorticity(self, seed: int = 0, amplitude: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        grid = rng.standard_normal((self.sht.J, self.sht.nlon))
        zeta_lm = self.sht.analysis(grid)
        zeta_lm *= self.init_slope[None, :]
        zeta_lm = self.filter_coeffs(zeta_lm)
        zeta_lm[0, 0] = 0.0
        return amplitude * zeta_lm

    def kinetic_energy(self, zeta_lm: np.ndarray) -> float:
        psi_lm = self.streamfunction_from_vorticity(zeta_lm)
        zeta = self.sht.synthesis(zeta_lm)
        psi = self.sht.synthesis(psi_lm)
        integrand = -0.5 * psi * zeta
        return float((2.0 * np.pi / self.sht.nlon) * np.sum(self.sht.w[:, None] * integrand))

    def step_rk4(self, zeta_lm: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(zeta_lm)
        k2 = self.rhs(zeta_lm + 0.5 * dt * k1)
        k3 = self.rhs(zeta_lm + 0.5 * dt * k2)
        k4 = self.rhs(zeta_lm + dt * k3)
        out = zeta_lm + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.filter_coeffs(out)


def transform_self_test(lmax: int = 15) -> tuple[float, float]:
    """简单自检：band-limited 场的谱-网格-谱一致性 + 拉普拉斯逆一致性。"""
    sht = SphericalHarmonicTransform(lmax=lmax)
    rng = np.random.default_rng(1234)

    alm = rng.standard_normal((sht.M, sht.L)) + 1j * rng.standard_normal((sht.M, sht.L))
    alm[~sht.valid] = 0.0
    alm[0, :] = alm[0, :].real

    field = sht.synthesis(alm)
    field_rt = sht.synthesis(sht.analysis(field))
    rel_rt = np.linalg.norm(field_rt - field) / np.linalg.norm(field)

    alm2 = rng.standard_normal((sht.M, sht.L)) + 1j * rng.standard_normal((sht.M, sht.L))
    alm2[~sht.valid] = 0.0
    alm2[0, :] = alm2[0, :].real
    lap_inv = sht.invert_laplacian(sht.apply_laplacian(alm2))
    alm2_ref = alm2.copy()
    alm2_ref[0, 0] = 0.0
    rel_lap = np.linalg.norm(lap_inv - alm2_ref) / np.linalg.norm(alm2_ref)
    return float(rel_rt), float(rel_lap)


def main() -> None:
    # 纯 Python 显式球谐变换，建议先从 31 / 63 起步
    lmax = 63
    nlat = lmax + 1
    nlon = 2 * (lmax + 1)

    dt = 1.0e-2
    steps = 1000
    save_every = 20
    nu = 1.0e-7

    rel_rt, rel_lap = transform_self_test(lmax=min(15, lmax))
    print(f"transform self-test: roundtrip={rel_rt:.3e}, laplacian-inverse={rel_lap:.3e}")

    sht = SphericalHarmonicTransform(lmax=lmax, nlat=nlat, nlon=nlon)
    model = SphereTurbulence2D(sht=sht, nu=nu)
    zeta_lm = model.random_initial_vorticity(seed=42, amplitude=1.0)

    history: list[np.ndarray] = []
    energy: list[float] = []

    for n in range(steps + 1):
        if n % save_every == 0:
            zeta = sht.synthesis(zeta_lm)
            history.append(zeta.copy())
            energy.append(model.kinetic_energy(zeta_lm))
            print(f"step={n:5d}, energy={energy[-1]:.8e}")

        if n < steps:
            zeta_lm = model.step_rk4(zeta_lm, dt)

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(energy)) * save_every * dt, energy, lw=1.8)
    plt.xlabel("time")
    plt.ylabel("kinetic energy")
    plt.title("2D turbulence on the sphere (vectorized, no shtns)")
    plt.tight_layout()
    plt.show()

    zeta_last = history[-1]
    lon_deg = np.degrees(sht.phi)
    lat_deg = 90.0 - np.degrees(sht.theta)

    plt.figure(figsize=(9, 4))
    plt.pcolormesh(lon_deg, lat_deg, zeta_last, shading="auto", cmap="RdBu_r")
    plt.xlabel("longitude (deg)")
    plt.ylabel("latitude (deg)")
    plt.title("Final vorticity")
    plt.colorbar(label=r"$\zeta$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
