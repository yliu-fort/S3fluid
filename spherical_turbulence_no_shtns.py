#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维球面湍流（barotropic vorticity equation on the unit sphere）
不依赖 shtns，仅使用 NumPy + SciPy：
- 纬向：高斯-勒让德网格
- 经向：FFT
- 球谐变换：通过 Associated Legendre functions 显式构造

方程（单位球面）：
    ∂ζ/∂t + J(ψ, ζ) = ν ∇²ζ
    ζ = ∇²ψ

说明：
1. 这是教学/原型版实现，分辨率适合 lmax <= 63 或 127；
   若上到 500+，纯 Python 显式球谐矩阵会明显变慢。
2. 采用复球谐系数 a[l, m]（仅存 m>=0 部分），
   实场通过厄米共轭自动恢复。
3. 使用高斯-勒让德求积，变换在该网格上是谱方法。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv, gammaln


# =========================
# 球谐变换：Gauss-Legendre + FFT
# =========================


def _norm_lm(l: int, m: int) -> float:
    """复球谐 Y_l^m 的归一化系数（Condon-Shortley phase 已包含在 lpmv 中）。"""
    logn = 0.5 * (
        math.log(2 * l + 1)
        - math.log(4 * math.pi)
        + gammaln(l - m + 1)
        - gammaln(l + m + 1)
    )
    return float(math.exp(logn))


@dataclass
class SphericalHarmonicTransform:
    lmax: int
    nlat: int | None = None
    nlon: int | None = None

    def __post_init__(self) -> None:
        if self.nlat is None:
            # 对到 lmax 的球谐，常见选择是 nlat = lmax + 1
            self.nlat = self.lmax + 1
        if self.nlon is None:
            # 至少 2*lmax+1；取 2*(lmax+1) 便于 rfft
            self.nlon = 2 * (self.lmax + 1)

        self.nlat = int(self.nlat)
        self.nlon = int(self.nlon)

        # 高斯-勒让德节点：mu = cos(theta), 权重 w，对 dmu 积分精确
        self.mu, self.w = np.polynomial.legendre.leggauss(self.nlat)
        self.theta = np.arccos(self.mu)
        self.sin_theta = np.sqrt(np.maximum(1.0 - self.mu**2, 1e-30))
        self.phi = 2.0 * np.pi * np.arange(self.nlon) / self.nlon

        # 预计算 S_lm(theta_j) = N_lm P_l^m(mu_j)
        # 以及 d/dtheta[S_lm]
        # 存储方式：list[m] -> array(nlat, lmax-m+1)
        self.P = []
        self.dP_dtheta = []
        self.ell = []
        self.m = np.arange(self.lmax + 1)

        for m in range(self.lmax + 1):
            ell = np.arange(m, self.lmax + 1)
            Plm = np.empty((self.nlat, len(ell)), dtype=np.float64)
            dPlm_dtheta = np.empty_like(Plm)

            # lpmv(m, l, x) 对每个 l 单独求，便于兼容 SciPy 常见接口
            for idx, l in enumerate(ell):
                P_lm = lpmv(m, l, self.mu)  # 已含 Condon-Shortley phase
                N_lm = _norm_lm(l, m)
                S_lm = N_lm * P_lm
                Plm[:, idx] = S_lm

                # dP_l^m/dmu = (l*mu*P_l^m - (l+m) P_{l-1}^m) / (mu^2 - 1)
                if l == m:
                    P_lm_minus_1 = np.zeros_like(self.mu)
                else:
                    P_lm_minus_1 = lpmv(m, l - 1, self.mu)
                dP_dmu = (l * self.mu * P_lm - (l + m) * P_lm_minus_1) / (self.mu**2 - 1.0)
                dS_dtheta = -self.sin_theta * N_lm * dP_dmu
                dPlm_dtheta[:, idx] = dS_dtheta

            self.P.append(Plm)
            self.dP_dtheta.append(dPlm_dtheta)
            self.ell.append(ell)

        # 拉普拉斯本征值：Δ Y_lm = -l(l+1) Y_lm
        self.lap_eigs = np.zeros((self.lmax + 1, self.lmax + 1), dtype=np.float64)
        for m in range(self.lmax + 1):
            ell = self.ell[m]
            self.lap_eigs[m, m:] = -ell * (ell + 1.0)

    def analysis(self, field: np.ndarray) -> np.ndarray:
        """
        网格 -> 球谐系数
        返回 a[m, l]，只存 m>=0 区域；m>l 部分保持 0。
        """
        if field.shape != (self.nlat, self.nlon):
            raise ValueError(f"field shape must be {(self.nlat, self.nlon)}, got {field.shape}")

        # 经向 Fourier 系数：F_m(mu_j) = sum_k f_jk exp(-i m phi_k)
        F = np.fft.rfft(field, axis=1)

        a = np.zeros((self.lmax + 1, self.lmax + 1), dtype=np.complex128)
        prefac = 2.0 * np.pi / self.nlon
        weighted = self.w[:, None]

        for m in range(self.lmax + 1):
            # 积分 a_lm = ∫ f Y_lm^* dΩ ≈ sum_j w_j * (2π/nlon) * F_m(mu_j) * S_lm(mu_j)
            # S_lm 为实数，因此只需转置乘
            a[m, m:] = prefac * (self.P[m].T @ (weighted[:, 0] * F[:, m]))
        return a

    def synthesis(self, a: np.ndarray) -> np.ndarray:
        """
        球谐系数 -> 网格
        a 存储格式为 a[m, l], m>=0, l>=m
        """
        if a.shape != (self.lmax + 1, self.lmax + 1):
            raise ValueError(f"coeff shape must be {(self.lmax + 1, self.lmax + 1)}, got {a.shape}")

        freq = np.zeros((self.nlat, self.nlon // 2 + 1), dtype=np.complex128)
        for m in range(self.lmax + 1):
            # G_m(theta_j) = sum_l a_lm S_lm(theta_j)
            freq[:, m] = self.nlon * (self.P[m] @ a[m, m:])
        field = np.fft.irfft(freq, n=self.nlon, axis=1)
        return field.real

    def dphi(self, a: np.ndarray) -> np.ndarray:
        """谱空间求 ∂/∂phi 后回到网格。"""
        da = np.zeros_like(a)
        for m in range(self.lmax + 1):
            da[m, m:] = 1j * m * a[m, m:]
        return self.synthesis(da)

    def dtheta(self, a: np.ndarray) -> np.ndarray:
        """谱空间求 ∂/∂theta 后回到网格。"""
        freq = np.zeros((self.nlat, self.nlon // 2 + 1), dtype=np.complex128)
        for m in range(self.lmax + 1):
            freq[:, m] = self.nlon * (self.dP_dtheta[m] @ a[m, m:])
        return np.fft.irfft(freq, n=self.nlon, axis=1).real

    def apply_laplacian(self, a: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a)
        for m in range(self.lmax + 1):
            out[m, m:] = self.lap_eigs[m, m:] * a[m, m:]
        return out

    def invert_laplacian(self, a: np.ndarray) -> np.ndarray:
        """解 Δψ = a，对 l=0 模置 0。"""
        out = np.zeros_like(a)
        for m in range(self.lmax + 1):
            ell = self.ell[m]
            eig = -ell * (ell + 1.0)
            inv = np.zeros_like(eig, dtype=np.float64)
            mask = eig != 0.0
            inv[mask] = 1.0 / eig[mask]
            out[m, m:] = a[m, m:] * inv
        return out


# =========================
# 二维球面湍流求解器
# =========================


@dataclass
class SphereTurbulence2D:
    sht: SphericalHarmonicTransform
    nu: float = 1.0e-7
    filter_alpha: float = 36.0
    filter_order: int = 8

    def __post_init__(self) -> None:
        self.filter_l = np.arange(self.sht.lmax + 1)
        self.spec_filter = np.exp(-self.filter_alpha * (self.filter_l / self.sht.lmax) ** self.filter_order)
        self.spec_filter[0] = 1.0

        # 对应 notebook 里的 degree^(-1/3) 初始化斜率
        self.init_slope = np.ones(self.sht.lmax + 1)
        mask = self.filter_l > 0
        self.init_slope[mask] = self.filter_l[mask] ** (-1.0 / 3.0)

    def filter_coeffs(self, a: np.ndarray) -> np.ndarray:
        out = a.copy()
        for m in range(self.sht.lmax + 1):
            out[m, m:] *= self.spec_filter[m:]
        return out

    def streamfunction_from_vorticity(self, zeta_lm: np.ndarray) -> np.ndarray:
        return self.sht.invert_laplacian(zeta_lm)

    def velocity_from_streamfunction(self, psi_lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        返回 (u_theta, u_phi)，采用约定：
            u_theta =  (1/sinθ) ∂ψ/∂φ
            u_phi   = -(∂ψ/∂θ)
        与你原 notebook 的符号保持一致。
        """
        dpsi_dphi = self.sht.dphi(psi_lm)
        dpsi_dtheta = self.sht.dtheta(psi_lm)
        u_theta = dpsi_dphi / self.sht.sin_theta[:, None]
        u_phi = -dpsi_dtheta
        return u_theta, u_phi

    def rhs(self, zeta_lm: np.ndarray) -> np.ndarray:
        psi_lm = self.streamfunction_from_vorticity(zeta_lm)

        # 速度
        u_theta, u_phi = self.velocity_from_streamfunction(psi_lm)

        # 梯度
        dzeta_dtheta = self.sht.dtheta(zeta_lm)
        dzeta_dphi = self.sht.dphi(zeta_lm) / self.sht.sin_theta[:, None]

        adv = u_theta * dzeta_dtheta + u_phi * dzeta_dphi
        adv_lm = self.filter_coeffs(self.sht.analysis(adv))

        # ν Δζ
        diff_lm = self.nu * self.sht.apply_laplacian(zeta_lm)
        return -adv_lm + diff_lm

    def random_initial_vorticity(self, seed: int = 0, amplitude: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        grid = rng.standard_normal((self.sht.nlat, self.sht.nlon))
        zeta_lm = self.sht.analysis(grid)
        for m in range(self.sht.lmax + 1):
            zeta_lm[m, m:] *= self.init_slope[m:]
        zeta_lm = self.filter_coeffs(zeta_lm)
        zeta_lm[0, 0] = 0.0  # 去掉平均模
        return amplitude * zeta_lm

    def kinetic_energy(self, zeta_lm: np.ndarray) -> float:
        psi_lm = self.streamfunction_from_vorticity(zeta_lm)
        # E = -1/2 ∫ ψζ dΩ ；若取谱正交展开，则 E = 1/2 Σ l(l+1)|ψ_lm|^2
        # 这里直接做网格积分，更直观。
        zeta = self.sht.synthesis(zeta_lm)
        psi = self.sht.synthesis(psi_lm)
        integrand = -0.5 * psi * zeta
        return float((2.0 * np.pi / self.sht.nlon) * np.sum(self.sht.w[:, None] * integrand))

    def step_rk4(self, zeta_lm: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(zeta_lm)
        k2 = self.rhs(zeta_lm + 0.5 * dt * k1)
        k3 = self.rhs(zeta_lm + 0.5 * dt * k2)
        k4 = self.rhs(zeta_lm + dt * k3)
        out = zeta_lm + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.filter_coeffs(out)


# =========================
# 示例主程序
# =========================


def main() -> None:
    # 纯 Python 显式球谐变换，建议先从 31 / 63 起步
    lmax = 63
    nlat = lmax + 1
    nlon = 2 * (lmax + 1)

    dt = 1.0e-2
    steps = 1000
    save_every = 20
    nu = 1.0e-7

    sht = SphericalHarmonicTransform(lmax=lmax, nlat=nlat, nlon=nlon)
    model = SphereTurbulence2D(sht=sht, nu=nu)

    zeta_lm = model.random_initial_vorticity(seed=42, amplitude=1.0)

    history = []
    energy = []

    for n in range(steps + 1):
        if n % save_every == 0:
            zeta = sht.synthesis(zeta_lm)
            history.append(zeta.copy())
            energy.append(model.kinetic_energy(zeta_lm))
            print(f"step={n:5d}, energy={energy[-1]:.8e}")

        if n < steps:
            zeta_lm = model.step_rk4(zeta_lm, dt)

    # 诊断图 1：能量随时间
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(energy)) * save_every * dt, energy, lw=1.8)
    plt.xlabel("time")
    plt.ylabel("kinetic energy")
    plt.title("2D turbulence on the sphere (no shtns)")
    plt.tight_layout()
    plt.show()

    # 诊断图 2：末时刻涡度场
    zeta_last = history[-1]
    lon_deg = np.degrees(sht.phi)
    lat_deg = 90.0 - np.degrees(sht.theta)  # latitude = pi/2 - theta

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
