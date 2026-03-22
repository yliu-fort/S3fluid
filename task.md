# 二维球面 PDE 实时求解与可视化 TODO

## 0. 固定技术边界

* [ ] PDE 固定为
  [
  \partial_t \zeta + J(\psi,\zeta)=\nu \nabla^2 \zeta,\qquad \zeta=\nabla^2\psi
  ]
* [ ] 几何固定为单位球面
* [ ] 纬向离散固定为 `Gauss-Legendre`
* [ ] 经向离散固定为 `FFT`
* [ ] 球谐变换固定为显式 `P_l^m` / `dP_l^m/dtheta`
* [ ] 谱系数只存 `m >= 0`
* [ ] 时间推进固定为 `RK4`
* [ ] 稳定化固定为指数谱滤波
* [ ] 初始化固定为：随机网格场 → analysis → 乘 `l^(-1/3)` → 谱滤波 → 去掉 `(0,0)` 平均模
* [ ] 诊断固定为：球面涡度可视化 + kinetic energy 曲线

> 这些都来自 demo 本体，不要改路线。

---

## 1. 项目骨架

```text
src/
  app/
    main.ts
    loop.ts
    gui.ts

  solver/
    config.ts
    precompute.ts
    layout.ts
    buffers.ts
    pipeline.ts
    diagnostics.ts

  shaders/
    initRandom.wgsl
    fftForwardLon.wgsl
    fftInverseLon.wgsl
    legendreAnalysis.wgsl
    legendreSynthesis.wgsl
    legendreSynthesisDTheta.wgsl
    mulIM.wgsl
    applyLaplacian.wgsl
    invertLaplacian.wgsl
    filterSpectrum.wgsl
    velocityFromPsi.wgsl
    advectGrid.wgsl
    rhsCompose.wgsl
    rk4Stage.wgsl
    rk4Combine.wgsl
    energyIntegrand.wgsl
    reduceSum.wgsl
    sphere.vert.wgsl
    scalarColor.frag.wgsl

  render/
    sphereView.ts
    colormap.ts

  tests/
    cpu-reference/
    solver/
    shaders/
    visual/
```

---

## 2. Phase A：CPU 参考层

### `tests/cpu-reference/shtReference.ts`

* [ ] 实现 `_norm_lm(l,m)`
* [ ] 实现 `analysis(field)`
* [ ] 实现 `synthesis(a)`
* [ ] 实现 `dphi(a)`
* [ ] 实现 `dtheta(a)`
* [ ] 实现 `applyLaplacian(a)`
* [ ] 实现 `invertLaplacian(a)`

### `tests/cpu-reference/modelReference.ts`

* [ ] 实现 `filter_coeffs`
* [ ] 实现 `streamfunction_from_vorticity`
* [ ] 实现 `velocity_from_streamfunction`
* [ ] 实现 `rhs`
* [ ] 实现 `random_initial_vorticity`
* [ ] 实现 `kinetic_energy`
* [ ] 实现 `step_rk4`

### CPU 参考验收

* [ ] `nlat = lmax + 1`
* [ ] `nlon = 2 * (lmax + 1)`
* [ ] `sinTheta = sqrt(max(1 - mu^2, 1e-30))`
* [ ] `lap_eigs = -l(l+1)`
* [ ] `u_theta = dpsi_dphi / sinTheta`
* [ ] `u_phi = -dpsi_dtheta`
* [ ] `rhs = -adv_lm + nu * laplacian(zeta_lm)`
* [ ] `step_rk4` 使用 4 次 RHS 且最后再滤波
* [ ] kinetic energy 用网格积分
  [
  E \approx \frac{2\pi}{nlon}\sum_j w_j(-\frac12 \psi\zeta)
  ]

这些都与 demo 完全一致。

---

## 3. Phase B：预计算与布局

### `src/solver/config.ts`

* [ ] 定义 `lmax`
* [ ] 定义 `nlat = lmax + 1`
* [ ] 定义 `nlon = 2 * (lmax + 1)`
* [ ] 定义 `dt`
* [ ] 定义 `nu`
* [ ] 定义 `filterAlpha`
* [ ] 定义 `filterOrder`
* [ ] 定义 `stepsPerFrame`
* [ ] 定义 `seed`
* [ ] 定义 `amplitude`

### `src/solver/precompute.ts`

* [ ] 生成 `mu`
* [ ] 生成 `w`
* [ ] 生成 `theta`
* [ ] 生成 `sinTheta`
* [ ] 生成 `phi`
* [ ] 生成 `P_lm`
* [ ] 生成 `dP_lm_dtheta`
* [ ] 生成 `lapEigs`
* [ ] 生成 `specFilter = exp(-alpha * (l/lmax)^order)`
* [ ] 生成 `initSlope[l] = l^(-1/3)`，`l=0` 单独处理

### `src/solver/layout.ts`

* [ ] 定义 `(m,l)` 到线性地址映射
* [ ] 定义 `(j,k)` 到线性地址映射
* [ ] 定义复数 `vec2<f32>` 布局
* [ ] 明确 `l < m` 区域必须置零
* [ ] 明确 `m` 方向长度为 `lmax + 1`
* [ ] 明确 `l` 方向长度为 `lmax + 1`

### `src/solver/buffers.ts`

* [ ] 创建 `zetaLM_A`
* [ ] 创建 `zetaLM_B`
* [ ] 创建 `psiLM`
* [ ] 创建 `k1/k2/k3/k4`
* [ ] 创建 `tmpLM`
* [ ] 创建 `zetaGrid`
* [ ] 创建 `psiGrid`
* [ ] 创建 `dpsiDphiGrid`
* [ ] 创建 `dpsiDthetaGrid`
* [ ] 创建 `uThetaGrid`
* [ ] 创建 `uPhiGrid`
* [ ] 创建 `dzetaDphiGrid`
* [ ] 创建 `dzetaDthetaGrid`
* [ ] 创建 `advGrid`
* [ ] 创建 `energyTerms`
* [ ] 创建显示纹理

### 本阶段测试

* [ ] `sum(w)` 近似 2
* [ ] `mu` 与 `w` 对称性正确
* [ ] `lapEigs[m,l] == -l(l+1)`
* [ ] `specFilter[0] == 1`
* [ ] `specFilter` 对高阶模单调衰减
* [ ] `initSlope[l>0] ~ l^(-1/3)`
* [ ] 任意索引不越界
* [ ] ping-pong 正确切换

---

## 4. Phase C：经向 FFT

### `shaders/fftForwardLon.wgsl`

* [ ] 输入 `grid(j,k)`
* [ ] 输出 `F(j,m)`

### `shaders/fftInverseLon.wgsl`

* [ ] 输入 `F(j,m)`
* [ ] 输出 `grid(j,k)`

### `src/solver/pipeline.ts`

* [ ] 注册 forward FFT pass
* [ ] 注册 inverse FFT pass
* [ ] 把 FFT 从 Legendre 管线中独立出来

### FFT 测试

* [ ] `cos(mφ)` 只有频率 `m` 非零
* [ ] `sin(mφ)` 只有频率 `m` 非零
* [ ] 常数场只有 `m=0`
* [ ] `ifft(fft(x)) ≈ x`
* [ ] GPU/CPU FFT 逐频对照
* [ ] 实场重建后虚部误差在阈值内

---

## 5. Phase D：Legendre 变换

### `shaders/legendreAnalysis.wgsl`

* [ ] 输入 `F(j,m)`
* [ ] 使用 `w_j`、`P_lm(j,l,m)`
* [ ] 输出 `a(m,l)`

### `shaders/legendreSynthesis.wgsl`

* [ ] 输入 `a(m,l)`
* [ ] 输出 `freq(j,m)`

### `shaders/legendreSynthesisDTheta.wgsl`

* [ ] 输入 `a(m,l)`
* [ ] 使用 `dP_lm_dtheta`
* [ ] 输出 `dthetaFreq(j,m)`

### Legendre 测试

* [ ] 单模 round-trip：`analysis(synthesis(a)) ≈ a`
* [ ] `m=0` 轴对称模正确
* [ ] `l<m` 始终为零
* [ ] `dtheta` 与 CPU 参考一致
* [ ] 多模叠加线性性成立
* [ ] 极区不出现 NaN/Inf

---

## 6. Phase E：谱算子

### `shaders/mulIM.wgsl`

* [ ] 实现 `i * m * a(m,l)`

### `shaders/applyLaplacian.wgsl`

* [ ] 实现 `lapEigs * a`

### `shaders/invertLaplacian.wgsl`

* [ ] 实现 `a / (-l(l+1))`
* [ ] `l=0` 强制为零

### `shaders/filterSpectrum.wgsl`

* [ ] 实现按 `l` 乘 `specFilter`

### 谱算子测试

* [ ] `m=0` 经 `mulIM` 后全零
* [ ] `applyLaplacian` 精确乘以 `-l(l+1)`
* [ ] `invertLaplacian(applyLaplacian(a)) ≈ a`，忽略 `l=0`
* [ ] `filterSpectrum` 不改 `l=0`
* [ ] 高频模被压低
* [ ] GPU/CPU 一致

---

## 7. Phase F：初始化

### `shaders/initRandom.wgsl`

* [ ] 生成随机网格场
* [ ] 调用 analysis
* [ ] 乘 `initSlope`
* [ ] 乘 `specFilter`
* [ ] 把 `(0,0)` 置零
* [ ] 乘 `amplitude`

### 初始化测试

* [ ] 相同 `seed` 可复现
* [ ] 不同 `seed` 不同
* [ ] `(0,0) == 0`
* [ ] 低阶谱壳斜率接近 `l^(-1/3)`
* [ ] 高频已被滤波抑制

---

## 8. Phase G：RHS 主链

demo 的 RHS 流程是：

1. `psi_lm = invert_laplacian(zeta_lm)`
2. `u_theta, u_phi = velocity_from_streamfunction(psi_lm)`
3. `dzeta_dtheta = dtheta(zeta_lm)`
4. `dzeta_dphi = dphi(zeta_lm) / sinTheta`
5. `adv = u_theta * dzeta_dtheta + u_phi * dzeta_dphi`
6. `adv_lm = filter(analysis(adv))`
7. `diff_lm = nu * apply_laplacian(zeta_lm)`
8. `rhs = -adv_lm + diff_lm` 

### `shaders/velocityFromPsi.wgsl`

* [ ] 输入 `dpsiDphiGrid`
* [ ] 输入 `dpsiDthetaGrid`
* [ ] 输出 `uThetaGrid = dpsiDphiGrid / sinTheta`
* [ ] 输出 `uPhiGrid = -dpsiDthetaGrid`

### `shaders/advectGrid.wgsl`

* [ ] 输入 `uThetaGrid`
* [ ] 输入 `uPhiGrid`
* [ ] 输入 `dzetaDthetaGrid`
* [ ] 输入 `dzetaDphiGrid`
* [ ] 输出 `advGrid`

### `shaders/rhsCompose.wgsl`

* [ ] 输入 `advLM`
* [ ] 输入 `zetaLM`
* [ ] 输出 `rhsLM = -advLM + nu * laplacian(zetaLM)`

### RHS 测试

* [ ] 零场输入得到零 RHS
* [ ] 纯扩散下单模 RHS 正确
* [ ] 速度符号与 demo 一致
* [ ] `1/sinTheta` 修正后极区稳定
* [ ] `advGrid` 与 CPU 参考一致
* [ ] `rhsLM` 与 CPU 参考一致

---

## 9. Phase H：RK4

### `shaders/rk4Stage.wgsl`

* [ ] 生成 `z + coeff * dt * k`

### `shaders/rk4Combine.wgsl`

* [ ] 实现
  [
  z_{n+1}=z_n+\frac{dt}{6}(k_1+2k_2+2k_3+k_4)
  ]
* [ ] 末尾调用 `filterSpectrum`

### `src/app/loop.ts`

* [ ] 每帧执行 `stepsPerFrame` 次 RK4
* [ ] 暂停时不推进
* [ ] 推进后更新显示与诊断

### RK4 测试

* [ ] `dt=0` 时状态不变
* [ ] 只开扩散时单模衰减正确
* [ ] `dt` 减半后全局误差呈 4 阶收敛趋势
* [ ] GPU/CPU 单步结果一致
* [ ] 1000 步无 NaN/Inf

---

## 10. Phase I：诊断

### `src/solver/diagnostics.ts`

* [ ] 合成 `zetaGrid`
* [ ] 合成 `psiGrid`
* [ ] 管理能量历史数组
* [ ] 管理 snapshot 采样周期

### `shaders/energyIntegrand.wgsl`

* [ ] 计算局部项 `-0.5 * psi * zeta`
* [ ] 乘 `w_j`
* [ ] 预留 `2π/nlon` 因子

### `shaders/reduceSum.wgsl`

* [ ] 分阶段归约
* [ ] 输出总能量

### 诊断测试

* [ ] kinetic energy 非负
* [ ] 零场能量为零
* [ ] GPU/CPU 能量一致
* [ ] 历史长度正确
* [ ] reset 后历史清空

---

## 11. Phase J：three.js 可视化

### `render/sphereView.ts`

* [ ] 创建球体 mesh
* [ ] 建立 `(phi, theta)` 到 UV 的映射
* [ ] 把 `zetaGrid` 上传为纹理
* [ ] 支持每帧刷新纹理

### `shaders/sphere.vert.wgsl`

* [ ] 传位置
* [ ] 传法线
* [ ] 传 UV

### `shaders/scalarColor.frag.wgsl`

* [ ] 从纹理采样标量场
* [ ] 做发散色图映射
* [ ] 暴露 `displayScale`

### 可视化测试

* [ ] 经向 seam 闭合
* [ ] 两极不翻转
* [ ] 北纬/南纬方向正确
* [ ] 零值映射到中性色
* [ ] 正负值颜色对称
* [ ] 改相机不影响数据

---

## 12. Phase K：lil-gui

### `app/gui.ts`

* [ ] `lmax`
* [ ] `dt`
* [ ] `nu`
* [ ] `filterAlpha`
* [ ] `filterOrder`
* [ ] `seed`
* [ ] `amplitude`
* [ ] `stepsPerFrame`
* [ ] `displayScale`
* [ ] `pause`
* [ ] `reset`
* [ ] `showEnergy`

### GUI 测试

* [ ] 参数变更立即生效
* [ ] `pause` 后状态冻结
* [ ] `reset(seed)` 可复现
* [ ] 改 `lmax` 后完整重建全部资源
* [ ] 重建后能继续稳定推进

---

## 13. Shader 级逐项验收单

### 变换类

* [ ] `fftForwardLon.wgsl`
* [ ] `fftInverseLon.wgsl`
* [ ] `legendreAnalysis.wgsl`
* [ ] `legendreSynthesis.wgsl`
* [ ] `legendreSynthesisDTheta.wgsl`

### 谱算子类

* [ ] `mulIM.wgsl`
* [ ] `applyLaplacian.wgsl`
* [ ] `invertLaplacian.wgsl`
* [ ] `filterSpectrum.wgsl`

### PDE 类

* [ ] `initRandom.wgsl`
* [ ] `velocityFromPsi.wgsl`
* [ ] `advectGrid.wgsl`
* [ ] `rhsCompose.wgsl`
* [ ] `rk4Stage.wgsl`
* [ ] `rk4Combine.wgsl`

### 诊断与显示类

* [ ] `energyIntegrand.wgsl`
* [ ] `reduceSum.wgsl`
* [ ] `sphere.vert.wgsl`
* [ ] `scalarColor.frag.wgsl`

每个 shader 必须有：

* [ ] 输入维度测试
* [ ] 输出维度测试
* [ ] 零输入测试
* [ ] 单模测试
* [ ] GPU/CPU 对照测试
* [ ] 长时间运行稳定性测试

---

## 14. 最终验收

* [ ] 首个稳定分辨率为 `lmax=31`
* [ ] 目标演示分辨率为 `lmax=63`
* [ ] 能跑出实时球面涡度动画
* [ ] 能显示 kinetic energy 曲线
* [ ] 参数可由 lil-gui 调整
* [ ] 结果没有极区爆值
* [ ] 没有 seam 断裂
* [ ] 没有 NaN/Inf
* [ ] 数值路径没有偏离 demo

demo 确实建议从 `31 / 63` 起步，主程序默认示例也是 `lmax=63, nlat=lmax+1, nlon=2*(lmax+1), dt=1e-2, nu=1e-7`。