# 基于 compute shader + three.js + lil-gui 的二维球面 PDE 实时求解与可视化开发文档

## 1. 目标与范围

### 1.1 目标

实现一个浏览器端实时系统，用 **compute shader** 完成数值求解，用 **three.js** 完成球面可视化，用 **lil-gui** 完成参数控制。首个版本只实现 demo 对应 PDE：

[
\partial_t \zeta + J(\psi,\zeta) = \nu \nabla^2 \zeta,\qquad \zeta = \nabla^2 \psi
]

其中球面为单位球，状态变量为涡度 (\zeta)，流函数为 (\psi)。

### 1.2 数值路线约束

必须与 demo 保持一致：

* 纬向：**Gauss-Legendre** 节点与求积权重
* 经向：**FFT**
* 球谐变换：**显式构造 (P_l^m) 与 (dP_l^m/d\theta)**
* 系数存储：仅存 **(m \ge 0)** 的复系数
* 速度定义：
  [
  u_\theta=\frac{1}{\sin\theta}\partial_\phi \psi,\qquad
  u_\phi=-\partial_\theta \psi
  ]
* 非线性项：网格上计算平流，回到谱空间后加扩散
* 时间推进：**RK4**
* 稳定化：**指数谱滤波**
* 初始化：随机场分析到谱空间后乘以 (l^{-1/3}) 斜率并滤波，且置零 (l=0) 模
* 诊断：至少包含**动能曲线**与**球面涡度可视化**。

### 1.3 明确不做

* 不改成有限差分/有限体积
* 不改成普通经纬均匀网格主求解
* 不引入外部球谐库路线替代当前显式变换
* 不把主求解放回 CPU，只允许 CPU 做初始化/预计算/调度

---

## 2. 总体架构

## 2.1 运行时分层

1. **预计算层（CPU）**
   生成 `mu, w, theta, sinTheta, phi, P_lm, dP_lm_dtheta, lapEigs, specFilter, initSlope`。

2. **求解层（GPU compute shader）**
   完成：

   * 经向 FFT / iFFT
   * Legendre analysis / synthesis
   * 谱算子
   * RHS
   * RK4
   * 能量归约

3. **渲染层（three.js）**

   * 球体 mesh
   * 标量场贴图更新
   * 颜色映射
   * 动画循环

4. **交互层（lil-gui）**

   * `lmax / dt / nu / filterAlpha / filterOrder / seed / amplitude / stepsPerFrame / pause / reset`

## 2.2 最小目录

```text
src/
  app/
    main.ts
    loop.ts
    gui.ts
  solver/
    config.ts
    precompute.ts
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
    gpu/
    visual/
```

---

## 3. 数据布局

## 3.1 分辨率策略

严格沿用 demo 的默认关系：

* `nlat = lmax + 1`
* `nlon = 2 * (lmax + 1)`

建议路线：

* **P0**：`lmax = 31`
* **P1**：`lmax = 63`

这是最符合 demo 的启动分辨率。

## 3.2 缓冲区与纹理

### 谱空间

* `zetaLM_A`, `zetaLM_B`：当前/下一步涡度谱系数
* `psiLM`
* `k1`, `k2`, `k3`, `k4`
* `tmpLM`

### 网格空间

* `zetaGrid`
* `psiGrid`
* `dpsiDphiGrid`
* `dpsiDthetaGrid`
* `uThetaGrid`
* `uPhiGrid`
* `dzetaDphiGrid`
* `dzetaDthetaGrid`
* `advGrid`

### 常量查表

* `mu`
* `w`
* `theta`
* `sinTheta`
* `P_lm`
* `dP_lm_dtheta`
* `lapEigs`
* `specFilter`
* `initSlope`

## 3.3 复数存储

GPU 上复数统一打包为 `vec2<f32>`：

* `.x = real`
* `.y = imag`

索引统一采用：

* 光谱索引：`(m, l)`，只使用 `l >= m`
* 网格索引：`(j, k)`，对应 `(latIndex, lonIndex)`

---

## 4. 数值流程

## 4.1 初始化

1. CPU 生成高斯-勒让德节点与权重
2. CPU 生成 `P_lm`、`dP_lm_dtheta`
3. GPU 生成随机网格场
4. GPU 做 `analysis`
5. GPU 乘 `initSlope`
6. GPU 乘 `specFilter`
7. GPU 强制 `zetaLM[0,0] = 0`

## 4.2 单步 RHS

给定 `zeta_lm`：

1. `psi_lm = invertLaplacian(zeta_lm)`
2. `dpsi/dphi = synthesis(i m psi_lm)`
3. `dpsi/dtheta = synthesis_dtheta(psi_lm)`
4. `u_theta = (1/sinθ) * dpsi/dphi`
5. `u_phi = -dpsi/dtheta`
6. `dzeta/dtheta = synthesis_dtheta(zeta_lm)`
7. `dzeta/dphi = synthesis(i m zeta_lm) / sinθ`
8. `adv = u_theta * dzeta/dtheta + u_phi * dzeta/dphi`
9. `adv_lm = analysis(adv)`
10. `adv_lm = filter(adv_lm)`
11. `diff_lm = nu * applyLaplacian(zeta_lm)`
12. `rhs = -adv_lm + diff_lm`

以上完全对应 demo 的 `streamfunction_from_vorticity / velocity_from_streamfunction / rhs`。

## 4.3 时间推进

严格使用 RK4：

* `k1 = rhs(z)`
* `k2 = rhs(z + 0.5 dt k1)`
* `k3 = rhs(z + 0.5 dt k2)`
* `k4 = rhs(z + dt k3)`
* `z_next = filter(z + dt/6 * (k1 + 2k2 + 2k3 + k4))`

与 demo 一致。

## 4.4 诊断与显示

* 显示字段：`zetaGrid = synthesis(zetaLM)`
* 能量：
  [
  E = -\frac12 \int \psi \zeta , d\Omega
  ]
  用高斯-勒让德权重加经向均匀求和近似。

---

## 5. 模块设计与模块测试

## 5.1 `precompute.ts`

职责：

* 生成 `mu, w, theta, sinTheta`
* 生成 `P_lm`
* 生成 `dP_lm_dtheta`
* 生成 `lapEigs`
* 生成 `specFilter`
* 生成 `initSlope`

测试：

1. `mu` 关于 0 对称，`w` 关于中轴对称。
2. `sum(w)` 误差接近 2。
3. `sinTheta = sqrt(max(1-mu^2, eps))`，极区无 NaN。
4. `lapEigs[m,l] == -l(l+1)`。
5. `specFilter[0] == 1`，且随 `l` 非增。
6. `initSlope[l] ~ l^(-1/3)`，`l=0` 单独处理。
7. 用 CPU 参考脚本抽查 20 个 `(l,m,j)` 点，`P_lm` 和 `dP_lm_dtheta` 相对误差小于阈值。

## 5.2 `buffers.ts`

职责：

* 创建所有 storage buffer / texture
* 管理 ping-pong
* 管理谱空间和网格空间布局

测试：

1. 复数打包/解包往返一致。
2. `(m,l)` 映射到线性地址后无越界。
3. `l < m` 区域默认清零。
4. grid buffer 行主序与渲染采样坐标一致。
5. reset 后所有 tmp buffer 为零或指定初值。

## 5.3 `fft` 子模块

职责：

* 经向 forward FFT
* 经向 inverse FFT

测试：

1. 单模 `cos(mφ)`、`sin(mφ)` 的 forward FFT 只在对应频率有峰值。
2. `ifft(fft(x)) ≈ x`。
3. 常数场仅 `m=0` 分量非零。
4. 实场输入满足 Hermitian 对称恢复正确。
5. 随机场往返误差随 `nlon` 增大不恶化。

## 5.4 `legendre` 子模块

职责：

* analysis：`grid -> lm`
* synthesis：`lm -> grid`
* synthesis_dtheta：`lm -> d/dtheta grid`

测试：

1. 单个球谐模 `Y_l^m` 经 synthesis 后再 analysis，主模恢复正确。
2. 轴对称模 `m=0` 不依赖经向 FFT 高频项。
3. `analysis(synthesis(a)) ≈ a`，仅比较 `l>=m` 区域。
4. `dtheta` 对已知低阶模式与 CPU 参考一致。
5. 积分使用 `w_j * (2π/nlon)` 的权重后，低阶模投影误差达标。

## 5.5 `spectralOperators.ts`

职责：

* `mulIM`
* `applyLaplacian`
* `invertLaplacian`
* `filterSpectrum`

测试：

1. `mulIM` 对 `m=0` 输出零。
2. `invertLaplacian(applyLaplacian(a)) ≈ a`，忽略 `l=0`。
3. `applyLaplacian` 后每个模恰乘 `-l(l+1)`。
4. `filterSpectrum` 不改变 `l=0`。
5. 高频模经滤波后幅值严格下降。

## 5.6 `rhs.ts`

职责：

* 从 `zetaLM` 计算完整 RHS

测试：

1. 零场输入得到零 RHS。
2. 单一低阶模在只开扩散、关非线性时，RHS 与 `nu * Δζ` 一致。
3. `uTheta/uPhi` 的符号与 demo 一致。
4. `dzetaDphi` 的 `1/sinθ` 修正后极区无爆值。
5. `advGrid` 与 CPU 参考版逐点比较。

## 5.7 `rk4.ts`

职责：

* 4 次 RHS 调用
* 中间态合成
* 最终滤波

测试：

1. `dt=0` 时状态不变。
2. 只开扩散时，对单模振幅衰减与解析解近似一致。
3. `dt` 减半后，全局误差按 4 阶趋势下降。
4. 关闭随机初始化，用固定谱场时，GPU/CPU 单步结果一致。
5. 连续 100 步后无 NaN/Inf。

## 5.8 `diagnostics.ts`

职责：

* 合成 `zetaGrid`
* 动能计算
* 历史曲线缓存

测试：

1. `kineticEnergy >= 0`。
2. 对固定谱场，动能与 CPU 参考误差达标。
3. `history` 长度与采样周期匹配。
4. reset 后历史清空。

## 5.9 `sphereView.ts`

职责：

* 球体渲染
* UV 与 `(lon, lat)` 对应
* 标量场贴图刷新

测试：

1. 经向 seam 连续。
2. 北极/南极方向与 `theta` 定义一致。
3. 颜色映射零点位于中性色。
4. 改变相机后不影响数值缓冲。

## 5.10 `gui.ts`

职责：

* 参数绑定
* pause/resume
* reset/reseed
* stepsPerFrame

测试：

1. 改 `nu/dt/filterAlpha/filterOrder` 后下一帧生效。
2. `pause` 后状态冻结。
3. `reset(seed)` 结果可复现。
4. 改 `lmax` 时完整重建所有 buffer 与查表。

---

## 6. Shader 设计与逐个测试

下面的 shader 列表就是首版必须实现的最小集合。

## 6.1 `initRandom.wgsl`

输入：`seed, amplitude, initSlope, specFilter`
输出：`zetaLM`

职责：

* 生成随机网格场
* 做 analysis
* 乘 `initSlope`
* 乘 `specFilter`
* 清零 `(0,0)`

测试：

1. 相同 seed 结果完全一致。
2. 不同 seed 结果不同。
3. `zetaLM[0,0] == 0`。
4. 壳平均谱斜率接近 `l^(-1/3)`。
5. 高频端已被滤波压低。

## 6.2 `fftForwardLon.wgsl`

输入：`grid(j,k)`
输出：`F_m(mu_j)`

测试：

1. 输入 `cos(mφ)`，只有对应 `m` 非零。
2. 输入常数场，仅 `m=0` 非零。
3. 与 CPU FFT 逐频比较。

## 6.3 `fftInverseLon.wgsl`

输入：`freq(j,m)`
输出：`grid(j,k)`

测试：

1. `ifft(fft(x)) ≈ x`。
2. 纯实 Hermitian 频谱恢复为实场。
3. 与 CPU iFFT 比较最大误差。

## 6.4 `legendreAnalysis.wgsl`

输入：`F_m(mu_j), w_j, P_lm`
输出：`a[m,l]`

测试：

1. 低阶单模投影到正确 `(m,l)`。
2. 与 CPU analysis 比较。
3. `l<m` 区域始终为零。
4. 随机场分析后主能量分布合理。

## 6.5 `legendreSynthesis.wgsl`

输入：`a[m,l], P_lm`
输出：`freq(j,m)`，再接 `fftInverseLon`

测试：

1. 单一 `a[l,m]` 恢复正确空间形态。
2. 与 CPU synthesis 对比。
3. 多模叠加线性性成立。

## 6.6 `legendreSynthesisDTheta.wgsl`

输入：`a[m,l], dP_lm_dtheta`
输出：`∂θ field`

测试：

1. 对已知低阶模式与 CPU `dtheta` 对齐。
2. 零输入零输出。
3. 极区无 NaN/Inf。

## 6.7 `mulIM.wgsl`

输入：`a[m,l]`
输出：`i m a[m,l]`

测试：

1. `m=0` 全零。
2. 实部/虚部旋转正确。
3. 与 CPU 复乘一致。

## 6.8 `applyLaplacian.wgsl`

输入：`a[m,l], lapEigs`
输出：`Δa`

测试：

1. 每个模都只乘 `-l(l+1)`。
2. 与 CPU 对比。
3. 线性性成立。

## 6.9 `invertLaplacian.wgsl`

输入：`a[m,l], lapEigs`
输出：`Δ^{-1}a`

测试：

1. `l=0` 输出强制为零。
2. `invert(apply(a)) ≈ a`，忽略 `l=0`。
3. 与 CPU 对比。

## 6.10 `filterSpectrum.wgsl`

输入：`a[m,l], specFilter`
输出：`filtered a`

测试：

1. `l=0` 不变。
2. 高频幅值下降。
3. 多次调用幂等趋势符合预期。

## 6.11 `velocityFromPsi.wgsl`

输入：`dpsiDphiGrid, dpsiDthetaGrid, sinTheta`
输出：`uThetaGrid, uPhiGrid`

测试：

1. 符号与 demo 约定一致。
2. `uTheta = dpsiDphi / sinTheta`。
3. `uPhi = -dpsiDtheta`。
4. 极区数值稳定。

## 6.12 `advectGrid.wgsl`

输入：`uTheta, uPhi, dzetaDtheta, dzetaDphi`
输出：`advGrid`

测试：

1. 零速度场输出零。
2. 零梯度场输出零。
3. 与 CPU 平流项逐点比较。
4. 极区无尖峰。

## 6.13 `rhsCompose.wgsl`

输入：`advLM, zetaLM, nu`
输出：`rhsLM`

测试：

1. `rhs = -advLM + nu*lap(zetaLM)`。
2. 关闭 `nu` 后无扩散项。
3. 关闭平流后只剩扩散项。

## 6.14 `rk4Stage.wgsl`

输入：`zetaLM, k, dt, coeff`
输出：`zetaTemp`

测试：

1. `coeff=0.5`、`1.0` 分别生成正确中间态。
2. `dt=0` 不变。
3. 与 CPU 中间态一致。

## 6.15 `rk4Combine.wgsl`

输入：`zetaLM, k1..k4, dt`
输出：`zetaNext`

测试：

1. 系数是 `1,2,2,1`。
2. 结果再经过滤波后与 CPU 一致。
3. 单步后无非法值。

## 6.16 `energyIntegrand.wgsl`

输入：`psiGrid, zetaGrid, w_j`
输出：`localEnergyTerms`

测试：

1. 点值满足 `-0.5 * psi * zeta * weight`。
2. 与 CPU 局部积分项一致。
3. 常零场总和为零。

## 6.17 `reduceSum.wgsl`

输入：`localEnergyTerms`
输出：`energy`

测试：

1. 小数组手算可验证。
2. 随机数组与 CPU sum 比较。
3. 多阶段 reduction 次序不影响结果误差级。

## 6.18 `sphere.vert.wgsl`

职责：

* 传递球体顶点位置、法线、UV

测试：

1. 球体方向与经纬映射一致。
2. UV seam 在经度 0/2π 处闭合。
3. 旋转相机不改变 UV 对应关系。

## 6.19 `scalarColor.frag.wgsl`

职责：

* 从标量场贴图采样
* 做发散型 colormap 映射

测试：

1. 正负值颜色对称。
2. 零值颜色固定。
3. 颜色范围裁剪正确。
4. 改变 `displayScale` 后动态范围正确。

---

## 7. three.js 与 lil-gui 接入要求

## 7.1 three.js

* 使用 `WebGPURenderer`
* 主循环中顺序：

  1. 若未暂停，执行 `stepsPerFrame` 次 RK4
  2. 合成 `zetaGrid`
  3. 更新显示纹理
  4. 渲染球体
  5. 每 `diagEvery` 帧做一次能量归约

## 7.2 lil-gui 参数

必须有：

* `lmax`
* `dt`
* `nu`
* `filterAlpha`
* `filterOrder`
* `seed`
* `amplitude`
* `stepsPerFrame`
* `pause`
* `reset`
* `displayScale`
* `showEnergy`

测试：

1. 所有参数改动都能在正确阶段生效。
2. 改 `lmax` 会完整重建预计算与 GPU 资源。
3. `reset` 后能量曲线从头开始。

---

## 8. 详细路线图

## Phase 0：CPU 参考基线

交付：

* 从 demo 提炼出浏览器版 CPU 参考实现
* 固定数据布局和索引规则

退出标准：

* CPU 版 `analysis/synthesis/dtheta/apply/invert/rhs/rk4/energy` 全可运行
* 能复现 demo 的能量曲线趋势与末态涡度图。

## Phase 1：预计算与资源初始化

交付：

* `precompute.ts`
* `buffers.ts`

退出标准：

* 所有查表数据可生成并上传 GPU
* 单元测试全部通过

## Phase 2：经向 FFT 管线

交付：

* `fftForwardLon.wgsl`
* `fftInverseLon.wgsl`

退出标准：

* 随机网格场 FFT/iFFT 往返误差达标
* 模态纯度测试通过

## Phase 3：Legendre 变换管线

交付：

* `legendreAnalysis.wgsl`
* `legendreSynthesis.wgsl`
* `legendreSynthesisDTheta.wgsl`

退出标准：

* 单模 round-trip 通过
* 与 CPU 参考逐项误差达标

## Phase 4：谱算子与初始化

交付：

* `mulIM.wgsl`
* `applyLaplacian.wgsl`
* `invertLaplacian.wgsl`
* `filterSpectrum.wgsl`
* `initRandom.wgsl`

退出标准：

* 所有单算子测试通过
* 初始化谱形状与 demo 一致

## Phase 5：RHS 管线

交付：

* `velocityFromPsi.wgsl`
* `advectGrid.wgsl`
* `rhsCompose.wgsl`

退出标准：

* 零场测试、扩散单模测试、GPU/CPU 对照测试通过

## Phase 6：RK4 与实时循环

交付：

* `rk4Stage.wgsl`
* `rk4Combine.wgsl`
* 主循环调度

退出标准：

* `stepsPerFrame >= 1` 时稳定运行
* 长时间运行无 NaN/Inf
* `dt` 收敛趋势正确

## Phase 7：诊断与可视化

交付：

* `energyIntegrand.wgsl`
* `reduceSum.wgsl`
* `sphere.vert.wgsl`
* `scalarColor.frag.wgsl`

退出标准：

* 球面显示正确
* 动能曲线刷新正确
* 可与 CPU 参考对照

## Phase 8：GUI 与验收

交付：

* `gui.ts`
* 完整 demo 页面

退出标准：

* 所有参数可交互
* reset/pause/reseed 稳定
* 完整测试清单通过

---

## 9. 最终验收标准

项目只有在以下条件都满足时才算完成：

1. **数值路线未偏离 demo**
2. GPU 版 `analysis / synthesis / dtheta / dphi / laplacian / inverse / rhs / rk4 / energy` 全部可用
3. 每个模块都有自动化测试
4. 每个 shader 都有输入输出级测试
5. GPU 与 CPU 参考在低分辨率下逐项可对照
6. 实时渲染、GUI 控制、能量曲线、末态涡度球面图全部可用
7. 长时间运行不出现 NaN、Inf、纹理破裂、极区异常、经向接缝断裂

---

## 10. 开发顺序建议

严格按这个顺序做，不要跳：

`CPU参考 -> 预计算 -> FFT -> Legendre -> 谱算子 -> 初始化 -> RHS -> RK4 -> 诊断 -> 渲染 -> GUI`

这样能保证每一步都能和 demo 做局部对照，最快定位误差来源。

如果你要，我下一条可以直接把这份文档继续落成**“任务清单版 TODO markdown”**，方便你按文件逐个开工。
