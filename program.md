# 球面谱方法 Navier-Stokes 方程纯 WebGL (GPGPU) 可视化模拟器开发文档

## 1. 项目概述

本项目旨在开发 Web 端下的球面谱方法 2D 湍流模拟器，实现一个无需安装任何后端环境、基于 GPGPU 技术的纯浏览器运行的高性能交互式流体动力学模拟器。

### 1.1 技术栈选型

- **核心计算与渲染引擎 (Compute & Render Engine):** 纯 WebGL 2.0 (推荐结合 Three.js / TWGL)。
  - **方案:** 摒弃传统的 CPU (WASM) 计算。利用 WebGL 2.0 的 **多渲染目标 (MRT)**、**浮点纹理 (OES_texture_float)** 和 **帧缓冲对象 (FBO)** 技术。流体状态（涡量、速度场、谱系数）全部以浮点纹理的形式存储在 VRAM 中。通过“乒乓渲染”（Ping-Pong FBO）交替更新纹理来实现 RK4 时间步进。
- **UI 与交互:** 使用 React 进行应用级别的状态管理和组件化开发，配合 lil-gui 构建悬浮控制面板进行核心参数的实时调节。

### 1.2 开发原则
- **显式错误可见性原则**：严禁静默失败。针对 WebGL 极易出现的 Shader 编译失败、Framebuffer 状态不完整等错误，必须通过 `gl.getShaderInfoLog` 捕获并抛出 UI 级别的致命异常。
- **快速失败**：如果用户的设备不支持 WebGL 2.0 或必须的浮点纹理扩展（`EXT_color_buffer_float`），程序应在初始化阶段立即拦截并阻断运行。
- **配置完整性原则**：所有外部配置必须通过 Schema 校验。
- **流水线隔离原则**：GPGPU 的每个 Shader Pass（如：非线性项计算、SHT 变换、RK4 步进）必须是可独立测试的。
- **状态快照原则**：由于 GPU 调试困难，必须实现基于 `gl.readPixels` 的显存状态导出功能，以便在特定帧将 GPU 纹理数据读回 CPU 进行精度校验。
- **独立测试原则**: 每个函数（比如球面谐波变换，梯度计算，拉普拉斯算子计算，各种非线性项计算等）在实现后都要编写完整的测试用例并与CPU计算结果比较。

## 2. 系统架构设计

### 2.1 模块划分 (GPGPU 管线)

在纯 WebGL 架构下，计算和渲染被统一为一系列的 Shader Passes：

- **初始化阶段 (Initialization Pass):**
  - 使用 Shader 结合特定算法（或 CPU 预生成一次后上传）生成初始涡量场的网格数据纹理。
  - 将拉普拉斯算子特征值、逆算子等预计算常量生成 1D 或 2D DataTexture 上传至 GPU。
- **球面谐波变换管线 (SHT Pipeline):**
  - **Grid to Spectral (Forward SHT):** 通过 Fragment Shader 执行离散网格到谱系数的积分变换。这通常需要将勒让德多项式矩阵编码为纹理，执行纹理间的矩阵乘法。
  - **Spectral to Grid (Inverse SHT):** 从谱空间的系数 $\zeta_{lm}$ 变换回物理空间的网格数据。
- **时间积分管线 (RK4 Ping-Pong Pipeline):**
  - 创建两组 FBO（Read FBO 和 Write FBO）。
  - 对于单步 RK4，需要分配额外的浮点纹理暂存 $k_1, k_2, k_3, k_4$ 的中间态。
  - 每执行一步积分，交换 Read/Write 的绑定状态（Ping-Ponging）。
- **渲染管线 (Render Pipeline):**
  - **零拷贝渲染:** 直接将当前计算周期的物理空间数据纹理（Grid Texture）作为材质，映射到 Three.js 的 SphereGeometry 上，应用 Colormap Fragment Shader 直接着色。

### 2.2 伪谱法数学模型在 GPU 上的映射

- **拉普拉斯与逆算子:** $\nabla^2$ 和 $\nabla^{-2}$ 在谱空间中仅是对角矩阵的乘法。在 WebGL 中，表现为对谱系数纹理（Spectral Texture）与预计算特征值纹理（Eigenvalue Texture）的逐像素（Per-pixel）乘法。
- **非线性项:** $N(\zeta) = -\mathbf{u} \cdot \nabla \zeta$。必须在物理网格空间（Grid Space）计算。管线流转为：在 GPU 谱空间求导 $\rightarrow$ 逆变换到网格空间得到 $u, v$ 和 $\nabla \zeta$ $\rightarrow$ 在网格空间执行逐像素乘法 $\rightarrow$ 正变换回谱空间应用高斯截断（De-aliasing）。

## 3. 详细测试计划

GPGPU 开发中最大的痛点是“黑盒化”。测试必须通过 `gl.readPixels` 将 GPU 计算结果读回 CPU，与标准数学库的结果进行比对。

### 3.1 Shader 单元测试计划 (Shader Unit Testing)

| 测试用例 ID | 测试模块 | 测试内容描述 | 预期结果 (Pass Condition) |
| :--- | :--- | :--- | :--- |
| UT-01 | WebGL 浮点精度支持 | 验证当前设备和浏览器是否完全支持 32 位浮点纹理读写及线性过滤。 | 写入特定的浮点值（如 $1.234567 \times 10^{-5}$），读出后误差 $< 10^{-7}$。 |
| UT-02 | GPGPU 矩阵乘法 (SHT 基础) | 在 Shader 中实现纹理矩阵乘法，验证 $C = A \times B$。 | `gl.readPixels` 读回的计算结果与 CPU 端纯数学库计算的矩阵乘法结果一致。 |
| UT-03 | SHT 正逆变换 Shader | 给定解析分布 $\zeta = \cos(\theta)$，在 GPU 管线中走完 Grid -> Spectral -> Grid 流程。 | 读回的 Grid 纹理数据误差 $L_2$ 范数在 GPU 单精/双精阈值内。 |
| UT-04 | GPU RK4 步进器 | 关闭非线性项，仅保留线性耗散项。给入初始纹理，在 FBO 中循环 Ping-Pong $N$ 次。 | 提取涡量纹理，其衰减率严格匹配解析解 $e^{-\nu l(l+1) t}$。 |

*(物理特性测试 PT-01~04 与 数值收敛性测试 CT-01~03 与原文档保持一致，仅验证手段从内存读取变为 `gl.readPixels` 提取)*

### 3.2 物理特性测试计划 (Physical Testing Plan)

物理测试用于验证模拟器是否遵循流体动力学规律，这是计算流体动力学（CFD）软件不可或缺的环节。

| 测试用例 ID | 物理测试项 | 初始条件与参数设置 | 验证标准与物理意义 |
| :--- | :--- | :--- | :--- |
| PT-01 | 动能守恒测试 (Inviscid Limit) | 设粘性系数 $\nu = 0$（或极小）。关闭高斯滤波。给入随机初始涡量。 | 标准: 系统总动能 $E = \sum \frac{1}{2} |\mathbf{u}|^2$（或谱空间对应项）应保持常数。 |
| PT-02 | 拟能衰减测试 (Enstrophy Cascade) | 设适中的粘性 $\nu$，开启高斯滤波。给入原型代码中的 $k^{-1/3}$ 随机能谱。 | 标准: 监测总动能 $E$ 和总拟能 $Z = \int \zeta^2 dA$。在 2D 球面湍流中，应观察到典型的“能量逆级联”和“拟能正级联”，拟能 $Z$ 显著下降，但动能 $E$ 下降极其缓慢。 |
| PT-03 | 固体自转测试 (Solid Body Rotation) | 初始流函数设为 $\psi = -\omega \cos(\theta)$，代表流体围绕极轴以角速度 $\omega$ 刚体旋转。 | 标准: 随着时间推移，流场应保持绝对静止不变。任何涡量的产生或形态改变都表明计算格式存在非物理的数值误差。 |
| PT-04 | Rossby-Haurwitz 波测试 | 使用气象学中经典的 RH-Wave 初始条件（波数 $m=4$）。 | 标准: 涡量场应保持其初始的波浪形状，并在球面上以恒定的相速度向西平移，波形在几百个时间步内不应发生明显的畸变或破碎。 |

### 3.3 数值方法收敛性测试计划 (Numerical Convergence Testing Plan)

谱方法以其极高的精度著称，为验证我们在浏览器端实现的求解器依然保持了原有的数学严谨性，需执行以下收敛性（Convergence）测试。

| 测试用例 ID | 测试项 | 测试步骤与方法 | 验证标准 (Pass Condition) |
| :--- | :--- | :--- | :--- |
| CT-01 | 空间谱收敛性测试 (Spatial Spectral Convergence) | 给定一个足够平滑的初始场（例如低阶的 Rossby-Haurwitz 波）。固定一个极小的时间步长 $dt$，分别使用 $L_{max} = 16, 32, 64, 128$ 运行固定的物理时间 $T$。以 $L_{max}=256$ 的结果作为“精确解”，计算低分辨率下的 $L_2$ 误差。 | 标准: 绘制 $\log(Error)$ 与 $L_{max}$ 的关系图，应呈现出明显的线性下降趋势（即指数收敛），直到误差降至机器精度（约 $10^{-12}$）后趋于平缓。 |
| CT-02 | 频谱尾部衰减测试 (Energy Spectrum Tail) | 运行一段完全发展的湍流（如 PT-02），提取某一时刻的球面谐波系数 $a_{lm}$。 | 标准: 计算不同阶数 $l$ 的能量谱 $E(l)$。在耗散区（大 $l$ 值区域），能量谱应呈现指数级衰减，证明高频振荡被正确处理，没有发生数值混叠积聚（Aliasing pile-up）。 |
| CT-03 | 时间推进四阶收敛性 (Temporal RK4 Convergence) | 固定一个较高的分辨率（如 $L_{max}=128$ 以排除空间截断误差）。选用平滑流场，分别使用时间步长 $\Delta t, \Delta t/2, \Delta t/4, \Delta t/8$ 推进相同的时间 $T$。以 $\Delta t/16$ 的结果为参考解，计算各步长的 $L_2$ 误差。 | 标准: 计算收敛阶数 $p = \log_2(\frac{Error_{\Delta t}}{Error_{\Delta t/2}})$。结果 $p$ 应严格趋近于 4.0，证明实现的 Runge-Kutta 时间推进具有理论的四阶精度。 |


## 4. WebGL 数据流与交互设计

由于计算与渲染在同一个 GPU 上下文内，我们的解耦设计需要调整为纹理采样策略。

### 4.1 纯显存数据流转 (Zero-Copy Data Flow)

- **计算纹理 (Compute Textures):** 尺寸严格受限于求解分辨率 $N_{lat} \times N_{lon}$，格式为 `gl.RGBA32F`。用于存储精确的物理场。
- **渲染解耦:** 不再需要将数据传回主线程。在最终渲染帧时，将当前的计算纹理直接绑定到渲染 Shader 的 `uniform sampler2D`。
- **插值平滑:** 利用 WebGL 原生的纹理过滤（`gl.LINEAR`），在 3D 球面的 Fragment Shader 中，即便计算网格只有 $64 \times 128$，映射到高面数的三维球体上时也会自动实现硬件级别的双线性平滑插值。

### 4.2 极点奇点处理 (Pole Singularity)

对于 GPU 物理计算，两极奇点可能导致除以零（$\frac{1}{\sin\theta}$ 当 $\theta=0$ 时）。
- **计算域:** 确保高斯网格（Gaussian Grid）的纬度点避开绝对极点（不在 $\theta = 0$ 和 $\theta = \pi$ 处采样）。
- **渲染域:** Fragment Shader 中对极点 UV 进行保护性 Clamp 或混合过渡处理。

### 4.3 用户交互面板 (GUI)

- **仿真控制 (Simulation):** Play/Pause, dt, Viscosity ($\nu$)。由于在 GPU 上运算极快，可调节每帧执行的 RK4 步数（Steps per Frame）以加速视觉演化。
- **分辨率控制 (Resolution):**
  - **Grid Resolution:** GPGPU 的计算网格大小（这会改变 FBO 纹理尺寸，修改时需要销毁并重建所有计算纹理和预计算矩阵）。
  - **Mesh Resolution:** Three.js 渲染球体的几何精度。修改此项无缝生效。

## 5. 开发阶段里程碑

- **Phase 1: GPGPU 基础设施搭建 (Weeks 1-2)**
  - 验证设备的 WebGL 2.0 浮点纹理和 MRT 支持度。
  - 搭建 Ping-Pong FBO 框架，实现基于纹理的简单局部差分计算（作为热身验证）。
  - 实现基于 `gl.readPixels` 的自动化测试抓手。
- **Phase 2: 谱变换核心 Shader 攻坚 (Weeks 3-4) (最难点)**
  - 将球谐变换（SHT）的数学过程拆解为适合 WebGL Fragment Shader 执行的纹理读取和矩阵运算。
  - 将复杂的勒让德多项式和权重预先在 CPU 计算，编码为高精度纹理上传至 GPU。
  - 完成 UT-02 和 UT-03。
- **Phase 3: Navier-Stokes 方程组装 (Week 5)**
  - 实现非线性项（对流项）的网格空间计算 Shader。
  - 组装 RK4 时间积分器。实现 $\zeta \rightarrow \psi \rightarrow (u,v) \rightarrow \text{Nonlinear}$ 的完整 GPU 流水线。
  - 跑通物理测试 PT-01 至 PT-04。
- **Phase 4: 渲染优化与 UI 交付 (Week 6)**
  - 将计算纹理直接挂载到 Three.js 球面材质上（零拷贝渲染）。
  - 实现 Colormap 着色和光照效果。
  - 接入控制面板，进行性能 Profiling。