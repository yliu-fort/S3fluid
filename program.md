# 基于 Python + Kivy 的球面谱方法非线性 PDE (GPGPU) 可视化模拟器开发文档

## 1. 项目概述

本项目旨在开发一个基于 Python 桌面端/移动端跨平台框架 Kivy 的球面谱方法二维非线性 PDE（偏微分方程，包含湍流等）模拟器。通过 GPGPU 技术实现极高计算性能，并利用 OpenGL 实现从计算到渲染的“零拷贝”管线，支持多种时间积分格式（如 RK3, RK4）的动态切换。

### 1.1 技术栈选型

- **核心语言与界面**: Python 3.12 & Kivy 2.3+。利用 Kivy 的 Properties 系统进行响应式状态管理，使用 Kivy Canvas API 构建跨平台 UI。
- **计算与渲染引擎**: Kivy 的底层图形 API (`kivy.graphics`, `RenderContext`, `Fbo`) 结合原生 GLSL 着色器。
- **GPGPU 方案**: 利用 OpenGL 的 帧缓冲对象 (FBO) 和 32 位浮点纹理 (`GL_RGBA32F`)。
- **零拷贝 (Zero-Copy)**: 流体状态全部存储在 GPU 显存（VRAM）中，计算阶段的输出纹理直接作为 Kivy 3D 网格（Mesh）的贴图材质进行渲染，无需经历从 GPU 读取到 CPU 再传回 GPU 的性能损耗。
- **科学计算辅助 (预计算与测试)**: NumPy / SciPy 以及 `shtns` 库。利用 `shtns` 在 CPU 端预生成标准的高斯-勒让德（Gauss-Legendre）网格节点、勒让德多项式权重、拉普拉斯算子特征值矩阵，并作为参考基准验证 GPU 计算精度。

### 1.2 开发原则

- **显式扩展校验**: Kivy 启动时，必须探测底层的 OpenGL 环境。若缺乏 `GL_ARB_texture_float` 或 `GL_EXT_color_buffer_float`，应立即弹出 Kivy 致命错误弹窗并终止，严禁静默退化。
- **独立测试原则 (Independent GPU Testing)**: GPU 计算如同“黑盒”，极易出现由于纹理坐标偏移、精度截断或矩阵维度错位导致的隐蔽数学错误。对于 SHT（正反变换）、拉普拉斯/逆算子、梯度算子 (`synth_grad`)、非线性对流项等每一个独立运行在 GPU 上的 Shader 计算模块，必须在实现后立刻编写测试用例。严禁不经验证直接组装。
- **状态快照与断言**: 测试的核心手段是利用 `glReadPixels` 或 Kivy 的 `texture.pixels` 将 VRAM 数据读回为 NumPy 数组，与 CPU (`shtns` 和 NumPy) 的基准计算结果进行严苛的误差 $L_2$ 范数或 $L_\infty$ 范数校验。
- **策略模式集成 (Strategy Pattern)**: 时间步进器（Time Stepper）必须抽象为独立接口，使得 RK3、RK4 乃至 Euler 格式可以在运行时无缝切换，管线应自动管理所需中间态的 FBO 数量。
- **显存泄漏防范 (Crucial)**: Python 垃圾回收机制不管理 GPU 资源。重建网格或切换分辨率时，必须显式调用 Kivy `Fbo` 和 `Texture` 的 `release()` 方法彻底释放旧的 VRAM 资源。

### 1.3 环境依赖与构建工具

本项目采用 `uv` 作为极速的 Python 包管理器和虚拟环境管理工具，放弃传统的 `pip` 和 `requirements.txt`，转而使用标准的 `pyproject.toml` 文件来统一管理依赖配置。

`pyproject.toml` 示例:

```toml
[project]
name = "kivy-gpgpu-pde"
version = "0.1.0"
description = "基于 Kivy 和 GPGPU 的球面谱方法 PDE 模拟器"
readme = "README.md"
requires-python = "==3.12.*"
dependencies = [
    "kivy>=2.3.0",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
    "shtns>=3.6",
    "matplotlib>=3.8.0", # 用于测试验证阶段的断言与可视化
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
# uv 的专属配置，如果遇到 shtns 等包构建问题，可在此处配置系统级依赖的编译选项
```

环境初始化流程:
1. **安装 uv 工具**: `curl -LsSf https://astral.sh/uv/install.sh | sh` (或参考官方文档安装)。
2. **创建并同步环境**: 在项目根目录下执行 `uv sync`，`uv` 将自动创建 `.venv` (Python 3.12) 并急速安装所有 dependencies。
3. **激活环境**: `source .venv/bin/activate` (Linux/macOS) 或 `.venv\Scripts\activate` (Windows)。

## 2. 系统架构设计

### 2.1 模块划分 (Kivy GPGPU 管线)

系统被划分为四个高度解耦的模块：

1. **初始化调度器 (Initialization Manager)**:
   - 具备 Teardown & Rebuild 能力，响应运行时动态分辨率切换。
   - 利用 `shtns.set_grid()` 生成初始高斯-勒让德网格 (Gauss-Legendre Grid) 的纬度 (lats) 和经度 (lons)。
   - 预生成 $8$ 阶高斯谱滤波器 $\exp(-36.0 \cdot (l / l_{max})^{8.0})$ 用于去混叠（De-aliasing）。
   - 将这些预计算的变换矩阵、滤波系数、拉普拉斯特征值转换为 `kivy.graphics.texture.Texture`，上传至 GPU。

2. **球面谐波变换组件 (SHT Component)**:
   - 封装一对相互可逆的 Shader Passes：
     - **Forward SHT (物理 $\rightarrow$ 谱)**: 将网格空间的非线性对流项 $N(\zeta)$ 积分转换为谱系数 $\hat{N}_{lm}$。基于纹理矩阵乘法。
     - **Inverse SHT 与求导 (谱 $\rightarrow$ 物理)**:
       - 将涡量谱 $\zeta_{lm}$ 映射为网格数据。
       - 从 $\zeta_{lm}$ 计算流函数谱 $\psi_{lm}$。
       - 执行 `synth_grad` 的等效 Shader 操作，求取流函数梯度（即速度场 $u_\theta, u_\phi$）和涡量梯度 $\nabla \zeta$，输出至物理空间用于非线性项拼接。

3. **动态时间积分器 (Pluggable Time Integrator)**:
   - 采用状态机和 FBO 池（FBO Pool）管理 Ping-Pong 渲染。
   - **RK3 格式**: 动态分配/激活 3 个中间态 FBO，执行 3 次 Shader Pass 更新。
   - **RK4 格式**: 动态分配/激活 4 个中间态 FBO，执行 4 次 Shader Pass 更新（如 demo 中 k1 到 k4 的显式推进）。

4. **3D 渲染器 (Zero-Copy Renderer)**:
   - 自定义 Kivy `RenderContext`。
   - **同构网格渲染 (Isomorphic Mesh)**: 直接复用 CPU 端 `shtns` 导出的高斯-勒让德网格 lats 和 lons 数组，精确构建 Kivy 的 3D Mesh 顶点。实现 3D 几何顶点与 GPU 计算纹理像素的 1:1 严格对齐。
   - 将最新的物理场 FBO 纹理绑定到 Mesh 上，应用伪彩着色器（Colormap Shader）。

### 2.2 伪谱法数学模型在 GPU 上的映射

方程定义为：$\frac{\partial \zeta}{\partial t} = \nu \nabla^2 \zeta - \mathbf{u} \cdot \nabla \zeta$

- **拉普拉斯与逆算子**:
  - 算子 $\nabla^2$ 在谱空间表现为对角矩阵乘法：$-l(l+1)$。
  - 流函数 $\psi$ 的求解即为逆算子：$\psi_{lm} = \zeta_{lm} / [-l(l+1)]$。
  - 在 GPU 中表现为谱系数纹理（Spectral Texture）与预计算的一维度数 $l$ 纹理的逐像素乘法。

- **非线性对流项 ($N(\zeta) = -\mathbf{u} \cdot \nabla \zeta$)**: 必须在物理网格空间计算。
  1. 在 GPU 谱空间求得 $\psi_{lm}$。
  2. 逆变换（带求导算子）到网格空间得到速度场 $u_\theta = \frac{1}{\sin\theta}\frac{\partial \psi}{\partial \phi}, u_\phi = -\frac{\partial \psi}{\partial \theta}$ 以及涡量梯度 $\frac{\partial \zeta}{\partial \theta}, \frac{1}{\sin\theta}\frac{\partial \zeta}{\partial \phi}$。
  3. 在网格空间执行逐像素计算：$Adv = u_\theta \cdot \frac{\partial \zeta}{\partial \theta} + u_\phi \cdot \frac{1}{\sin\theta}\frac{\partial \zeta}{\partial \phi}$。
  4. 正变换回谱空间，应用 $8$ 阶指数谱滤波器截断（取代单纯的高斯截断）以维持数值稳定性。

### 2.3 核心 SHT 与代数算子的 GPU 实现指南

为了在 GPU 上实现真正的“零拷贝”闭环计算，必须将 `shtns` 库中最核心的球面谐波变换及其相关函数翻译为 GPU 的 Shader Pass：

1. **正向球面谐波变换 (`sht.analys` 的 GPU 映射)**
   - **作用**: 将物理空间的网格数据（Grid）转换为谱空间的球谐系数（Spectral Coefficients）。
   - **用途**: 在计算完非线性对流项（$u \cdot \nabla \zeta$）后，调用此模块将其从物理网格空间变回谱空间，以便与线性耗散项相加并在积分器中更新谱系数。
   - **GPU 实现思路**: 在 Fragment Shader 中实现数值积分（离散傅里叶变换 + 高斯-勒让德求积）。输入为物理场 RGBA32F 纹理，采样预计算好的勒让德权重纹理，执行张量收缩/矩阵乘积，输出谱系数纹理（复数可拆分为实部和虚部存入不同通道）。

2. **逆向球面谐波变换 (`sht.synth` 的 GPU 映射)**
   - **作用**: 将谱空间的球谐系数还原为物理空间的网格数据。
   - **用途**: 用于渲染前的状态提取，或者当需要直接在物理空间获取涡量场 $\zeta$ 的分布时使用。
   - **GPU 实现思路**: 输入谱系数纹理，与预计算的勒让德多项式基函数纹理进行张量收缩/求和，输出网格场纹理。

3. **带梯度的逆向变换 (`sht.synth_grad` 的 GPU 映射) (🌟 最核心且最复杂)**
   - **作用**: 输入谱系数，直接在逆变换的同时计算出其在球面物理网格上的梯度分量。
   - **用途**: 在求解 Navier-Stokes 方程中需调用两次：一是求速度场（从流函数谱系数 $\psi_{lm}$ 算出 $u_\phi, u_\theta$），二是求涡量梯度（从 $\zeta_{lm}$ 算出 $\nabla \zeta$ 分量）。
   - **GPU 实现思路**: 严禁在粗糙的物理网格上用有限差分求导（精度太低且极点会产生奇点）。必须将勒让德多项式的导数项预先在 CPU 算好并作为纹理上传，在 Shader 中执行带有导数权重的逆变换矩阵乘法，输出一张包含 2 个通道（例如 R 通道存 $\partial / \partial \theta$，G 通道存 $1/\sin\theta \partial / \partial \phi$）的浮点纹理。

4. **谱空间逐像素代数算子**
   - **包含**: 流函数反演 (invlap)、计算耗散项 (lap)、应用高斯滤波截断 (spec_filter)。
   - **GPU 实现思路**: 在 GPGPU 架构中，只需把 invlap、lap、spec_filter 作为一个 1D 或 2D 的浮点纹理初始化时传给 GPU。计算时只需用极简的 Shader 将状态系数纹理（如 zeta_coeffs）与这些算子纹理进行对应位置的“逐像素乘法”即可。

## 3. 详细测试计划 (全模块覆盖)

由于 GPU 编程的“黑盒”属性，所有的数学管线必须被拆解成独立的单元进行断言测试。开发必须遵循 TDD（测试驱动开发）的思想，先对比 CPU 实现，再进行管线组装。

### 3.1 GPGPU 模块单元与集成测试矩阵

| 测试用例 ID | 测试模块 (GPU Pass) | 测试内容描述与输入 | 验证基准 (Ground Truth) 与预期结果 |
| :--- | :--- | :--- | :--- |
| UT-01 | FBO 环境与精度 | 创建 32F FBO，通过 Shader 写入大动态范围数值（如 $10^{-8}$ 到 $10^5$），使用 `fbo.pixels` 读回转 NumPy 校验。 | 读回数值与写入值之间的相对误差 $< 10^{-6}$ (单精度极限)。 |
| UT-02 | 预计算数据纹理化 | 加载 CPU (`shtns`) 计算的勒让德矩阵序列、求积权重、拉普拉斯算子。 | 随机抽样验证 GPU 纹理的 RGBA 通道解包浮点数与 NumPy 原始数组完全一致。 |
| UT-03 | 正向 SHT 算子 (Forward SHT) | 输入：解析的网格场数据（如 $\cos^2(\theta)\sin(\phi)$）。 操作：网格空间积分映射到谱空间。 | 从 GPU 读回的谱系数 $\hat{f}_{lm}$ 必须与 `shtns.analys(data)` 计算的系数残差极小。 |
| UT-04 | 逆向 SHT 算子 (Inverse SHT) | 输入：解析分布的谱系数数组。 操作：执行勒让德多项式求和回到高斯网格。 | 从 GPU 读回的网格场必须与 `shtns.synth(coeffs)` 结果的误差 $L_\infty$ 范数在容许阈值内。 |
| UT-05 | 线性算子 (Laplacian) | 在 GPU 谱空间执行 $\zeta \rightarrow \nabla^2 \zeta$ 以及 $\zeta \rightarrow \psi$ (除以 $-l(l+1)$)。 | 读回结果必须与 NumPy 中的纯代数乘法完全对齐。 |
| UT-06 | 梯度算子 (synth_grad) | 输入：给定的流函数谱系数 $\psi_{lm}$。 操作：在 Shader 中计算导数并执行 SHT 逆变换得到 $(u_\theta, u_\phi)$。 | 读回的速度场必须与 `shtns.synth_grad(psi)` 的结果一致，确保极点不出现奇点爆炸。 |
| UT-07 | 非线性对流项 (Grid Math) | 输入：由 Shader 生成的速度场和涡量梯度场。 操作：物理空间逐像素计算 $u \cdot \nabla \zeta$。 | 与 NumPy 直接计算的相应矩阵点积+逐元素乘法结果精度对齐。 |
| UT-08 | RHS 单步闭环 (RHS Closure) | 输入：初始涡量谱 $\zeta_{lm}$。 操作：连贯执行 $\zeta \rightarrow \psi \rightarrow u, \nabla\zeta \rightarrow N(\zeta) \rightarrow \hat{N}_{lm} \rightarrow$ 滤波加耗散。 | **极度重要**: 单次右端项求值的结果，必须与 demo 代码中的 `nonl(zeta_coeffs)` python 函数输出一致！ |
| UT-09 | 积分器切换 (Time Steppers) | 分别使用 RK3 和 RK4 积分器，对纯线性耗散方程 $\frac{\partial \zeta}{\partial t} = \nu \nabla^2 \zeta$ 步进相同时间 $T$。 | RK4 结果的耗散率严格匹配解析解 $e^{-\nu l(l+1)T}$，并比较 RK3 差异。且不允许任何显存泄漏。 |

## 4. 数据流与 UI 交互设计 (Kivy 特性)

### 4.1 零拷贝与精确纹理映射 (Zero-Copy & Exact Mapping)

- **计算阶段**: `Clock.schedule_interval` 触发 `update(dt)`。写入内部 FBO (`fbo_pong`)。
- **获取纹理**: 提取 `active_texture = fbo_pong.texture`。计算纹理尺寸严格为 $N_{lon} \times N_{lat}$。
- **赋值渲染**: 将纹理绑定给同构高斯网格生成的 Mesh。
- **无插值损耗**: 由于 3D 网格顶点的 UV 坐标正好落在计算纹理的像素中心（或通过 nearest 采样对齐），渲染时可以直接读取流体场的精确浮点数值进行伪彩映射，避免了传统投影插值带来的模糊和误差。

### 4.2 高斯网格渲染：线框、极点与闭合处理

高斯-勒让德网格为计算设计，在直接用于 3D 球面渲染时需要做特殊的图形学处理：

- **经度缝合线 (Longitude Seam)**: 计算网格经度通常划分在 $[0, 2\pi-\Delta\phi]$。为了在 3D 中无缝闭合，构建 Mesh 顶点时，需在经度方向额外复制一列起始顶点（令其物理位置重合），并分配 UV 坐标 $U=1.0$。
- **两极孔洞 (Pole Hole)**: 高斯求积节点天然避开绝对极点 $\theta=0$ 和 $\theta=\pi$（防止计算域除以零）。这意味着直接渲染时，南北极会存在一个小圆孔。
  - **解决方案**: 在构建渲染 Mesh 时，手动在南北极处各增加一个顶点 $(0,0,\pm R)$。
  - **极点着色**: 将极点顶点的 UV 的 $V$ 坐标分别设为 $0.0$ 和 $1.0$。在着色器中，Kivy 的纹理采样器由于设置为 `clamp_to_edge`，极点区域将平滑延续最接近的高斯纬度圈上的颜色，在视觉上完美闭合球体。
- **线框高亮覆盖 (Wireframe Overlay)**: 为了在物理场颜色之上显示计算网格的边界，在 Kivy Canvas 中需将网格绘制两次。
  1. 第一遍：使用 `mode='triangles'`，挂载 Colormap 着色器进行实体填充。
  2. 第二遍：使用 `mode='lines'` 或 `mode='line_strip'`，取消纹理绑定，并设置单一纯黑材质（或黑色纯色着色器）重新绘制同构的 Mesh 顶点，以此实现网格边界的高亮叠加。

### 4.3 Kivy UI 交互面板

利用 Kivy 的布局 (Box/Grid Layout) 构建控制层（悬浮在 3D 渲染层之上）：

- **仿真控制 (Simulation Properties)**:
  - **运行/暂停**: `ToggleButton` 绑定 `Clock.unschedule` / `Clock.schedule`。
  - **分辨率切换 (Resolution)**: `Spinner` (下拉菜单)，选项如 `['lmax=63 (Low)', 'lmax=127 (Med)', 'lmax=255 (High)']`。触发 4.4 节定义的动态分辨率切换机制。
  - **积分格式选择**: `Spinner`，选项为 `['RK3', 'RK4', 'Euler']`。切换时重构 FBO 池。
  - **参数调节**: `Slider` 控制时间步长 $dt$，黏性系数 $\nu$。通过 Kivy Properties 的 `on_value` 同步至 Shader Uniform。
  - **网格线框开关 (Show Grid)**: `Switch` 控件。控制“第二遍绘制 Mesh”的颜色通道透明度，用于实时开启/关闭黑色网格线的覆盖显示。

- **性能与状态监控 (Metrics Monitor)**:
  - **帧率 (FPS)**: 实时通过 `Clock.get_fps()` 提取帧率。
  - **能量监控 (Total Energy)**: 监控总动能 $E / E_0$。
  - **CFL 数 (Courant-Friedrichs-Lewy Number)**: 在仿真循环中定期评估：$CFL = U_{max} \cdot \frac{\Delta t}{\Delta x_{min}}$。

### 4.4 动态分辨率切换与谱插值机制 (Dynamic Resolution Switching)

在运行时动态更改网格分辨率是一项高级特性。在基于网格的差分法中，这需要复杂的物理场双线性/双三次插值；但在谱方法中，这可以通过优雅的谱截断 (Spectral Truncation) 或 补零 (Zero-padding) 实现完美的数学重采样。由于此操作并不频繁（响应用户 UI 点击），为了管线稳定，该过程需由 CPU (`shtns`) 介入辅助完成：

1. **暂停计算与状态捕获**: `Clock.unschedule` 暂停仿真器。利用 `fbo.pixels` 将当前的涡量场 $\zeta$ 从 GPU (旧分辨率) 下载回 CPU NumPy 数组。
2. **CPU 提取当前谱系数**: 利用旧的 `shtns` 实例计算当前的谱系数：$\zeta_{lm}^{old} = \text{sht\_old.analys}(\zeta_{grid})$。
3. **彻底清理 VRAM (Teardown)**: 调用 Kivy 的所有相关 `Fbo.release()`、`Texture.release()`，并清空当前的三维 Mesh 顶点，防止切换分辨率导致显存爆炸。
4. **管线重建 (Rebuild)**:
   - 以用户选定的新 $l_{max}$ 初始化新的 `shtns` 实例，获取新的 $N_{lat} \times N_{lon}$ 网格。
   - 重新生成所有预计算纹理（勒让德权重、新拉普拉斯特征值等），上传至 GPU。
   - 重新构建更高（或更低）面数的高斯 3D Mesh。
   - 重新分配对应新尺寸的 FBO 池。
5. **谱空间重采样 (Spectral Resampling)**:
   - **如果分辨率升高 (Up-scaling)**: 创建全零的新谱系数数组 $\zeta_{lm}^{new}$，将 $\zeta_{lm}^{old}$ 的低频部分直接复制进去（高频部分补零）。
   - **如果分辨率降低 (Down-scaling)**: 创建新的 $\zeta_{lm}^{new}$，直接截取 $\zeta_{lm}^{old}$ 中对应低频部分（高频部分自动截断丢弃）。
6. **状态恢复与继续**: 利用新的 `shtns` 将 $\zeta_{lm}^{new}$ 还原到新的物理网格：$\zeta_{grid}^{new} = \text{sht\_new.synth}(\zeta_{lm}^{new})$。将其作为初始场上传至新建的 GPU FBO，恢复 `Clock.schedule` 继续执行。

## 5. 开发阶段里程碑

- **Phase 1: 环境构建与基础 GPGPU 设施 (Weeks 1-2)**
  - 使用 `uv` 初始化 Python 3.12 虚拟环境并配置 `pyproject.toml`。
  - 构建底层 FBO 包装类。跑通基础 Ping-Pong 渲染并完成 UT-01 到 UT-02。
- **Phase 2: 谱计算核心与 GPU 张量化 (Weeks 3-4) (最难点)**
  - 集成 `shtns`。编写 Kivy Shader，实现正逆映射及指数滤波 (`spec_filter`)。
  - 完成求导管线的 GPU 移植，强制通过 UT-03 到 UT-06 测试。
- **Phase 3: Navier-Stokes 组装与策略模式积分器 (Weeks 5-6)**
  - 实现非线性对流项逻辑。通过终极闭环测试 UT-08。
  - 实现 RK3/RK4 架构切换，通过 UT-09 测试。
- **Phase 4: 3D 渲染器、动态响应与交互交付 (Week 7-8)**
  - 同构渲染与线框: 构建基于高斯经纬度导出的同构 Kivy 3D Mesh，实现极点缝合与网格边界的黑色线框高亮。
  - 监控告警: 接入实时 CFL 数计算。
  - 动态重采样: 完成 4.4 节定义的“暂停->显存清理->CPU补零/截断重采样->管线重建”的运行时分辨率热切换功能。