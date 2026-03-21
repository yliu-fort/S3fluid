# 1. 项目概述

本项目旨在开发一个基于 Python 桌面端/移动端跨平台框架 Kivy 的球面谱方法二维非线性 PDE（偏微分方程，包含湍流等）模拟器。通过 GPGPU 技术实现极高计算性能，并利用 OpenGL 实现从计算到渲染的“零拷贝”管线，支持多种时间积分格式（如 RK3, RK4）的动态切换。

## 1.1 技术栈选型

- **核心语言与界面**: Python 3.12 & Kivy 2.3+。利用 Kivy 的 Properties 系统进行响应式状态管理，使用 Kivy Canvas API 构建跨平台 UI。
- **计算与渲染引擎**: Kivy 的底层图形 API (`kivy.graphics`, `RenderContext`, `Fbo`) 结合原生 GLSL 着色器。
- **GPGPU 方案**: 利用 OpenGL 的 帧缓冲对象 (FBO) 和 32 位浮点纹理 (`GL_RGBA32F`)。
- **零拷贝 (Zero-Copy)**: 流体状态全部存储 in GPU 显存（VRAM）中，计算阶段的输出纹理直接作为 Kivy 3D 网格（Mesh）的贴图材质进行渲染，无需经历从 GPU 读取到 CPU 再传回 GPU 的性能损耗。
- **科学计算辅助 (NumPy-SHT)**: 利用 NumPy 和 SciPy 完全自主实现球谐变换库。利用 NumPy 数组操作在 CPU 端预生成标准的高斯-勒让德（Gauss-Legendre）网格节点、勒让德多项式权重、拉普拉斯算子特征值矩阵，并作为参考基准验证 OpenGL Shader 管线的计算精度。（本项目严禁导入和使用传统的 `shtns` 库）。

## 1.2 开发原则

- **显式扩展校验**: Kivy 启动时，必须探测底层的 OpenGL 环境。若缺乏 `GL_ARB_texture_float` 或 `GL_EXT_color_buffer_float`，应立即弹出 Kivy 致命错误弹窗并终止，严禁静默退化。
- **独立测试原则 (Independent GPU Testing)**: GPU 计算如同“黑盒”，极易出现由于纹理坐标偏移、精度截断或矩阵维度错位导致的隐蔽数学错误。对于 SHT（正反变换）、拉普拉斯/逆算子、梯度算子 (`synth_grad`)、非线性对流项等每一个独立运行在 GPU 上的 Shader 计算模块，必须在实现后立刻编写测试用例。严禁不经验证直接组装。
- **状态快照与断言**: 测试的核心手段是利用 `glReadPixels` 或 Kivy 的 `texture.pixels` 将 VRAM 数据读回为 NumPy 数组，与自定义 NumPy-SHT 引擎的基准计算结果进行严苛的误差 $L_2$ 范数或 $L_\infty$ 范数校验。
- **策略模式集成 (Strategy Pattern)**: 时间步进器（Time Stepper）必须抽象为独立接口，使得 RK3、RK4 乃至 Euler 格式可以在运行时无缝切换，管线应自动管理所需中间态的 FBO 数量。
- **显存泄漏防范 (Crucial)**: Python 垃圾回收机制不管理 GPU 资源。重建网格或切换分辨率时，必须显式调用 Kivy `Fbo` 和 `Texture` 的 `release()` 方法彻底释放旧的 VRAM 资源。
- **泛用性与强解耦 (Generality & Decoupling)**: 核心算法库（NumPy-SHT）与图形框架必须严格物理隔离。NumPy-SHT 应被设计为一个独立、泛用的球面谱方法工具包，不依赖任何 GUI 状态；同时，GPU 管线中的 SHT 算子和具体的 PDE 方程定义（如 Navier-Stokes 的 RHS）必须通过标准化的 FBO 接口分离，确保系统能轻松扩展以支持浅水方程等其他物理系统。

## 1.3 环境依赖与构建工具

本项目采用 `uv` 作为极速的 Python 包管理器和虚拟环境管理工具，使用标准的 `pyproject.toml` 文件来统一管理依赖配置。

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
    "numpy>=1.26.0", # 用于替代 shtns 开发纯 NumPy SHT 核心
    "scipy>=1.12.0", # 核心依赖：利用其 special 模块极大简化球谐变换基底的生成
    "matplotlib>=3.8.0", # 用于测试验证阶段的断言与可视化
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

环境初始化流程:
1. 安装 uv 工具: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. 创建并同步环境: 执行 `uv sync`，安装依赖项。
3. 激活环境: `source .venv/bin/activate` 或 `.venv\Scripts\activate`

# 2. 自定义 NumPy 球谐变换库设计 (NumPy-SHT)

由于严禁使用第三方 `shtns` 库，我们将基于 NumPy 和 SciPy 开发一套球谐变换库作为整个 PDE 模拟器的底层数学基石与真值校验基准。该库被设计为高度泛用的科学计算组件，可在剥离 Kivy 环境后单独被外部 Python 项目调用。

## 2.1 模块定位与职责

- **预计算引擎 (Pre-computation Engine)**: 计算 Gauss-Legendre 节点及求积权重，生成缔合勒让德多项式（Associated Legendre Polynomials, ALP）$P_l^m(\cos\theta)$ 及其导数矩阵。（直接利用 `scipy.special.lpmv` 或 `scipy.special.sph_harm` 替代手动编写递推公式，大幅减轻工作量与除错成本）。将结果封装为 NumPy `ndarray`，稍后转化为 Kivy `Texture` 注入 OpenGL 管线。
- **真值校验器 (Ground Truth Validator)**: 作为 Kivy Shader 管线的“数学标尺”。在独立测试中，利用 NumPy 的高精度浮点运算验证 OpenGL FBO 输出结果。
- **热切换干预者 (Resolution Switcher)**: 在动态分辨率切换时，通过 NumPy 介入执行精确的谱空间截断与补零重采样。

## 2.2 核心算法实现策略

### 高斯-勒让德网格与基函数初始化:
1. 利用 `scipy.special.roots_legendre(N_lat)` 获取高斯节点 $x = \cos\theta$ 与求积权重 $w$，并转换为纬度网格 `lats = np.arccos(x)`。
2. **快速基函数生成**: 遍历波矢 $l$ 和 $m$，调用 `scipy.special.lpmv(m, l, x)` 或通过 `scipy.special.sph_harm` 提取经度 $\phi=0$ 时的实部，直接获取归一化的伴随勒让德多项式数组，避免了从底层手写容易溢出的三项递推公式。

### 正向球谐变换 (Forward SHT / analys):
$f(\theta, \phi)$ 到 $f_{lm}$ 的转换包含两步：经度方向的傅里叶变换 + 纬度方向的勒让德高斯求积。
- 实现: 调用 `np.fft.rfft(data, axis=-1)` 执行经度 FFT 获取 $f_m(\theta)$；利用 `np.einsum` 或 `np.matmul` 将 $f_m(\theta)$ 与预计算的 ALP 权重数组 $W_{l, m, \theta}$ 执行数组收缩积完成积分。

### 逆向球谐变换 (Inverse SHT / synth):
- 实现: 先利用 `np.matmul` 将谱系数 $f_{lm}$ 与 ALP 基函数 $P_l^m(\cos\theta)$ 矩阵相乘还原出 $f_m(\theta)$，再调用 `np.fft.irfft(f_m, n=N_lon, axis=-1)` 还原物理网格数据。

### 带导数的逆变换 (synth_grad):
- 经向导数 $\partial_\phi$: 在谱空间直接乘以波矢 $i \cdot m$。
- 纬向导数 $\partial_\theta$: 求流场速度或梯度需要计算 $\partial_\theta P_l^m$。在初始化阶段，利用 SciPy 生成的伴随勒让德多项式值，直接代入标准的勒让德解析导数关系式（如包含 $P_{l}^{m+1}$ 与 $P_l^m$ 的组合），算出准确的导数矩阵，随后与谱系数执行收缩积。
- 分别对上述两项执行 `irfft` 还原至网格空间。

# 3. 系统架构设计 (Kivy GPGPU 管线)

系统严格遵循“高内聚、低耦合”的泛用性架构模式，划分为四个独立模块。模块间仅通过标准接口或浮点纹理/多维数组协议进行通信：

### 初始化调度器 (Initialization Manager):
- 具备 Teardown & Rebuild能力，响应运行时动态分辨率切换。
- 利用 NumPySHT 生成初始高斯-勒让德网格 (Gauss-Legendre Grid) 的纬度 (lats) 和经度 (lons)。
- 预生成 $8$ 阶高斯谱滤波器 $\exp(-36.0 \cdot (l / l_{max})^{8.0})$ 用于去混叠（De-aliasing）。
- 将由 NumPy 计算的变换矩阵、滤波系数、拉普拉斯特征值转换为 `kivy.graphics.texture.Texture`，上传至 GPU。

### 球体谐波变换组件 (SHT Component):
作为泛用的纯数学着色器工具箱，封装一对相互可逆的 Kivy Shader Passes：
1. **Forward SHT (物理 $\rightarrow$ 谱)**: 将网格空间的物理场积分转换为谱系数 $\hat{f}_{lm}$。基于纹理矩阵乘法。
2. **Inverse SHT 与求导 (谱 $\rightarrow$ 物理)**:
   - 将任意谱场 $\hat{f}_{lm}$ 映射为网格数据。
   - 提供求解流函数与梯度的通用算子接口。
   - 执行等效于 `NumPySHT.inverse_sht_grad` 的 Shader 操作，求取梯度或速度场，输出至物理空间供后续物理方程层调用。

### 动态时间积分器 (Pluggable Time Integrator):
- 作为高度解耦的插件化模块，定义通用的方程右端项 (RHS) 函数接口，完全与特定物理方程解耦。
- 采用状态机和 FBO 池（FBO Pool）管理 Ping-Pong 渲染。
- 将具体的非线性方程组（如涡量-流函数对流项）封装为可插拔的着色器 Pass，供积分器子步调用。

### 3D 渲染器 (Zero-Copy Renderer):
- 自定义 Kivy `RenderContext`。
- **同构网格渲染 (Isomorphic Mesh)**: 直接复用由 NumPySHT 导出的高斯-勒让德网格 lats 和 lons 数组，精确构建 Kivy 的 3D Mesh 顶点。
- 将最新的物理场 FBO 纹理绑定到 Mesh 上，应用伪彩着色器（Colormap Shader）。

## 3.1 核心 SHT 与代数算子的 GPU 实现指南

为了在 GPU 上实现真正的“零拷贝”闭闭环计算，必须将 NumPySHT 中定义的数学逻辑映射为 GPU 的 Shader Pass：
- **正向球面谐波变换 (NumPySHT.forward_sht 的 GPU 映射)**: 在 Fragment Shader 中实现数值积分（离散傅里叶变换 + 高斯-勒让德求积）。采样预计算好的勒让德权重纹理，执行数组收缩。
- **逆向球面谐波变换 (NumPySHT.inverse_sht 的 GPU 映射)**: 输入谱系数纹理，与预计算的勒让德多项式基函数纹理进行数组收缩/求和，输出网格场纹理。
- **带梯度的逆向变换 (NumPySHT.inverse_sht_grad 的 GPU 映射) (🌟 最复杂)**: 严禁在粗糙的物理网格上用有限差分求导。必须将 NumPy 生成的勒让德多项式导数项作为纹理上传，在 Shader 中执行带有导数权重的逆变换矩阵乘法。

# 4. 详细测试计划 (全模块覆盖)

测试的核心是将 GLSL Shader 管线输出的 32 位浮点纹理读回，与 NumPySHT 的双精度前向传播结果对齐。

## 4.1 GPGPU 模块单元与集成测试矩阵

| 测试用例 ID | 测试模块 (GPU Pass) | 测试内容描述与输入 | 验证基准 (Ground Truth) 与预期结果 |
| :--- | :--- | :--- | :--- |
| UT-01 | FBO 环境与精度 | 写入大动态范围数值（如 $10^{-8}$ 到 $10^5$），使用 `fbo.pixels` 读回校验。 | 读回数值与写入值相对误差 $< 10^{-6}$ (单精度极限)。 |
| UT-02 | 预计算数据纹理化 | 加载 NumPySHT 计算的勒让德矩阵序列、求积权重。 | 随机抽样验证 GPU 纹理与 NumPy 数组完全一致。 |
| UT-03 | 正向 SHT 算子 | 输入：解析的网格场数据（如 $\cos^2(\theta)\sin(\phi)$）。 | 从 GPU 读回的谱系数 $\hat{f}_{lm}$ 必须与 `NumPySHT.forward_sht(data)` 残差极小。 |
| UT-04 | 逆向 SHT 算子 | 输入：解析分布的谱系数数组。 | 从 GPU 读回的网格场必须与 `NumPySHT.inverse_sht(coeffs)` 误差在容许内。 |
| UT-05 | 线性算子 | 在 GPU 谱空间执行泛用的代数运算（如求解拉普拉斯或流函数）。 | 读回结果必须与 NumPy 中的代数乘法对齐。 |
| UT-06 | 梯度算子 | 输入：流函数谱系数 $\psi_{lm}$。执行 Shader 求导与逆变换。 | 读回的速度场必须与 `NumPySHT.inverse_sht_grad(psi)` 的结果一致。 |
| UT-07 | 非线性对流项 | 输入：Shader 速度场和标量梯度场。执行泛用逐像素 $u \cdot \nabla f$。 | 与 NumPy 点积计算结果精度对齐。 |
| UT-08 | RHS 单步闭环 | 输入：特定物理方程的初始态谱 $\zeta_{lm}$。 | **极度重要**: 单次 Shader RHS 结果必须与纯 NumPy 编写的 `numpy_nonl(zeta_coeffs)` 输出一致！ |
| UT-09 | 积分器切换 | 将积分器模块拔插到简单的线性衰减测试系统验证。 | RK4 耗散率严格匹配解析解，证明模块解耦彻底且无显存泄漏。 |

# 5. 数据流与 UI 交互设计 (Kivy 特性)

## 5.1 零拷贝与高斯网格渲染

- **同构网格设计**: 由于 3D 网格顶点的 UV 坐标正好落在计算纹理的像素中心，渲染时直接读取 GPU FBO 中流体场的精确浮点数值进行伪彩映射。
- **两极与缝合线处理**: 额外复制一列经度顶点缝合网格；两极单独增加极点顶点并利用 Kivy 的 `clamp_to_edge` 特性延续高斯纬度圈的颜色闭合球体。

## 5.2 动态分辨率切换与谱插值机制 (NumPy 介入)

在运行时动态更改网格分辨率，需由 CPU 端的 NumPySHT 引擎介入完成优雅的谱截断 (Spectral Truncation) 或补零 (Zero-padding)：
1. **暂停计算与状态捕获**: `Clock.unschedule` 暂停仿真。利用 `fbo.pixels` 将当前的涡量场 $\zeta$ 下载回主内存转换为 ndarray。
2. **提取当前谱系数**: 利用旧配置的 NumPySHT 实例获取谱系数：$\zeta_{lm}^{old} = \text{sht\_old.forward\_sht}(\zeta_{grid})$。
3. **清理 VRAM (Teardown)**: 调用 Kivy 相关 `Fbo.release()` 防止显存泄漏。
4. **管线重建 (Rebuild)**:
   - 以用户选定的新 $l_{max}$ 实例化新的 NumPySHT 对象。
   - 重新生成所有勒让德矩阵并上传至 GPU；重构 Kivy 3D Mesh。
5. **谱空间重采样**:
   - 利用 NumPy 灵活的数组索引：新建尺寸为新 $l_{max}$ 的空 `ndarray`。
   - 若升高分辨率，将 $\zeta_{lm}^{old}$ 填入低频切片，高频补零；若降低分辨率，直接截取 $\zeta_{lm}^{old}$ 低频切片丢弃高频。
6. **状态恢复**: 利用新的 NumPySHT 将新谱系数还原到网格。作为初始场上传至新 FBO，恢复仿真循环。

# 6. 开发阶段里程碑

- **Phase 1: 基础设施与自定义 NumPy-SHT (Weeks 1-2)**
  - 使用 `uv` 初始化环境。
  - **核心开发**: 完成 `NumPySHT` 类的编写（集成 `scipy.special` 库直接获取 Gauss-Legendre 求积节点与多维数组化的伴随勒让德多项式，彻底替代易错的手写递推）。
  - 实现基于 `np.fft` 和 `np.matmul` 的 `forward_sht`, `inverse_sht`, `inverse_sht_grad` 并完成严密的内部正确性测试（确保作为独立的数学包可用）。

- **Phase 2: GPU 数组化与 Kivy Shader 映射 (Weeks 3-4) (最难点)**
  - 将 NumPy 预计算的权重编码为 `GL_RGBA32F` 纹理。
  - 编写 Kivy Shader 执行通用 OpenGL 前后向映射，强制通过基准测试 (UT-03 到 UT-06)。

- **Phase 3: 物理模块接入与积分器解耦 (Weeks 5-6)**
  - 实现非线性对流项等特定物理 Shader 逻辑并作为插件接入系统。通过终极闭环验证 UT-08。
  - 实现可动态插拔的 RK3/RK4 时间步进状态机。

- **Phase 4: 渲染、动态响应与交互交付 (Week 7-8)**
  - 构建极点闭合的 Kivy 3D Mesh 与网格线框高亮覆盖。
  - 接入实时 CFL 数计算与 Kivy UI。
  - 完成基于 NumPy 数组补零与截断的动态分辨率运行时热切换功能。