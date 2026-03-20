# 球面谱方法 Navier-Stokes 方程 WebGL 可视化模拟器开发文档

## 1. 项目概述

本项目旨在开发 Web 端下的球面谱方法 2D 湍流模拟器，实现一个无需安装任何后端环境、纯浏览器运行的交互式流体动力学模拟器。

### 1.1 技术栈选型

- **核心计算引擎 (Math Engine):** WebAssembly (WASM) + Web Workers。
  - **方案:** 使用 Emscripten 将 C 语言编写的高性能球面谐波变换库（如 SHTns）编译为 WASM。将 RK4 积分和非线性项计算放在 Web Worker 中运行，避免阻塞主线程 UI。
- **渲染引擎 (Render Engine):** WebGL 2.0 (推荐结合 Three.js)。
  - **方案:** 将 WASM 计算出的网格空间（Grid Space）数据（如速度场 $u, v$ 或涡量 $\zeta$）作为浮点纹理（Float Texture）上传至 GPU，通过自定义 Shader 映射到 3D 球体模型上。
- **UI 与交互:** React。
  - **方案:** 使用 React 进行应用级别的状态管理和组件化开发，配合 lil-gui 构建悬浮控制面板进行核心参数的实时调节。
- **依赖管理:** uv project manager + toml 进行依赖管理。

### 1.2 开发原则
- **显式错误可见性原则（至关重要）**：严禁静默失败。 对于无法恢复的错误（例如：配置解析失败、组件缺失、非法状态），程序必须抛出异常，绝不允许仅通过日志记录或返回 null 来敷衍处理。
- **快速失败**：如果配置文件不正确，服务应在启动时立即崩溃，而不是带着错误的配置继续运行。
- **禁止吞噬异常**：严禁在 catch 块中捕获错误后不进行处理便让程序继续执行。
- **配置完整性原则**：所有外部 JSON 或脚本配置必须通过 Schema 校验。测试不仅要覆盖代码逻辑，还必须涵盖配置文件的合法性校验，以防止因拼写错误导致运行时崩溃。
- **流水线隔离原则**：每个流水线节点必须是可独立测试的。
- **确定性随机原则**：所有随机数生成必须使用严格受控且基于种子的随机实例，绝不允许使用系统全局随机函数。这确保了任何状态都可被完美回放，以便于调试和校验。
- **累积回归原则（至关重要）**：在每个里程碑交付之前，必须重新运行之前所有阶段的测试套件。 测试不仅是为了验证新功能，更是为了确保新代码没有破坏原有的基础。如果第三阶段的开发导致第一阶段的测试失败，这将被视为阻塞性问题。


## 2. 系统架构设计

### 2.1 模块划分

- **模拟器内核 (WASM Module):**
  - `init_grid(lmax)`: 初始化球面高斯网格。
  - `init_field(spectrum_slope)`: 基于给定的能量谱斜率（如原型的 $k^{-1/3}$）和高斯滤波初始化涡量场。
  - `step(dt, nu)`: 执行单步 RK4 时间推进。包含 $\zeta \rightarrow \psi \rightarrow (u, v) \rightarrow \text{Nonlinear} \rightarrow \zeta_{new}$ 的完整计算流。
  - `get_grid_data()`: 导出当前帧的速度场或涡量场的网格数据（经纬度矩阵）。
- **渲染管线 (WebGL Pipeline):**
  - 生成一个 high-precision 的 3D 球体几何体（SphereGeometry）。
  - **Data Texture:** 将计算引擎输出的 $[N_{lat} \times N_{lon}]$ 数组转换为 DataTexture。
  - **Fragment Shader:** 读取纹理值，根据速度场模长 $|U| = \sqrt{u^2 + v^2}$ 或涡量值，应用色带（Colormap，如 RdBu_r）进行像素级着色。
- **主控循环 (Main Loop):**
  - 使用 `requestAnimationFrame` 同步渲染和计算逻辑。

### 2.2 伪谱法数学模型对应预计算项

- **拉普拉斯算子特征值:** $\nabla^2 Y_l^m = -l(l+1) Y_l^m$。即代码中的 `lap = -l*(l+1)`。
- **逆拉普拉斯算子:** $\nabla^{-2}$，即 `invlap = -1 / (l*(l+1))`，用于从涡量 $\zeta$ 求流函数 $\psi$。
- **非线性项:** $N(\zeta) = -\mathbf{u} \cdot \nabla \zeta = - (u_\theta \frac{1}{R} \frac{\partial \zeta}{\partial \theta} + u_\phi \frac{1}{R \sin\theta} \frac{\partial \zeta}{\partial \phi})$。在网格空间进行乘法，然后变换回谱空间，应用高斯滤波 `spec_filter` 消除混叠现象（De-aliasing）。

## 3. 详细测试计划

为了确保数值模拟的准确性和稳定性，必须进行严格的单元测试（针对纯数学和代码逻辑）、物理测试（针对流体力学特性）以及数值收敛性测试。

### 3.1 代码单元测试计划 (Unit Testing Plan)

框架建议：使用 Jest 测试 JS/WASM 接口调用，使用 C++ 原生测试框架（如 GoogleTest）测试编译前的核心计算逻辑。

| 测试用例 ID | 测试模块 | 测试内容描述 | 预期结果 (Pass Condition) |
| :--- | :--- | :--- | :--- |
| UT-01 | SHT 正逆变换 | 生成一个已知的球面解析函数（如 $f(\theta, \phi) = \cos(\theta)$），执行 Grid -> Spectral -> Grid 变换。 | 转换前后的 Grid 数据误差的 $L_2$ 范数趋近于机器精度（$< 10^{-12}$）。 |
| UT-02 | 拉普拉斯逆算子 | 给定解析分布 $\zeta = 2\cos(\theta)$，在谱空间乘以 invlap，反变换求 $\psi$ | 计算结果 $\psi$ 应严格等于解析解（误差 $< 10^{-12}$）。 |
| UT-03 | 球面梯度计算 | 测试 sht.synth_grad(zeta_coeffs)。给定特定的低阶球面调和函数，计算其在 $\theta$ 和 $\phi$ 方向的导数。 | 与理论导数场的数据比对，误差在合理阈值内。 |
| UT-04 | RK4 积分器 | 关闭非线性项（对流项设为 0），仅保留线性耗散项 $\frac{\partial \zeta}{\partial t} = \nu \nabla^2 \zeta$。给入初始场，演化 $T$ 时间。 | 涡量场的衰减率应严格匹配解析的热传导衰减解 $e^{-\nu l(l+1) t}$。 |
| UT-05 | WebGL 纹理映射 | 将包含已知极值（如赤道上为1，两极为0）的 Dummy Grid 数据传入 WebGL 渲染管线。 | Shader 输出的颜色应准确对应预设的 Colormap，极点无缝隙或撕裂异常（Pole Singularity 处理）。 |

### 3.2 物理特性测试计划 (Physical Testing Plan)

物理测试用于验证模拟器是否遵循流体动力学规律，这是计算流体力学（CFD）软件不可或缺的环节。

| 测试用例 ID | 物理测试项 | 初始条件与参数设置 | 验证标准与物理意义 |
| :--- | :--- | :--- | :--- |
| PT-01 | 动能守恒测试 (Inviscid Limit) | 设粘性系数 $\nu = 0$（或极小）。关闭高斯滤波。给入随机初始涡量。 | 标准: 系统总动能 $E = \sum \frac{1}{2}$ |
| PT-02 | 拟能衰减测试 (Enstrophy Cascade) | 设适中的粘性 $\nu$，开启高斯滤波。给入原型代码中的 $k^{-1/3}$ 随机能谱。 | 标准: 监测总动能 $E$ 和总拟能 $Z = \int \zeta^2 dA$。在 2D 球面湍流中，应观察到典型的“能量逆级联”（大涡变大）和“拟能正级联”（细小涡流被耗散），拟能 $Z$ 显著下降，但动能 $E$ 下降极其缓慢。 |
| PT-03 | 固体自转测试 (Solid Body Rotation) | 初始流函数设为 $\psi = -\omega \cos(\theta)$，代表流体围绕极轴以角速度 $\omega$ 刚体旋转。 | 标准: 随着时间推移，流场应保持绝对静止不变。任何涡量的产生或形态改变都表明计算格式存在非物理的数值误差。 |
| PT-04 | Rossby-Haurwitz 波测试 | 使用气象学中经典的 RH-Wave 初始条件（波数 $m=4$）。 | 标准: 涡量场应保持其初始的波浪形状，并在球面上以恒定的相速度向西平移，波形在几百个时间步内不应发生明显的畸变或破碎。这是验证球面 NS 求解器时空精度的黄金标准。 |

### 3.3 数值方法收敛性测试计划 (Numerical Convergence Testing Plan)

谱方法以其极高的精度著称，为验证我们在浏览器端实现的求解器依然保持了原有的数学严谨性，需执行以下收敛性（Convergence）测试。

| 测试用例 ID | 测试项 | 测试步骤与方法 | 验证标准 (Pass Condition) |
| :--- | :--- | :--- | :--- |
| CT-01 | 空间谱收敛性测试 (Spatial Spectral Convergence) | 给定一个足够平滑的初始场（例如低阶的 Rossby-Haurwitz 波）。固定一个极小的时间步长 $dt$，分别使用 $L_{max} = 16, 32, 64, 128$ 运行固定的物理时间 $T$。以 $L_{max}=256$ 的结果作为“精确解”，计算低分辨率下的 $L_2$ 误差。 | 标准: 绘制 $\log(Error)$ 与 $L_{max}$ 的关系图，应呈现出明显的线性下降趋势（即指数收敛），直到误差降至机器精度（约 $10^{-12}$）后趋于平缓。 |
| CT-02 | 频谱尾部衰减测试 (Energy Spectrum Tail) | 运行一段完全发展的湍流（如 PT-02），提取某一时刻的球面谐波系数 $a_{lm}$。 | 标准: 计算不同阶数 $l$ 的能量谱 $E(l)$。在耗散区（大 $l$ 值区域），能量谱应呈现指数级衰减，证明高频振荡被正确处理，没有发生数值混叠积聚（Aliasing pile-up）。 |
| CT-03 | 时间推进四阶收敛性 (Temporal RK4 Convergence) | 固定一个较高的分辨率（如 $L_{max}=128$ 以排除空间截断误差）。选用平滑流场，分别使用时间步长 $\Delta t, \Delta t/2, \Delta t/4, \Delta t/8$ 推进相同的时间 $T$。以 $\Delta t/16$ 的结果为参考解，计算各步长的 $L_2$ 误差。 | 标准: 计算收敛阶数 $p = \log_2(\frac{Error_{\Delta t}}{Error_{\Delta t/2}})$。结果 $p$ 应严格趋近于 4.0，证明实现的 Runge-Kutta 时间推进具有理论的四阶精度。 |

## 4. WebGL 可视化与交互设计

原型代码使用 Matplotlib 的 Orthographic 投影。在 WebGL 中，我们将直接在 3D 空间中渲染。

### 4.1 Shader 与分辨率解耦设计

为了达到高性能和最佳的视觉平滑度，物理求解分辨率与渲染分辨率应当解耦：

- **计算数据流:** WASM 根据当前求解分辨率（$L_{max}$）输出长度为 $N_{lat} \times N_{lon}$ 的 Float32Array 数组。
- **GPU 纹理映射:** 将其作为单通道浮点纹理 (RED 格式，FLOAT 类型) 绑定到 GPU。通过设置纹理过滤为 `gl.LINEAR`，即便在较低的求解分辨率下，GPU 也能利用双线性插值渲染出平滑的过渡色彩。
- **着色管线:**
  - **顶点着色器 (Vertex Shader):** 计算标准球体的 UV 坐标。
  - **片元着色器 (Fragment Shader):** 根据 UV 坐标采样上述数据纹理。读取到的变量（如速度模长）传入自定义的 Colormap 函数（如 RdBu_r 蓝到红的插值），输出最终像素颜色。

### 4.2 极点奇点处理 (Pole Singularity)

经纬度网格在南北极存在奇点（无数个经度收敛于一点）。在 Three.js 中构建球体时，需要注意 UV 映射在顶部和底部可能产生的视觉伪影（Artifacts）。建议在 Fragment Shader 中，当纬度接近 $\pm 90^\circ$ 时，进行微小的像素级平滑过滤。

### 4.3 用户交互面板 (GUI)

开发面板应包含以下可动态调节的参数（调节后实时反馈到引擎）：

- **仿真控制:**
  - **Play / Pause:** 控制时间演化。
  - **Time Step (dt):** 时间步长。如果过大，会触发 CFL 条件导致模拟爆炸。
  - **Viscosity (nu):** 运动粘性系数 $\nu$。
- **分辨率控制（解耦设计）:**
  - **Solver Resolution (L_max):** 求解分辨率（如 64, 128, 256, 512, ...）。控制底层谱方法截断阶数和物理网格 $N_{lat} \times N_{lon}$ 的大小。此参数直接影响计算耗时和物理精确度。注意：更改此项需要重置模拟状态（清空内存并重建 WASM 实例）。
  - **Visual Resolution:** 可视化（渲染）分辨率（如 128, 256, 512, 1024, ...）。控制 Three.js 中 SphereGeometry 的细分数（widthSegments / heightSegments）。此参数仅影响 3D 渲染的网格面数以控制 GPU 渲染负载，不改变底层流体数据。更改此项无缝切换，无需重置模拟。
- **可视化选项:**
  - **Color Variable:** 选择可视化变量（Velocity Magnitude 速度模长, Vorticity 涡量 $\zeta$, Streamfunction 流函数 $\psi$）。

## 5. 开发阶段里程碑

- **Phase 1: 核心算法移植与验证 (Weeks 1-2)**
  - 将 shtns 编译为 WASM 环境可用的库。
  - 在 JS/TS 端复现原型 Jupyter Notebook 中的数据流结构和 RK4 逻辑。
  - 完成 UT-01 到 UT-04，以及 CT-01 到 CT-03 (收敛性验证极其重要)。
- **Phase 2: 渲染管线搭建 (Week 3)**
  - 构建 Three.js 场景，实现 DataTexture 到 3D 球面的映射。
  - 实现物理求解数据生成与 GPU 纹理过滤解耦渲染，编写 Colormap Shader。
  - 完成 UT-05。
- **Phase 3: 物理测试与性能调优 (Week 4)**
  - 完成 PT-01 到 PT-04。
  - 将核心计算移入 Web Worker 防止主线程掉帧。
  - 优化数据传输瓶颈（尽可能减少 WASM Heap 与 JS Array 之间的复制）。
- **Phase 4: UI/UX 完善 (Week 5)**
  - 添加用户控制面板，接入对双重分辨率（Solver 与 Visual）的控制逻辑。
  - 添加预设场景（随机湍流、Rossby波等）。