# 球面谱方法 Navier-Stokes 方程 WebGL 可视化模拟器

## 项目概述

本项目旨在将 Python 环境下的球面谱方法 2D 湍流模拟器（基于 shtns）移植到 Web 端，实现一个无需安装任何后端环境、纯浏览器运行的交互式流体动力学模拟器。

原项目背景：Solving unstructured mesh on 3D sphere (S3). The solver is currently 1st-order accuracy. Supports: Acoustic wave equation, diffusion equation, linear advection equation, and incompressible Navier-Stokes equation.

### 技术栈选型

- **核心计算引擎 (Math Engine):** WebAssembly (WASM) + Web Workers。使用 Emscripten 将 C 语言编写的高性能球面谐波变换库编译为 WASM。将 RK4 积分和非线性项计算放在 Web Worker 中运行。
- **渲染引擎 (Render Engine):** WebGL 2.0 (结合 Three.js)。将 WASM 计算出的网格空间数据作为浮点纹理上传至 GPU，通过自定义 Shader 映射到 3D 球体模型上。
- **UI 与交互:** React + lil-gui。构建悬浮控制面板进行核心参数的实时调节。
- **依赖管理:** uv project manager + toml 进行 Python 依赖管理；npm 进行前端依赖管理。

## 系统架构设计

### 模块划分

- **模拟器内核 (WASM Module):** 负责初始化高斯网格、涡量场，执行单步 RK4 时间推进（包含非线性计算完整流），并导出网格数据。
- **渲染管线 (WebGL Pipeline):** 生成高精度 3D 球体，将网格数据转换为 DataTexture，在 Fragment Shader 中应用色带进行像素级着色。
- **主控循环 (Main Loop):** 使用 `requestAnimationFrame` 同步渲染和计算逻辑。

## WebGL 可视化与交互设计

- **分辨率解耦:** 物理求解分辨率 ($L_{max}$) 与渲染（视觉）分辨率解耦，以达到高性能和最佳视觉平滑度。
- **极点奇点处理:** 在 Fragment Shader 中对两极区域进行像素级平滑过滤。
- **用户交互面板 (GUI):** 支持动态调节时间步长 (dt)、粘性系数 (nu)、求解分辨率、渲染分辨率和可视化变量（速度、涡量等）。

## 开发里程碑

- **Phase 1: 核心算法移植与验证**
  - 将 shtns 编译为 WASM 环境可用的库，并在 JS/TS 端复现 RK4 逻辑。完成单元测试和收敛性验证。
- **Phase 2: 渲染管线搭建**
  - 构建 Three.js 场景，实现 DataTexture 到 3D 球面的映射。编写 Colormap Shader。
- **Phase 3: 物理测试与性能调优**
  - 完成物理流体特性测试。将核心计算移入 Web Worker 防止主线程掉帧。
- **Phase 4: UI/UX 完善**
  - 添加用户控制面板，接入对双重分辨率的控制逻辑。添加预设场景。