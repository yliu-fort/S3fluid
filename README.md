# S3Fluid — 球面谱方法 Navier-Stokes WebGL 可视化模拟器

**纯浏览器运行 · 无需安装 · 零后端依赖**

球面 2D 湍流实时仿真器，在 3D 球面上可视化 Navier-Stokes 方程的谱方法数值解。

## 快速开始

### 安装与运行

主程序为纯静态 Web 应用，无需打包构建，直接在浏览器中即可运行。项目目录下的 `package.json` 仅用于管理**开发与测试依赖**。

```bash
# 运行本地开发服务器
# 你可以使用任意静态文件服务器（必须通过 HTTP 加载，直接打开 index.html 因 CORS 限制无法使用）
npx serve .
# 或
python -m http.server 8080
# 然后打开 http://localhost:8080
```

### 运行测试 (可选)

本项目使用 Jest 进行单元测试，并依赖 `gl` (headless-gl) 来在 Node.js 中模拟 WebGL 上下文。

```bash
# 安装开发与测试依赖
npm install

# 运行测试
npm test
```

> **⚠️ Windows 系统安装测试依赖注意事项:**
>
> 在运行 `npm install` 安装 `gl` 测试包时，它需要编译原生 C++ 模块。如果你的系统报错 `Could not find any Visual Studio installation to use`，这说明你的电脑缺乏 C++ 编译环境。
>
> **解决办法：**
> 如果你只是想运行和体验项目，**完全不需要执行 npm install 或运行测试**，直接开启 HTTP 服务即可！
> 如果你必须运行测试，你需要以管理员身份运行 PowerShell 并安装构建工具：
> `npm install --global windows-build-tools` 或者手动安装 Visual Studio 并勾选 "使用 C++ 的桌面开发" 工作负载。

## 当前开发进度

**Phase 1 阶段已完成**：
- [x] 验证设备的 WebGL 2.0 浮点纹理和 MRT 支持度。
- [x] 搭建 Ping-Pong FBO 框架，实现基于纹理的简单局部差分计算（作为热身验证）。
- [x] 实现基于 `gl.readPixels` 的自动化测试抓手。

## 功能特性

| 功能 | 描述 |
|------|------|
| **谱方法求解器** | 纯 JS 球面谐波变换（SHT），高斯-勒让德积分，RK4 时间推进 |
| **WebGL 渲染** | Three.js 球体 + 自定义 Fragment Shader，4 种色带（RdBu_r / Viridis / Inferno / Plasma） |
| **Web Worker** | 数值计算完全在后台线程运行，不阻塞 UI |
| **实时控制** | 粘性系数 ν、时间步 dt、求解分辨率 Lmax、渲染分辨率独立调节 |
| **多种预设** | 随机湍流、刚体旋转（PT-03 验证）、Rossby-Haurwitz 波（m=4） |
| **可视化变量** | 涡量 ζ、速度模长 \|U\|、流函数 ψ |

## 技术架构

```
index.html          # 主入口：Three.js 场景、GUI、渲染循环
src/
  sht.js            # 球面谐波变换（高斯-勒让德 + 关联勒让德多项式）
  solver.js         # NS 求解器（RK4 + 非线性项 + 高斯滤波器）
  worker.js         # Web Worker 消息路由
examples/
  demo_spherical_spectral_turbulence.ipynb  # Python 原型（shtns + numpy）
program.md          # 项目设计文档与开发里程碑
```

## 数学模型

涡量-流函数形式的 2D 球面不可压缩 N-S 方程：

```
∂ζ/∂t = -u·∇ζ + ν∇²ζ
∇²ψ = ζ    (通过谱空间逆拉普拉斯)
u_φ = (1/sinθ) ∂ψ/∂φ,  u_θ = -∂ψ/∂θ
```

谱方法精度：球面谐波截断至 Lmax，网格点数 nlat = Lmax+1，nlon = 2(Lmax+1)。

## 参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| Lmax | 16–64 | 谱截断阶数（调大需重置模拟，JS 计算较慢）|
| dt | 0.05–2.0 | 时间步长（过大会触发 CFL 不稳定） |
| ν | 1e-7–1e-3 | 运动粘性系数 |
| 渲染精度 | 64–512 | Three.js 球体细分数（不影响物理） |

## 浏览器兼容性

需要支持 ES Module、WebGL 2.0、Web Workers 的现代浏览器（Chrome 80+、Firefox 75+、Edge 80+）。

## Python 原型

`examples/demo_spherical_spectral_turbulence.ipynb` 包含基于 [SHTns](https://users.isterre.fr/nschaeff/SHTns/) 的高性能 Python 实现，可作为 Web 版本的数学参考和精度基准。

依赖：`numpy`, `shtns`, `matplotlib`, `cartopy`, `tqdm`
