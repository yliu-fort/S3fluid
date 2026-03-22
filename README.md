# S3Fluid — 球面谱方法 Navier-Stokes WebGL 可视化模拟器

## 简介
S3Fluid 是一个基于浏览器端技术构建的二维球面流体偏微分方程 (PDE) 实时模拟器与可视化工具。项目在 GPU 上使用 WebGPU 的 Compute Shader 进行核心的偏微分方程求解运算（包括球面上的流体动力学、RK4 时间步进、显式的高斯-勒让德球谐波谱方法、以及使用 FFT 算法等）。渲染端利用 Three.js 完成球面网格与动态标量场的可视化映射。它能够在网页内提供平滑、实时的球面涡度演化和数值诊断。

## 使用说明
要运行此项目，您需要 Node.js 环境（推荐版本 18+）。由于项目依赖 WebGPU 计算，请使用支持 WebGPU 的最新版浏览器（例如 Chrome 113+ 或 Edge 113+）。

1. **安装依赖**
   在项目根目录下执行：
   ```bash
   npm install
   ```
2. **启动本地开发服务器**
   执行以下命令以启动 Vite 服务器：
   ```bash
   npm run dev
   ```
   然后通过浏览器访问输出的本地地址（通常是 http://localhost:5173/ ）。
3. **交互界面 (GUI)**
   在页面的控制面板中，您可以实时修改：
   - 模拟精度 `lmax`
   - 时间步长 `dt`
   - 扩散系数 `nu`
   - 谱滤波参数 `filterAlpha`、`filterOrder`
   - 显示比例及其他仿真控制（如暂停、重置）。

## 测试方法
项目的算法和组件使用了 Jest 测试框架进行单元测试与逻辑验证。CPU 端的数学和流体物理逻辑也包含了与 GPU shader 同步逻辑对齐的参考测试。

1. **运行测试套件**
   在项目根目录下执行：
   ```bash
   npm test
   ```
2. **构建测试**
   执行 TypeScript 构建校验：
   ```bash
   npm run build
   ```

## 测试结果
所有的 14 个测试用例和相关的单元测试已经全部通过，包括但不限于以下模块：
- CPU 参考模型验证 (`shtReference.test.ts`, `modelReference.test.ts`)
- WebGPU Buffer 内存布局分配与 Precompute 预计算模块
- 谱算子模块 (`spectralOperators.test.ts`)
- FFT 正逆变换模拟
- RK4 时间推进与 RHS 计算逻辑 (`rk4.test.ts`, `rhs.test.ts`)
- 显示渲染和 Colormap 测试 (`sphere.test.ts`, `colormap.test.ts`)

目前系统能够稳定在 `lmax=63` 分辨率下模拟流体演化，不出现极区异常或数值崩溃。