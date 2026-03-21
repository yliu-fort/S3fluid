# kivy-gpgpu-pde — 基于 Python + Kivy 的球面谱方法非线性 PDE (GPGPU) 模拟器

## 1. 简介
本项目是一个基于 Python 桌面端/移动端跨平台框架 Kivy 的球面谱方法二维非线性 PDE（偏微分方程，包含湍流等）模拟器。通过 GPGPU 技术实现极高计算性能，并利用 OpenGL 实现从计算到渲染的“零拷贝”管线。流体状态全部存储在 GPU 显存（VRAM）中，计算阶段的输出纹理直接作为 Kivy 3D 网格的贴图材质进行渲染。

## 2. 依赖安装方法
本项目采用 `uv` 作为极速的 Python 包管理器和虚拟环境管理工具。

### 2.1 安装系统级依赖
在 Linux 环境下，编译 `shtns` 库需要依赖 `fftw3`：
```bash
sudo apt-get update
sudo apt-get install -y libfftw3-dev
```

### 2.2 安装 uv
如果未安装 `uv`，可通过官方脚本安装：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.3 初始化并同步环境
在项目根目录下执行以下命令，`uv` 将自动创建 `.venv` (Python 3.12) 并急速安装 `pyproject.toml` 中的所有依赖：
```bash
uv sync
```

## 3. 测试方法
本项目严格遵循测试驱动开发 (TDD) 原则。GPU 计算模块 (如 SHT 变换、拉普拉斯算子) 在实现后必须通过断言测试，与 CPU (`shtns` 和 NumPy) 的基准计算结果进行严苛校验。

要运行所有 GPGPU 单元测试（如 FBO 精度测试 UT-01 和预计算纹理测试 UT-02），请在激活的环境中执行：
```bash
uv run pytest tests/ -s
```
*注：测试脚本已配置环境变量以确保 Kivy 在无头的 SDL2 环境下安全运行。*

## 4. 运行方法
项目的主入口（暂未完全实现 3D 渲染，目前处于 Phase 1 阶段）将通过以下命令启动：
```bash
uv run python src/main.py
```
*(在后续阶段中，我们将完善 main.py 提供 UI 交互面板和 3D 球面网格渲染。)*
