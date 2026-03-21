# S3Fluid — 球面谱方法 Navier-Stokes WebGL 可视化模拟器
## Development Progress
- [x] Phase 1: Infrastructure and Custom NumPy-SHT implementation.
      Includes `NumPySHT` with forward, inverse, and inverse-gradient SHT functions perfectly passing unit tests (testing against $L_2$ errors < $1e^{-10}$).
- [x] Phase 2: GPU Arrays & Kivy Shader mapping.
      Implemented GL_RGBA32F mapping utilizing ctypes. Translated matrix operations to `forward_sht.glsl`, `inverse_sht.glsl`, and `inverse_sht_grad.glsl` via accurate `texelFetch` sampling.
- [x] Phase 3: Physical Module Integration & Pluggable Integrator.
      Implemented specific physical Shader logic (e.g. nonlinear convective term `convection.glsl`) and pure Python RHS `NavierStokesRHS` for baseline comparison.
      Implemented pluggable RK3/RK4 time-stepping state machine decoupled from pure physics.
- [x] Phase 4: Rendering & Dynamic Res Switching.
      Implemented 3D Isomorphic Mesh rendering with accurate Gaussian grid texel mapping using Kivy Mesh & RenderContext. Integrated dynamic resolution switching logic involving VRAM Teardown, NumPy-assisted spectral zero-padding, and FBO rebuilding. Implemented real-time interactive UI with CFL calculation.

## How to Test
Execute tests using the `uv` environment with `pytest`:
```bash
# Execute within XVFB for headless execution
PYTHONPATH=$(pwd) xvfb-run uv run pytest tests/
```
