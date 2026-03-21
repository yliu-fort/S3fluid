# S3Fluid — 球面谱方法 Navier-Stokes WebGL 可视化模拟器
## Development Progress
- [x] Phase 1: Infrastructure and Custom NumPy-SHT implementation.
      Includes `NumPySHT` with forward, inverse, and inverse-gradient SHT functions perfectly passing unit tests (testing against $L_2$ errors < $1e^{-10}$).
- [ ] Phase 2: GPU Arrays & Kivy Shader mapping.
- [ ] Phase 3: Physical Module Integration & Pluggable Integrator.
- [ ] Phase 4: Rendering & Dynamic Res Switching.

## How to Test
Execute tests using the `uv` environment with `pytest`:
```bash
# Execute within XVFB for headless execution
PYTHONPATH=$(pwd) xvfb-run uv run pytest tests/
```
