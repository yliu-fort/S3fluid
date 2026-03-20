# Spherical Spectral Method Navier-Stokes Solver

Solving unstructured mesh on 3D sphere (S3).
The solver is currently 1st-order accuracy.
Supports: Acoustic wave equation, diffusion equation, linear advection equation, and incompressible Navier-Stokes equation.

---

## WebGL Visualizer Development Plan (Work In Progress)

The project is currently being ported to the Web, creating an interactive fluid dynamics simulator running purely in the browser without any backend environment.

### 1. Technology Stack
* **Math Engine**: WebAssembly (WASM) + Web Workers. C-based Spherical Harmonic Transform library (SHTns) will be compiled to WASM. RK4 integration and non-linear computations will run in Web Workers.
* **Render Engine**: WebGL 2.0 (Three.js). Grid space data (velocity/vorticity) will be uploaded as float textures and mapped to a 3D sphere via custom shaders.
* **UI & Interaction**: React. Application state management and control panel for real-time parameter adjustments.

### 2. System Architecture
1. **WASM Module**: Handles initialization (`init_grid`, `init_field`), single-step RK4 time advancement (`step(dt, nu)`), and grid data export (`get_grid_data`).
2. **WebGL Pipeline**: Generates a 3D sphere, converts computational arrays to `DataTexture`, and applies a colormap (e.g., RdBu_r) via Fragment Shader based on velocity magnitude or vorticity.
3. **Main Loop**: Uses `requestAnimationFrame` to sync rendering and computation.

### 3. Testing Plans
Rigorous testing is planned to ensure numerical accuracy and stability:
* **Unit Testing**: SHT Transforms, Inverse Laplacian, Spherical Gradients, RK4 Integrator (Linear Dissipation), and WebGL Texture Mapping.
* **Physical Testing**: Inviscid Limit (Kinetic Energy Conservation), Enstrophy Cascade (Turbulence), Solid Body Rotation, and Rossby-Haurwitz Wave propagation.
* **Numerical Convergence**: Spatial Spectral Convergence (Exponential), Energy Spectrum Tail (Dissipation), and Temporal RK4 Convergence (4th Order).

### 4. WebGL UI & Interaction
* High-performance 3D rendering bypassing CPU drawing. Float32Array arrays will be mapped directly to GPU textures.
* Custom fragment shaders to handle pole singularities (smoothing artifacts near poles).
* Dynamic Control Panel allowing adjustments to Resolution ($l_{max}$), Time Step ($dt$), Viscosity ($\nu$), and Color Variables.
