# Spherical Mesh Solver & WebGL Visualizer

Solving unstructured mesh on 3D sphere (S3).
The solver is currently 1st-order accuracy.
Supports: Acoustic wave equation, diffusion equation, linear advection equation, and incompressible Navier-Stokes equation.

## WebGL Visualizer Porting Project

This project is currently being ported to an interactive WebGL visualizer running in-browser via React, Three.js, and WASM, enabling a pure browser-based interactive fluid dynamics simulator without backend environment dependencies.

### Architecture

- **Math Engine (WASM + Web Workers):** Uses Emscripten to compile high-performance spherical harmonic transform libraries (e.g., SHTns) to WASM. RK4 integration and nonlinear term calculations run in Web Workers.
- **Render Engine (WebGL 2.0 / Three.js):** Maps grid space data (velocity field, vorticity) as Float Textures to a 3D sphere using custom shaders.
- **UI & Interaction (React):** Uses React for application state management and component-based development, alongside lil-gui for real-time parameter tuning.
- **Dependency Management:** Managed using `uv` and `pyproject.toml`.

### Development Milestones

- **Phase 1: Core Algorithm Porting & Validation (Weeks 1-2)**
  - Compile `shtns` to WASM.
  - Replicate prototype data flow and RK4 logic in JS/TS.
  - Perform unit testing and convergence testing.
- **Phase 2: Render Pipeline Setup (Week 3)**
  - Build Three.js scene and implement DataTexture mapping.
  - Decouple physics solver resolution and visual rendering resolution.
- **Phase 3: Physics Testing & Performance Tuning (Week 4)**
  - Validate physical invariants (Kinetic Energy, Enstrophy).
  - Move core compute to Web Workers to prevent UI blocking.
- **Phase 4: UI/UX Completion (Week 5)**
  - Add user control panels.
  - Implement preset scenes (e.g., random turbulence, Rossby waves).

## Tests
Testing is handled via `pytest`. Tests are located in `examples/tests/`.
