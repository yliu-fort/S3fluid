# WebGL 球面谱方法 NS 流体模拟器

## Phase 0: Planning & Architecture
- [ ] Read program.md and understand requirements
- [ ] Examine existing Python prototype notebook
- [ ] Create implementation plan

## Phase 1: Pure-JS Spherical Spectral Solver (No WASM)
> Strategy: Since SHTns has no pre-built WASM binary, implement SHT in pure JS using associated Legendre polynomials for a self-contained demo that runs fully in the browser without compilation.
- [ ] Implement spherical harmonics transform (SHT) in JavaScript
  - [ ] Gauss-Legendre quadrature nodes & weights
  - [ ] Associated Legendre polynomials P_l^m
  - [ ] Forward transform (grid → spectral)
  - [ ] Inverse transform / gradient synthesis
- [ ] Implement NS solver core (RK4 + nonlinear advection)
- [ ] Implement spectral filter (Gaussian)

## Phase 2: WebGL Rendering Pipeline
- [ ] Set up Three.js scene with high-res SphereGeometry
- [ ] Implement DataTexture upload from Float32Array
- [ ] Write fragment shader with RdBu_r colormap
- [ ] Handle pole singularity in UV mapping

## Phase 3: Main Loop & UI
- [ ] requestAnimationFrame loop coordinating solver and renderer
- [ ] lil-gui control panel (play/pause, dt, nu, Lmax, visual resolution, color variable)
- [ ] Web Worker offloading for solver computation

## Phase 4: Testing & Polish
- [ ] Verify solid-body rotation stability (PT-03)
- [ ] Verify energy conservation behavior
- [ ] Responsive layout, loading state
- [ ] Update README.md with project description and usage
