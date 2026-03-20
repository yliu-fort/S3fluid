# WebGL 球面谱方法 NS 流体模拟器

## Phase 0: Planning & Architecture
- [x] Read program.md and understand requirements
- [x] Examine existing Python prototype
- [x] Create implementation plan

## Phase 1: Pure-JS Spherical Spectral Solver (No WASM)
> Strategy: Since SHTns has no pre-built WASM binary, implement SHT in pure JS using associated Legendre polynomials for a self-contained demo that runs fully in the browser without compilation.
- [x] Implement spherical harmonics transform (SHT) in JavaScript
  - [x] Gauss-Legendre quadrature nodes & weights
  - [x] Associated Legendre polynomials P_l^m
  - [x] Forward transform (grid → spectral)
  - [x] Inverse transform / gradient synthesis
  - [x] Comprehensive unit tests for each function
- [x] Implement NS solver core (RK4 + nonlinear advection)
- [x] Implement spectral filter (Gaussian)

## Phase 2: WebGL Rendering Pipeline
- [x] Set up Three.js scene with high-res SphereGeometry
- [x] Implement DataTexture upload from Float32Array
- [x] Write fragment shader with RdBu_r colormap
- [x] Handle pole singularity in UV mapping

## Phase 3: Main Loop & UI
- [x] requestAnimationFrame loop coordinating solver and renderer
- [x] lil-gui control panel (play/pause, dt, nu, Lmax, visual resolution, color variable)
- [x] Web Worker offloading for solver computation

## Phase 4: Testing & Polish
- [x] Verify solid-body rotation stability (PT-03)
- [x] Verify energy conservation behavior
- [x] Responsive layout, loading state
- [x] Update README.md with project description and usage
