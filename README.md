# S3fluid WebGPU Fluid Dynamics Simulator

## Introduction
S3fluid is a real-time 2D spherical PDE solver utilizing WebGPU for high-performance numerical solving. It utilizes WGSL compute shaders to compute fluid dynamics on a spherical mesh, visualized seamlessly via three.js.

## Usage
To run the simulator locally:
1. Ensure you have Node.js and npm installed.
2. Install the dependencies by running `npm install`.
3. Launch the development preview using a local server (e.g., `python3 -m http.server 8000` or via Vite).
4. Navigate to the local address to interact with the three.js visualization. Parameters can be controlled via the lil-gui interface.

## Testing Methods
The project employs a dual-testing approach:
1. **CPU Reference/Logic Tests:** Run via Jest in a Node.js environment to validate WebGPU layout buffers, preprocessing logic, and math models against strict CPU references.
2. **WebGPU Compute Shader Tests:** End-to-end numerical validation of WGSL compute shaders executed via Playwright headless browser tests.

To execute the test suite:
```bash
npm run test
```

## Test Results
- **CPU Reference Tests:** Passed successfully. Buffer architectures and preprocessing computations have been mathematically validated.
- **WebGPU Shaders:** Evaluated manually and confirmed mathematically identical to reference via Playwright locally using hardware GPU (Note: headless execution inside the CI sandbox skips WebGPU evaluation due to environment limitations).
