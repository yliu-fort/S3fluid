import { ShaderPass } from './gpgpu.js';

// Precompute Gaussian Quadrature weights and Legendre polynomials for Spherical Harmonics Transform (SHT)
// The GPGPU SHT requires matrix multiplication:
// Spectral = Forward(Grid) -> Integral over sphere
// Grid = Inverse(Spectral) -> Summation over harmonics

export class SHT {
  constructor(gpgpu, latRes = 64, lonRes = 128) {
    this.gpgpu = gpgpu;
    this.latRes = latRes;
    this.lonRes = lonRes;
    // Spectral resolution is typically 1/3 of the grid resolution for anti-aliasing (de-aliasing rule)
    this.lMax = Math.floor(this.lonRes / 3);

    // Prepare Legendre Polynomial Textures (Mock implementation for now)
    // Normally, we compute P_l^m(cos theta) and Gaussian weights on CPU and upload
    this.legendreData = new Float32Array(this.latRes * this.lMax * 4); // Example size
    this.weightsData = new Float32Array(this.latRes * 4);

    this.legendreTex = this.gpgpu.createTexture(this.lMax, this.latRes, this.legendreData);
    this.weightsTex = this.gpgpu.createTexture(this.latRes, 1, this.weightsData);

    // UT-02: Generic Matrix Multiplication Shader
    // C = A * B
    // A: texA, B: texB
    // We assume A is MxK, B is KxN, C is MxN
    // In GPGPU, C's dimensions are given by gl.viewport.
    const matMulFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D texA;
      uniform sampler2D texB;
      uniform int K;

      void main() {
        // v_uv gives the (x, y) in [0, 1] for C
        // corresponding to column x of B and row y of A
        vec4 sum = vec4(0.0);
        float dK = 1.0 / float(K);
        for(int k = 0; k < K; k++) {
           float u = (float(k) + 0.5) * dK;
           vec4 valA = texture(texA, vec2(u, v_uv.y));
           vec4 valB = texture(texB, vec2(v_uv.x, u));
           // For simplicity, we just do single channel multiply
           sum += valA * valB;
        }
        outColor = sum;
      }
    `;
    this.matMulPass = new ShaderPass(gpgpu, matMulFrag);

    // UT-03: SHT Forward Transform Shader (Grid -> Spectral)
    // zeta_lm = sum_theta sum_phi zeta(theta, phi) Y_lm(theta, phi) sin(theta) dtheta dphi
    // Here we simplify as a generic shader that mimics the process.
    const forwardSHTFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D gridTex;
      uniform sampler2D legendreTex;
      uniform sampler2D weightsTex;
      uniform int latRes;
      uniform int lonRes;

      void main() {
        // For UT-03 testing, we just output a mock transformation
        // In actual implementation, this integrates over gridTex using weightsTex
        vec4 sum = vec4(0.0);
        float dLat = 1.0 / float(latRes);
        for(int i = 0; i < latRes; i++) {
           float thetaUv = (float(i) + 0.5) * dLat;
           vec4 gridVal = texture(gridTex, vec2(v_uv.x, thetaUv)); // Simplified lookup
           vec4 weight = texture(weightsTex, vec2(thetaUv, 0.5));
           vec4 leg = texture(legendreTex, vec2(v_uv.x, thetaUv));
           sum += gridVal * leg * weight;
        }
        outColor = sum;
      }
    `;
    this.forwardPass = new ShaderPass(gpgpu, forwardSHTFrag);

    // UT-03: SHT Inverse Transform Shader (Spectral -> Grid)
    const inverseSHTFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D spectralTex;
      uniform sampler2D legendreTex;
      uniform int lMax;

      void main() {
        // Mock inverse transformation
        vec4 sum = vec4(0.0);
        float dL = 1.0 / float(lMax);
        for(int l = 0; l < lMax; l++) {
           float lUv = (float(l) + 0.5) * dL;
           vec4 specVal = texture(spectralTex, vec2(v_uv.x, lUv));
           vec4 leg = texture(legendreTex, vec2(lUv, v_uv.y));
           sum += specVal * leg;
        }
        outColor = sum;
      }
    `;
    this.inversePass = new ShaderPass(gpgpu, inverseSHTFrag);
  }

  // UT-02 Helper
  multiplyMatrices(texA, texB, M, K, N) {
    const texC = this.gpgpu.createTexture(N, M);
    const fboC = this.gpgpu.createFBO(texC);

    this.matMulPass.render(fboC, N, M, { K: K }, { texA: texA, texB: texB });

    const resultData = this.gpgpu.readPixels(fboC, N, M);
    return { tex: texC, fbo: fboC, data: resultData };
  }

  // UT-03 Helper: Forward
  forwardTransform(gridTex) {
    const texSpec = this.gpgpu.createTexture(this.lMax, this.lMax); // Assuming square spectral space
    const fboSpec = this.gpgpu.createFBO(texSpec);

    this.forwardPass.render(fboSpec, this.lMax, this.lMax,
      { latRes: this.latRes, lonRes: this.lonRes },
      { gridTex: gridTex, legendreTex: this.legendreTex, weightsTex: this.weightsTex }
    );

    return { tex: texSpec, fbo: fboSpec, data: this.gpgpu.readPixels(fboSpec, this.lMax, this.lMax) };
  }

  // UT-03 Helper: Inverse
  inverseTransform(spectralTex) {
    const texGrid = this.gpgpu.createTexture(this.lonRes, this.latRes);
    const fboGrid = this.gpgpu.createFBO(texGrid);

    this.inversePass.render(fboGrid, this.lonRes, this.latRes,
      { lMax: this.lMax },
      { spectralTex: spectralTex, legendreTex: this.legendreTex }
    );

    return { tex: texGrid, fbo: fboGrid, data: this.gpgpu.readPixels(fboGrid, this.lonRes, this.latRes) };
  }
}
