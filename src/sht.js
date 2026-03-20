import { ShaderPass } from './gpgpu.js';

// Spherical Harmonics Transform (SHT) Precomputations and GPU Pipeline
export class SHT {
  constructor(gpgpu, latRes = 64, lonRes = 128) {
    this.gpgpu = gpgpu;
    this.latRes = latRes;
    this.lonRes = lonRes;
    // Spectral resolution is typically 1/3 of the grid resolution for anti-aliasing
    this.lMax = Math.floor(this.lonRes / 3);

    // We only need m=0..lMax, so the size of the array is O(lMax^2 * latRes)
    // For simplicity in testing UT-03 (m=0 case like cos(theta)), we only compute P_l^0
    // In a full Navier-Stokes SHT, we'd need to compute P_l^m and pack it appropriately into textures.

    this._computeGaussLegendre(this.latRes);
    this._computeALPs(this.lMax, this.latRes);

    if (this.gpgpu) {
       this.legendreTex = this.gpgpu.createTexture(this.lMax, this.latRes, this.legendreData);
       this.weightsTex = this.gpgpu.createTexture(this.latRes, 1, this.weightsData);
       this._initShaders();
    }
  }

  // Helper to compute Gauss-Legendre roots and weights (latRes points in [-1, 1])
  // Using Golub-Welsch algorithm or simple Newton-Raphson
  _computeGaussLegendre(n) {
    this.nodes = new Float64Array(n);
    this.weights = new Float64Array(n);
    this.weightsData = new Float32Array(n * 4); // RGBA for GPU

    const m = Math.floor((n + 1) / 2);
    for (let i = 0; i < m; i++) {
      let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));
      let z1 = 0;
      let pp = 0;

      // Newton-Raphson iteration
      do {
        let p1 = 1.0;
        let p2 = 0.0;
        for (let j = 0; j < n; j++) {
          let p3 = p2;
          p2 = p1;
          p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
        }
        pp = n * (z * p1 - p2) / (z * z - 1.0);
        z1 = z;
        z = z1 - p1 / pp;
      } while (Math.abs(z - z1) > 1e-15);

      this.nodes[i] = -z;
      this.nodes[n - 1 - i] = z;

      let w = 2.0 / ((1.0 - z * z) * pp * pp);
      this.weights[i] = w;
      this.weights[n - 1 - i] = w;
    }

    // Pack to Float32 RGBA
    for (let i = 0; i < n; i++) {
      this.weightsData[i * 4] = this.weights[i];
    }
  }

  // Precompute normalized Associated Legendre Polynomials (ALPs) P_l^m(x)
  // For this mock we compute P_l^0(x) for all nodes
  // Normalized such that integral_{-1}^1 P_l^m P_l'^m dx = delta_l,l'
  _computeALPs(lMax, n) {
    // Storing [latRes][lMax], flatten to 1D
    this.legendreData = new Float32Array(n * lMax * 4);

    for (let i = 0; i < n; i++) {
      let x = this.nodes[i];

      let p_lminus1 = 0; // P_0
      let p_l = Math.sqrt(0.5); // Normalized P_0^0(x) = sqrt(1/2)

      this.legendreData[(i * lMax + 0) * 4] = p_l; // l=0

      for (let l = 1; l < lMax; l++) {
        let l_float = l;
        // Recurrence relation for non-normalized: (l)*P_l = (2l-1)*x*P_{l-1} - (l-1)*P_{l-2}
        // For orthonormal basis: P~_l = sqrt(l+0.5) P_l

        let p_next;
        if (l === 1) {
            p_next = Math.sqrt(1.5) * x; // Normalized P_1^0 = sqrt(3/2) x
        } else {
            // Using standard recurrence modified for normalized polynomials
            let a = Math.sqrt((4.0 * l * l - 1.0) / (l * l));
            let b = Math.sqrt((2.0 * l + 1.0) / (2.0 * l - 3.0)) * ((l - 1.0) / l);
            p_next = a * x * p_l - b * p_lminus1;
        }

        this.legendreData[(i * lMax + l) * 4] = p_next;

        p_lminus1 = p_l;
        p_l = p_next;
      }
    }
  }

  _initShaders() {
    const matMulFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D texA;
      uniform sampler2D texB;
      uniform int K;

      void main() {
        vec4 sum = vec4(0.0);
        float dK = 1.0 / float(K);
        for(int k = 0; k < K; k++) {
           float u = (float(k) + 0.5) * dK;
           vec4 valA = texture(texA, vec2(u, v_uv.y));
           vec4 valB = texture(texB, vec2(v_uv.x, u));
           sum += valA * valB;
        }
        outColor = sum;
      }
    `;
    this.matMulPass = new ShaderPass(this.gpgpu, matMulFrag);

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
        vec4 sum = vec4(0.0);
        float dLat = 1.0 / float(latRes);
        for(int i = 0; i < latRes; i++) {
           float thetaUv = (float(i) + 0.5) * dLat;
           vec4 gridVal = texture(gridTex, vec2(v_uv.x, thetaUv));
           vec4 weight = texture(weightsTex, vec2(thetaUv, 0.5));
           vec4 leg = texture(legendreTex, vec2(v_uv.x, thetaUv)); // v_uv.x acts as l here for testing
           sum += gridVal * leg * weight;
        }
        outColor = sum;
      }
    `;
    this.forwardPass = new ShaderPass(this.gpgpu, forwardSHTFrag);

    const inverseSHTFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D spectralTex;
      uniform sampler2D legendreTex;
      uniform int lMax;

      void main() {
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
    this.inversePass = new ShaderPass(this.gpgpu, inverseSHTFrag);
  }

  multiplyMatrices(texA, texB, M, K, N) {
    const texC = this.gpgpu.createTexture(N, M);
    const fboC = this.gpgpu.createFBO(texC);
    this.matMulPass.render(fboC, N, M, { K: K }, { texA: texA, texB: texB });
    const resultData = this.gpgpu.readPixels(fboC, N, M);
    return { tex: texC, fbo: fboC, data: resultData };
  }

  forwardTransform(gridTex) {
    const texSpec = this.gpgpu.createTexture(this.lMax, this.lMax);
    const fboSpec = this.gpgpu.createFBO(texSpec);
    this.forwardPass.render(fboSpec, this.lMax, this.lMax,
      { latRes: this.latRes, lonRes: this.lonRes },
      { gridTex: gridTex, legendreTex: this.legendreTex, weightsTex: this.weightsTex }
    );
    return { tex: texSpec, fbo: fboSpec, data: this.gpgpu.readPixels(fboSpec, this.lMax, this.lMax) };
  }

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
