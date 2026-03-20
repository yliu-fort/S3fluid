import { ShaderPass } from './gpgpu.js';

// Spherical Harmonics Transform (SHT) Precomputations and GPU Pipeline
export class SHT {
  constructor(gpgpu, latRes = 64, lonRes = 128) {
    this.gpgpu = gpgpu;
    this.latRes = latRes;
    this.lonRes = lonRes;
    // Spectral resolution is typically 1/3 of the grid resolution for anti-aliasing
    this.lMax = Math.floor(this.lonRes / 3);
    this.mMax = this.lMax;

    // We need m=0..lMax, l=m..lMax, so the size of the array is O(lMax^2 * latRes)
    // We compute P_l^m(x)
    this._computeGaussLegendre(this.latRes);
    this._computeALPs(this.lMax, this.latRes);

    if (this.gpgpu) {
       // Create a 2D texture where width=LMax*MMax, height=latRes
       // Since LMax*MMax can be large, we pack (l, m) into 1D index
       // But LMax=42, 42*42=1764, well within WebGL max texture size 2048 or 4096.
       // However, to be safe, we just use a 2D texture where each row is a latitude,
       // and columns represent a flat 1D array of (l, m) pairs.
       // Number of valid pairs: m=0..lMax-1, l=m..lMax-1.
       // Size is lMax * lMax / 2 approx. Let's just allocate lMax * lMax to simplify indexing in shader.

       this.legendreTex = this.gpgpu.createTexture(this.lMax * this.mMax, this.latRes, this.legendreData);
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
  // Normalized such that integral_{-1}^1 P_l^m P_l'^m dx = delta_l,l'
  _computeALPs(lMax, n) {
    let mMax = lMax;
    // Flattened array: size = n * (lMax * mMax) * 4
    this.legendreData = new Float32Array(n * (lMax * mMax) * 4);

    for (let i = 0; i < n; i++) {
      let x = this.nodes[i];
      let y = Math.sqrt(Math.max(0.0, 1.0 - x * x)); // sin(theta)

      let p_mm = Math.sqrt(0.5); // P_0^0
      for (let m = 0; m < mMax; m++) {
        // Compute P_m^m
        // Formula: P_m^m = - P_{m-1}^{m-1} * y * sqrt((2m+1)/(2m))
        // Wait, m=0 is already P_0^0 = sqrt(0.5).

        let val_mm = p_mm;

        // Write P_m^m
        let idx_mm = i * (lMax * mMax) + (m * lMax + m);
        this.legendreData[idx_mm * 4] = val_mm;

        if (m < lMax - 1) {
            // Compute P_{m+1}^m
            // P_{m+1}^m = x * sqrt(2m+3) * P_m^m
            let val_mp1 = x * Math.sqrt(2.0 * m + 3.0) * val_mm;
            let idx_mp1 = i * (lMax * mMax) + (m * lMax + (m + 1));
            this.legendreData[idx_mp1 * 4] = val_mp1;

            // Compute P_l^m for l > m+1
            let p_lminus2 = val_mm;
            let p_lminus1 = val_mp1;

            for (let l = m + 2; l < lMax; l++) {
                let a = Math.sqrt(((4.0 * l * l - 1.0) / (l * l - m * m)));
                let b = Math.sqrt(((2.0 * l + 1.0) * ((l - 1.0) * (l - 1.0) - m * m)) / ((2.0 * l - 3.0) * (l * l - m * m)));

                let p_l = a * x * p_lminus1 - b * p_lminus2;
                let idx_l = i * (lMax * mMax) + (m * lMax + l);
                this.legendreData[idx_l * 4] = p_l;

                p_lminus2 = p_lminus1;
                p_lminus1 = p_l;
            }
        }

        // Prepare for next m (P_{m+1}^{m+1})
        // P_{m+1}^{m+1} = - y * sqrt((2m+3)/(2m+2)) * P_m^m
        p_mm = -y * Math.sqrt((2.0 * m + 3.0) / (2.0 * m + 2.0)) * p_mm;
      }
    }
  }

  _initShaders() {
    // Passes for full 2D transform

    // 1. Forward Fourier (Longitude)
    // gridTex: (lonRes, latRes) - Real grid values in Red channel
    // Output: fourierTex: (mMax, latRes) - Complex Fourier coefficients (Re: Red, Im: Green)
    const forwardFourierFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D gridTex;
      uniform int lonRes;
      uniform int mMax;

      const float PI = 3.1415926535897932384626433832795;

      void main() {
        int m = int(floor(v_uv.x * float(mMax)));
        float sumRe = 0.0;
        float sumIm = 0.0;
        float dLon = 2.0 * PI / float(lonRes);
        float du = 1.0 / float(lonRes);

        for(int j = 0; j < lonRes; j++) {
           float u = (float(j) + 0.5) * du;
           float phi = float(j) * dLon;
           float val = texture(gridTex, vec2(u, v_uv.y)).r;

           float angle = -float(m) * phi;
           sumRe += val * cos(angle);
           sumIm += val * sin(angle);
        }

        // Multiply by 2pi/N for continuous integral, standard DFT scaling
        float scale = dLon;
        outColor = vec4(sumRe * scale, sumIm * scale, 0.0, 0.0);
      }
    `;
    this.forwardFourierPass = new ShaderPass(this.gpgpu, forwardFourierFrag);

    // 2. Forward Legendre (Latitude)
    // fourierTex: (mMax, latRes)
    // Output: spectralTex: (lMax, mMax) - Complex spectral coefficients (Re: Red, Im: Green)
    const forwardLegendreFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D fourierTex;
      uniform sampler2D legendreTex;
      uniform sampler2D weightsTex;
      uniform int latRes;
      uniform int lMax;
      uniform int mMax;

      void main() {
        int m = int(floor(v_uv.x * float(mMax)));
        int l = int(floor(v_uv.y * float(lMax)));

        if (l < m) {
            outColor = vec4(0.0);
            return;
        }

        float sumRe = 0.0;
        float sumIm = 0.0;
        float dLat = 1.0 / float(latRes);

        // The legendreTex is 2D: width = lMax*mMax, height = latRes
        float dL_LM = 1.0 / float(lMax * mMax);
        float u_lm = (float(m * lMax + l) + 0.5) * dL_LM;

        for(int i = 0; i < latRes; i++) {
           float v_theta = (float(i) + 0.5) * dLat;

           vec2 fourierUv = vec2((float(m) + 0.5) / float(mMax), v_theta);
           vec2 fm = texture(fourierTex, fourierUv).rg;

           float weight = texture(weightsTex, vec2(v_theta, 0.5)).r;
           float plm = texture(legendreTex, vec2(u_lm, v_theta)).r;

           sumRe += fm.r * plm * weight;
           sumIm += fm.g * plm * weight;
        }

        // SHT integration factor: 1/2 from integral over cos(theta) = [-1, 1], and 1/(2pi) from phi integral
        // Actually, the integral is simply sum_i w_i F_m(theta_i) P_l^m(cos theta_i)
        // Normalization is handled in P_l^m.
        // Need to divide by 2*PI since Fourier step multiplied by dLon. Wait, integral is int_0^2pi int_-1^1 f Y_l^m d(cos t) dphi
        // Fourier sum computed int f e^{-im phi} dphi
        // Then sum computed int F_m P_l^m d(cos t)
        // So just a factor of 1/(2pi) to get standard coefficient?
        // Let's not add constant factors if we balance them in inverse transform.
        // Actually, orthonormal basis Y_l^m implies:
        // f(theta, phi) = sum_l sum_m f_l^m P_l^m(cos t) e^{im phi}
        // Then f_l^m = 1/(2pi) int_0^2pi int_-1^1 f(theta, phi) P_l^m(cos t) e^{-im phi} d(cos t) dphi
        float scale = 1.0 / (2.0 * 3.141592653589793);
        outColor = vec4(sumRe * scale, sumIm * scale, 0.0, 0.0);
      }
    `;
    this.forwardLegendrePass = new ShaderPass(this.gpgpu, forwardLegendreFrag);

    // 3. Inverse Legendre (Latitude)
    // spectralTex: (lMax, mMax)
    // Output: fourierTex: (mMax, latRes)
    const inverseLegendreFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D spectralTex;
      uniform sampler2D legendreTex;
      uniform int lMax;
      uniform int mMax;

      void main() {
        int m = int(floor(v_uv.x * float(mMax)));

        float sumRe = 0.0;
        float sumIm = 0.0;
        float dL = 1.0 / float(lMax);
        float dL_LM = 1.0 / float(lMax * mMax);

        for(int l = m; l < lMax; l++) {
           vec2 specUv = vec2((float(m) + 0.5) / float(mMax), (float(l) + 0.5) * dL);
           vec2 flm = texture(spectralTex, specUv).rg;

           float u_lm = (float(m * lMax + l) + 0.5) * dL_LM;
           float plm = texture(legendreTex, vec2(u_lm, v_uv.y)).r;

           sumRe += flm.r * plm;
           sumIm += flm.g * plm;
        }

        outColor = vec4(sumRe, sumIm, 0.0, 0.0);
      }
    `;
    this.inverseLegendrePass = new ShaderPass(this.gpgpu, inverseLegendreFrag);

    // 4. Inverse Fourier (Longitude)
    // fourierTex: (mMax, latRes)
    // Output: gridTex: (lonRes, latRes)
    const inverseFourierFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;
      uniform sampler2D fourierTex;
      uniform int mMax;
      uniform int lonRes;

      const float PI = 3.1415926535897932384626433832795;

      void main() {
        int j = int(floor(v_uv.x * float(lonRes)));
        float phi = float(j) * (2.0 * PI / float(lonRes));

        float sumRe = 0.0;
        float dm = 1.0 / float(mMax);

        for(int m = 0; m < mMax; m++) {
           vec2 fourierUv = vec2((float(m) + 0.5) * dm, v_uv.y);
           vec2 fm = texture(fourierTex, fourierUv).rg;

           float angle = float(m) * phi;
           float term = fm.r * cos(angle) - fm.g * sin(angle);

           if (m > 0) {
               // Since real function, F_{-m} = F_m^*, we add 2 * Re(F_m e^{im phi})
               sumRe += 2.0 * term;
           } else {
               sumRe += term;
           }
        }

        outColor = vec4(sumRe, 0.0, 0.0, 0.0);
      }
    `;
    this.inverseFourierPass = new ShaderPass(this.gpgpu, inverseFourierFrag);

    // For backwards compatibility in testing (e.g. UT-02)
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
  }

  multiplyMatrices(texA, texB, M, K, N) {
    const texC = this.gpgpu.createTexture(N, M);
    const fboC = this.gpgpu.createFBO(texC);
    this.matMulPass.render(fboC, N, M, { K: K }, { texA: texA, texB: texB });
    const resultData = this.gpgpu.readPixels(fboC, N, M);
    return { tex: texC, fbo: fboC, data: resultData };
  }

  forwardTransform(gridTex) {
    const texFourier = this.gpgpu.createTexture(this.mMax, this.latRes);
    const fboFourier = this.gpgpu.createFBO(texFourier);
    this.forwardFourierPass.render(fboFourier, this.mMax, this.latRes,
      { lonRes: this.lonRes, mMax: this.mMax },
      { gridTex: gridTex }
    );

    const texSpec = this.gpgpu.createTexture(this.mMax, this.lMax);
    const fboSpec = this.gpgpu.createFBO(texSpec);
    this.forwardLegendrePass.render(fboSpec, this.mMax, this.lMax,
      { latRes: this.latRes, lMax: this.lMax, mMax: this.mMax },
      { fourierTex: texFourier, legendreTex: this.legendreTex, weightsTex: this.weightsTex }
    );

    // Clean up intermediate FBO/Tex
    this.gpgpu.gl.deleteFramebuffer(fboFourier);
    this.gpgpu.gl.deleteTexture(texFourier);

    return { tex: texSpec, fbo: fboSpec, data: this.gpgpu.readPixels(fboSpec, this.mMax, this.lMax) };
  }

  inverseTransform(spectralTex) {
    const texFourier = this.gpgpu.createTexture(this.mMax, this.latRes);
    const fboFourier = this.gpgpu.createFBO(texFourier);
    this.inverseLegendrePass.render(fboFourier, this.mMax, this.latRes,
      { lMax: this.lMax, mMax: this.mMax },
      { spectralTex: spectralTex, legendreTex: this.legendreTex }
    );

    const texGrid = this.gpgpu.createTexture(this.lonRes, this.latRes);
    const fboGrid = this.gpgpu.createFBO(texGrid);
    this.inverseFourierPass.render(fboGrid, this.lonRes, this.latRes,
      { mMax: this.mMax, lonRes: this.lonRes },
      { fourierTex: texFourier }
    );

    // Clean up intermediate FBO/Tex
    this.gpgpu.gl.deleteFramebuffer(fboFourier);
    this.gpgpu.gl.deleteTexture(texFourier);

    return { tex: texGrid, fbo: fboGrid, data: this.gpgpu.readPixels(fboGrid, this.lonRes, this.latRes) };
  }
}
