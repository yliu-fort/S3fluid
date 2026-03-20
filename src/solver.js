import { GPGPU, ShaderPass, PingPongFBO } from './gpgpu.js';
import { SHT } from './sht.js';

export class NSSolver {
  constructor(gpgpu, sht) {
    this.gpgpu = gpgpu;
    this.sht = sht;
    this.latRes = sht.latRes;
    this.lonRes = sht.lonRes;
    this.lMax = sht.lMax;
    this.mMax = sht.mMax;

    // Create textures for eigenvalues and predefined constants
    this._initConstants();

    // Create intermediate FBOs for grid operations
    // U and V (velocity fields)
    this.uTex = this.gpgpu.createTexture(this.lonRes, this.latRes);
    this.fboU = this.gpgpu.createFBO(this.uTex);

    this.vTex = this.gpgpu.createTexture(this.lonRes, this.latRes);
    this.fboV = this.gpgpu.createFBO(this.vTex);

    // Grad Zeta (dZeta/dx, dZeta/dy)
    this.dZetaDxTex = this.gpgpu.createTexture(this.lonRes, this.latRes);
    this.fboDZetaDx = this.gpgpu.createFBO(this.dZetaDxTex);

    this.dZetaDyTex = this.gpgpu.createTexture(this.lonRes, this.latRes);
    this.fboDZetaDy = this.gpgpu.createFBO(this.dZetaDyTex);

    // Nonlinear term grid texture
    this.nonlinearTex = this.gpgpu.createTexture(this.lonRes, this.latRes);
    this.fboNonlinear = this.gpgpu.createFBO(this.nonlinearTex);

    // Intermediate textures for velocity components and streamfunction
    this.psiGridTex = this.gpgpu.createTexture(this.lonRes, this.latRes);
    this.fboPsiGrid = this.gpgpu.createFBO(this.psiGridTex);

    this.velocityTex = this.gpgpu.createTexture(this.lonRes, this.latRes);
    this.fboVelocity = this.gpgpu.createFBO(this.velocityTex);

    this.gradZetaTex = this.gpgpu.createTexture(this.lonRes, this.latRes);
    this.fboGradZeta = this.gpgpu.createFBO(this.gradZetaTex);

    this._initShaders();
  }

  _initShaders() {
    // 1. Inverse Laplacian Shader (Spectral space)
    // zeta (vorticity spectral coefficients) -> psi (streamfunction spectral coefficients)
    // psi_l^m = zeta_l^m / (-l*(l+1))
    const inverseLaplacianFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;

      uniform sampler2D spectralTex;
      uniform sampler2D eigenTex;

      void main() {
        // Read zeta coefficient
        vec2 zeta_lm = texture(spectralTex, v_uv).rg;

        // Read inverse eigenvalue (stored in Green channel)
        float invEigen = texture(eigenTex, v_uv).g;

        outColor = vec4(zeta_lm.r * invEigen, zeta_lm.g * invEigen, 0.0, 0.0);
      }
    `;
    this.inverseLaplacianPass = new ShaderPass(this.gpgpu, inverseLaplacianFrag);

    // 2. Compute Derivatives Shader (Spectral space)
    // Given psi (or any field), compute its gradients
    // Note: velocity components on sphere are v_theta (southward), v_phi (eastward)
    // u (zonal, phi) = -d psi / d theta
    // v (meridional, theta) = (1/sin theta) d psi / d phi
    // In spectral space, d/d phi of Y_l^m is i*m * Y_l^m
    // For theta derivative, we need relations between P_l^m and P_{l+-1}^m, but
    // to keep the GPU pipeline simple and aligned with the plan's grid-space emphasis,
    // we can compute gradients in grid space via finite differences or exact spectral derivatives
    // Let's use simple finite differences in grid space for the velocities and non-linear terms first to build the pipeline,
    // as spectral derivatives in latitude are complex to pack into the simple texture multiply pass.

    // Grid-space velocities and derivatives from Streamfunction psi and Vorticity zeta
    const gridVelocityFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;

      uniform sampler2D psiTex;
      uniform int lonRes;
      uniform int latRes;

      const float PI = 3.1415926535897932384626433832795;

      void main() {
        float dLon = 2.0 * PI / float(lonRes);
        float dLat = PI / float(latRes);

        float d_u = 1.0 / float(lonRes);
        float d_v = 1.0 / float(latRes);

        float theta = v_uv.y * PI; // [0, PI]
        float sinTheta = sin(theta);

        // Prevent division by zero at poles
        sinTheta = max(sinTheta, 1e-4);

        // Central differences for psi
        // d psi / d theta (y direction)
        float psi_up = texture(psiTex, vec2(v_uv.x, min(v_uv.y + d_v, 1.0))).r;
        float psi_dn = texture(psiTex, vec2(v_uv.x, max(v_uv.y - d_v, 0.0))).r;
        float dPsi_dTheta = (psi_up - psi_dn) / (2.0 * dLat);

        // d psi / d phi (x direction)
        // Wrap around longitude
        float x_right = v_uv.x + d_u;
        if (x_right > 1.0) x_right -= 1.0;

        float x_left = v_uv.x - d_u;
        if (x_left < 0.0) x_left += 1.0;

        float psi_right = texture(psiTex, vec2(x_right, v_uv.y)).r;
        float psi_left = texture(psiTex, vec2(x_left, v_uv.y)).r;
        float dPsi_dPhi = (psi_right - psi_left) / (2.0 * dLon);

        // Velocity components
        // u (zonal, eastward) = -d psi / d theta
        // v (meridional, southward) = d psi / d phi / sin(theta)
        float u = -dPsi_dTheta;
        float v_vel = dPsi_dPhi / sinTheta;

        // Output: R: u, G: v
        outColor = vec4(u, v_vel, 0.0, 0.0);
      }
    `;
    this.gridVelocityPass = new ShaderPass(this.gpgpu, gridVelocityFrag);

    // Grid-space gradients of Zeta
    const gridGradZetaFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;

      uniform sampler2D zetaTex;
      uniform int lonRes;
      uniform int latRes;

      const float PI = 3.1415926535897932384626433832795;

      void main() {
        float dLon = 2.0 * PI / float(lonRes);
        float dLat = PI / float(latRes);

        float d_u = 1.0 / float(lonRes);
        float d_v = 1.0 / float(latRes);

        float theta = v_uv.y * PI;
        float sinTheta = sin(theta);
        sinTheta = max(sinTheta, 1e-4);

        // d zeta / d theta (y direction)
        float zeta_up = texture(zetaTex, vec2(v_uv.x, min(v_uv.y + d_v, 1.0))).r;
        float zeta_dn = texture(zetaTex, vec2(v_uv.x, max(v_uv.y - d_v, 0.0))).r;
        float dZeta_dTheta = (zeta_up - zeta_dn) / (2.0 * dLat);

        // d zeta / d phi (x direction)
        float x_right = v_uv.x + d_u;
        if (x_right > 1.0) x_right -= 1.0;

        float x_left = v_uv.x - d_u;
        if (x_left < 0.0) x_left += 1.0;

        float zeta_right = texture(zetaTex, vec2(x_right, v_uv.y)).r;
        float zeta_left = texture(zetaTex, vec2(x_left, v_uv.y)).r;
        float dZeta_dPhi = (zeta_right - zeta_left) / (2.0 * dLon);

        // Gradient components on sphere
        // Grad_phi zeta = d zeta / d phi / sin(theta)
        // Grad_theta zeta = d zeta / d theta
        float gradZetaPhi = dZeta_dPhi / sinTheta;
        float gradZetaTheta = dZeta_dTheta;

        // Output: R: dZeta/dPhi / sinTheta, G: dZeta/dTheta
        outColor = vec4(gradZetaPhi, gradZetaTheta, 0.0, 0.0);
      }
    `;
    this.gridGradZetaPass = new ShaderPass(this.gpgpu, gridGradZetaFrag);

    // 3. Nonlinear term Shader (Grid space)
    // - (u * grad_phi zeta + v * grad_theta zeta)
    const nonlinearFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;

      uniform sampler2D velocityTex; // R: u, G: v
      uniform sampler2D gradZetaTex; // R: gradZetaPhi, G: gradZetaTheta

      void main() {
        vec2 vel = texture(velocityTex, v_uv).rg;
        vec2 gradZeta = texture(gradZetaTex, v_uv).rg;

        // Advection term: N(zeta) = - (u * gradZetaPhi + v * gradZetaTheta)
        float n_zeta = -(vel.r * gradZeta.r + vel.g * gradZeta.g);

        outColor = vec4(n_zeta, 0.0, 0.0, 0.0);
      }
    `;
    this.nonlinearPass = new ShaderPass(this.gpgpu, nonlinearFrag);

    // 4. RK4 Step Shader (Spectral space)
    // Computes intermediate steps and adds dissipation
    // k = dt * (N_l^m + nu * lambda_l * zeta_l^m)
    const rk4StepFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;

      uniform sampler2D zetaSpectralTex; // Current zeta state
      uniform sampler2D nSpectralTex;    // Nonlinear term N(zeta) transformed to spectral
      uniform sampler2D eigenTex;        // Laplacian eigenvalues

      uniform float dt;
      uniform float nu;
      uniform float kWeight; // e.g. 1.0 for k1, 0.5 for intermediate state sum
      uniform float stateWeight; // 1.0 or 0.0

      void main() {
        vec2 zeta_lm = texture(zetaSpectralTex, v_uv).rg;
        vec2 n_lm = texture(nSpectralTex, v_uv).rg;
        float eigen = texture(eigenTex, v_uv).r; // -l*(l+1)

        // Dissipation term: nu * nabla^2 zeta = nu * (-l*(l+1)) * zeta_l^m
        vec2 diss_lm = nu * eigen * zeta_lm;

        // d(zeta)/dt = N(zeta) + nu * nabla^2 zeta
        vec2 dZeta_dt = n_lm + diss_lm;

        // For RK4 intermediate evaluations, we often need:
        // zeta_intermediate = zeta_0 + dt * kWeight * dZeta_dt
        // If stateWeight = 1.0, it's zeta_0 + ...
        // If stateWeight = 0.0, we just want to output the k term: dt * dZeta_dt
        vec2 out_lm = stateWeight * zeta_lm + dt * kWeight * dZeta_dt;

        outColor = vec4(out_lm.r, out_lm.g, 0.0, 0.0);
      }
    `;
    this.rkStepPass = new ShaderPass(this.gpgpu, rk4StepFrag);

    // RK4 Accumulator Shader
    // finalZeta = zeta_0 + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    const rk4AccumFrag = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 outColor;

      uniform sampler2D zeta0Tex;
      uniform sampler2D k1Tex;
      uniform sampler2D k2Tex;
      uniform sampler2D k3Tex;
      uniform sampler2D k4Tex;

      void main() {
        vec2 z0 = texture(zeta0Tex, v_uv).rg;
        vec2 k1 = texture(k1Tex, v_uv).rg;
        vec2 k2 = texture(k2Tex, v_uv).rg;
        vec2 k3 = texture(k3Tex, v_uv).rg;
        vec2 k4 = texture(k4Tex, v_uv).rg;

        vec2 finalZeta = z0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        outColor = vec4(finalZeta.r, finalZeta.g, 0.0, 0.0);
      }
    `;
    this.rkAccumPass = new ShaderPass(this.gpgpu, rk4AccumFrag);
  }

  // --- Pipeline Execution ---

  // Computes N(zeta) in spectral space for a given spectral zeta
  computeNonlinearSpectral(zetaSpectralTex) {
    // 1. Inverse Laplacian to get Psi (spectral)
    const texPsiSpec = this.gpgpu.createTexture(this.mMax, this.lMax);
    const fboPsiSpec = this.gpgpu.createFBO(texPsiSpec);
    this.inverseLaplacianPass.render(fboPsiSpec, this.mMax, this.lMax, {}, {
        spectralTex: zetaSpectralTex,
        eigenTex: this.eigenTex
    });

    // 2. Inverse SHT: Psi (spectral) -> Psi (grid)
    const psiGrid = this.sht.inverseTransform(texPsiSpec);

    // 3. Inverse SHT: Zeta (spectral) -> Zeta (grid)
    const zetaGrid = this.sht.inverseTransform(zetaSpectralTex);

    // 4. Compute Velocity (grid) from Psi (grid)
    this.gridVelocityPass.render(this.fboVelocity, this.lonRes, this.latRes, {
        lonRes: this.lonRes,
        latRes: this.latRes
    }, {
        psiTex: psiGrid.tex
    });

    // 5. Compute Grad Zeta (grid) from Zeta (grid)
    this.gridGradZetaPass.render(this.fboGradZeta, this.lonRes, this.latRes, {
        lonRes: this.lonRes,
        latRes: this.latRes
    }, {
        zetaTex: zetaGrid.tex
    });

    // 6. Compute N(zeta) (grid)
    this.nonlinearPass.render(this.fboNonlinear, this.lonRes, this.latRes, {}, {
        velocityTex: this.velocityTex,
        gradZetaTex: this.gradZetaTex
    });

    // 7. Forward SHT: N(zeta) (grid) -> N(zeta) (spectral)
    const nSpectral = this.sht.forwardTransform(this.nonlinearTex);

    // Cleanup intermediates
    this.gpgpu.gl.deleteFramebuffer(fboPsiSpec);
    this.gpgpu.gl.deleteTexture(texPsiSpec);
    this.gpgpu.gl.deleteFramebuffer(psiGrid.fbo);
    this.gpgpu.gl.deleteTexture(psiGrid.tex);
    this.gpgpu.gl.deleteFramebuffer(zetaGrid.fbo);
    this.gpgpu.gl.deleteTexture(zetaGrid.tex);

    return nSpectral;
  }

  // Single step of RK4
  step(pingPongFBO, dt, nu) {
    const zeta0Tex = pingPongFBO.read;

    // We need textures for k1, k2, k3, k4 and intermediate states
    const k1Tex = this.gpgpu.createTexture(this.mMax, this.lMax);
    const fboK1 = this.gpgpu.createFBO(k1Tex);

    const k2Tex = this.gpgpu.createTexture(this.mMax, this.lMax);
    const fboK2 = this.gpgpu.createFBO(k2Tex);

    const k3Tex = this.gpgpu.createTexture(this.mMax, this.lMax);
    const fboK3 = this.gpgpu.createFBO(k3Tex);

    const k4Tex = this.gpgpu.createTexture(this.mMax, this.lMax);
    const fboK4 = this.gpgpu.createFBO(k4Tex);

    const zetaInterTex = this.gpgpu.createTexture(this.mMax, this.lMax);
    const fboZetaInter = this.gpgpu.createFBO(zetaInterTex);

    // --- k1 ---
    let nSpec = this.computeNonlinearSpectral(zeta0Tex);
    this.rkStepPass.render(fboK1, this.mMax, this.lMax, {
        dt: dt, nu: nu, kWeight: 1.0, stateWeight: 0.0
    }, {
        zetaSpectralTex: zeta0Tex, nSpectralTex: nSpec.tex, eigenTex: this.eigenTex
    });

    // Zeta_mid1 = zeta_0 + 0.5 * k1
    this.rkStepPass.render(fboZetaInter, this.mMax, this.lMax, {
        dt: 1.0, nu: 0.0, kWeight: 0.5, stateWeight: 1.0
    }, {
        zetaSpectralTex: zeta0Tex, nSpectralTex: k1Tex, eigenTex: this.eigenTex
    });
    this.gpgpu.gl.deleteFramebuffer(nSpec.fbo);
    this.gpgpu.gl.deleteTexture(nSpec.tex);

    // --- k2 ---
    nSpec = this.computeNonlinearSpectral(zetaInterTex);
    this.rkStepPass.render(fboK2, this.mMax, this.lMax, {
        dt: dt, nu: nu, kWeight: 1.0, stateWeight: 0.0
    }, {
        zetaSpectralTex: zetaInterTex, nSpectralTex: nSpec.tex, eigenTex: this.eigenTex
    });

    // Zeta_mid2 = zeta_0 + 0.5 * k2
    this.rkStepPass.render(fboZetaInter, this.mMax, this.lMax, {
        dt: 1.0, nu: 0.0, kWeight: 0.5, stateWeight: 1.0
    }, {
        zetaSpectralTex: zeta0Tex, nSpectralTex: k2Tex, eigenTex: this.eigenTex
    });
    this.gpgpu.gl.deleteFramebuffer(nSpec.fbo);
    this.gpgpu.gl.deleteTexture(nSpec.tex);

    // --- k3 ---
    nSpec = this.computeNonlinearSpectral(zetaInterTex);
    this.rkStepPass.render(fboK3, this.mMax, this.lMax, {
        dt: dt, nu: nu, kWeight: 1.0, stateWeight: 0.0
    }, {
        zetaSpectralTex: zetaInterTex, nSpectralTex: nSpec.tex, eigenTex: this.eigenTex
    });

    // Zeta_end = zeta_0 + k3
    this.rkStepPass.render(fboZetaInter, this.mMax, this.lMax, {
        dt: 1.0, nu: 0.0, kWeight: 1.0, stateWeight: 1.0
    }, {
        zetaSpectralTex: zeta0Tex, nSpectralTex: k3Tex, eigenTex: this.eigenTex
    });
    this.gpgpu.gl.deleteFramebuffer(nSpec.fbo);
    this.gpgpu.gl.deleteTexture(nSpec.tex);

    // --- k4 ---
    nSpec = this.computeNonlinearSpectral(zetaInterTex);
    this.rkStepPass.render(fboK4, this.mMax, this.lMax, {
        dt: dt, nu: nu, kWeight: 1.0, stateWeight: 0.0
    }, {
        zetaSpectralTex: zetaInterTex, nSpectralTex: nSpec.tex, eigenTex: this.eigenTex
    });
    this.gpgpu.gl.deleteFramebuffer(nSpec.fbo);
    this.gpgpu.gl.deleteTexture(nSpec.tex);

    // --- Accumulate ---
    // pingPongFBO.write = zeta_0 + (k1 + 2*k2 + 2*k3 + k4)/6
    this.rkAccumPass.render(pingPongFBO.writeFBO, this.mMax, this.lMax, {}, {
        zeta0Tex: zeta0Tex,
        k1Tex: k1Tex,
        k2Tex: k2Tex,
        k3Tex: k3Tex,
        k4Tex: k4Tex
    });

    // Cleanup k1..k4 and inter
    this.gpgpu.gl.deleteFramebuffer(fboK1); this.gpgpu.gl.deleteTexture(k1Tex);
    this.gpgpu.gl.deleteFramebuffer(fboK2); this.gpgpu.gl.deleteTexture(k2Tex);
    this.gpgpu.gl.deleteFramebuffer(fboK3); this.gpgpu.gl.deleteTexture(k3Tex);
    this.gpgpu.gl.deleteFramebuffer(fboK4); this.gpgpu.gl.deleteTexture(k4Tex);
    this.gpgpu.gl.deleteFramebuffer(fboZetaInter); this.gpgpu.gl.deleteTexture(zetaInterTex);

    pingPongFBO.swap();
  }

  _initConstants() {
    // We need Laplacian eigenvalues: -l*(l+1)
    // and Inverse Laplacian: -1/(l*(l+1)) for l>0, 0 for l=0
    // We can store these in a 2D texture (mMax x lMax) matching spectral coefficients
    const eigenData = new Float32Array(this.mMax * this.lMax * 4);

    for (let m = 0; m < this.mMax; m++) {
      for (let l = m; l < this.lMax; l++) {
        const idx = (l * this.mMax + m) * 4;
        const eigen = -l * (l + 1);
        const invEigen = l === 0 ? 0 : 1.0 / eigen;

        // R: eigen, G: invEigen, B: 0, A: 0
        eigenData[idx] = eigen;
        eigenData[idx + 1] = invEigen;
        eigenData[idx + 2] = 0.0;
        eigenData[idx + 3] = 0.0;
      }
    }

    this.eigenTex = this.gpgpu.createTexture(this.mMax, this.lMax, eigenData);
  }
}
