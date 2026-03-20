/**
 * solver.js - Spherical Navier-Stokes Solver
 *
 * Implements the 2D incompressible NS solver on the sphere using
 * the vorticity-streamfunction formulation:
 *   ∂ζ/∂t = -u·∇ζ + ν∇²ζ
 * with RK4 time integration.
 *
 * All computations are in JS (no WASM); this module is designed to run
 * inside a Web Worker.
 */

import { SHT } from './sht.js';

export class NSolver {
  /**
   * @param {number} lmax - Max spectral degree
   * @param {number} dt   - Time step
   * @param {number} nu   - Kinematic viscosity
   */
  constructor(lmax, dt = 0.5, nu = 1e-5) {
    this.lmax = lmax;
    this.dt   = dt;
    this.nu   = nu;
    this.sht  = new SHT(lmax);

    const { ncoeffs } = this.sht;
    this.ncoeffs = ncoeffs;
    const lArr = this.sht.degree;

    // Laplacian eigenvalues: ∇²Y_l^m = -l(l+1) Y_l^m
    this.lap    = new Float64Array(ncoeffs);
    this.invlap = new Float64Array(ncoeffs);
    for (let k = 0; k < ncoeffs; k++) {
      const l = lArr[k];
      if (l === 0) {
        this.lap[k]    = 0;
        this.invlap[k] = 0;
      } else {
        this.lap[k]    = -l * (l + 1);
        this.invlap[k] = 1.0 / this.lap[k];
      }
    }

    // Gaussian spectral filter: exp(-36*(l/lmax)^8)
    this.specFilter = new Float64Array(ncoeffs);
    for (let k = 0; k < ncoeffs; k++) {
      const l = lArr[k];
      this.specFilter[k] = Math.exp(-36.0 * Math.pow(l / lmax, 8));
    }

    // Initialize vorticity coefficients to zero
    this.zetaCoeffs = new Float64Array(2 * ncoeffs);

    this.step = 0;
    this.time = 0;
  }

  /**
   * Initialize with random turbulence using k^(-1/3) energy spectrum.
   */
  initRandom(seed = 42) {
    const rng = mulberry32(seed);
    const { ncoeffs, degree } = this.sht;
    const { specFilter } = this;
    const zetaCoeffs = this.zetaCoeffs;

    for (let k = 0; k < ncoeffs; k++) {
      const l = degree[k];
      // Energy slope: k^(-1/3) in spherical harmonic space
      const slope = l > 0 ? Math.pow(l, -1.0 / 3.0) : 0;
      const phase = 2 * Math.PI * rng();
      const amp   = rng() * slope * specFilter[k];
      zetaCoeffs[2 * k]     = amp * Math.cos(phase);
      zetaCoeffs[2 * k + 1] = amp * Math.sin(phase);
    }

    // Normalize to unit initial energy
    const zetaGrid = this.sht.synth(zetaCoeffs);
    const psiCoeffs = this._streamFromZeta(zetaCoeffs);
    const psiGrid   = this.sht.synth(psiCoeffs);
    let E = 0;
    for (let i = 0; i < zetaGrid.length; i++) E += Math.abs(zetaGrid[i] * psiGrid[i]);
    const scale = E > 0 ? 1.0 / Math.sqrt(E / zetaGrid.length) : 1;
    for (let i = 0; i < 2 * ncoeffs; i++) zetaCoeffs[i] *= scale;
  }

  /**
   * Initialize solid-body rotation: ψ = -ω cos(θ)  ↔  ζ modes.
   * l=1, m=0 mode of the normalized P_1^0 = sqrt(3/4π) * cos(θ)
   */
  initSolidBodyRotation(omega = 1.0) {
    const { ncoeffs } = this.sht;
    this.zetaCoeffs = new Float64Array(2 * ncoeffs);
    // ψ = -ω cos θ → ζ = ∇²ψ = ω * l(l+1) * psi_coeff for l=1,m=0
    // Normalized P_1^0(cos θ) = sqrt(3/(4π)) cos θ → amplitude for cos theta
    // We set psi_10 = -ω * sqrt(4π/3) so that ψ = -ω cos θ in grid space
    const psiCoeffs = new Float64Array(2 * ncoeffs);
    const k10 = this.sht.idx(1, 0);
    psiCoeffs[2 * k10] = -omega * Math.sqrt(4 * Math.PI / 3);
    // ζ = ∇²ψ: ζ_lm = lap_l * psi_lm
    for (let k = 0; k < ncoeffs; k++) {
      this.zetaCoeffs[2 * k]     = this.lap[k] * psiCoeffs[2 * k];
      this.zetaCoeffs[2 * k + 1] = this.lap[k] * psiCoeffs[2 * k + 1];
    }
  }

  /**
   * Initialize Rossby-Haurwitz wave (wave number m=4).
   */
  initRossbyHaurwitz(waveNumber = 4) {
    const { ncoeffs } = this.sht;
    this.zetaCoeffs = new Float64Array(2 * ncoeffs);
    // Set a single mode: l=waveNumber, m=waveNumber
    const k = this.sht.idx(waveNumber, waveNumber);
    if (k < ncoeffs) {
      this.zetaCoeffs[2 * k]     = 1.0;
      this.zetaCoeffs[2 * k + 1] = 0.0;
    }
  }

  /** Compute stream function from vorticity: ψ = ζ / ∇² = ζ * invlap */
  _streamFromZeta(zetaCoeffs) {
    const { ncoeffs, invlap } = this;
    const psi = new Float64Array(2 * ncoeffs);
    for (let k = 0; k < ncoeffs; k++) {
      psi[2 * k]     = zetaCoeffs[2 * k]     * invlap[k];
      psi[2 * k + 1] = zetaCoeffs[2 * k + 1] * invlap[k];
    }
    return psi;
  }

  /**
   * Compute the nonlinear RHS: -u·∇ζ + ν∇²ζ
   * @param {Float64Array} zetaCoeffs - 2*ncoeffs
   * @returns {Float64Array} rhs - 2*ncoeffs
   */
  _nonlinear(zetaCoeffs) {
    const { ncoeffs, lap, specFilter, nu, sht } = this;

    // ψ from ζ
    const psiCoeffs = this._streamFromZeta(zetaCoeffs);

    // Velocity field from ψ: uPhi = (1/sinθ)∂ψ/∂φ, uTheta = ∂ψ/∂θ
    const { uPhi, uTheta } = sht.synthGrad(psiCoeffs);

    // Gradient of ζ: dPhi = (1/sinθ)∂ζ/∂φ, dTheta = ∂ζ/∂θ
    const { uPhi: dPhi, uTheta: dTheta } = sht.synthGrad(zetaCoeffs);

    // Advection: u·∇ζ = u_theta * ∂ζ/∂θ + u_phi * (1/sinθ)∂ζ/∂φ
    // where u_theta = (1/sinθ)∂ψ/∂φ = uPhi  and  u_phi = -∂ψ/∂θ = -uTheta
    const nlat = sht.nlat;
    const nlon = sht.nlon;
    const adv = new Float64Array(nlat * nlon);
    for (let i = 0; i < nlat * nlon; i++) {
      adv[i] = uPhi[i] * dTheta[i] - uTheta[i] * dPhi[i];
    }

    // Back to spectral
    let advCoeffs = sht.analys(adv);

    // Apply spectral filter and add viscous term
    const rhs = new Float64Array(2 * ncoeffs);
    for (let k = 0; k < ncoeffs; k++) {
      const filt = specFilter[k];
      rhs[2 * k]     = -advCoeffs[2 * k]     * filt + nu * lap[k] * zetaCoeffs[2 * k];
      rhs[2 * k + 1] = -advCoeffs[2 * k + 1] * filt + nu * lap[k] * zetaCoeffs[2 * k + 1];
    }
    return rhs;
  }

  /**
   * Advance one RK4 time step.
   */
  advance() {
    const { zetaCoeffs, dt, ncoeffs: nc } = this;
    const n = 2 * this.sht.ncoeffs;

    const addScaled = (a, b, scale) => {
      const r = new Float64Array(n);
      for (let i = 0; i < n; i++) r[i] = a[i] + scale * b[i];
      return r;
    };

    const k1 = this._nonlinear(zetaCoeffs);
    const k2 = this._nonlinear(addScaled(zetaCoeffs, k1, 0.5 * dt));
    const k3 = this._nonlinear(addScaled(zetaCoeffs, k2, 0.5 * dt));
    const k4 = this._nonlinear(addScaled(zetaCoeffs, k3, dt));

    for (let i = 0; i < n; i++) {
      zetaCoeffs[i] += (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }

    this.step++;
    this.time += dt;
  }

  /**
   * Get grid data for rendering.
   * @param {'vorticity'|'velocity'|'streamfunction'} variable
   * @returns {{ grid: Float32Array, min: number, max: number }}
   */
  getGridData(variable = 'vorticity') {
    const { sht, zetaCoeffs } = this;
    let raw;

    if (variable === 'vorticity') {
      raw = sht.synth(zetaCoeffs);
    } else if (variable === 'streamfunction') {
      const psiCoeffs = this._streamFromZeta(zetaCoeffs);
      raw = sht.synth(psiCoeffs);
    } else {
      // velocity magnitude
      const psiCoeffs = this._streamFromZeta(zetaCoeffs);
      const { uPhi, uTheta } = sht.synthGrad(psiCoeffs);
      raw = new Float64Array(uPhi.length);
      for (let i = 0; i < raw.length; i++) {
        raw[i] = Math.sqrt(uPhi[i] * uPhi[i] + uTheta[i] * uTheta[i]);
      }
    }

    // Convert to Float32 and compute min/max
    const grid = new Float32Array(raw.length);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < raw.length; i++) {
      grid[i] = raw[i];
      if (raw[i] < mn) mn = raw[i];
      if (raw[i] > mx) mx = raw[i];
    }
    return { grid, min: mn, max: mx, nlat: sht.nlat, nlon: sht.nlon };
  }

  /**
   * Compute energy: E = Σ |ζ_lm|² / (l(l+1)) (half the enstrophy spectrum integral)
   */
  getEnergy() {
    const { zetaCoeffs, invlap, sht } = this;
    let E = 0;
    for (let k = 0; k < sht.ncoeffs; k++) {
      if (invlap[k] === 0) continue;
      const re = zetaCoeffs[2 * k];
      const im = zetaCoeffs[2 * k + 1];
      const mult = sht.order[k] === 0 ? 1 : 2; // account for m and -m
      E += mult * (re * re + im * im) * (-invlap[k]);
    }
    return E;
  }
}

// Simple 32-bit mulberry RNG for reproducible random init
function mulberry32(seed) {
  let s = seed >>> 0;
  return function() {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
