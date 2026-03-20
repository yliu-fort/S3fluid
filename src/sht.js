/**
 * sht.js - Spherical Harmonic Transform (SHT) in Pure JavaScript
 *
 * Implements a truncated spherical harmonic transform up to degree lmax,
 * on a Gauss-Legendre latitude grid and uniform longitude grid.
 *
 * Convention: real spherical harmonics, complex spectral coefficients stored
 * in a 1D array indexed by (l, m) using the triangular packing:
 *   index(l, m) = l*(l+1)/2 + m,  0 <= m <= l
 * Negative-m modes are obtained by conjugate symmetry.
 *
 * This implementation is intentionally straightforward over optimized.
 */

// ============================================================
// Gauss-Legendre quadrature
// ============================================================

/**
 * Compute N-point Gauss-Legendre nodes (x ∈ [-1,1]) and weights.
 * Uses Newton's method on Legendre polynomials.
 */
export function gaussLegendre(N) {
  const x = new Float64Array(N);
  const w = new Float64Array(N);
  const half = Math.floor((N + 1) / 2);

  for (let i = 0; i < half; i++) {
    // Initial guess
    let xi = Math.cos(Math.PI * (i + 0.75) / (N + 0.5));
    let pp, p1, p2, p3;
    for (let iter = 0; iter < 100; iter++) {
      p1 = 1.0;
      p2 = 0.0;
      for (let j = 1; j <= N; j++) {
        p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * xi * p2 - (j - 1) * p3) / j;
      }
      // p1 is P_N(xi), pp is dP_N/dx
      pp = N * (xi * p1 - p2) / (xi * xi - 1.0);
      const dx = p1 / pp;
      xi -= dx;
      if (Math.abs(dx) < 1e-15) break;
    }
    x[i] = -xi;
    x[N - 1 - i] = xi;
    w[i] = 2.0 / ((1.0 - xi * xi) * pp * pp);
    w[N - 1 - i] = w[i];
  }
  return { x, w };
}

// ============================================================
// Associated Legendre Polynomials  P_l^m(cos θ)
// ============================================================

/**
 * Compute normalized associated Legendre polynomials P_l^m(x)
 * for all l from 0..lmax and a fixed m, evaluated at x.
 * Returns array plm[l-m] for l = m..lmax.
 *
 * Uses the standard 3-term recurrence with full normalization so that
 *   ∫_{-1}^{1} (P_l^m)^2 dx = 1  (orthonormal for m > 0 includes √2 factor)
 */
function computePlm(x, lmax, m) {
  const size = lmax - m + 1;
  const plm = new Float64Array(size);  // plm[l-m] = P_l^m(x)

  // Compute P_m^m first (starting value)
  let pmm = 1.0;
  // Normalization: sqrt( (2m+1)/(4π) * (2m)! / (m!)^2 ) * sqrt(2) for m>0
  // We use the recurrence: pmm = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
  const sinTheta = Math.sqrt(Math.max(0, 1.0 - x * x));
  let sfact = 1.0;
  for (let i = 1; i <= m; i++) {
    sfact *= sinTheta * Math.sqrt((2.0 * i + 1) / (2.0 * i));
  }
  pmm = sfact * Math.sqrt((2.0 * m + 1) / (4.0 * Math.PI));
  if (m > 0) pmm *= Math.SQRT2;

  if (m === lmax) {
    plm[0] = pmm;
    return plm;
  }

  // P_{m+1}^m
  let pmp1m = x * Math.sqrt(2.0 * m + 3) * pmm;

  plm[0] = pmm;
  if (size > 1) plm[1] = pmp1m;

  // Recurrence for l > m+1
  for (let l = m + 2; l <= lmax; l++) {
    const a = Math.sqrt(((2.0 * l + 1) * (2.0 * l - 1)) / ((l - m) * (l + m)));
    const b = Math.sqrt(((2.0 * l + 1) * (l - m - 1) * (l + m - 1)) / ((l - m) * (l + m) * (2.0 * l - 3)));
    const pnew = a * x * pmp1m - b * pmm;
    plm[l - m] = pnew;
    pmm = pmp1m;
    pmp1m = pnew;
  }
  return plm;
}

// ============================================================
// SHT class
// ============================================================

export class SHT {
  /**
   * @param {number} lmax - Maximum spherical harmonic degree
   */
  constructor(lmax) {
    this.lmax = lmax;
    // Number of coefficients (triangular): (lmax+1)*(lmax+2)/2
    this.ncoeffs = ((lmax + 1) * (lmax + 2)) / 2;

    // Grid dimensions: nlat = lmax+1, nlon = 2*(lmax+1)
    this.nlat = lmax + 1;
    this.nlon = 2 * (lmax + 1);

    // Gauss-Legendre quadrature on cos(theta) ∈ [-1, 1]
    const { x, w } = gaussLegendre(this.nlat);
    this.cosTheta = x;   // cos(theta) values, north-to-south typically
    this.glWeights = w;

    // Latitude angles (for reference)
    this.lats = new Float64Array(this.nlat);
    for (let i = 0; i < this.nlat; i++) {
      this.lats[i] = Math.asin(x[i]);
    }

    // Longitude grid: uniform [0, 2π)
    this.lons = new Float64Array(this.nlon);
    for (let j = 0; j < this.nlon; j++) {
      this.lons[j] = (2.0 * Math.PI * j) / this.nlon;
    }

    // Degree and order arrays (length = ncoeffs)
    this.degree = new Int32Array(this.ncoeffs);
    this.order  = new Int32Array(this.ncoeffs);
    let idx = 0;
    for (let l = 0; l <= lmax; l++) {
      for (let m = 0; m <= l; m++) {
        this.degree[idx] = l;
        this.order[idx]  = m;
        idx++;
      }
    }

    // Precompute Plm tables for each latitude and m
    // plmTable[i][m] = Float64Array of P_{m..lmax}^m evaluated at cosTheta[i]
    // Store as flat 2D: plmTable[i * (lmax+1) + m] = Float64Array
    this._plmTable = [];
    for (let i = 0; i < this.nlat; i++) {
      const row = [];
      for (let m = 0; m <= lmax; m++) {
        row.push(computePlm(x[i], lmax, m));
      }
      this._plmTable.push(row);
    }

    // Precompute longitude trig for FFT (we use DFT for simplicity)
    // expTable[j][m] = { cos: cos(m * lon_j), sin: sin(m * lon_j) }
    // We'll compute on-the-fly from longitude values.

    console.log(`SHT initialized: lmax=${lmax}, nlat=${this.nlat}, nlon=${this.nlon}, ncoeffs=${this.ncoeffs}`);
  }

  /** Map (l,m) to flat coefficient index */
  idx(l, m) {
    return l * (l + 1) / 2 + m;
  }

  /**
   * Forward transform: grid → spectral coefficients
   * @param {Float64Array} grid - [nlat × nlon] row-major (lat outer, lon inner)
   * @returns {Float64Array} coeffs - 2*ncoeffs reals: [re0,im0, re1,im1, ...]
   */
  analys(grid) {
    const { lmax, nlat, nlon, ncoeffs, cosTheta, glWeights, _plmTable } = this;
    const coeffs = new Float64Array(2 * ncoeffs);

    for (let i = 0; i < nlat; i++) {
      const wt = glWeights[i] * (2.0 * Math.PI / nlon);
      const row = grid.subarray(i * nlon, (i + 1) * nlon);

      // DFT along longitude for each m
      for (let m = 0; m <= lmax; m++) {
        let re = 0.0, im = 0.0;
        const dPhi = 2.0 * Math.PI * m / nlon;
        for (let j = 0; j < nlon; j++) {
          const angle = dPhi * j;
          re += row[j] * Math.cos(angle);
          im -= row[j] * Math.sin(angle);
        }

        const plmRow = _plmTable[i][m];  // P_{m..lmax}^m(cosTheta[i])
        for (let l = m; l <= lmax; l++) {
          const k = this.idx(l, m);
          const plm = plmRow[l - m];
          coeffs[2 * k]     += wt * plm * re;
          coeffs[2 * k + 1] += wt * plm * im;
        }
      }
    }
    return coeffs;
  }

  /**
   * Inverse transform: spectral coefficients → grid
   * @param {Float64Array} coeffs - 2*ncoeffs reals
   * @returns {Float64Array} grid - [nlat × nlon]
   */
  synth(coeffs) {
    const { lmax, nlat, nlon, _plmTable } = this;
    const grid = new Float64Array(nlat * nlon);

    for (let i = 0; i < nlat; i++) {
      // Accumulate Fourier coefficients F_m for this latitude
      // F(lon) = sum_m F_m * exp(i*m*lon)
      // We compute F_m = sum_{l>=m} c_lm * P_l^m(cos theta_i)
      const Fre = new Float64Array(lmax + 1);
      const Fim = new Float64Array(lmax + 1);

      for (let m = 0; m <= lmax; m++) {
        const plmRow = _plmTable[i][m];
        let sumRe = 0.0, sumIm = 0.0;
        for (let l = m; l <= lmax; l++) {
          const k = this.idx(l, m);
          const plm = plmRow[l - m];
          sumRe += coeffs[2 * k]     * plm;
          sumIm += coeffs[2 * k + 1] * plm;
        }
        Fre[m] = sumRe;
        Fim[m] = sumIm;
      }

      // Inverse DFT: reconstruct real function from positive-freq Fourier modes
      for (let j = 0; j < nlon; j++) {
        let val = Fre[0];  // m=0 is real
        const lon = 2.0 * Math.PI * j / nlon;
        for (let m = 1; m <= lmax; m++) {
          const angle = m * lon;
          val += 2.0 * (Fre[m] * Math.cos(angle) - Fim[m] * Math.sin(angle));
        }
        grid[i * nlon + j] = val;
      }
    }
    return grid;
  }

  /**
   * Gradient synthesis: from streamfunction coefficients → (dPhi, dTheta) velocity components
   * Computes:  u_phi = (1/sinθ) ∂ψ/∂φ  and  u_theta = ∂ψ/∂θ
   * Returned as gradient in (phi, theta) directions on the grid.
   *
   * Uses: ∂Y_l^m/∂φ = i*m * Y_l^m  (multiplication by im)
   *       ∂Y_l^m/∂θ from recurrence (simplified via sinθ factor)
   *
   * We return:
   *   dPhi   = (1/sinθ) * ∂ψ/∂φ  → multiply spectral by i*m, then synth / sinθ
   *   dTheta = ∂ψ/∂θ              → spectral differentiation w.r.t. theta
   *
   * For the NS solver convenience, we return (uphi, utheta) matching the Python prototype:
   *   u_phi, u_theta = sht.synth_grad(psi_coeffs)
   */
  synthGrad(psiCoeffs) {
    const { lmax, nlat, nlon, _plmTable, cosTheta } = this;

    // u_phi = (1/sinθ) ∂ψ/∂φ
    // In spectral space: multiply by (i*m), then synth, then divide by sinθ
    const dPhiCoeffs = new Float64Array(2 * this.ncoeffs);
    for (let l = 0; l <= lmax; l++) {
      for (let m = 0; m <= l; m++) {
        const k = this.idx(l, m);
        // multiply complex coeff by (i*m): re' = -m*im, im' = m*re
        dPhiCoeffs[2 * k]     = -m * psiCoeffs[2 * k + 1];
        dPhiCoeffs[2 * k + 1] =  m * psiCoeffs[2 * k];
      }
    }
    const uPhiGrid = this.synth(dPhiCoeffs);

    // Divide by sinθ at each latitude (avoid singularity at poles)
    for (let i = 0; i < nlat; i++) {
      const sinTheta = Math.sqrt(Math.max(1e-20, 1.0 - cosTheta[i] * cosTheta[i]));
      for (let j = 0; j < nlon; j++) {
        uPhiGrid[i * nlon + j] /= sinTheta;
      }
    }

    // u_theta = ∂ψ/∂θ
    // dP_l^m/dθ = -sinθ * dP_l^m/d(cosθ)
    // Using recurrence: sinθ * dP_l^m/d(cosθ) = ... this is complex.
    // Simpler: use the identity:
    //   sinθ * dP_l^m/dθ = ... rewrite via associated Legendre gradient recurrence.
    // We'll use: ∂Y_l^m/∂θ = (l * cot θ * Y_l^m - sqrt((l+m)(l-m+1)/(2l+1)/(2l-1)) * (2l+1) * Y_{l-1}^m / sinθ ... )
    // Simpler approach: use finite-difference approximation on uTheta is wrong.
    // Instead, we use the SH gradient formula:
    //   ∂P_l^m/∂θ = m * cosθ/sinθ * P_l^m - sqrt((l+m)(l-m+1)) * P_l^{m+1}
    //             (with the convention P_l^{lmax+1} = 0)
    // Actually the cleanest route: compute dPsi/dTheta directly from spectral.

    const uThetaGrid = new Float64Array(nlat * nlon);

    for (let i = 0; i < nlat; i++) {
      const ct = cosTheta[i];
      const sinT = Math.sqrt(Math.max(1e-20, 1.0 - ct * ct));

      // For each m, accumulate F_m(theta) = sum_l c_lm * dP_l^m/dTheta
      const Fre = new Float64Array(lmax + 1);
      const Fim = new Float64Array(lmax + 1);

      for (let m = 0; m <= lmax; m++) {
        const plmRow = _plmTable[i][m];
        let sumRe = 0.0, sumIm = 0.0;
        for (let l = m; l <= lmax; l++) {
          const k = this.idx(l, m);
          const re = psiCoeffs[2 * k];
          const im = psiCoeffs[2 * k + 1];

          // dP_l^m/dθ = (m * ct / sinT) * P_l^m(ct) - sqrt((l+m)(l-m+1)) * P_l^{m+1}(ct)
          // (the second term requires P_l^{m+1})
          let dplm = 0.0;
          if (l > 0 || m === 0) {
            dplm = m * (ct / sinT) * plmRow[l - m];
            if (m < l) {
              // Need P_l^{m+1}: in plmTable[i][m+1], index l-(m+1) = l-m-1
              const plmPlus1Row = _plmTable[i][m + 1];
              const plmPlus1 = plmPlus1Row[l - m - 1];
              // Normalization correction factor between our normalized Plm and the gradient formula
              // Factor: sqrt((l-m)*(l+m+1))
              const fac = Math.sqrt((l - m) * (l + m + 1.0));
              dplm -= fac * plmPlus1;
            }
          }

          sumRe += re * dplm;
          sumIm += im * dplm;
        }
        Fre[m] = sumRe;
        Fim[m] = sumIm;
      }

      // Inverse DFT
      for (let j = 0; j < nlon; j++) {
        let val = Fre[0];
        const lon = 2.0 * Math.PI * j / nlon;
        for (let m = 1; m <= lmax; m++) {
          const angle = m * lon;
          val += 2.0 * (Fre[m] * Math.cos(angle) - Fim[m] * Math.sin(angle));
        }
        uThetaGrid[i * nlon + j] = val;
      }
    }

    return { uPhi: uPhiGrid, uTheta: uThetaGrid };
  }
}
