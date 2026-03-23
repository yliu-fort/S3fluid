export function gammaln(x: number): number {
    // Lanczos approximation of ln(Gamma(x))
    const p = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ];
    let y = x;
    const t = y + 7.5;
    let sum = p[0];
    for (let i = 1; i < p.length; i++) {
        sum += p[i] / (y + i);
    }
    return (y + 0.5) * Math.log(t) - t + Math.log(2.5066282746310005 * sum / y);
}

export function leggauss(n: number): { x: Float64Array, w: Float64Array } {
    const x = new Float64Array(n);
    const w = new Float64Array(n);
    const m = Math.floor((n + 1) / 2);

    for (let i = 0; i < m; i++) {
        let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));
        let z1 = 0;
        let pp = 0;

        // Newton-Raphson iteration
        while (Math.abs(z - z1) > Number.EPSILON) {
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
        }

        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    return { x, w };
}

// Compute P_l^m(x) where x is usually cos(theta)
export function lpmv(m: number, l: number, x: number): number {
    if (m < 0 || m > l || Math.abs(x) > 1.0) {
        return 0;
    }
    let pmm = 1.0;
    if (m > 0) {
        let somx2 = Math.sqrt((1.0 - x) * (1.0 + x));
        let fact = 1.0;
        for (let i = 1; i <= m; i++) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if (l === m) {
        return pmm;
    }
    let pmmp1 = x * (2 * m + 1) * pmm;
    if (l === m + 1) {
        return pmmp1;
    }
    let pll = 0;
    for (let ll = m + 2; ll <= l; ll++) {
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

export function _norm_matrix(lmax: number): { norm: Float64Array, valid: boolean[] } {
    const M = lmax + 1;
    const L = lmax + 1;
    const norm = new Float64Array(M * L);
    const valid = new Array(M * L).fill(false);

    for (let m = 0; m < M; m++) {
        for (let l = 0; l < L; l++) {
            let idx = m * L + l;
            if (l >= m) {
                valid[idx] = true;
                let logn = 0.5 * (
                    Math.log(2.0 * l + 1.0)
                    - Math.log(4.0 * Math.PI)
                    + gammaln(l - m + 1.0)
                    - gammaln(l + m + 1.0)
                );
                norm[idx] = Math.exp(logn);
            } else {
                norm[idx] = 0;
            }
        }
    }
    return { norm, valid };
}

// Very simple naive DFT for the reference implementation since we need exactness
function rfft(realInput: Float64Array, N: number, M: number): { real: Float64Array, imag: Float64Array } {
    const real = new Float64Array(M);
    const imag = new Float64Array(M);

    for (let m = 0; m < M; m++) {
        let sumRe = 0;
        let sumIm = 0;
        for (let n = 0; n < N; n++) {
            let angle = -2 * Math.PI * m * n / N;
                sumRe += (realInput[n] || 0.0) * Math.cos(angle);
                sumIm += (realInput[n] || 0.0) * Math.sin(angle);
        }
        real[m] = sumRe;
        imag[m] = sumIm;
    }
    return { real, imag };
}

function irfft(realFreq: Float64Array, imagFreq: Float64Array, N: number, M: number): Float64Array {
    const out = new Float64Array(N);
    for (let n = 0; n < N; n++) {
            let sum = realFreq[0] || 0.0; // m = 0 term
        for (let m = 1; m < M; m++) {
            let angle = 2 * Math.PI * m * n / N;
            // Since input was real, we take 2 * Real(X[m] * exp(i * angle))
                sum += 2 * ((realFreq[m] || 0.0) * Math.cos(angle) - (imagFreq[m] || 0.0) * Math.sin(angle));
        }
        out[n] = sum / N;
    }
    return out;
}

export class SphericalHarmonicTransform {
    lmax: number;
    nlat: number;
    nlon: number;

    M: number;
    L: number;
    J: number;
    K: number;

    mu: Float64Array;
    w: Float64Array;
    theta: Float64Array;
    sin_theta: Float64Array;
    phi: Float64Array;

    P: Float64Array;
    dP: Float64Array;
    Pw: Float64Array;

    lap: Float64Array;
    inv_lap: Float64Array;
    im: Float64Array;
    valid: boolean[];

    constructor(lmax: number, nlat?: number, nlon?: number) {
        this.lmax = lmax;
        this.nlat = nlat || lmax + 1;
        this.nlon = nlon || 2 * (lmax + 1);

        const gauss = leggauss(this.nlat);
        this.mu = gauss.x;
        this.w = gauss.w;

        this.theta = new Float64Array(this.nlat);
        this.sin_theta = new Float64Array(this.nlat);
        for (let i = 0; i < this.nlat; i++) {
            this.theta[i] = Math.acos(this.mu[i]);
            this.sin_theta[i] = Math.sqrt(Math.max(1.0 - this.mu[i] * this.mu[i], 1e-30));
        }

        this.phi = new Float64Array(this.nlon);
        for (let i = 0; i < this.nlon; i++) {
            this.phi[i] = 2.0 * Math.PI * i / this.nlon;
        }

        this.M = this.lmax + 1;
        this.L = this.lmax + 1;
        this.J = this.nlat;
        this.K = Math.floor(this.nlon / 2) + 1;

        const normData = _norm_matrix(this.lmax);
        this.valid = normData.valid;
        const norm = normData.norm;

        this.P = new Float64Array(this.J * this.M * this.L);
        this.dP = new Float64Array(this.J * this.M * this.L);
        this.Pw = new Float64Array(this.J * this.M * this.L);

        for (let j = 0; j < this.J; j++) {
            for (let m = 0; m < this.M; m++) {
                for (let l = 0; l < this.L; l++) {
                    let idx = j * (this.M * this.L) + m * this.L + l;
                    let nm_idx = m * this.L + l;
                    if (this.valid[nm_idx]) {
                        let p_raw = lpmv(m, l, this.mu[j]);
                        let p_val = norm[nm_idx] * p_raw;
                        this.P[idx] = p_val;
                        this.Pw[idx] = p_val * this.w[j];

                        let l_prev = Math.max(l - 1, 0);
                        let p_prev = lpmv(m, l_prev, this.mu[j]);
                        let dp_dmu_raw = (l * this.mu[j] * p_raw - (l + m) * p_prev) / (this.mu[j] * this.mu[j] - 1.0);
                        this.dP[idx] = -this.sin_theta[j] * norm[nm_idx] * dp_dmu_raw;
                    }
                }
            }
        }

        this.lap = new Float64Array(this.M * this.L);
        this.inv_lap = new Float64Array(this.M * this.L);
        for (let m = 0; m < this.M; m++) {
            for (let l = 0; l < this.L; l++) {
                let idx = m * this.L + l;
                if (this.valid[idx]) {
                    this.lap[idx] = -l * (l + 1.0);
                    if (this.lap[idx] !== 0) {
                        this.inv_lap[idx] = 1.0 / this.lap[idx];
                    }
                }
            }
        }

        this.im = new Float64Array(this.M);
        for (let m = 0; m < this.M; m++) {
            this.im[m] = m;
        }
    }

    // field is expected to be [J][nlon] continuous array
    analysis(field: Float64Array): { real: Float64Array, imag: Float64Array } {
        const out_re = new Float64Array(this.M * this.L);
        const out_im = new Float64Array(this.M * this.L);

        const F_re = new Float64Array(this.J * this.M);
        const F_im = new Float64Array(this.J * this.M);

        for (let j = 0; j < this.J; j++) {
            let row = new Float64Array(this.nlon);
            for (let k = 0; k < this.nlon; k++) {
                row[k] = field[j * this.nlon + k] || 0.0;
            }
            let row_fft = rfft(row, this.nlon, this.M);
            for (let m = 0; m < this.M; m++) {
                F_re[j * this.M + m] = row_fft.real[m] || 0.0;
                F_im[j * this.M + m] = row_fft.imag[m] || 0.0;
            }
        }

        for (let m = 0; m < this.M; m++) {
            for (let l = 0; l < this.L; l++) {
                let ml_idx = m * this.L + l;
                if (this.valid[ml_idx]) {
                    let sum_re = 0;
                    let sum_im = 0;
                    for (let j = 0; j < this.J; j++) {
                        let p_w = this.Pw[j * (this.M * this.L) + m * this.L + l] || 0.0;
                        sum_re += p_w * (F_re[j * this.M + m] || 0.0);
                        sum_im += p_w * (F_im[j * this.M + m] || 0.0);
                    }
                    out_re[ml_idx] = sum_re * (2.0 * Math.PI / this.nlon);
                    out_im[ml_idx] = sum_im * (2.0 * Math.PI / this.nlon);
                }
            }
        }

        return { real: out_re, imag: out_im };
    }

    synthesis(a_re: Float64Array, a_im: Float64Array): Float64Array {
        const freq_re = new Float64Array(this.J * this.M);
        const freq_im = new Float64Array(this.J * this.M);

        for (let j = 0; j < this.J; j++) {
            for (let m = 0; m < this.M; m++) {
                let sum_re = 0;
                let sum_im = 0;
                for (let l = 0; l < this.L; l++) {
                    let ml_idx = m * this.L + l;
                    if (this.valid[ml_idx]) {
                        let p_val = this.P[j * (this.M * this.L) + m * this.L + l] || 0.0;
                        sum_re += p_val * (a_re[ml_idx] || 0.0);
                        sum_im += p_val * (a_im[ml_idx] || 0.0);
                    }
                }
                freq_re[j * this.M + m] = sum_re * this.nlon;
                freq_im[j * this.M + m] = sum_im * this.nlon;
            }
        }

        const field = new Float64Array(this.J * this.nlon);
        for (let j = 0; j < this.J; j++) {
            let row_freq_re = new Float64Array(this.M);
            let row_freq_im = new Float64Array(this.M);
            for (let m = 0; m < this.M; m++) {
                row_freq_re[m] = freq_re[j * this.M + m];
                row_freq_im[m] = freq_im[j * this.M + m];
            }
            let row_ifft = irfft(row_freq_re, row_freq_im, this.nlon, this.M);
            for (let k = 0; k < this.nlon; k++) {
                field[j * this.nlon + k] = row_ifft[k] || 0.0;
            }
        }
        return field;
    }

    dphi(a_re: Float64Array, a_im: Float64Array): Float64Array {
        const out_re = new Float64Array(this.M * this.L);
        const out_im = new Float64Array(this.M * this.L);

        for (let m = 0; m < this.M; m++) {
            for (let l = 0; l < this.L; l++) {
                let idx = m * this.L + l;
                if (this.valid[idx]) {
                    // a * (i*m) = (a_re + i*a_im) * i*m = -a_im*m + i*a_re*m
                    out_re[idx] = -(a_im[idx] || 0.0) * m;
                    out_im[idx] = (a_re[idx] || 0.0) * m;
                }
            }
        }
        return this.synthesis(out_re, out_im);
    }

    dtheta(a_re: Float64Array, a_im: Float64Array): Float64Array {
        const freq_re = new Float64Array(this.J * this.M);
        const freq_im = new Float64Array(this.J * this.M);

        for (let j = 0; j < this.J; j++) {
            for (let m = 0; m < this.M; m++) {
                let sum_re = 0;
                let sum_im = 0;
                for (let l = 0; l < this.L; l++) {
                    let ml_idx = m * this.L + l;
                    if (this.valid[ml_idx]) {
                        let dp_val = this.dP[j * (this.M * this.L) + m * this.L + l] || 0.0;
                        sum_re += dp_val * (a_re[ml_idx] || 0.0);
                        sum_im += dp_val * (a_im[ml_idx] || 0.0);
                    }
                }
                freq_re[j * this.M + m] = sum_re * this.nlon;
                freq_im[j * this.M + m] = sum_im * this.nlon;
            }
        }

        const field = new Float64Array(this.J * this.nlon);
        for (let j = 0; j < this.J; j++) {
            let row_freq_re = new Float64Array(this.M);
            let row_freq_im = new Float64Array(this.M);
            for (let m = 0; m < this.M; m++) {
                row_freq_re[m] = freq_re[j * this.M + m];
                row_freq_im[m] = freq_im[j * this.M + m];
            }
            let row_ifft = irfft(row_freq_re, row_freq_im, this.nlon, this.M);
            for (let k = 0; k < this.nlon; k++) {
                field[j * this.nlon + k] = row_ifft[k] || 0.0;
            }
        }
        return field;
    }

    apply_laplacian(a_re: Float64Array, a_im: Float64Array): { real: Float64Array, imag: Float64Array } {
        const out_re = new Float64Array(this.M * this.L);
        const out_im = new Float64Array(this.M * this.L);
        for (let i = 0; i < this.M * this.L; i++) {
            if (this.valid[i]) {
                out_re[i] = (a_re[i] || 0.0) * (this.lap[i] || 0.0);
                out_im[i] = (a_im[i] || 0.0) * (this.lap[i] || 0.0);
            }
        }
        return { real: out_re, imag: out_im };
    }

    invert_laplacian(a_re: Float64Array, a_im: Float64Array): { real: Float64Array, imag: Float64Array } {
        const out_re = new Float64Array(this.M * this.L);
        const out_im = new Float64Array(this.M * this.L);
        for (let i = 0; i < this.M * this.L; i++) {
            if (this.valid[i]) {
                out_re[i] = (a_re[i] || 0.0) * (this.inv_lap[i] || 0.0);
                out_im[i] = (a_im[i] || 0.0) * (this.inv_lap[i] || 0.0);
            }
        }
        out_re[0] = 0.0;
        out_im[0] = 0.0;
        return { real: out_re, imag: out_im };
    }
}
