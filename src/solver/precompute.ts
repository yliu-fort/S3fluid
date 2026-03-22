import type { SimulationConfig } from "./config";
import type { SimulationBuffers } from "./buffers";

export function gammaln(x: number): number {
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

export interface PrecomputedData {
    mu: Float32Array;
    w: Float32Array;
    theta: Float32Array;
    sinTheta: Float32Array;
    phi: Float32Array;
    P_lm: Float32Array;
    dP_lm_dtheta: Float32Array;
    lapEigs: Float32Array;
    specFilter: Float32Array;
    initSlope: Float32Array;
}

export function precompute(config: SimulationConfig): PrecomputedData {
    const { lmax, nlat, nlon, filterAlpha, filterOrder } = config;
    const M = lmax + 1;
    const L = lmax + 1;
    const J = nlat;

    const gauss = leggauss(J);
    const mu = new Float32Array(J);
    const w = new Float32Array(J);
    const theta = new Float32Array(J);
    const sinTheta = new Float32Array(J);

    for (let j = 0; j < J; j++) {
        mu[j] = gauss.x[j] || 0.0;
        w[j] = gauss.w[j] || 0.0;
        theta[j] = Math.acos(mu[j]!);
        sinTheta[j] = Math.sqrt(Math.max(1.0 - mu[j]! * mu[j]!, 1e-30));
    }

    const phi = new Float32Array(nlon);
    for (let i = 0; i < nlon; i++) {
        phi[i] = 2.0 * Math.PI * i / nlon;
    }

    // precompute norms and valid flags
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

    // P_lm and dP_lm_dtheta
    const P_lm = new Float32Array(J * M * L);
    const dP_lm_dtheta = new Float32Array(J * M * L);

    for (let j = 0; j < J; j++) {
        for (let m = 0; m < M; m++) {
            for (let l = 0; l < L; l++) {
                let idx = j * (M * L) + m * L + l;
                let nm_idx = m * L + l;

                if (valid[nm_idx]) {
                    let p_raw = lpmv(m, l, mu[j]!);
                    let p_val = norm[nm_idx]! * p_raw;
                    P_lm[idx] = p_val;

                    let l_prev = Math.max(l - 1, 0);
                    let p_prev = lpmv(m, l_prev, mu[j]!);
                    let dp_dmu_raw = (l * mu[j]! * p_raw - (l + m) * p_prev) / (mu[j]! * mu[j]! - 1.0);
                    dP_lm_dtheta[idx] = -sinTheta[j]! * norm[nm_idx]! * dp_dmu_raw;
                } else {
                    P_lm[idx] = 0;
                    dP_lm_dtheta[idx] = 0;
                }
            }
        }
    }

    // Laplacian eigenvalues
    const lapEigs = new Float32Array(M * L);
    for (let m = 0; m < M; m++) {
        for (let l = 0; l < L; l++) {
            let idx = m * L + l;
            if (valid[idx]) {
                lapEigs[idx] = -l * (l + 1.0);
            }
        }
    }

    // Spectrum filter
    const specFilter = new Float32Array(M * L);
    for (let m = 0; m < M; m++) {
        for (let l = 0; l < L; l++) {
            let idx = m * L + l;
            if (valid[idx]) {
                specFilter[idx] = Math.exp(-filterAlpha * Math.pow(l / lmax, filterOrder));
            }
        }
    }
    specFilter[0] = 1.0;

    // Initial slope
    const initSlope = new Float32Array(M * L);
    for (let m = 0; m < M; m++) {
        for (let l = 0; l < L; l++) {
            let idx = m * L + l;
            if (valid[idx]) {
                initSlope[idx] = l > 0 ? Math.pow(l, -1.0 / 3.0) : 1.0;
            }
        }
    }

    return {
        mu,
        w,
        theta,
        sinTheta,
        phi,
        P_lm,
        dP_lm_dtheta,
        lapEigs,
        specFilter,
        initSlope
    };
}

export async function initPrecomputeBuffers(device: GPUDevice, config: SimulationConfig, buffers: SimulationBuffers) {
    const precomp = precompute(config);

    // allocate buffers that are strictly read-only after init
    buffers.w = device.createBuffer({
        size: precomp.w.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buffers.w, 0, precomp.w as any);

    buffers.P_lm = device.createBuffer({
        size: precomp.P_lm.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buffers.P_lm, 0, precomp.P_lm as any);

    buffers.dP_lm_dtheta = device.createBuffer({
        size: precomp.dP_lm_dtheta.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buffers.dP_lm_dtheta, 0, precomp.dP_lm_dtheta as any);
}
