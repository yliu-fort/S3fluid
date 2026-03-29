import { SOLVER_CONFIG } from './config';
import { getMIndex, isLValid } from './layout';
import { leggauss, lpmv, _norm_matrix } from '../../tests/cpu-reference/shtReference';

export type PrecomputedData = {
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

export function generatePrecomputedData(lmax: number, nlat: number, nlon: number): PrecomputedData {
    const { x: mu_f64, w: w_f64 } = leggauss(nlat);
    const mu = new Float32Array(mu_f64);
    const w = new Float32Array(w_f64);

    const theta = new Float32Array(nlat);
    const sinTheta = new Float32Array(nlat);
    for (let i = 0; i < nlat; i++) {
        theta[i] = Math.acos(mu[i]);
        sinTheta[i] = Math.sqrt(Math.max(1.0 - mu[i] * mu[i], 1e-30));
    }

    const phi = new Float32Array(nlon);
    for (let i = 0; i < nlon; i++) {
        phi[i] = 2.0 * Math.PI * i / nlon;
    }

    const L = lmax + 1;
    const M = lmax + 1;
    const J = nlat;

    const { norm, valid } = _norm_matrix(lmax);

    const P_lm = new Float32Array(J * M * L);
    const dP_lm_dtheta = new Float32Array(J * M * L);

    for (let j = 0; j < J; j++) {
        for (let m = 0; m < M; m++) {
            for (let l = 0; l < L; l++) {
                let idx = j * (M * L) + m * L + l;
                let nm_idx = m * L + l;
                if (valid[nm_idx]) {
                    let p_raw = lpmv(m, l, mu[j]);
                    let p_val = norm[nm_idx] * p_raw;
                    P_lm[idx] = p_val;

                    let l_prev = Math.max(l - 1, 0);
                    let p_prev = lpmv(m, l_prev, mu[j]);
                    let dp_dmu_raw = (l * mu[j] * p_raw - (l + m) * p_prev) / (mu[j] * mu[j] - 1.0);
                    dP_lm_dtheta[idx] = -sinTheta[j] * norm[nm_idx] * dp_dmu_raw;
                } else {
                    P_lm[idx] = 0.0;
                    dP_lm_dtheta[idx] = 0.0;
                }
            }
        }
    }

    const lapEigs = new Float32Array(M * L);
    for (let m = 0; m < M; m++) {
        for (let l = 0; l < L; l++) {
            let idx = m * L + l;
            if (isLValid(m, l)) {
                lapEigs[idx] = -l * (l + 1.0);
            } else {
                lapEigs[idx] = 0.0;
            }
        }
    }

    const specFilter = new Float32Array(M * L);
    const filterAlpha = SOLVER_CONFIG.filterAlpha;
    const filterOrder = SOLVER_CONFIG.filterOrder;
    for (let m = 0; m < M; m++) {
        for (let l = 0; l < L; l++) {
            let idx = m * L + l;
            if (isLValid(m, l)) {
                // specFilter = exp(-alpha * (l/lmax)^order)
                specFilter[idx] = Math.exp(-filterAlpha * Math.pow(l / lmax, filterOrder));
            } else {
                specFilter[idx] = 0.0;
            }
        }
    }

    const initSlope = new Float32Array(M * L);
    for (let m = 0; m < M; m++) {
        for (let l = 0; l < L; l++) {
            let idx = m * L + l;
            if (isLValid(m, l)) {
                if (l === 0) {
                    initSlope[idx] = 0.0;
                } else {
                    initSlope[idx] = Math.pow(l, -1.0 / 3.0);
                }
            } else {
                initSlope[idx] = 0.0;
            }
        }
    }

    return {
        mu, w, theta, sinTheta, phi, P_lm, dP_lm_dtheta, lapEigs, specFilter, initSlope
    };
}
