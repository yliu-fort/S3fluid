import { SOLVER_CONFIG } from './config';

export function getMIndex(m: number, l: number, lmax: number): number {
    return m * (lmax + 1) + l;
}

export function getGridIndex(j: number, k: number, nlon: number): number {
    return j * nlon + k;
}

export function isLValid(m: number, l: number): boolean {
    return l >= m;
}

// Complex numbers are represented as vec2<f32>, so sizes are in number of f32s (2 per complex number)
// Layout for spectral coefficients: m from 0 to lmax, l from 0 to lmax.
export const SPECTRAL_COEFFS_F32_COUNT = (SOLVER_CONFIG.lmax + 1) * (SOLVER_CONFIG.lmax + 1) * 2;
// Layout for grid: j from 0 to nlat-1, k from 0 to nlon-1
export const GRID_POINTS_F32_COUNT = SOLVER_CONFIG.nlat * SOLVER_CONFIG.nlon;
