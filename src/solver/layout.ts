import type { SimulationConfig } from "./config";

export function getLmIndex(m: number, l: number, config: SimulationConfig): number {
    const L = config.lmax + 1;
    return m * L + l;
}

export function isValidLm(m: number, l: number, config: SimulationConfig): boolean {
    const L = config.lmax + 1;
    const M = config.lmax + 1;
    return m >= 0 && m < M && l >= m && l < L;
}

export function getGridIndex(j: number, k: number, config: SimulationConfig): number {
    return j * config.nlon + k;
}

export function getGridSize(config: SimulationConfig): number {
    return config.nlat * config.nlon;
}

export function getSpectralSize(config: SimulationConfig): number {
    const L = config.lmax + 1;
    const M = config.lmax + 1;
    return M * L; // we allocate rectangular (M, L) and ignore l < m. complex size means it's usually vec2
}
