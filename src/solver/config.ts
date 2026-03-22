export interface SimulationConfig {
    lmax: number;
    nlat: number;
    nlon: number;
    dt: number;
    nu: number;
    filterAlpha: number;
    filterOrder: number;
    stepsPerFrame: number;
    seed: number;
    amplitude: number;
}

export const defaultConfig: SimulationConfig = {
    lmax: 31,
    nlat: 32, // lmax + 1
    nlon: 64, // 2 * (lmax + 1)
    dt: 1e-2,
    nu: 1e-7,
    filterAlpha: 36.0,
    filterOrder: 8,
    stepsPerFrame: 10,
    seed: 0,
    amplitude: 1.0,
};

export function createConfig(overrides: Partial<SimulationConfig> = {}): SimulationConfig {
    const config = { ...defaultConfig, ...overrides };

    // enforce consistency if only lmax is provided
    if (overrides.lmax !== undefined) {
        if (overrides.nlat === undefined) {
            config.nlat = config.lmax + 1;
        }
        if (overrides.nlon === undefined) {
            config.nlon = 2 * (config.lmax + 1);
        }
    }

    return config;
}
