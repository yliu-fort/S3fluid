export const SOLVER_CONFIG = {
    lmax: 63,
    get nlat() { return this.lmax + 1; },
    get nlon() { return 2 * (this.lmax + 1); },
    dt: 1e-2,
    nu: 1e-7,
    filterAlpha: 36.0,
    filterOrder: 8,
    stepsPerFrame: 10,
    seed: 42,
    amplitude: 1.0
};
