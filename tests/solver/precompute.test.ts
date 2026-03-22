import { SOLVER_CONFIG } from '../../src/solver/config';
import { generatePrecomputedData } from '../../src/solver/precompute';
import { getMIndex, isLValid } from '../../src/solver/layout';

describe('Precomputation', () => {
    it('generates correct mu and w properties', () => {
        const { mu, w } = generatePrecomputedData(SOLVER_CONFIG.lmax, SOLVER_CONFIG.nlat, SOLVER_CONFIG.nlon);
        let sumW = 0;
        for (let i = 0; i < w.length; i++) {
            sumW += w[i];
        }
        expect(sumW).toBeCloseTo(2.0, 5);

        // Check symmetry
        for (let i = 0; i < Math.floor(mu.length / 2); i++) {
            expect(mu[i]).toBeCloseTo(-mu[mu.length - 1 - i], 5);
            expect(w[i]).toBeCloseTo(w[w.length - 1 - i], 5);
        }
    });

    it('generates correct laplacian eigenvalues', () => {
        const { lapEigs } = generatePrecomputedData(SOLVER_CONFIG.lmax, SOLVER_CONFIG.nlat, SOLVER_CONFIG.nlon);
        const lmax = SOLVER_CONFIG.lmax;
        for (let m = 0; m <= lmax; m++) {
            for (let l = 0; l <= lmax; l++) {
                const idx = m * (lmax + 1) + l;
                if (l >= m) {
                    expect(lapEigs[idx]).toBe(-l * (l + 1));
                } else {
                    expect(lapEigs[idx]).toBe(0);
                }
            }
        }
    });

    it('generates correct spec filter properties', () => {
        const { specFilter } = generatePrecomputedData(SOLVER_CONFIG.lmax, SOLVER_CONFIG.nlat, SOLVER_CONFIG.nlon);
        const lmax = SOLVER_CONFIG.lmax;
        for (let m = 0; m <= lmax; m++) {
            for (let l = 0; l <= lmax; l++) {
                const idx = m * (lmax + 1) + l;
                if (l >= m) {
                    if (l === 0) {
                        expect(specFilter[idx]).toBe(1.0);
                    } else if (l === lmax) {
                        expect(specFilter[idx]).toBeCloseTo(Math.exp(-SOLVER_CONFIG.filterAlpha), 5);
                    }
                }
            }
        }
    });

    it('generates correct init slope', () => {
        const { initSlope } = generatePrecomputedData(SOLVER_CONFIG.lmax, SOLVER_CONFIG.nlat, SOLVER_CONFIG.nlon);
        const lmax = SOLVER_CONFIG.lmax;
        for (let m = 0; m <= lmax; m++) {
            for (let l = 0; l <= lmax; l++) {
                const idx = m * (lmax + 1) + l;
                if (l >= m) {
                    if (l === 0) {
                        expect(initSlope[idx]).toBe(0.0);
                    } else {
                        expect(initSlope[idx]).toBeCloseTo(Math.pow(l, -1.0 / 3.0), 5);
                    }
                }
            }
        }
    });
});
