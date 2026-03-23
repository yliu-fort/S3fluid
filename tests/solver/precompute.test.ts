import { createConfig } from "../../src/solver/config";
import { precompute } from "../../src/solver/precompute";
import { getLmIndex, isValidLm } from "../../src/solver/layout";

describe("Precompute Tests", () => {
    it("should compute correct quadrature nodes and weights", () => {
        const config = createConfig({ lmax: 31 });
        const data = precompute(config);

        // sum(w) should be close to 2.0
        let sumW = 0;
        for (let i = 0; i < data.w.length; i++) {
            sumW += data.w[i];
        }
        expect(sumW).toBeCloseTo(2.0, 5);

        // mu should be symmetric
        const J = config.nlat;
        for (let j = 0; j < Math.floor(J / 2); j++) {
            expect(data.mu[j]).toBeCloseTo(-data.mu[J - 1 - j], 5);
            expect(data.w[j]).toBeCloseTo(data.w[J - 1 - j], 5);
        }

        // no NaN in sinTheta at poles
        for (let j = 0; j < J; j++) {
            expect(data.sinTheta[j]).toBeGreaterThan(0);
            expect(Number.isNaN(data.sinTheta[j])).toBe(false);
        }
    });

    it("should compute correct eigenvalues and filters", () => {
        const config = createConfig({ lmax: 31, filterAlpha: 36.0, filterOrder: 8 });
        const data = precompute(config);

        for (let m = 0; m <= config.lmax; m++) {
            for (let l = m; l <= config.lmax; l++) {
                const idx = getLmIndex(m, l, config);

                // lapEigs[m,l] = -l(l+1)
                expect(data.lapEigs[idx]).toBeCloseTo(-l * (l + 1), 5);

                // specFilter logic
                if (l === 0 && m === 0) {
                    expect(data.specFilter[idx]).toBeCloseTo(1.0, 5);
                } else {
                    const expectedFilter = Math.exp(-config.filterAlpha * Math.pow(l / config.lmax, config.filterOrder));
                    expect(data.specFilter[idx]).toBeCloseTo(expectedFilter, 5);
                }

                // initSlope logic
                if (l > 0) {
                    expect(data.initSlope[idx]).toBeCloseTo(Math.pow(l, -1 / 3), 5);
                }
            }
        }

        // specFilter should be non-increasing with l (for a fixed m=0)
        for (let l = 1; l < config.lmax; l++) {
            const idx1 = getLmIndex(0, l, config);
            const idx2 = getLmIndex(0, l + 1, config);
            expect(data.specFilter[idx1]).toBeGreaterThanOrEqual(data.specFilter[idx2]);
        }
    });
});
