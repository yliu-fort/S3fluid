import { createConfig } from "../../src/solver/config";
import { getLmIndex, isValidLm, getGridIndex, getGridSize, getSpectralSize } from "../../src/solver/layout";

describe("Layout Logic Tests", () => {
    it("should compute accurate array sizes", () => {
        const config = createConfig({ lmax: 31 });

        const L = config.lmax + 1;
        const M = config.lmax + 1;

        expect(getSpectralSize(config)).toBe(M * L);

        const gridNodes = config.nlat * config.nlon;
        expect(getGridSize(config)).toBe(gridNodes);
    });

    it("should correctly validate LM pairs", () => {
        const config = createConfig({ lmax: 31 });

        expect(isValidLm(0, 0, config)).toBe(true);
        expect(isValidLm(31, 31, config)).toBe(true);
        expect(isValidLm(0, 31, config)).toBe(true);

        expect(isValidLm(1, 0, config)).toBe(false); // l < m
        expect(isValidLm(-1, 0, config)).toBe(false); // m < 0
        expect(isValidLm(0, 32, config)).toBe(false); // l out of bounds
        expect(isValidLm(32, 32, config)).toBe(false); // m and l out of bounds
    });

    it("should map indices accurately", () => {
        const config = createConfig({ lmax: 31 });

        expect(getLmIndex(0, 0, config)).toBe(0);

        // the max valid l and m is 31
        expect(getLmIndex(31, 31, config)).toBe(31 * 32 + 31);

        // test grid mappings
        expect(getGridIndex(0, 0, config)).toBe(0);

        const J = config.nlat;
        const K = config.nlon;
        expect(getGridIndex(J - 1, K - 1, config)).toBe((J - 1) * K + (K - 1));
    });
});
