import { SphericalHarmonicTransform, _norm_matrix } from './shtReference';

describe('CPU Reference Acceptance', () => {

    test('Grid dimensions and polar safety logic', () => {
        const lmax = 15;
        const sht = new SphericalHarmonicTransform(lmax);

        // Check dimensions
        expect(sht.nlat).toBe(lmax + 1);
        expect(sht.nlon).toBe(2 * (lmax + 1));

        // Check polar safety logic: sinTheta = sqrt(max(1 - mu^2, 1e-30))
        for (let i = 0; i < sht.nlat; i++) {
            const expectedSinTheta = Math.sqrt(Math.max(1.0 - sht.mu[i] * sht.mu[i], 1e-30));
            expect(sht.sin_theta[i]).toBeCloseTo(expectedSinTheta, 10);
        }
    });

    test('Laplacian eigenvalues', () => {
        const lmax = 15;
        const sht = new SphericalHarmonicTransform(lmax);

        for (let m = 0; m < sht.M; m++) {
            for (let l = 0; l < sht.L; l++) {
                const idx = m * sht.L + l;
                if (sht.valid[idx]) {
                    expect(sht.lap[idx]).toBeCloseTo(-l * (l + 1), 10);
                }
            }
        }
    });

    test('Roundtrip transform and math stability', () => {
        const lmax = 15;
        const sht = new SphericalHarmonicTransform(lmax);

        // Simple single mode validation
        const a_re = new Float64Array(sht.M * sht.L);
        const a_im = new Float64Array(sht.M * sht.L);

        // Mode l=2, m=1
        const test_m = 1;
        const test_l = 2;
        const test_idx = test_m * sht.L + test_l;
        a_re[test_idx] = 1.0;

        const field = sht.synthesis(a_re, a_im);
        const analyzed = sht.analysis(field);

        expect(analyzed.real[test_idx]).toBeCloseTo(1.0, 5);
        expect(analyzed.imag[test_idx]).toBeCloseTo(0.0, 5);
    });
});
