import { SphericalHarmonicTransform } from './shtReference';
import { SphereTurbulence2D } from './modelReference';
import { generatePrecomputedData } from '../../src/solver/precompute';

describe('Analytical Solutions', () => {
    test('Solid body rotation analytical decay (Zonal Flow l=1, m=0)', () => {
        const lmax = 127;
        const sht = new SphericalHarmonicTransform(lmax);
        const model = new SphereTurbulence2D(sht);

        // Zonal flow represented by spherical harmonic Y_1^0
        // decays analytically at exactly the rate exp(-nu * l * (l+1) * t) without nonlinear advection.

        model.nu = 1e-3;
        const dt = 1.0; // Large step for noticeable decay

        let a_re = new Float64Array(sht.M * sht.L);
        let a_im = new Float64Array(sht.M * sht.L);

        const m = 0;
        const l = 1;
        const idx = m * sht.L + l;
        a_re[idx] = 1.0; // Initial value

        let initial_val = a_re[idx];

        // Step forward once
        const res = model.step_rk4(a_re, a_im, dt);

        // Analytical solution
        const analytical = initial_val * Math.exp(-model.nu * l * (l + 1) * dt);

        // Check difference
        const diff = Math.abs(res.real[idx] - analytical);

        // The difference should be very small for a good integrator
        expect(diff).toBeLessThan(1e-5);
    });
});
