import { SphericalHarmonicTransform } from './shtReference';
import { SphereTurbulence2D } from './modelReference';

describe('SphereTurbulence2D model tests', () => {
    test('Kinetic energy and RHS sanity', () => {
        const lmax = 15;
        const sht = new SphericalHarmonicTransform(lmax);
        const model = new SphereTurbulence2D(sht);

        const a_re = new Float64Array(sht.M * sht.L);
        const a_im = new Float64Array(sht.M * sht.L);

        // mode l=1, m=0 -> constant velocity, predictable behavior
        a_re[1] = 1.0;

        // evaluate kinetic energy bounds (should be non-negative)
        const ke = model.kinetic_energy(a_re, a_im);
        expect(ke).toBeGreaterThan(0);

        // evaluate basic RHS diffusion decay
        const rhs = model.rhs(a_re, a_im);
        expect(rhs.real[1]).toBeLessThan(0); // diff_lm = nu * -l(l+1) * a_re -> should be negative
    });

    test('Velocity from streamfunction mapping logic', () => {
        const lmax = 15;
        const sht = new SphericalHarmonicTransform(lmax);
        const model = new SphereTurbulence2D(sht);

        const psi_re = new Float64Array(sht.M * sht.L);
        const psi_im = new Float64Array(sht.M * sht.L);

        // simple mode
        psi_re[1] = 1.0;

        const vels = model.velocity_from_streamfunction(psi_re, psi_im);
        expect(vels.u_theta.length).toBe(sht.J * sht.nlon);
        expect(vels.u_phi.length).toBe(sht.J * sht.nlon);

        // Based on logic u_phi = -dpsi_dtheta
        // u_theta = dpsi_dphi / sin_theta
        // A pure m=0 mode (l=1, m=0) has dphi = 0
        let max_u_theta = 0;
        for (let i = 0; i < vels.u_theta.length; i++) {
            max_u_theta = Math.max(max_u_theta, Math.abs(vels.u_theta[i]));
        }
        expect(max_u_theta).toBeCloseTo(0, 10);
    });
});