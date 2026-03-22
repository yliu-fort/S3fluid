import { SphericalHarmonicTransform } from "./shtReference";

export class SphereTurbulence2D {
    sht: SphericalHarmonicTransform;
    nu: number;
    filter_alpha: number;
    filter_order: number;

    spec_filter: Float64Array;
    init_slope: Float64Array;

    constructor(sht: SphericalHarmonicTransform, nu: number = 1.0e-7, filter_alpha: number = 36.0, filter_order: number = 8) {
        this.sht = sht;
        this.nu = nu;
        this.filter_alpha = filter_alpha;
        this.filter_order = filter_order;

        this.spec_filter = new Float64Array(sht.M * sht.L);
        for (let m = 0; m < sht.M; m++) {
            for (let l = 0; l < sht.L; l++) {
                let idx = m * sht.L + l;
                if (sht.valid[idx]) {
                    this.spec_filter[idx] = Math.exp(-this.filter_alpha * Math.pow(l / sht.lmax, this.filter_order));
                }
            }
        }
        this.spec_filter[0] = 1.0;

        this.init_slope = new Float64Array(sht.M * sht.L);
        for (let m = 0; m < sht.M; m++) {
            for (let l = 0; l < sht.L; l++) {
                let idx = m * sht.L + l;
                if (sht.valid[idx]) {
                    this.init_slope[idx] = l > 0 ? Math.pow(l, -1.0 / 3.0) : 1.0;
                }
            }
        }
    }

    filter_coeffs(a_re: Float64Array, a_im: Float64Array): { real: Float64Array, imag: Float64Array } {
        const out_re = new Float64Array(this.sht.M * this.sht.L);
        const out_im = new Float64Array(this.sht.M * this.sht.L);
        for (let i = 0; i < this.sht.M * this.sht.L; i++) {
            if (this.sht.valid[i]) {
                out_re[i] = a_re[i] * this.spec_filter[i];
                out_im[i] = a_im[i] * this.spec_filter[i];
            }
        }
        return { real: out_re, imag: out_im };
    }

    streamfunction_from_vorticity(zeta_re: Float64Array, zeta_im: Float64Array): { real: Float64Array, imag: Float64Array } {
        return this.sht.invert_laplacian(zeta_re, zeta_im);
    }

    velocity_from_streamfunction(psi_re: Float64Array, psi_im: Float64Array): { u_theta: Float64Array, u_phi: Float64Array } {
        const dpsi_dphi = this.sht.dphi(psi_re, psi_im);
        const dpsi_dtheta = this.sht.dtheta(psi_re, psi_im);

        const u_theta = new Float64Array(this.sht.J * this.sht.nlon);
        const u_phi = new Float64Array(this.sht.J * this.sht.nlon);

        for (let j = 0; j < this.sht.J; j++) {
            for (let k = 0; k < this.sht.nlon; k++) {
                let idx = j * this.sht.nlon + k;
                u_theta[idx] = dpsi_dphi[idx] / this.sht.sin_theta[j];
                u_phi[idx] = -dpsi_dtheta[idx];
            }
        }
        return { u_theta, u_phi };
    }

    rhs(zeta_re: Float64Array, zeta_im: Float64Array): { real: Float64Array, imag: Float64Array } {
        const psi = this.streamfunction_from_vorticity(zeta_re, zeta_im);
        const vels = this.velocity_from_streamfunction(psi.real, psi.imag);

        const dzeta_dtheta = this.sht.dtheta(zeta_re, zeta_im);
        const dzeta_dphi = this.sht.dphi(zeta_re, zeta_im);

        const adv = new Float64Array(this.sht.J * this.sht.nlon);
        for (let j = 0; j < this.sht.J; j++) {
            for (let k = 0; k < this.sht.nlon; k++) {
                let idx = j * this.sht.nlon + k;
                // u_theta * dzeta_dtheta + u_phi * (dzeta_dphi / sin_theta)
                adv[idx] = vels.u_theta[idx] * dzeta_dtheta[idx] + vels.u_phi[idx] * (dzeta_dphi[idx] / this.sht.sin_theta[j]);
            }
        }

        const adv_analyzed = this.sht.analysis(adv);
        const adv_lm = this.filter_coeffs(adv_analyzed.real, adv_analyzed.imag);

        const diff_lm = this.sht.apply_laplacian(zeta_re, zeta_im);

        const out_re = new Float64Array(this.sht.M * this.sht.L);
        const out_im = new Float64Array(this.sht.M * this.sht.L);
        for (let i = 0; i < this.sht.M * this.sht.L; i++) {
            if (this.sht.valid[i]) {
                out_re[i] = -adv_lm.real[i] + this.nu * diff_lm.real[i];
                out_im[i] = -adv_lm.imag[i] + this.nu * diff_lm.imag[i];
            }
        }
        return { real: out_re, imag: out_im };
    }

    // Box-Muller transform for standard normal variables
    random_standard_normal(rng: () => number): number {
        const u = 1.0 - rng(); // Converting [0,1) to (0,1]
        const v = rng();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    random_initial_vorticity(seed: number = 0, amplitude: number = 1.0): { real: Float64Array, imag: Float64Array } {
        // very simple LCG for seeded RNG
        let current_seed = seed;
        const rng = () => {
            current_seed = (current_seed * 1664525 + 1013904223) % 4294967296;
            return current_seed / 4294967296;
        };

        const grid = new Float64Array(this.sht.J * this.sht.nlon);
        for (let i = 0; i < grid.length; i++) {
            grid[i] = this.random_standard_normal(rng);
        }

        const zeta_analyzed = this.sht.analysis(grid);
        for (let i = 0; i < this.sht.M * this.sht.L; i++) {
            if (this.sht.valid[i]) {
                zeta_analyzed.real[i] *= this.init_slope[i];
                zeta_analyzed.imag[i] *= this.init_slope[i];
            }
        }

        const zeta_filtered = this.filter_coeffs(zeta_analyzed.real, zeta_analyzed.imag);
        zeta_filtered.real[0] = 0.0;
        zeta_filtered.imag[0] = 0.0;

        for (let i = 0; i < this.sht.M * this.sht.L; i++) {
            if (this.sht.valid[i]) {
                zeta_filtered.real[i] *= amplitude;
                zeta_filtered.imag[i] *= amplitude;
            }
        }
        return zeta_filtered;
    }

    kinetic_energy(zeta_re: Float64Array, zeta_im: Float64Array): number {
        const psi = this.streamfunction_from_vorticity(zeta_re, zeta_im);
        const zeta_grid = this.sht.synthesis(zeta_re, zeta_im);
        const psi_grid = this.sht.synthesis(psi.real, psi.imag);

        let total_energy = 0;
        for (let j = 0; j < this.sht.J; j++) {
            let row_sum = 0;
            for (let k = 0; k < this.sht.nlon; k++) {
                let idx = j * this.sht.nlon + k;
                row_sum += -0.5 * psi_grid[idx] * zeta_grid[idx];
            }
            total_energy += this.sht.w[j] * row_sum;
        }
        return total_energy * (2.0 * Math.PI / this.sht.nlon);
    }

    step_rk4(zeta_re: Float64Array, zeta_im: Float64Array, dt: number): { real: Float64Array, imag: Float64Array } {
        const k1 = this.rhs(zeta_re, zeta_im);

        const z2_re = new Float64Array(this.sht.M * this.sht.L);
        const z2_im = new Float64Array(this.sht.M * this.sht.L);
        for (let i = 0; i < z2_re.length; i++) {
            if (this.sht.valid[i]) {
                z2_re[i] = zeta_re[i] + 0.5 * dt * k1.real[i];
                z2_im[i] = zeta_im[i] + 0.5 * dt * k1.imag[i];
            }
        }
        const k2 = this.rhs(z2_re, z2_im);

        const z3_re = new Float64Array(this.sht.M * this.sht.L);
        const z3_im = new Float64Array(this.sht.M * this.sht.L);
        for (let i = 0; i < z3_re.length; i++) {
            if (this.sht.valid[i]) {
                z3_re[i] = zeta_re[i] + 0.5 * dt * k2.real[i];
                z3_im[i] = zeta_im[i] + 0.5 * dt * k2.imag[i];
            }
        }
        const k3 = this.rhs(z3_re, z3_im);

        const z4_re = new Float64Array(this.sht.M * this.sht.L);
        const z4_im = new Float64Array(this.sht.M * this.sht.L);
        for (let i = 0; i < z4_re.length; i++) {
            if (this.sht.valid[i]) {
                z4_re[i] = zeta_re[i] + dt * k3.real[i];
                z4_im[i] = zeta_im[i] + dt * k3.imag[i];
            }
        }
        const k4 = this.rhs(z4_re, z4_im);

        const out_re = new Float64Array(this.sht.M * this.sht.L);
        const out_im = new Float64Array(this.sht.M * this.sht.L);
        for (let i = 0; i < out_re.length; i++) {
            if (this.sht.valid[i]) {
                out_re[i] = zeta_re[i] + (dt / 6.0) * (k1.real[i] + 2.0 * k2.real[i] + 2.0 * k3.real[i] + k4.real[i]);
                out_im[i] = zeta_im[i] + (dt / 6.0) * (k1.imag[i] + 2.0 * k2.imag[i] + 2.0 * k3.imag[i] + k4.imag[i]);
            }
        }

        return this.filter_coeffs(out_re, out_im);
    }
}
