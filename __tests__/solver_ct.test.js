import { describe, expect, test } from '@jest/globals';
import { NSolver } from '../src/solver.js';

describe('Solver Convergence Tests', () => {
  test('CT-03 Temporal RK4 Convergence', () => {
    const lmax = 16;
    const nu = 1e-4;
    const T = 0.4;

    const runSim = (dt) => {
        const solver = new NSolver(lmax, dt, nu);
        let k21 = -1;
        for(let i=0; i<solver.ncoeffs; i++) {
            if(solver.sht.degree[i] === 2 && solver.sht.order[i] === 1) {
                k21 = i;
                break;
            }
        }
        solver.zetaCoeffs[2*k21] = 1.0;

        const steps = Math.round(T / dt);
        for(let i=0; i<steps; i++) {
            solver.advance();
        }
        return new Float64Array(solver.zetaCoeffs);
    };

    const res_ref = runSim(0.0125);
    const res1 = runSim(0.1);
    const res2 = runSim(0.05);

    const calcErr = (res) => {
        let err = 0;
        for(let i=0; i<res.length; i++) {
            err += Math.pow(res[i] - res_ref[i], 2);
        }
        return Math.sqrt(err);
    };

    const err1 = calcErr(res1);
    const err2 = calcErr(res2);

    const order = Math.log2(err1 / err2);

    expect(order).toBeGreaterThan(3.5); // Should be roughly 4th order
  });
});
