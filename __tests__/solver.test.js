import { describe, expect, test } from '@jest/globals';
import { NSolver } from '../src/solver.js';
import { SHT } from '../src/sht.js';

describe('Solver tests', () => {
  test('UT-02 Inverse Laplace Operator', () => {
    // zeta = 2*cos(theta) -> Y_1^0
    const lmax = 16;
    const sht = new SHT(lmax);

    // zeta = 2 * cos(theta) = 2 * sqrt(4pi/3) * Y_1^0
    const zetaCoeffs = new Float64Array(2 * sht.ncoeffs);
    let k10 = -1;
    for(let i=0; i<sht.ncoeffs; i++) {
        if(sht.degree[i] === 1 && sht.order[i] === 0) {
            k10 = i;
            break;
        }
    }

    zetaCoeffs[2*k10] = 2 * Math.sqrt(4 * Math.PI / 3);

    const solver = new NSolver(lmax);
    const psiCoeffs = solver._streamFromZeta(zetaCoeffs);

    const expectedPsiCoeffs = new Float64Array(2 * sht.ncoeffs);
    expectedPsiCoeffs[2*k10] = zetaCoeffs[2*k10] / -2.0;

    let maxErr = 0;
    for(let i=0; i<psiCoeffs.length; i++) {
        maxErr = Math.max(maxErr, Math.abs(psiCoeffs[i] - expectedPsiCoeffs[i]));
    }
    expect(maxErr).toBeLessThan(1e-12);

    // Check grid space psi
    const psiGrid = sht.synth(psiCoeffs);
    let maxErrGrid = 0;
    for(let i=0; i<sht.nlat; i++) {
        const ct = sht.cosTheta[i];
        const expectedPsi = -ct; // since zeta = 2 cos(theta), psi = -cos(theta)
        for(let j=0; j<sht.nlon; j++) {
            maxErrGrid = Math.max(maxErrGrid, Math.abs(psiGrid[i*sht.nlon+j] - expectedPsi));
        }
    }
    expect(maxErrGrid).toBeLessThan(1e-12);
  });

  test('UT-04 RK4 Integrator with Viscosity', () => {
    // Disable nonlinear term
    const lmax = 16;
    const nu = 0.1;
    const dt = 0.5;
    const solver = new NSolver(lmax, dt, nu);

    solver._nonlinear = function(zetaCoeffs) {
        const rhs = new Float64Array(2 * solver.ncoeffs);
        for(let k=0; k<solver.ncoeffs; k++) {
            rhs[2*k] = nu * solver.lap[k] * zetaCoeffs[2*k];
            rhs[2*k+1] = nu * solver.lap[k] * zetaCoeffs[2*k+1];
        }
        return rhs;
    };

    let k21 = -1;
    for(let i=0; i<solver.ncoeffs; i++) {
        if(solver.sht.degree[i] === 2 && solver.sht.order[i] === 1) {
            k21 = i;
            break;
        }
    }
    solver.zetaCoeffs[2*k21] = 1.0;

    for(let i=0; i<10; i++) {
        solver.advance();
    }

    const t = 10 * dt;
    const l = 2;
    const expectedCoeff = Math.exp(-nu * l * (l+1) * t);

    expect(Math.abs(solver.zetaCoeffs[2*k21] - expectedCoeff)).toBeLessThan(1e-4);
  });

  test('PT-03 Solid Body Rotation', () => {
      const lmax = 16;
      const solver = new NSolver(lmax, 0.1, 0.0); // nu=0
      solver.initSolidBodyRotation(1.0);

      const initialZeta = new Float64Array(solver.zetaCoeffs);

      for(let i=0; i<10; i++) {
          solver.advance();
      }

      let maxErr = 0;
      for(let i=0; i<solver.zetaCoeffs.length; i++) {
          maxErr = Math.max(maxErr, Math.abs(solver.zetaCoeffs[i] - initialZeta[i]));
      }
      expect(maxErr).toBeLessThan(1e-10);
  });

  test('PT-04 Rossby-Haurwitz Wave', () => {
      const lmax = 16;
      const solver = new NSolver(lmax, 0.05, 0.0);
      solver.initRossbyHaurwitz(4);

      const E_start = solver.getEnergy();

      for(let i=0; i<5; i++) {
          solver.advance();
      }

      const E_end = solver.getEnergy();
      // Should be relatively stable
      expect(Math.abs(E_end - E_start) / E_start).toBeLessThan(0.05);
  });
});
