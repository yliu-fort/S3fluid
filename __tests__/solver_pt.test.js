import { describe, expect, test } from '@jest/globals';
import { NSolver } from '../src/solver.js';

describe('Solver Physical Tests', () => {
  test('PT-01 Energy Conservation (Inviscid Limit)', () => {
    const lmax = 16;
    const solver = new NSolver(lmax, 0.005, 0.0);
    // Use a very smooth initial condition (single mode)
    let k21 = -1;
    for(let i=0; i<solver.ncoeffs; i++) {
        if(solver.sht.degree[i] === 2 && solver.sht.order[i] === 1) {
            k21 = i;
            break;
        }
    }
    solver.zetaCoeffs[2*k21] = 1.0;
    // disable spectral filter
    for(let i=0; i<solver.specFilter.length; i++) {
        solver.specFilter[i] = 1.0;
    }

    const initialEnergy = solver.getEnergy();

    // Advance a few steps
    for(let i=0; i<5; i++) {
        solver.advance();
    }

    const finalEnergy = solver.getEnergy();
    // Energy should be conserved exactly (within small numerical error of RK4)
    expect(Math.abs(finalEnergy - initialEnergy) / initialEnergy).toBeLessThan(0.01);
  });

  test('PT-02 Enstrophy Cascade', () => {
      const lmax = 16;
      const solver = new NSolver(lmax, 0.01, 1e-3);
      solver.initRandom(42);

      const E_start = solver.getEnergy();
      const getEnstrophy = () => {
          let Z = 0;
          for(let k=0; k<solver.sht.ncoeffs; k++) {
              const re = solver.zetaCoeffs[2*k];
              const im = solver.zetaCoeffs[2*k+1];
              const mult = solver.sht.order[k] === 0 ? 1 : 2;
              Z += mult * (re*re + im*im);
          }
          return Z;
      };

      const Z_start = getEnstrophy();

      for(let i=0; i<20; i++) {
          solver.advance();
      }

      const E_end = solver.getEnergy();
      const Z_end = getEnstrophy();

      // Energy drops slowly, Enstrophy drops faster
      expect(E_end).toBeLessThan(E_start);
      expect(Z_end).toBeLessThan(Z_start);

      const dE = (E_start - E_end) / E_start;
      const dZ = (Z_start - Z_end) / Z_start;
      expect(dZ).toBeGreaterThan(dE * 2); // Enstrophy cascades faster due to viscosity acting on high modes
  });
});
