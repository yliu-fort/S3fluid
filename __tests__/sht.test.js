import { describe, expect, test } from '@jest/globals';
import { gaussLegendre, SHT } from '../src/sht.js';

describe('SHT tests', () => {
  test('gaussLegendre test', () => {
    const { x, w } = gaussLegendre(2);
    expect(x[0]).toBeCloseTo(-0.57735, 4);
    expect(x[1]).toBeCloseTo(0.57735, 4);
    expect(w[0]).toBeCloseTo(1, 4);
    expect(w[1]).toBeCloseTo(1, 4);
  });

  test('UT-01 SHT Forward and Inverse Transform', () => {
    const lmax = 16;
    const sht = new SHT(lmax);
    const grid = new Float64Array(sht.nlat * sht.nlon);

    for(let i=0; i<sht.nlat; i++) {
        for(let j=0; j<sht.nlon; j++) {
            grid[i*sht.nlon + j] = sht.cosTheta[i];
        }
    }

    const coeffs = sht.analys(grid);
    const newGrid = sht.synth(coeffs);

    let maxErr = 0;
    for(let i=0; i<grid.length; i++) {
        maxErr = Math.max(maxErr, Math.abs(grid[i] - newGrid[i]));
    }

    expect(maxErr).toBeLessThan(1e-12);
  });

  test('UT-03 SHT Synth Grad', () => {
    const lmax = 16;
    const sht = new SHT(lmax);

    const psiCoeffs = new Float64Array(2 * sht.ncoeffs);
    let k10 = -1;
    for(let i=0; i<sht.ncoeffs; i++) {
        if(sht.degree[i] === 1 && sht.order[i] === 0) {
            k10 = i;
            break;
        }
    }

    psiCoeffs[2*k10] = Math.sqrt(4 * Math.PI / 3);

    const {uPhi, uTheta} = sht.synthGrad(psiCoeffs);

    let maxErrPhi = 0;
    let maxErrTheta = 0;
    for(let i=0; i<sht.nlat; i++) {
        const sinT = Math.sqrt(Math.max(0, 1 - sht.cosTheta[i]*sht.cosTheta[i]));
        const expectedUTheta = -sinT;

        // P_1^0(x) = sqrt(3/(4pi)) x
        // grad_theta(Y_1^0) = -sin(theta) sqrt(3/(4pi))
        // So with coefficient sqrt(4pi/3), gradient in theta is -sin(theta)

        for(let j=0; j<sht.nlon; j++) {
            maxErrPhi = Math.max(maxErrPhi, Math.abs(uPhi[i*sht.nlon+j]));
            maxErrTheta = Math.max(maxErrTheta, Math.abs(uTheta[i*sht.nlon+j] - expectedUTheta));
        }
    }

    expect(maxErrPhi).toBeLessThan(1e-12);
    expect(maxErrTheta).toBeLessThan(1e-12);
  });
});
