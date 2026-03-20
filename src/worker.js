/**
 * worker.js - Web Worker for NS Solver
 *
 * Runs the NS solver in a background thread to prevent UI blocking.
 * Messages from main thread:
 *   { type: 'init',    lmax, dt, nu, preset }
 *   { type: 'step',    stepsPerFrame, variable }
 *   { type: 'setParam', key, value }
 *   { type: 'reset',   preset }
 *
 * Messages to main thread:
 *   { type: 'ready' }
 *   { type: 'frame', grid: Float32Array, min, max, nlat, nlon, step, time, energy }
 *   { type: 'error', message }
 */

import { NSolver } from './solver.js';

let solver = null;

self.onmessage = function(e) {
  const msg = e.data;

  try {
    if (msg.type === 'init') {
      const { lmax = 32, dt = 0.5, nu = 1e-4, preset = 'random' } = msg;
      solver = new NSolver(lmax, dt, nu);
      applyPreset(solver, preset);
      self.postMessage({ type: 'ready', nlat: solver.sht.nlat, nlon: solver.sht.nlon });

    } else if (msg.type === 'step') {
      if (!solver) return;
      const { stepsPerFrame = 1, variable = 'vorticity' } = msg;
      for (let i = 0; i < stepsPerFrame; i++) {
        solver.advance();
      }
      const { grid, min, max, nlat, nlon } = solver.getGridData(variable);
      const energy = solver.getEnergy();
      self.postMessage(
        { type: 'frame', grid, min, max, nlat, nlon,
          step: solver.step, time: solver.time, energy },
        [grid.buffer]
      );

    } else if (msg.type === 'setParam') {
      if (!solver) return;
      const { key, value } = msg;
      if (key === 'dt')  solver.dt  = value;
      if (key === 'nu')  solver.nu  = value;

    } else if (msg.type === 'reset') {
      if (!solver) return;
      const { preset = 'random', lmax, dt, nu } = msg;
      if (lmax !== undefined && lmax !== solver.lmax) {
        solver = new NSolver(lmax, dt ?? solver.dt, nu ?? solver.nu);
      }
      if (dt !== undefined) solver.dt = dt;
      if (nu !== undefined) solver.nu = nu;
      applyPreset(solver, preset);
      self.postMessage({ type: 'ready', nlat: solver.sht.nlat, nlon: solver.sht.nlon });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message + '\n' + err.stack });
  }
};

function applyPreset(solver, preset) {
  if (preset === 'solid-body') {
    solver.initSolidBodyRotation(1.0);
  } else if (preset === 'rossby') {
    solver.initRossbyHaurwitz(4);
  } else {
    solver.initRandom(42);
  }
  solver.step = 0;
  solver.time = 0;
}
