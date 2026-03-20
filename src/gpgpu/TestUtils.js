/**
 * Reads 32-bit floating point pixels from the currently bound framebuffer.
 * @param {WebGL2RenderingContext} gl
 * @param {number} width
 * @param {number} height
 * @returns {Float32Array} Array of length width * height * 4 (RGBA)
 */
export function readPixelsFloat(gl, width, height) {
  const result = new Float32Array(width * height * 4);
  // Read using RGBA format and FLOAT type
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, result);
  return result;
}

/**
 * Asserts that two Float32Arrays are almost equal within a given tolerance.
 */
export function assertFloatArraysEqual(actual, expected, tolerance = 1e-6) {
  if (actual.length !== expected.length) {
    throw new Error(`Assertion failed: Array lengths differ. Actual ${actual.length}, Expected ${expected.length}`);
  }
  let maxDiff = 0;
  let maxIdx = -1;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
      maxIdx = i;
    }
  }
  if (maxDiff > tolerance) {
    throw new Error(`Assertion failed: Max difference ${maxDiff.toExponential(3)} at index ${maxIdx} exceeds tolerance ${tolerance}. Actual: ${actual[maxIdx]}, Expected: ${expected[maxIdx]}`);
  }
}

/**
 * Basic test runner UI helper.
 */
export class TestRunner {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = containerId;
      document.body.appendChild(this.container);
    }
    this.container.style.fontFamily = 'monospace';
    this.container.style.padding = '1rem';
    this.container.style.background = '#1e1e1e';
    this.container.style.color = '#fff';

    const title = document.createElement('h2');
    title.textContent = "GPGPU Phase 1 Tests";
    this.container.appendChild(title);
  }

  log(msg, type = 'info') {
    const el = document.createElement('div');
    el.textContent = `[${type.toUpperCase()}] ${msg}`;
    if (type === 'pass') el.style.color = '#4ade80';
    if (type === 'fail') el.style.color = '#f87171';
    if (type === 'warn') el.style.color = '#facc15';
    this.container.appendChild(el);
  }

  async run(name, fn) {
    this.log(`Running ${name}...`, 'info');
    try {
      await fn();
      this.log(`${name} PASSED`, 'pass');
    } catch (e) {
      this.log(`${name} FAILED: ${e.message}`, 'fail');
      console.error(e);
      const trace = document.createElement('pre');
      trace.textContent = e.stack;
      trace.style.color = '#f87171';
      trace.style.fontSize = '0.8em';
      this.container.appendChild(trace);
    }
  }
}
