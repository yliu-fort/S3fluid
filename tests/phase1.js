import { GPGPUSystem } from '../src/gpgpu/System.js';
import { GPGPUShader } from '../src/gpgpu/Shader.js';
import { GPGPUFBO, PingPongFBO } from '../src/gpgpu/FBO.js';
import { readPixelsFloat, assertFloatArraysEqual, TestRunner } from '../src/gpgpu/TestUtils.js';

const runner = new TestRunner('test-runner');

async function runTests() {
  let system;

  await runner.run('UT-00: Initialize System', () => {
    system = new GPGPUSystem();
  });

  if (!system) return;

  await runner.run('UT-01: Float Texture Read/Write Accuracy', () => {
    const w = 4;
    const h = 4;
    const len = w * h * 4;
    const input = new Float32Array(len);

    // Fill with very small/large floats to test precision
    for (let i = 0; i < len; i += 4) {
      input[i + 0] = (i + 1) * 1.234567e-5; // R
      input[i + 1] = -(i + 2) * 9.87654e5;  // G
      input[i + 2] = Math.PI;               // B
      input[i + 3] = 1.0;                   // A
    }

    const fbo = new GPGPUFBO(system.gl, w, h, input);
    fbo.bind();

    const output = readPixelsFloat(system.gl, w, h);
    fbo.unbind();

    // The tolerance needs to cover standard 32-bit float quantization.
    assertFloatArraysEqual(output, input, 1e-6);
  });

  await runner.run('UT-01.5: Shader Basic Compilation & Ping-Pong FBO Warmup', () => {
    const w = 2;
    const h = 2;
    const len = w * h * 4;

    // Initial state: A simple grid of values
    const initial = new Float32Array(len);
    for (let i = 0; i < len; i += 4) {
      initial[i] = i / 4.0; // R channel holds index 0, 1, 2, 3
      initial[i+3] = 1.0;   // A
    }

    const ppFBO = new PingPongFBO(system.gl, w, h, initial);

    // Fragment shader: Read from texture and add a constant (e.g. 10.0)
    const fsSource = `
      uniform sampler2D uTexture;
      uniform float uAdd;
      in vec2 vUv;
      out vec4 fragColor;

      void main() {
        vec4 val = texture(uTexture, vUv);
        fragColor = vec4(val.r + uAdd, val.g, val.b, 1.0);
      }
    `;

    const shader = new GPGPUShader(system.gl, fsSource);
    shader.use();

    // Step 1: initial -> step 1
    ppFBO.write.bind();
    shader.setUniform1i('uTexture', 0); // texture unit 0
    shader.setUniform1f('uAdd', 10.0);

    system.gl.activeTexture(system.gl.TEXTURE0);
    system.gl.bindTexture(system.gl.TEXTURE_2D, ppFBO.read.texture);

    system.runPass();

    let out = readPixelsFloat(system.gl, w, h);

    // Check first step: R should be index + 10.0
    for(let i=0; i<w*h; i++) {
        if(Math.abs(out[i*4] - (i + 10.0)) > 1e-5) throw new Error(`Step 1 failed at ${i}: expected ${i + 10.0}, got ${out[i*4]}`);
    }

    ppFBO.write.unbind();
    ppFBO.swap();

    // Step 2: step 1 -> step 2
    ppFBO.write.bind();
    system.gl.bindTexture(system.gl.TEXTURE_2D, ppFBO.read.texture);
    system.runPass();

    out = readPixelsFloat(system.gl, w, h);
    for(let i=0; i<w*h; i++) {
        if(Math.abs(out[i*4] - (i + 20.0)) > 1e-5) throw new Error(`Step 2 failed at ${i}: expected ${i + 20.0}, got ${out[i*4]}`);
    }

    ppFBO.write.unbind();
  });
}

runTests();
