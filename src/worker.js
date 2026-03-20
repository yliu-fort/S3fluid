import { GPGPU, ShaderPass, PingPongFBO } from './gpgpu.js';
import { SHT } from './sht.js';
import { NSSolver } from './solver.js';

let gpgpu, sht, solver, stateFBO;
let stepCount = 0;
let simTime = 0.0;
let currentDt = 0.0;
let currentNu = 0.0;
let currentLMax = 0;

self.onmessage = (e) => {
    const msg = e.data;

    switch (msg.type) {
        case 'init':
        case 'reset':
            initSimulation(msg.lmax, msg.dt, msg.nu, msg.preset);
            break;
        case 'setParam':
            if (msg.key === 'dt') currentDt = msg.value;
            if (msg.key === 'nu') currentNu = msg.value;
            break;
        case 'step':
            runStep(msg.stepsPerFrame, msg.variable);
            break;
    }
};

function initSimulation(lmax, dt, nu, preset) {
    try {
        currentLMax = lmax;
        currentDt = dt;
        currentNu = nu;
        stepCount = 0;
        simTime = 0.0;

        // Cleanup old instances
        // We assume re-instantiating GPGPU might be expensive but clean.
        // Actually, workers can't share WebGL contexts across resets easily if not careful,
        // but since this is pure JS/WebGL context inside an OffscreenCanvas (if available)
        // Wait, WebGL in worker requires OffscreenCanvas.
        // We need to verify if the frontend passed an OffscreenCanvas or if we just create one.
        // GPGPU uses document.createElement('canvas'), which throws in Worker.
        // So we need to patch GPGPU to use OffscreenCanvas in worker context.

        // Wait, GPGPU in src/gpgpu.js:
        // const canvas = document.createElement('canvas');
        // This will fail in Worker. Let's fix that by providing a mock or OffscreenCanvas

        const canvas = new OffscreenCanvas(256, 256); // Size doesn't matter for pure GPGPU (we just use FBOs)
        const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });

        gpgpu = new GPGPU(gl);

        // Anti-aliasing rule: grid lonRes = 3 * lMax, latRes = 1.5 * lMax approx (actually let's say latRes = 1.5*lMax)
        // Wait, sht.js defaults mMax=lMax=lonRes/3, so lonRes = 3*lmax.
        const lonRes = lmax * 3;
        const latRes = Math.floor(lmax * 1.5);

        sht = new SHT(gpgpu, latRes, lonRes);
        solver = new NSSolver(gpgpu, sht);

        // Initial conditions
        stateFBO = new PingPongFBO(gpgpu, sht.mMax, sht.lMax);

        // Generate initial preset in grid space and transform
        const initialGrid = new Float32Array(lonRes * latRes * 4);

        for (let j = 0; j < latRes; j++) {
            const theta = (j + 0.5) * Math.PI / latRes;
            for (let i = 0; i < lonRes; i++) {
                const phi = i * 2.0 * Math.PI / lonRes;
                const idx = (j * lonRes + i) * 4;

                let val = 0.0;

                if (preset === 'random') {
                    // Random noise
                    val = (Math.random() - 0.5) * 2.0;
                } else if (preset === 'solid-body') {
                    // Vorticity for psi = -omega * cos(theta) is zeta = -2 omega cos(theta)
                    val = -2.0 * Math.cos(theta);
                } else if (preset === 'rossby') {
                    // Rossby wave m=4 approx
                    val = Math.cos(theta) + Math.pow(Math.sin(theta), 4) * Math.cos(4.0 * phi);
                }

                initialGrid[idx] = val; // R channel
            }
        }

        const gridTex = gpgpu.createTexture(lonRes, latRes, initialGrid);

        // Transform to spectral
        const specInitial = sht.forwardTransform(gridTex);

        // Copy to stateFBO
        const pass = new ShaderPass(gpgpu, `#version 300 es
            precision highp float;
            in vec2 v_uv;
            out vec4 outColor;
            uniform sampler2D tex;
            void main() { outColor = texture(tex, v_uv); }
        `);
        pass.render(stateFBO.writeFBO, sht.mMax, sht.lMax, {}, { tex: specInitial.tex });
        stateFBO.swap();

        gpgpu.gl.deleteTexture(gridTex);
        gpgpu.gl.deleteFramebuffer(specInitial.fbo);
        gpgpu.gl.deleteTexture(specInitial.tex);

        self.postMessage({ type: 'ready' });
    } catch (err) {
        self.postMessage({ type: 'error', message: err.toString() });
    }
}

function runStep(steps, variable) {
    if (!solver || !stateFBO) return;

    try {
        for (let i = 0; i < steps; i++) {
            solver.step(stateFBO, currentDt, currentNu);
            stepCount++;
            simTime += currentDt;
        }

        // Read back data to send to main thread
        // Instead of spectral, we send back physical grid
        const zetaGrid = sht.inverseTransform(stateFBO.read);

        // Calculate energy as 0.5 * sum(zeta_l^m ^ 2 / l(l+1)) -- wait, simply calculate L2 norm of grid
        // Actually, we'll just extract grid values
        const rawPixels = gpgpu.readPixels(zetaGrid.fbo, sht.lonRes, sht.latRes);
        const floatPixels = new Float32Array(rawPixels.buffer); // It's already float32

        // Extract Red channel (vorticity)
        const outputGrid = new Float32Array(sht.lonRes * sht.latRes);
        let minVal = 1e9, maxVal = -1e9;
        let energy = 0.0;

        for (let i = 0; i < outputGrid.length; i++) {
            const val = floatPixels[i * 4];
            outputGrid[i] = val;
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
            energy += val * val;
        }
        energy /= outputGrid.length; // rough proxy for enstrophy

        gpgpu.gl.deleteFramebuffer(zetaGrid.fbo);
        gpgpu.gl.deleteTexture(zetaGrid.tex);

        self.postMessage({
            type: 'frame',
            step: stepCount,
            time: simTime,
            energy: energy,
            min: minVal,
            max: maxVal,
            grid: outputGrid,
            nlon: sht.lonRes,
            nlat: sht.latRes
        }, [outputGrid.buffer]);

    } catch (err) {
        self.postMessage({ type: 'error', message: err.toString() });
    }
}
