import { GPGPU } from '../src/gpgpu.js';
import { SHT } from '../src/sht.js';

// The analytical function:
// f(theta, phi) = 1.0 + cos(theta) + sin(theta)cos(phi) + sin^2(theta)cos(2phi) + sin^3(theta)sin(3phi) + cos^4(theta) - sin^4(theta)cos(4phi)
function analyticFunction(theta, phi) {
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    return 1.0
         + cosT
         + sinT * Math.cos(phi)
         + Math.pow(sinT, 2) * Math.cos(2.0 * phi)
         + Math.pow(sinT, 3) * Math.sin(3.0 * phi)
         + Math.pow(cosT, 4)
         - Math.pow(sinT, 4) * Math.cos(4.0 * phi);
}

async function runTest() {
    const resultsDiv = document.getElementById('results');
    const statusDiv = document.getElementById('status');

    try {
        const gpgpu = new GPGPU();
        const latRes = 128;
        const lonRes = 256;
        const sht = new SHT(gpgpu, latRes, lonRes);

        // 1. Generate Input Data
        const inputData = new Float32Array(lonRes * latRes * 4);
        const PI = Math.PI;

        for (let i = 0; i < latRes; i++) {
            // Note: Nodes are roughly mapped to latRes
            // For a perfectly exact match to continuous, we use the arccos of Gauss-Legendre roots
            let theta = Math.acos(sht.nodes[i]);

            for (let j = 0; j < lonRes; j++) {
                let phi = (j + 0.5) * (2.0 * PI / lonRes); // cell-centered longitude

                let val = analyticFunction(theta, phi);
                let idx = (i * lonRes + j) * 4;
                inputData[idx] = val; // R channel
                // other channels remain 0
            }
        }

        const inputTex = gpgpu.createTexture(lonRes, latRes, inputData);

        // 2. Perform Transform
        // Forward
        const spectralResult = sht.forwardTransform(inputTex);

        // Inverse
        const outputResult = sht.inverseTransform(spectralResult.tex);
        const outputData = outputResult.data;

        // 3. Compute Metrics
        let maxError = 0;
        let l2ErrorSum = 0;

        for (let i = 0; i < latRes; i++) {
            let theta = Math.acos(sht.nodes[i]);

            for (let j = 0; j < lonRes; j++) {
                let phi = (j + 0.5) * (2.0 * PI / lonRes);
                let expected = analyticFunction(theta, phi);

                let idx = (i * lonRes + j) * 4;
                let actual = outputData[idx];

                let err = Math.abs(expected - actual);
                maxError = Math.max(maxError, err);
                l2ErrorSum += err * err;
            }
        }

        let l2Error = Math.sqrt(l2ErrorSum / (latRes * lonRes));

        const resultHTML = `
            <h2>Results</h2>
            <ul>
                <li><strong>Max Error:</strong> <span id="maxError">${maxError}</span></li>
                <li><strong>L2 Error:</strong> <span id="l2Error">${l2Error}</span></li>
            </ul>
        `;

        resultsDiv.innerHTML = resultHTML;

        // Given Float32 texture limits and float precision in shaders, a max error of < 1e-3 is considered acceptable
        if (maxError < 1e-3) {
            statusDiv.innerHTML = "<span style='color:green'>Test Passed!</span>";
        } else {
            statusDiv.innerHTML = "<span style='color:red'>Test Failed! Max error is too high.</span>";
        }

        // Mark DOM for playwright hook
        window.__testComplete = true;
        window.__testMaxError = maxError;
        window.__testL2Error = l2Error;

    } catch (e) {
        statusDiv.innerHTML = `<span style='color:red'>Error: ${e.message}</span>`;
        console.error(e);
        window.__testComplete = true;
        window.__testError = e.message;
    }
}

runTest();