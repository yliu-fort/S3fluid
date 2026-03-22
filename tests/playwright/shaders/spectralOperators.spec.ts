import { test, expect } from '@playwright/test';
import { createConfig } from '../../../src/solver/config';

test.describe('WebGPU Spectral Operators Functionality', () => {
    test.beforeEach(async ({ page }) => {
        page.on('pageerror', error => console.error(error));
        await page.goto('/');
        await page.waitForFunction(() => (window as any).TestRunnerReady === true);
    });

    test('m=0 after mulIM is all zero', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const runner = (window as any).testRunner;
            const config = runner.config;
            const spectralSize = runner.buffers.zetaLM_A.size / 4;
            const inputLM = new Float32Array(spectralSize);

            // Populate some m=0, l=0, l=1, etc.
            const M = config.lmax + 1;
            const L = config.lmax + 1;
            for (let l = 0; l < L; l++) {
                const idx = (0 * L + l) * 2;
                inputLM[idx] = 1.0; // real part
                inputLM[idx + 1] = 1.0; // imag part
            }

            const outputLM = await runner.testMulIM(inputLM);
            return Array.from(outputLM);
        });

        const config = createConfig({ lmax: 31 });
        const L = config.lmax + 1;

        for (let l = 0; l < L; l++) {
            const idx = (0 * L + l) * 2;
            expect(result[idx]).toBeCloseTo(0.0, 5);
            expect(result[idx + 1]).toBeCloseTo(0.0, 5);
        }
    });

    test('applyLaplacian accurately multiplies by -l(l+1)', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const runner = (window as any).testRunner;
            const config = runner.config;
            const spectralSize = runner.buffers.zetaLM_A.size / 4;
            const inputLM = new Float32Array(spectralSize);

            const M = config.lmax + 1;
            const L = config.lmax + 1;
            for (let m = 0; m < M; m++) {
                for (let l = 0; l < L; l++) {
                    const idx = (m * L + l) * 2;
                    // the initial data is multiplied by the laplacian, we populate this buffer
                    inputLM[idx] = 1.0; // real part
                    inputLM[idx + 1] = -1.0; // imag part
                }
            }

            const outputLM = await runner.testApplyLaplacian(inputLM);
            return Array.from(outputLM);
        });

        const config = createConfig({ lmax: 31 });
        const M = config.lmax + 1;
        const L = config.lmax + 1;

        for (let m = 0; m < M; m++) {
            for (let l = 0; l < L; l++) {
                const idx = (m * L + l) * 2;
                const lapEig = -l * (l + 1);

                // The shader will output 0 for l < m, or apply the eigenvalue otherwise
                if (l < m) {
                    expect(result[idx]).toBeCloseTo(0.0, 5);
                    expect(result[idx + 1]).toBeCloseTo(0.0, 5);
                } else {
                    expect(result[idx]).toBeCloseTo(1.0 * lapEig, 5);
                    expect(result[idx + 1]).toBeCloseTo(-1.0 * lapEig, 5);
                }
            }
        }
    });

    test('invertLaplacian(applyLaplacian(a)) ≈ a (ignoring l=0)', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const runner = (window as any).testRunner;
            const config = runner.config;
            const spectralSize = runner.buffers.zetaLM_A.size / 4;
            const inputLM = new Float32Array(spectralSize);

            const M = config.lmax + 1;
            const L = config.lmax + 1;
            for (let m = 0; m < M; m++) {
                for (let l = 0; l < L; l++) {
                    const idx = (m * L + l) * 2;
                    // Create stable random data, not true random, to ensure predictable precision.
                    inputLM[idx] = (m + l) * 0.1; // real part
                    inputLM[idx + 1] = (m - l) * 0.1; // imag part
                }
            }

            // testApplyLaplacian reads from zetaLM_A and outputs to tmpLM
            const lapLM = await runner.testApplyLaplacian(inputLM);

            // testInvertLaplacian also needs to read from zetaLM_A and output to tmpLM
            // So we need to put lapLM into zetaLM_A before calling it.
            // But wait, the runner does:
            // this.uploadFloat32Array(this.buffers.zetaLM_A, inputLM);
            // which is good, we can just pass lapLM in.
            const outputLM = await runner.testInvertLaplacian(lapLM);

            return {
                input: Array.from(inputLM),
                output: Array.from(outputLM)
            };
        });

        const config = createConfig({ lmax: 31 });
        const M = config.lmax + 1;
        const L = config.lmax + 1;

        const input = result.input;
        const output = result.output;

        for (let m = 0; m < M; m++) {
            for (let l = 0; l < L; l++) {
                const idx = (m * L + l) * 2;
                if (l === 0) {
                    expect(output[idx]).toBe(0.0);
                    expect(output[idx + 1]).toBe(0.0);
                } else {
                    if (l < m) {
                        expect(output[idx]).toBeCloseTo(0.0, 5);
                        expect(output[idx + 1]).toBeCloseTo(0.0, 5);
                    } else {
                        // Output should match input because we inverted the laplacian exactly
                        expect(output[idx]).toBeCloseTo(input[idx], 4); // Precision loss across WGSL math
                        expect(output[idx + 1]).toBeCloseTo(input[idx + 1], 4);
                    }
                }
            }
        }
    });

    test('filterSpectrum does not change l=0', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const runner = (window as any).testRunner;
            const config = runner.config;
            const spectralSize = runner.buffers.zetaLM_A.size / 4;
            const inputLM = new Float32Array(spectralSize);

            const L = config.lmax + 1;
            const idx = (0 * L + 0) * 2;
            inputLM[idx] = 42.0; // real part
            inputLM[idx + 1] = -42.0; // imag part

            const outputLM = await runner.testFilterSpectrum(inputLM);
            return Array.from(outputLM);
        });

        const config = createConfig({ lmax: 31 });
        const L = config.lmax + 1;
        const idx = (0 * L + 0) * 2;

        expect(result[idx]).toBeCloseTo(42.0, 5);
        expect(result[idx + 1]).toBeCloseTo(-42.0, 5);
    });

    test('High-frequency modes are reduced by filterSpectrum', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const runner = (window as any).testRunner;
            const config = runner.config;
            const spectralSize = runner.buffers.zetaLM_A.size / 4;
            const inputLM = new Float32Array(spectralSize);

            const M = config.lmax + 1;
            const L = config.lmax + 1;
            // set all to 1.0 to easily check reduction
            for (let m = 0; m < M; m++) {
                for (let l = m; l < L; l++) {
                    const idx = (m * L + l) * 2;
                    inputLM[idx] = 1.0; // real part
                    inputLM[idx + 1] = 0.0; // imag part
                }
            }

            const outputLM = await runner.testFilterSpectrum(inputLM);
            return Array.from(outputLM);
        });

        const config = createConfig({ lmax: 31 });
        const M = config.lmax + 1;
        const L = config.lmax + 1;

        let prevMag = 1.0;
        // Check for m=0
        for (let l = 1; l < L; l++) {
            const idx = (0 * L + l) * 2;
            const magnitude = Math.sqrt(result[idx] * result[idx] + result[idx + 1] * result[idx + 1]);

            // Should be smaller than or equal to the lower mode
            expect(magnitude).toBeLessThanOrEqual(prevMag);
            prevMag = magnitude;
        }

        // Check specifically that the highest frequency is heavily filtered
        const highestL = config.lmax;
        const highestIdx = (0 * L + highestL) * 2;
        const highestMag = Math.sqrt(result[highestIdx] * result[highestIdx] + result[highestIdx + 1] * result[highestIdx + 1]);

        // expected factor is roughly exp(-alpha * (l/lmax)^order) = exp(-36.0 * 1^8) = exp(-36) which is tiny
        expect(highestMag).toBeLessThan(1e-5);
    });
});
