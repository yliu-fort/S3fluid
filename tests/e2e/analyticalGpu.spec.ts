import { test, expect } from '@playwright/test';

test('Solid body rotation (analytical Y_1^0) decays exactly according to nu on WebGPU', async ({ page }) => {
    // Navigate to the app
    await page.goto('http://localhost:5173/');

    // Pause immediately and inject Y_1^0 initial condition
    const result = await page.evaluate(async () => {
        const app = (window as any).app;
        const config = (window as any).SOLVER_CONFIG;

        if (!app || !config) return { error: "App not loaded" };

        // 1. Pause GUI
        app.gui.state.pause = true;

        // Wait a bit for device to be fully ready
        await new Promise(r => setTimeout(r, 1000));

        // 2. Set lmax=15, nu=0.01, dt=0.01
        config.lmax = 15;
        config.nu = 0.01;
        config.dt = 0.01;
        config.filterAlpha = 0; // Disable filter for pure analytical test

        await app.rebuildSimulation();

        // 3. Inject pure Y_1^0 mode
        const M = config.lmax + 1;
        const L = config.lmax + 1;

        // Complex float arrays (M * L * 2)
        const initial_zeta = new Float32Array(M * L * 2);

        // Mode l=1, m=0
        const target_m = 0;
        const target_l = 1;
        const target_idx = (target_m * L + target_l) * 2;

        initial_zeta[target_idx] = 1.0; // Real part = 1.0

        // Upload to GPU buffer zetaLM_A
        app.device.queue.writeBuffer(app.buffers.zetaLM_A, 0, initial_zeta);

        // We run EXACTLY one RK4 step
        // We override stepsPerFrame = 1
        config.stepsPerFrame = 1;
        app.gui.state.stepsPerFrame = 1;

        // Step once manually
        app.gui.state.pause = false; // unpause to allow step
        app.step();
        app.gui.state.pause = true; // repause

        // Read back the state from zetaLM_A
        const readBuffer = app.device.createBuffer({
            size: initial_zeta.byteLength,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const commandEncoder = app.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(app.buffers.zetaLM_A, 0, readBuffer, 0, readBuffer.size);
        app.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const final_zeta = new Float32Array(readBuffer.getMappedRange());

        // We read the final values
        const final_val = final_zeta[target_idx];

        // Also check that other modes are 0 (e.g. l=2, m=0)
        const l2_idx = (0 * L + 2) * 2;
        const noise_val = final_zeta[l2_idx];

        readBuffer.unmap();

        return {
            final_val,
            noise_val,
            expected: Math.exp(-2.0 * config.nu * config.dt)
        };
    });

    console.log("Analytical WebGPU Result:", result);

    expect(result).not.toHaveProperty('error');

    // Using closeTo for 32-bit float accuracy
    expect(result.final_val).toBeCloseTo(result.expected, 4);
    expect(result.noise_val).toBeCloseTo(0.0, 5);
});
