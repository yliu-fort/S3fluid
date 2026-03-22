import { test, expect } from '@playwright/test';

test('check scalar data is not all zero', async ({ page }) => {
    page.on('console', msg => console.log('BROWSER CONSOLE:', msg.text()));
    page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));

    await page.goto('http://localhost:5173/');

    // Pause immediately
    await page.evaluate(() => {
        if ((window as any).app) {
            (window as any).app.gui.state.pause = true;
            (window as any).app.gui.state.lmax = 127;
            if ((window as any).SOLVER_CONFIG) {
                (window as any).SOLVER_CONFIG.lmax = 127;
            }
            (window as any).app.gui.state.reset();
        }
    });

    // Wait for initial WebGPU calculations to start
    await page.waitForTimeout(4000);

    // Evaluate window.app.scalarData again to get sum
    const dataSum = await page.evaluate(() => {
        const app = (window as any).app;
        if (!app) return { error: 'app is null' };
        if (!app.scalarData) return { error: 'app.scalarData is null' };

        let sum = 0;
        let hasNaN = false;
        let countNonZero = 0;

        for (let i = 0; i < app.scalarData.length; i++) {
            const v = app.scalarData[i];
            if (Number.isNaN(v)) hasNaN = true;
            if (v !== 0) countNonZero++;
            sum += Math.abs(v);
        }

        return { sum, hasNaN, countNonZero, length: app.scalarData.length };
    });

    console.log("Scalar Data Report:", dataSum);
    expect(dataSum).not.toBeNull();
    expect(dataSum!.countNonZero).toBeGreaterThan(0);
});
