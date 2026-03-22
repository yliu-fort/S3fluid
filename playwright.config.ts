import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30 * 1000,
  expect: {
    timeout: 5000
  },
  use: {
    headless: true,
    launchOptions: {
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,UseSkiaRenderer,WebGPU',
        '--use-vulkan=native',
        '--disable-vulkan-fallback-to-gl-for-testing',
        '--ignore-gpu-blocklist',
        '--use-angle=vulkan'
      ],
    },
  },
});
