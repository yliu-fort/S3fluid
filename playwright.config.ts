import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/playwright',
  webServer: {
    command: 'npx vite --port 8080 --host 127.0.0.1',
    url: 'http://127.0.0.1:8080',
    reuseExistingServer: true,
  },
  use: {
    baseURL: 'http://127.0.0.1:8080',
    browserName: 'chromium',
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,UseSkiaRenderer,WebGPU',
        '--use-vulkan=native',
        '--disable-vulkan-fallback-to-gl-for-testing',
        '--ignore-gpu-blocklist',
        '--no-sandbox',
        '--disable-setuid-sandbox'
      ]
    }
  },
});
