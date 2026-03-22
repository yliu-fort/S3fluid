import { defineConfig } from 'vite';

export default defineConfig({
  root: 'tests/playwright',
  server: {
    host: '127.0.0.1',
    port: 8080,
  },
  esbuild: {
    keepNames: true
  }
});
