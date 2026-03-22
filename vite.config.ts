
import { UserConfig } from 'vite';

const config: UserConfig = {
  root: 'tests/playwright',
  server: {
    host: '127.0.0.1',
    port: 8080,
  },
  esbuild: {
    keepNames: true
  }
};
export default config;
