import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: true,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/personas': 'http://127.0.0.1:8000',
      '/sessions': 'http://127.0.0.1:8000',
      '/artifacts': 'http://127.0.0.1:8000',
    },
  },
});
