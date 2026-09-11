import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  css: {
    postcss: './postcss.config.js',
  },
  optimizeDeps: {
    // Prevent Vite from trying to pre-bundle the worker chunk
    exclude: ['maplibre-gl'],
  },
  server: {
    // Proxy FastAPI backend so /api/* calls don't hit CORS in dev
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path,
      },
    },
  },
})
