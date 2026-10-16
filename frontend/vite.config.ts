import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '')
  const apiTarget = env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8780'

  return {
    plugins: [react()],
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.indexOf('node_modules/recharts/') >= 0) return 'recharts'
            if (id.indexOf('node_modules/d3-') >= 0) return 'd3'
          },
        },
      },
    },
    server: {
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: path => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
