import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'esbuild'
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/samples': 'http://localhost:8000'
    }
  }
})
