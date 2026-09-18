import { defineConfig } from 'vite';

// COOP/COEP make the page crossOriginIsolated, enabling SharedArrayBuffer —
// onnxruntime-web's wasm build deadlocks without it in some browsers.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  base: './',
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 4096,
    manifest: true,
  },
});
