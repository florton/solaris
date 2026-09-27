import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { defaultClientConditions, defineConfig, type Plugin } from 'vite';

// COOP/COEP make the page crossOriginIsolated, enabling SharedArrayBuffer —
// onnxruntime-web's wasm build deadlocks without it in some browsers.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

function filesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true, recursive: true })
    .filter((e) => e.isFile())
    .map((e) => join(e.parentPath, e.name))
    .sort();
}

/** Stamps dist/sw.js with a cache name derived from the unhashed files in
 *  public/ (weights, ORT runtime) — the only URLs that can go stale under a
 *  cache-first SW. Hashed assets/ get new URLs on change and don't need it. */
function swCacheVersion(): Plugin {
  let publicDir = '';
  let outDir = '';
  return {
    name: 'solaris-sw-cache-version',
    apply: 'build',
    configResolved(c) {
      publicDir = c.publicDir;
      outDir = resolve(c.root, c.build.outDir);
    },
    closeBundle() {
      const h = createHash('sha256');
      for (const f of filesUnder(publicDir)) {
        const rel = relative(publicDir, f).replaceAll('\\', '/');
        // .htaccess is server config, never fetched, so it can't go stale.
        if (rel === 'sw.js' || rel === '.htaccess') continue;
        h.update(rel).update('\0').update(readFileSync(f));
      }
      const swPath = join(outDir, 'sw.js');
      const sw = readFileSync(swPath, 'utf8');
      if (!sw.includes('__SOLARIS_CACHE__')) throw new Error('sw.js: __SOLARIS_CACHE__ placeholder missing');
      writeFileSync(swPath, sw.replace('__SOLARIS_CACHE__', `solaris-${h.digest('hex').slice(0, 12)}`));
    },
  };
}

export default defineConfig({
  base: './',
  // ORT's default entry inlines its wasm as a Vite asset (a 21 MB duplicate);
  // this condition picks the build that loads it from wasmPaths (public/ort/).
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions] },
  plugins: [swCacheVersion()],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 4096,
    manifest: true,
  },
});
