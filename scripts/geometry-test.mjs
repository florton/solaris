// Sanity check for src/geometry.ts (compiled to .probe/geometry.mjs by esbuild).
import { cosineSimilarity, spectralEmbed, selectEdges, layerGeometry, jacobiEigen } from '../.probe/geometry.mjs';

let failures = 0;
const check = (name, cond) => { console.log(`${cond ? 'ok' : 'FAIL'}  ${name}`); if (!cond) failures++; };

// --- cosine similarity ---
const d = new Float32Array([1, 2, 3, 2, 4, 6]); // 2 rows x 3 dims, same direction
const sim = cosineSimilarity(d, 2, 3);
check('cos same-direction rows = 1', Math.abs(sim[1] - 1) < 1e-6);
const dOrtho = new Float32Array([1, 0, 0, 0, 1, 0]);
check('cos orthogonal rows = 0', Math.abs(cosineSimilarity(dOrtho, 2, 3)[1]) < 1e-6);

// --- jacobi on a known matrix: diag(1,2,3) ---
const n = 3;
const m = new Float64Array([1, 0, 0, 0, 2, 0, 0, 0, 3]);
const { values } = jacobiEigen(m, n);
check('jacobi diag eigenvalues', Math.abs(values[0] - 1) < 1e-9 && Math.abs(values[2] - 3) < 1e-9);

// --- jacobi on a rotation-symmetric matrix ---
const m2 = new Float64Array([2, 1, 0, 1, 2, 0, 0, 0, 5]);
const e2 = jacobiEigen(m2, 3);
check('jacobi 2x2 block eigenvalues 1,3,5', Math.abs(e2.values[0] - 1) < 1e-9 && Math.abs(e2.values[1] - 3) < 1e-9 && Math.abs(e2.values[2] - 5) < 1e-9);

// --- spectral embedding separates a cluster from an outlier ---
// tokens 0,1,2 near direction A; token 3 near orthogonal direction B
const rows = [];
const rand = (() => { let s = 42; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
const dims = 16;
const dirA = Array.from({ length: dims }, () => rand() - 0.5);
const dirB = Array.from({ length: dims }, () => rand() - 0.5);
const norm = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
const na = norm(dirA), nb = norm(dirB);
for (let i = 0; i < 4; i++) {
  const base = i < 3 ? dirA : dirB;
  const nb0 = i < 3 ? na : nb;
  rows.push(base.map((x) => (x / nb0) * 10 + (rand() - 0.5) * 0.5));
}
const hidden = new Float32Array(rows.flat());
const g = layerGeometry(hidden, 4, dims);
// outlier (token 3) should be far from the cluster centroid in embedding space
const c = [0, 1, 2].map((k) => [0, 1, 2].reduce((a, i) => a + g.positions[i * 3 + k], 0) / 3);
const distOutlier = Math.hypot(g.positions[9] - c[0], g.positions[10] - c[1], g.positions[11] - c[2]);
const distMember = Math.hypot(g.positions[0] - c[0], g.positions[1] - c[1], g.positions[2] - c[2]);
console.log(`  outlier distance ${distOutlier.toFixed(3)} vs member distance ${distMember.toFixed(3)}`);
check('outlier separated in embedding', distOutlier > distMember);

// --- determinism: same input twice ---
const g2 = layerGeometry(Float32Array.from(hidden), 4, dims);
check('deterministic positions', g.positions.every((v, i) => v === g2.positions[i]));

// --- edges ---
const edges = selectEdges(g.similarity, 4, 3, 0.2);
check('cluster members interconnected', edges.some(([i, j]) => i < 3 && j < 3));
console.log('  edges:', edges.map(([i, j, w]) => `${i}-${j}:${w.toFixed(2)}`).join(' '));

// --- n=1 and n=128 extremes ---
check('n=1 ok', layerGeometry(new Float32Array(dims), 1, dims).positions.length === 3);
const big = new Float32Array(128 * dims).map(() => rand());
const t0 = performance.now();
layerGeometry(big, 128, dims);
console.log(`  n=128 layerGeometry: ${(performance.now() - t0).toFixed(1)}ms`);

process.exit(failures ? 1 : 0);
