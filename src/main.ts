// Solaris — a thought goes in, a hologram of what the model holds together
// comes out. Boot: load weights -> run -> render -> idle drift.
import { SolarisModel } from './model.ts';
import { buildSculpture, type Sculpture } from './sculpture.ts';
import { Renderer } from './renderer.ts';
import { UI, encodePermalink, decodePermalink } from './ui.ts';

const W_SPAN = 1.5;
const canvas = document.getElementById('gl') as HTMLCanvasElement;
const ui = new UI(runThought);

let renderer: Renderer;
let model: SolarisModel;
let sculpture: Sculpture | null = null;
let loadMs = 0;
let lastForwardMs = 0;
let lastEigenMs = 0;
let lastFormMs: number | undefined;
let lastFormGrid: number | undefined;

// interaction state
let dragging = false;
let lastX = 0;
let lastY = 0;
let downX = 0;
let downY = 0;
let focusTarget = 0;
let lastInteract = 0;
let fps = 60;

async function runThought(text: string): Promise<void> {
  ui.unpin();
  ui.hideLabel(true);
  const result = await model.forward(text);
  sculpture = buildSculpture(result);
  renderer.resetVolumes();
  renderer.setSculpture(sculpture);
  // the sculpture *appears*: sweep up the layer axis from below the stack
  renderer.focus = -(W_SPAN + 0.6);
  focusTarget = 0;
  renderer.render();
  ui.buildGauge(sculpture.nLayers);
  lastForwardMs = result.forwardMs;
  lastEigenMs = sculpture.eigenMs;
  lastFormMs = undefined;
  lastFormGrid = undefined;
  history.replaceState(null, '', encodePermalink(text));
  updateStats();

  // kick the imagination: pooled layer embeddings -> latents -> SDF grids,
  // arriving per layer while the thought cloud holds the stage
  const sc = sculpture;
  const mat = model.materialize(result, (layer, sdf, grid) => {
    if (sculpture !== sc) return; // superseded by a newer thought
    if (renderer.volCount === 0) renderer.initVolumes(sc.nLayers, grid, model.imagination!.bound);
    renderer.setLayerVolume(layer, sdf);
    if (renderer.volCount === sc.nLayers) renderer.matTarget = 1; // condense
  });
  void mat?.then((r) => {
    if (r && sculpture === sc) {
      lastFormMs = r.ms;
      lastFormGrid = r.grid;
      updateStats();
    }
  });
}

function pickToken(x: number, y: number): { token: string; sx: number; sy: number } | null {
  if (!sculpture) return null;
  const proj = renderer.projectVertices();
  // candidates: near the pointer and near the focal layer
  const hits: number[] = [];
  let nearestDepth = Infinity;
  for (let v = 0; v < sculpture.vertexCount; v++) {
    if (proj[v * 4 + 2] < 0.25) continue;
    const d = Math.hypot(proj[v * 4] - x, proj[v * 4 + 1] - y);
    if (d > 30) continue;
    hits.push(v);
    nearestDepth = Math.min(nearestDepth, proj[v * 4 + 3]);
  }
  if (!hits.length) return null;
  // of the candidates, prefer the front shell of the form
  let best = -1;
  let bestD = Infinity;
  for (const v of hits) {
    if (proj[v * 4 + 3] > nearestDepth + 1.0) continue;
    const d = Math.hypot(proj[v * 4] - x, proj[v * 4 + 1] - y);
    if (d < bestD) {
      bestD = d;
      best = v;
    }
  }
  if (best < 0) return null;
  return { token: sculpture.tokens[sculpture.tokenOf[best]], sx: proj[best * 4], sy: proj[best * 4 + 1] };
}

function wireInteraction(): void {
  const hoverAt = (x: number, y: number) => {
    if (ui.pinned) return;
    const hit = pickToken(x, y);
    if (hit) ui.showLabel(hit.token, hit.sx + 14, hit.sy - 10);
    else ui.hideLabel();
  };
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = downX = e.clientX;
    lastY = downY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    lastInteract = performance.now();
  });
  canvas.addEventListener('pointermove', (e) => {
    lastInteract = performance.now();
    if (dragging) {
      renderer.yaw += (e.clientX - lastX) * 0.005;
      renderer.pitch = Math.max(-1.4, Math.min(1.4, renderer.pitch + (e.clientY - lastY) * 0.005));
      lastX = e.clientX;
      lastY = e.clientY;
      if (!ui.pinned) ui.hideLabel();
    } else {
      hoverAt(e.clientX, e.clientY);
    }
  });
  // some drivers send only mousemove; hover-pick is idempotent
  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) hoverAt(e.clientX, e.clientY);
  });
  canvas.addEventListener('pointerup', (e) => {
    dragging = false;
    // tap (not drag): reveal + pin the token under the pointer — the touch
    // equivalent of hover
    const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
    if (moved < 6) {
      if (ui.pinned) {
        ui.unpin();
        ui.hideLabel(true);
      } else {
        hoverAt(e.clientX, e.clientY);
        ui.togglePin();
      }
    }
  });
  window.addEventListener(
    'wheel',
    (e) => {
      focusTarget = Math.max(-W_SPAN - 0.4, Math.min(W_SPAN + 0.4, focusTarget + e.deltaY * 0.0016));
      lastInteract = performance.now();
    },
    { passive: true },
  );
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') focusTarget = Math.min(W_SPAN + 0.4, focusTarget + 0.22);
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') focusTarget = Math.max(-W_SPAN - 0.4, focusTarget - 0.22);
    lastInteract = performance.now();
  });
}

function updateStats(): void {
  if (!sculpture || !model) return;
  ui.setStats({
    modelName: 'MiniLM-L6',
    params: model.meta.params,
    dtype: model.meta.dtype,
    backend: model.backend,
    loadMs,
    forwardMs: lastForwardMs,
    eigenMs: lastEigenMs,
    tokens: sculpture.nTokens,
    layers: sculpture.nLayers,
    fps: Math.round(fps),
    formMs: lastFormMs,
    formGrid: lastFormGrid,
    offline: cacheReport || undefined,
  });
}

let lastFrame = 0;
function frame(now: number): void {
  const idle = now - lastInteract > 3000;
  if (idle) {
    renderer.yaw += 0.00022; // glacial rotation
    focusTarget = Math.sin(now * 0.00011) * (W_SPAN + 0.2); // slow depth drift
    renderer.rotXW = Math.sin(now * 0.000043) * 0.3;
  }
  renderer.focus += (focusTarget - renderer.focus) * 0.08;
  renderer.render();
  ui.setGauge((renderer.focus / W_SPAN + 1) / 2);

  if (lastFrame) fps = fps * 0.95 + 0.05 * (1000 / Math.max(1, now - lastFrame));
  lastFrame = now;
  updateStats();
  requestAnimationFrame(frame);
}

async function boot(): Promise<void> {
  try {
    renderer = new Renderer(canvas);
  } catch {
    ui.showError('this piece needs WebGL2 — your browser said no.');
    return;
  }
  ui.setStatus('first visit: ~23 MB of weights, then it is all yours. nothing leaves this machine.');
  window.addEventListener('error', (e) => ui.setStatus(`error: ${e.message}`));
  window.addEventListener('unhandledrejection', (e) => ui.setStatus(`error: ${e.reason instanceof Error ? e.reason.message : String(e.reason)}`));
  try {
    const t0 = performance.now();
    model = await SolarisModel.load('./', (msg) => ui.setStatus(msg));
    loadMs = performance.now() - t0;
  } catch (e) {
    ui.showError(`model failed to load: ${e instanceof Error ? e.message : String(e)}`);
    return;
  }
  ui.hideStatus();
  wireInteraction();

  const shared = decodePermalink(location.hash);
  if (shared) {
    ui.setInput(shared);
    await runThought(shared);
  }
  requestAnimationFrame(frame);
}

if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}

/** After a successful first load, hand the service worker every URL the
 *  piece needs; the SW owns the cache (single source for the cache name). */
async function cacheEverything(): Promise<string> {
  if (!('serviceWorker' in navigator)) return 'no service worker — online only';
  try {
    const urls = new Set(performance.getEntriesByType('resource').map((r) => r.name));
    const base = new URL('./', document.baseURI).href;
    urls.add(base);
    if (model?.workerUrl) urls.add(model.workerUrl);
    for (const p of [
      'models/minilm-l6/solaris_model.json',
      'models/minilm-l6/vocab.txt',
      'models/minilm-l6/onnx/model_quantized.onnx',
      'models/imagination/imagination_model.json',
      'models/imagination/decoder.onnx',
      'ort/ort-wasm-simd-threaded.jsep.wasm',
      'ort/ort-wasm-simd-threaded.jsep.mjs',
      'ort/ort-wasm-simd-threaded.wasm',
      'ort/ort-wasm-simd-threaded.mjs',
    ]) {
      urls.add(new URL(p, base).href);
    }
    const total = urls.size;
    const reg = await navigator.serviceWorker.ready;
    const sw = reg.active;
    if (!sw) return 'offline caching unavailable';
    return await new Promise<string>((resolve) => {
      const onMsg = (e: MessageEvent) => {
        if (e.data?.type === 'precache-done') {
          navigator.serviceWorker.removeEventListener('message', onMsg);
          resolve(`${e.data.ok}/${e.data.total} files cached for offline`);
        }
      };
      navigator.serviceWorker.addEventListener('message', onMsg);
      sw.postMessage({ type: 'precache', urls: [...urls] });
      setTimeout(() => resolve(`caching timed out (0/${total})`), 30000);
    });
  } catch (e) {
    return `cache failed: ${e instanceof Error ? e.message : String(e)}`;
  }
}

let cacheReport = '';
void boot().then(() => {
  if (import.meta.env.PROD)
    void cacheEverything().then((r) => {
      cacheReport = r;
      updateStats();
    });
});
