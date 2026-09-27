// Solaris — a thought goes in, a hologram of what the model holds together
// comes out. Boot: load weights -> tour the preset thoughts (each: cloud ->
// dreamed forms -> condense -> hold) -> a typed thought takes over, and the
// tour resumes once the viewer goes idle.
import { SolarisModel, type ForwardResult } from './model.ts';
import { buildSculpture, type Sculpture } from './sculpture.ts';
import { Renderer } from './renderer.ts';
import { UI, encodePermalink, decodePermalink } from './ui.ts';
import { PRESETS, TOUR } from './presets.ts';

const W_SPAN = 1.5;
const canvas = document.getElementById('gl') as HTMLCanvasElement;
const ui = new UI((text) => void runUserThought(text));

let renderer: Renderer;
let model: SolarisModel;
let sculpture: Sculpture | null = null;
let loadMs = 0;

// interaction state
let dragging = false;
let lastX = 0;
let lastY = 0;
let downX = 0;
let downY = 0;
let focusTarget = 0;
let tiltTarget = 0; // x-w tilt of the slicing hyperplane (renderer.rotXW eases to it)
const clampTilt = (a: number) => Math.max(-0.7, Math.min(0.7, a));
let lastInteract = 0;
let fps = 60;

/** A thought, encoded, with its dream arriving layer by layer. */
interface Prepared {
  text: string;
  result: ForwardResult;
  layers: (Uint16Array | null)[]; // half-float full SDFs from the dream worker
  sources: string[]; // per layer: the library form it was dreamed from
  forms: number[]; // per layer: that form's library index (for its credit)
  melts: number[]; // per neighbouring pair: the melt's half-way inflation
  dreamId: number | null; // while the dream is queued or running
  dreamMs?: number; // set once every layer has arrived
  failed?: string;
}

let current: Prepared | null = null;
let shownAt = 0;
let condensedAt = 0;

/** Encode the thought and queue its dream (the dream worker runs them in order). */
async function prepare(text: string): Promise<Prepared> {
  const result = await model.forward(text);
  const p: Prepared = { text, result, layers: new Array(result.nLayers).fill(null), sources: [], forms: [], melts: [], dreamId: null };
  p.dreamId = model.requestDream(text, result, {
    onLayer: (layer, vol, source, index) => {
      p.layers[layer] = vol;
      p.sources[layer] = source;
      p.forms[layer] = index;
      if (current === p) uploadLayer(p, layer);
    },
    onDone: (ms, melts) => {
      p.dreamMs = ms;
      p.melts = melts;
      p.dreamId = null;
      if (current === p) renderer.setMelts(melts);
    },
    onFail: (message) => {
      p.failed = message;
      p.dreamId = null;
      console.warn(`dream failed for "${text}":`, message);
    },
  });
  return p;
}

function uploadLayer(p: Prepared, layer: number): void {
  renderer.setLayerVolume(layer, p.layers[layer]!);
}

/** Condense once the whole body is in (every layer and the melts between
 *  them) and the cloud has had its moment. */
function maybeCondense(now: number): void {
  if (!current || renderer.matTarget !== 0 || current.dreamMs === undefined) return;
  if (renderer.volCount < current.layers.length) return;
  if (now - shownAt < TOUR.cloudMs) return;
  renderer.matTarget = 1;
  condensedAt = now;
}

/** Put a prepared thought on stage; any layers already dreamed go up at once. */
function show(p: Prepared, permalink: boolean): void {
  current = p;
  ui.unpin();
  ui.hideLabel(true);
  ui.setInput(p.text);
  sculpture = buildSculpture(p.result);
  renderer.resetVolumes();
  renderer.setSculpture(sculpture);
  // the sculpture *appears*: sweep up the layer axis from below the stack
  renderer.focus = -(W_SPAN + 0.6);
  focusTarget = 0;
  renderer.render();
  ui.buildGauge(sculpture.nLayers);
  shownAt = performance.now();
  condensedAt = 0;
  if (model.dream) {
    renderer.initVolumes(p.result.nLayers, model.dream.grid, model.dream.bound);
    p.layers.forEach((v, l) => v && uploadLayer(p, l));
    renderer.setMelts(p.melts);
  }
  history.replaceState(null, '', permalink ? encodePermalink(p.text) : location.pathname + location.search);
  updateStats();
}

// --- the tour: preset thoughts, the next one dreamed while this one is on stage
const tour = { on: true, index: -1, busy: false, resumeAfter: 0 };
const tourCache = new Map<string, Promise<Prepared>>();

function tourPrepare(i: number): Promise<Prepared> {
  const text = PRESETS[((i % PRESETS.length) + PRESETS.length) % PRESETS.length];
  let p = tourCache.get(text);
  if (!p) {
    p = prepare(text);
    tourCache.set(text, p);
  }
  return p;
}

async function tourShow(i: number): Promise<void> {
  if (tour.busy) return;
  tour.busy = true;
  try {
    const p = await tourPrepare(i);
    if (!tour.on) return; // someone typed a thought meanwhile
    tour.index = i;
    show(p, false);
    void tourPrepare(i + 1);
  } finally {
    tour.busy = false;
  }
}

/** Stop the tour and drop its unfinished dreams so the typed one runs next. */
function pauseTour(): void {
  tour.on = false;
  tour.resumeAfter = performance.now() + TOUR.resumeMs;
  for (const [text, pending] of tourCache) {
    void pending.then((p) => {
      if (p.dreamId !== null) {
        model.cancelDream(p.dreamId);
        tourCache.delete(text);
      }
    });
  }
}

let userSeq = 0;
let userPrepared: Prepared | null = null;
// a thought typed while the weights are still loading waits here instead of
// being dropped (and keeps the tour from starting at all)
let modelLoaded: () => void = () => {};
const whenModel = new Promise<void>((r) => (modelLoaded = r));

async function runUserThought(text: string): Promise<void> {
  tour.on = false;
  tour.resumeAfter = performance.now() + TOUR.resumeMs;
  const seq = ++userSeq;
  if (!model) ui.setStatus(`"${text.length > 40 ? text.slice(0, 40) + '…' : text}" is next, as soon as the model is in`);
  await whenModel;
  if (seq !== userSeq) return;
  pauseTour();
  if (userPrepared && userPrepared.dreamId !== null) model.cancelDream(userPrepared.dreamId); // superseded
  const p = await prepare(text);
  if (seq !== userSeq) {
    if (p.dreamId !== null) model.cancelDream(p.dreamId);
    return;
  }
  userPrepared = p;
  show(p, true);
}

function tourTick(): void {
  if (!model) return;
  const now = performance.now();
  // state, not animation: runs here too so a paused rAF (hidden tab) can't stall it
  maybeCondense(now);
  ui.setDreamBanner(dreamState().banner ?? null);
  if (tour.busy) return;
  if (!tour.on) {
    if (now > tour.resumeAfter && now - lastInteract > TOUR.resumeMs) {
      tour.on = true;
      void tourShow(tour.index + 1);
    }
    return;
  }
  if (!current) return;
  // hold from the moment it condenses; a thought that cannot dream holds as a cloud
  const cannotDream = current.failed !== undefined || (current.dreamId === null && current.dreamMs === undefined);
  const since = condensedAt || (cannotDream ? shownAt : 0);
  if (since && now - since > TOUR.holdMs && now - lastInteract > TOUR.idleMs) void tourShow(tour.index + 1);
}

/** The imagination's state in words: a HUD line, and a banner while it matters. */
function dreamState(): { hud?: string; banner?: string } {
  const s = model.dreamStatus;
  if (s.state === 'off') return {};
  if (s.state === 'unavailable') return { hud: `imagination unavailable: ${s.message}`, banner: 'the imagination could not wake — cloud only' };
  if (s.state === 'loading') {
    const pct = s.total ? Math.round((100 * s.loaded) / s.total) : 0;
    return { hud: `imagination loading ${pct}% of ${(s.total / 1e6).toFixed(0)} MB`, banner: `the imagination is waking… ${pct}%` };
  }
  if (!current) return {};
  if (current.failed !== undefined) return { hud: `dream failed: ${current.failed}`, banner: 'this dream failed — cloud only' };
  const n = current.layers.filter(Boolean).length;
  if (current.dreamMs === undefined) return { hud: `imagination dreaming ${n}/${current.layers.length}`, banner: `dreaming… ${n}/${current.layers.length}` };
  return {
    hud:
      `imagination · 7 forms in ${(current.dreamMs / 1000).toFixed(1)} s · ${model.dream!.n.toLocaleString()} forms · ` +
      `${model.dream!.grid}³ · wasm ×${s.threads}${s.library ? '' : ' · library streaming'}`,
  };
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
    if (dragging && e.shiftKey) {
      // shift-drag: tilt the slicing hyperplane into w
      tiltTarget = clampTilt(tiltTarget + (e.clientX - lastX) * 0.004);
      lastX = e.clientX;
      lastY = e.clientY;
    } else if (dragging) {
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
      // (shift turns the wheel sideways in some browsers: deltaX)
      if (e.shiftKey) tiltTarget = clampTilt(tiltTarget + (e.deltaY || e.deltaX) * 0.0012);
      else focusTarget = Math.max(-W_SPAN - 0.4, Math.min(W_SPAN + 0.4, focusTarget + e.deltaY * 0.0016));
      lastInteract = performance.now();
    },
    { passive: true },
  );
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') focusTarget = Math.min(W_SPAN + 0.4, focusTarget + 0.22);
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') focusTarget = Math.max(-W_SPAN - 0.4, focusTarget - 0.22);
    if (e.key === '[') tiltTarget = clampTilt(tiltTarget - 0.1);
    if (e.key === ']') tiltTarget = clampTilt(tiltTarget + 0.1);
    if (e.key === '0') tiltTarget = 0;
    lastInteract = performance.now();
  });
}

// [uid, name, author, license] per library form; fetched after boot, only for the stats line
let credits: string[][] | null = null;
const LICENSES: Record<string, string> = { by: 'CC-BY', 'by-sa': 'CC-BY-SA', cc0: 'CC0' };

/** The focal layer's form: what it is, and whose model it was. */
function focalSource(): string | undefined {
  if (!current || !sculpture) return undefined;
  const l = Math.round(Math.max(0, Math.min(1, (renderer.focus / W_SPAN + 1) / 2)) * (sculpture.nLayers - 1));
  const caption = current.sources[l];
  if (caption === undefined) return undefined;
  const c = credits?.[current.forms[l]];
  return c ? `${caption} — “${c[1]}” by ${c[2]} (${LICENSES[c[3]] ?? c[3]})` : caption;
}

function updateStats(): void {
  if (!sculpture || !model) return;
  ui.setStats({
    modelName: 'MiniLM-L6',
    params: model.meta.params,
    dtype: model.meta.dtype,
    backend: model.backend,
    loadMs,
    forwardMs: current?.result.forwardMs ?? 0,
    eigenMs: sculpture.eigenMs,
    tokens: sculpture.nTokens,
    layers: sculpture.nLayers,
    fps: Math.round(fps),
    imagination: dreamState().hud,
    formSource: focalSource(),
    tilt: renderer.rotXW,
    tour: tour.on ? `tour ${(tour.index % PRESETS.length) + 1}/${PRESETS.length}` : 'tour paused — resumes when idle',
    offline: cacheReport || undefined,
  });
}

let lastFrame = 0;
function frame(now: number): void {
  const idle = now - lastInteract > 3000;
  if (idle) {
    renderer.yaw += 0.00022; // glacial rotation
    focusTarget = Math.sin(now * 0.00011) * (W_SPAN + 0.2); // slow drift along w
    tiltTarget = Math.sin(now * 0.000043) * 0.4; // ...and the slice slowly tilting into it
  }
  maybeCondense(performance.now());
  renderer.focus += (focusTarget - renderer.focus) * 0.08;
  renderer.rotXW += (tiltTarget - renderer.rotXW) * 0.06;
  renderer.render();
  ui.setGauge((renderer.focus / W_SPAN + 1) / 2);

  if (lastFrame) fps = fps * 0.95 + 0.05 * (1000 / Math.max(1, now - lastFrame));
  lastFrame = now;
  updateStats();
  ui.setDreamBanner(dreamState().banner ?? null);
  requestAnimationFrame(frame);
}

async function boot(): Promise<void> {
  try {
    renderer = new Renderer(canvas);
  } catch {
    ui.showError('this piece needs WebGL2 — your browser said no.');
    return;
  }
  ui.setStatus('first visit: ~37 MB of weights, then it is all yours. nothing leaves this machine. type while it loads.');
  (document.getElementById('thought-input') as HTMLInputElement).focus({ preventScroll: true });
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

  // typing counts as interaction: the tour never advances under someone's cursor
  document.getElementById('thought-input')!.addEventListener('input', () => (lastInteract = performance.now()));
  setInterval(tourTick, 500);
  requestAnimationFrame(frame);
  modelLoaded(); // a thought typed during the load goes first
  if (model.dream)
    void fetch('./models/dream/credits.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((c) => (credits = c))
      .catch(() => {});
  // then a shared link; the tour only starts if nobody has typed anything
  const shared = decodePermalink(location.hash);
  if (shared && userSeq === 0) await runUserThought(shared);
  else if (tour.on) await tourShow(0);
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
      'models/dream/dream_model.json',
      'models/dream/decoder.onnx',
      'models/dream/library_lat.bin',
      'models/dream/library_emb.bin',
      'models/dream/captions.json',
      'models/dream/credits.json',
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
