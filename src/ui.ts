// DOM chrome: status overlay, readout strip, layer-axis gauge, token label,
// about panel, permalink. No framework; everything cached at construction.

export interface Stats {
  modelName: string;
  params: number;
  dtype: string;
  backend: string;
  loadMs: number;
  forwardMs: number;
  eigenMs: number;
  tokens: number;
  layers: number;
  fps: number;
  offline?: string;
}

export class UI {
  private status = document.getElementById('status')!;
  private statusMsg = document.getElementById('status-msg')!;
  private readout = document.getElementById('readout')!;
  private gauge = document.getElementById('gauge')!;
  private label = document.getElementById('token-label')!;
  private about = document.getElementById('about')!;
  private input = document.getElementById('thought-input') as HTMLInputElement;
  private gaugeDots: HTMLElement[] = [];
  private labelPinned = false;

  constructor(onSubmit: (text: string) => void) {
    document.getElementById('thought')!.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = this.input.value.trim();
      if (text) onSubmit(text);
      this.input.blur();
    });
    document.getElementById('about-toggle')!.addEventListener('click', () => {
      this.about.classList.toggle('open');
    });
  }

  setStatus(msg: string): void {
    this.statusMsg.textContent = msg;
  }

  hideStatus(): void {
    this.status.classList.add('gone');
  }

  showError(msg: string): void {
    this.statusMsg.textContent = msg;
    this.status.classList.add('error');
  }

  setStats(s: Stats): void {
    const m = (v: number) => `${v.toFixed(v < 10 ? 1 : 0)}ms`;
    this.readout.textContent =
      `${s.modelName} · ${(s.params / 1e6).toFixed(1)}M params · ${s.dtype} · ${s.backend}\n` +
      `load ${m(s.loadMs)} · fwd ${m(s.forwardMs)} · eigen ${m(s.eigenMs)} · ` +
      `${s.tokens} tokens × ${s.layers} layers · ${s.fps} fps` +
      (s.offline ? `\n${s.offline}` : '');
  }

  buildGauge(nLayers: number): void {
    this.gauge.replaceChildren();
    this.gaugeDots = [];
    for (let i = 0; i < nLayers; i++) {
      const dot = document.createElement('div');
      dot.className = 'gauge-dot';
      this.gauge.appendChild(dot);
      this.gaugeDots.push(dot);
    }
  }

  /** focusNorm: 0..1 along the layer axis */
  setGauge(focusNorm: number): void {
    const pos = focusNorm * (this.gaugeDots.length - 1);
    this.gaugeDots.forEach((dot, i) => {
      const d = Math.abs(i - pos);
      const b = Math.exp(-d * d * 1.2);
      dot.style.opacity = `${0.18 + 0.82 * b}`;
      dot.style.transform = `scale(${0.7 + 0.7 * b})`;
    });
  }

  showLabel(text: string, x: number, y: number): void {
    this.label.textContent = text;
    this.label.style.left = `${x}px`;
    this.label.style.top = `${y}px`;
    this.label.classList.add('visible');
  }

  hideLabel(force = false): void {
    if (this.labelPinned && !force) return;
    this.label.classList.remove('visible');
  }

  togglePin(): boolean {
    this.labelPinned = !this.labelPinned && this.label.classList.contains('visible');
    this.label.classList.toggle('pinned', this.labelPinned);
    return this.labelPinned;
  }

  unpin(): void {
    this.labelPinned = false;
    this.label.classList.remove('pinned');
  }

  get pinned(): boolean {
    return this.labelPinned;
  }

  setInput(text: string): void {
    this.input.value = text;
  }
}

export function encodePermalink(text: string): string {
  const b64 = btoa(String.fromCharCode(...new TextEncoder().encode(text))).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
  return `#t=${b64}`;
}

export function decodePermalink(hash: string): string | null {
  const m = hash.match(/#t=([A-Za-z0-9_-]+)/);
  if (!m) return null;
  try {
    const b64 = m[1].replaceAll('-', '+').replaceAll('_', '/');
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  } catch {
    return null;
  }
}
