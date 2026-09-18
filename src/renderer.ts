// Sculptural renderer: WebGL2. The piece is a raymarched implicit surface —
// token positions (interpolated between the two layers bracketing the focal
// plane, put through the same 4D -> 3D transform as the picking projection)
// become metaballs fused with a smooth-min into one glossy iridescent form
// that morphs as the layer axis is scrubbed. Marched into an offscreen
// target at adaptive resolution and composited with a chromatic fringe.
// Hover picking stays on the CPU (projectVertices mirrors the transform).
import type { Sculpture } from './sculpture.ts';

const W_SPAN = 1.5;
const MAX_BALLS = 64;

const VERT_FULLSCREEN = `#version 300 es
void main() {
  vec2 v = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0);
}
`;

const FRAG_RAYMARCH = `#version 300 es
precision highp float;
precision highp int;
uniform vec2 uRes;
uniform vec3 uEye;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uFwd;
uniform float uTanFov;
uniform float uAspect;
uniform highp sampler2D uBalls; // 64x2 RGBA32F: row 0 = xyz+radius, row 1 = rgb
uniform int uCount;
uniform float uSmin;
uniform float uBound;
out vec4 outColor;

float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float map(vec3 p) {
  float d = 1e9;
  for (int i = 0; i < ${MAX_BALLS}; i++) {
    if (i >= uCount) break;
    vec4 b = texelFetch(uBalls, ivec2(i, 0), 0);
    if (b.w <= 0.0) continue;
    d = smin(d, length(p - b.xyz) - b.w, uSmin);
  }
  return d;
}

vec3 bodyColor(vec3 p) {
  vec3 c = vec3(0.0);
  float wsum = 0.0;
  for (int i = 0; i < ${MAX_BALLS}; i++) {
    if (i >= uCount) break;
    vec4 b = texelFetch(uBalls, ivec2(i, 0), 0);
    if (b.w <= 0.0) continue;
    float dd = length(p - b.xyz) / max(b.w, 1e-3);
    float w = exp(-dd * dd * 2.5);
    c += texelFetch(uBalls, ivec2(i, 1), 0).rgb * w;
    wsum += w;
  }
  return c / max(wsum, 1e-4);
}

vec3 normalAt(vec3 p) {
  vec2 e = vec2(0.0025, -0.0025);
  return normalize(
    e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) +
    e.yxy * map(p + e.yxy) + e.xxx * map(p + e.xxx)
  );
}

void main() {
  vec2 uv = (gl_FragCoord.xy / uRes) * 2.0 - 1.0;
  vec3 rd = normalize(uFwd + uRight * (uv.x * uTanFov * uAspect) + uUp * (uv.y * uTanFov));
  vec3 ro = uEye;
  vec3 bg = vec3(0.016, 0.012, 0.024);
  vec3 hdr = vec3(0.0);

  // bounding sphere early-out
  float b = dot(ro, rd);
  float c = dot(ro, ro) - uBound * uBound;
  float disc = b * b - c;
  if (disc > 0.0) {
    float sq = sqrt(disc);
    float t = max(-b - sq, 0.0);
    float t1 = -b + sq;
    float dmin = 1e9;
    bool hit = false;
    for (int s = 0; s < 64 && t < t1; s++) {
      float d = map(ro + rd * t);
      dmin = min(dmin, d);
      if (d < 0.002) {
        hit = true;
        break;
      }
      t += d * 0.95;
    }
    if (hit) {
      vec3 p = ro + rd * t;
      vec3 nrm = normalAt(p);
      vec3 base = bodyColor(p);
      // crevice shading: how fast the field opens up along the normal
      float ao = clamp(map(p + nrm * 0.25) / 0.25, 0.0, 1.0);
      ao = 0.3 + 0.7 * ao;
      vec3 l1 = normalize(vec3(0.6, 0.8, 0.5));
      vec3 l2 = normalize(vec3(-0.7, -0.2, -0.4));
      float dif = max(0.0, dot(nrm, l1)) * 0.9 + max(0.0, dot(nrm, l2)) * 0.3;
      float fres = pow(1.0 - max(0.0, dot(nrm, -rd)), 3.0);
      float spec = pow(max(0.0, dot(reflect(rd, nrm), l1)), 60.0) * 0.7 +
                   pow(max(0.0, dot(reflect(rd, nrm), l2)), 24.0) * 0.25;
      // iridescent rim: hue cycles with the fresnel angle
      vec3 irid = 0.5 + 0.5 * cos(6.2832 * (fres * 1.3 + vec3(0.0, 0.33, 0.67)));
      hdr = base * (0.07 + dif * ao) + base * fres * 0.8 + irid * fres * 0.35 + vec3(1.0) * spec;
    } else {
      // proximity aura around the form
      hdr = vec3(0.32, 0.28, 0.44) * exp(-dmin * dmin * 2.5) * 0.2;
    }
  }
  outColor = vec4(bg + 1.0 - exp(-hdr * 1.6), 1.0);
}
`;

const FRAG_BLIT = `#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform vec2 uInvRes;
uniform float uFringe;
out vec4 outColor;
void main() {
  vec2 uv = gl_FragCoord.xy * uInvRes;
  vec2 dir = (uv - 0.5) * uFringe;
  float r = texture(uTex, uv + dir).r;
  float g = texture(uTex, uv).g;
  float b = texture(uTex, uv - dir).b;
  outColor = vec4(r, g, b, 1.0);
}
`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? 'shader error');
  return sh;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link error');
  return p;
}

const FOV_Y = (50 * Math.PI) / 180;

export class Renderer {
  private gl: WebGL2RenderingContext;
  private rmProg: WebGLProgram;
  private blitProg: WebGLProgram;
  private emptyVAO: WebGLVertexArrayObject;
  private ballTex: WebGLTexture;
  private fbo: WebGLFramebuffer | null = null;
  private fboTex: WebGLTexture | null = null;
  private fboW = 0;
  private fboH = 0;
  private sculpture: Sculpture | null = null;
  private ballSel: Uint16Array = new Uint16Array(0);
  private ballData = new Float32Array(MAX_BALLS * 2 * 4);
  // adaptive march resolution: drop the scale when frames run long
  private scale = 1;
  private frameEma = 16;
  private frameCount = 0;
  private lastT = 0;

  yaw = 0.6;
  pitch = 0.25;
  distance = 5.4;
  focus = 0;
  rotXW = 0;
  sigma = 1.15;
  persp4 = 3.2;
  private targetDistance = 5.4;
  // per-frame body stats, recomputed in updateBalls: robust framing (median),
  // exact bounding-sphere radius (frameMax), fusion tracker (spreadK)
  private frameMedian = 1.6;
  private frameMax = 3;
  private spreadK = 1;
  private ballR = new Float32Array(MAX_BALLS);

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.rmProg = link(gl, VERT_FULLSCREEN, FRAG_RAYMARCH);
    this.blitProg = link(gl, VERT_FULLSCREEN, FRAG_BLIT);
    this.emptyVAO = gl.createVertexArray()!;
    this.ballTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.ballTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, MAX_BALLS, 2, 0, gl.RGBA, gl.FLOAT, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.disable(gl.DEPTH_TEST);
    this.resize();
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
    const fw = Math.max(1, Math.floor(w * this.scale));
    const fh = Math.max(1, Math.floor(h * this.scale));
    if (fw !== this.fboW || fh !== this.fboH) {
      const gl = this.gl;
      this.fboW = fw;
      this.fboH = fh;
      if (this.fboTex) gl.deleteTexture(this.fboTex);
      if (this.fbo) gl.deleteFramebuffer(this.fbo);
      this.fboTex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, fw, fh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      this.fbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
  }

  setSculpture(s: Sculpture): void {
    this.sculpture = s;
    // tokens feeding the body, most salient first (capped)
    const sal = new Float64Array(s.nTokens);
    for (let v = 0; v < s.vertexCount; v++) sal[s.tokenOf[v]] += s.brightness[v];
    this.ballSel = Uint16Array.from(
      Array.from({ length: s.nTokens }, (_, i) => i)
        .sort((a, b) => sal[b] - sal[a] || a - b)
        .slice(0, MAX_BALLS),
    );
  }

  private camera(): { eye: [number, number, number]; right: number[]; up: number[]; fwd: number[] } {
    const cp = Math.cos(this.pitch);
    const eye: [number, number, number] = [
      this.distance * Math.sin(this.yaw) * cp,
      this.distance * Math.sin(this.pitch),
      this.distance * Math.cos(this.yaw) * cp,
    ];
    const len = Math.hypot(eye[0], eye[1], eye[2]) || 1;
    const z = [eye[0] / len, eye[1] / len, eye[2] / len]; // backward
    const x0 = [-z[2], 0, z[0]]; // cross((0,1,0), z)
    const xl = Math.hypot(x0[0], x0[1], x0[2]) || 1;
    const right = [x0[0] / xl, x0[1] / xl, x0[2] / xl];
    const up = [
      z[1] * right[2] - z[2] * right[1],
      z[2] * right[0] - z[0] * right[2],
      z[0] * right[1] - z[1] * right[0],
    ];
    return { eye, right, up, fwd: [-z[0], -z[1], -z[2]] };
  }

  /** Mirror of the 4D transform, for hover picking.
   *  Returns [x, y, alpha, eyeDistance] per vertex, x/y in CSS pixels. */
  projectVertices(): Float32Array {
    const s = this.sculpture;
    if (!s) return new Float32Array(0);
    const { eye, right, up, fwd } = this.camera();
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const f = 1 / Math.tan(FOV_Y / 2);
    const out = new Float32Array(s.vertexCount * 4);
    const cs = Math.cos(this.rotXW);
    const sn = Math.sin(this.rotXW);
    const dpr = this.canvas.width / Math.max(1, this.canvas.clientWidth);
    for (let v = 0; v < s.vertexCount; v++) {
      const px = s.positions[v * 3];
      const py = s.positions[v * 3 + 1];
      const pz = s.positions[v * 3 + 2];
      const w0 = s.wCoords[v];
      const x = px * cs - w0 * sn;
      const w = px * sn + w0 * cs;
      const scale = this.persp4 / (this.persp4 - w * 0.35);
      const cx = x * scale - eye[0];
      const cy = py * scale - eye[1];
      const cz = pz * scale - eye[2];
      const vx = cx * right[0] + cy * right[1] + cz * right[2];
      const vy = cx * up[0] + cy * up[1] + cz * up[2];
      const vz = cx * fwd[0] + cy * fwd[1] + cz * fwd[2]; // forward distance
      const ndcX = (vx * f) / aspect / vz;
      const ndcY = (vy * f) / vz;
      out[v * 4] = (ndcX * 0.5 + 0.5) * this.canvas.width / dpr;
      out[v * 4 + 1] = (0.5 - ndcY * 0.5) * this.canvas.height / dpr;
      const dw = w - this.focus;
      out[v * 4 + 2] = Math.exp((-dw * dw) / (2 * this.sigma * this.sigma));
      out[v * 4 + 3] = vz;
    }
    return out;
  }

  /** Metaballs for the body: tokens lerped between the layers bracketing the
   *  focal plane, through the same 4D transform as the picking projection so
   *  hover targets sit on the visible form. */
  private updateBalls(): void {
    const s = this.sculpture;
    if (!s) return;
    const L = s.nLayers;
    const n = s.nTokens;
    const fw = Math.max(-W_SPAN, Math.min(W_SPAN, this.focus));
    const t = L <= 1 ? 0 : (fw / W_SPAN + 1) * 0.5 * (L - 1);
    const l0 = Math.min(L - 1, Math.floor(t));
    const l1 = Math.min(L - 1, l0 + 1);
    const fr = Math.min(1, Math.max(0, t - l0));
    const cs = Math.cos(this.rotXW);
    const sn = Math.sin(this.rotXW);
    const d = this.ballData;
    const m = this.ballSel.length;
    // pass 1: transformed positions + per-ball distance from the origin, so
    // radii, framing and the bounding sphere can all track how dispersed the
    // current layer pair is — the form stays fused and in frame instead of
    // dissolving into separate bubbles on spread-out layers
    for (let k = 0; k < m; k++) {
      const i = this.ballSel[k];
      const v0 = l0 * n + i;
      const v1 = l1 * n + i;
      const x = s.positions[v0 * 3] + (s.positions[v1 * 3] - s.positions[v0 * 3]) * fr;
      const y = s.positions[v0 * 3 + 1] + (s.positions[v1 * 3 + 1] - s.positions[v0 * 3 + 1]) * fr;
      const z = s.positions[v0 * 3 + 2] + (s.positions[v1 * 3 + 2] - s.positions[v0 * 3 + 2]) * fr;
      const w = s.wCoords[v0] + (s.wCoords[v1] - s.wCoords[v0]) * fr;
      const xr = x * cs - w * sn;
      const wr = x * sn + w * cs;
      const sc = this.persp4 / (this.persp4 - wr * 0.35);
      d[k * 4] = xr * sc;
      d[k * 4 + 1] = y * sc;
      d[k * 4 + 2] = z * sc;
      d[k * 4 + 3] = sc; // stashed perspective scale, used for the radius in pass 2
      this.ballR[k] = Math.hypot(d[k * 4], d[k * 4 + 1], d[k * 4 + 2]);
    }
    const sorted = this.ballR.subarray(0, m).sort();
    this.frameMedian = m ? sorted[m >> 1] : 1.6;
    const spreadK = Math.min(1.6, Math.max(0.8, this.frameMedian / 1.7));
    this.spreadK = spreadK;
    let maxRadius = 0;
    for (let k = 0; k < MAX_BALLS; k++) {
      if (k < m) {
        const i = this.ballSel[k];
        const v0 = l0 * n + i;
        const v1 = l1 * n + i;
        const bright = s.brightness[v0] + (s.brightness[v1] - s.brightness[v0]) * fr;
        d[k * 4 + 3] = (0.16 + 0.38 * bright) * spreadK * d[k * 4 + 3];
        maxRadius = Math.max(maxRadius, d[k * 4 + 3]);
        const co = MAX_BALLS * 4 + k * 4;
        d[co] = s.colors[v0 * 3] + (s.colors[v1 * 3] - s.colors[v0 * 3]) * fr;
        d[co + 1] = s.colors[v0 * 3 + 1] + (s.colors[v1 * 3 + 1] - s.colors[v0 * 3 + 1]) * fr;
        d[co + 2] = s.colors[v0 * 3 + 2] + (s.colors[v1 * 3 + 2] - s.colors[v0 * 3 + 2]) * fr;
        d[co + 3] = 1;
      } else {
        d[k * 4 + 3] = 0;
      }
    }
    this.frameMax = (m ? sorted[m - 1] : 0) + maxRadius;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.ballTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, MAX_BALLS, 2, gl.RGBA, gl.FLOAT, d);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  private adaptQuality(): void {
    const now = performance.now();
    if (this.lastT) this.frameEma = this.frameEma * 0.95 + (now - this.lastT) * 0.05;
    this.lastT = now;
    if (++this.frameCount % 90 !== 0) return;
    if (this.frameEma > 24 && this.scale > 0.5) this.scale = Math.max(0.5, this.scale - 0.25);
    else if (this.frameEma < 13 && this.scale < 1) this.scale = Math.min(1, this.scale + 0.25);
  }

  render(): void {
    const gl = this.gl;
    this.adaptQuality();
    this.resize();
    if (!this.sculpture) {
      gl.clearColor(0.016, 0.012, 0.024, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    this.updateBalls();
    // robust framing: follow the median ball distance so outlier tokens
    // can't push the body out of frame and the form fills the view on
    // compressed and dispersed layers alike
    this.targetDistance = Math.min(10, Math.max(4, this.frameMedian * 3.4));
    this.distance += (this.targetDistance - this.distance) * 0.05;
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const cam = this.camera();

    // 1: march the body into the offscreen target
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.fboW, this.fboH);
    gl.useProgram(this.rmProg);
    const ru = (name: string) => gl.getUniformLocation(this.rmProg, name);
    gl.uniform2f(ru('uRes'), this.fboW, this.fboH);
    gl.uniform3f(ru('uEye'), cam.eye[0], cam.eye[1], cam.eye[2]);
    gl.uniform3f(ru('uRight'), cam.right[0], cam.right[1], cam.right[2]);
    gl.uniform3f(ru('uUp'), cam.up[0], cam.up[1], cam.up[2]);
    gl.uniform3f(ru('uFwd'), cam.fwd[0], cam.fwd[1], cam.fwd[2]);
    gl.uniform1f(ru('uTanFov'), Math.tan(FOV_Y / 2));
    gl.uniform1f(ru('uAspect'), aspect);
    gl.uniform1i(ru('uCount'), Math.min(this.ballSel.length, MAX_BALLS));
    gl.uniform1f(ru('uSmin'), 0.38 * this.spreadK);
    gl.uniform1f(ru('uBound'), this.frameMax + 0.3);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.ballTex);
    gl.uniform1i(ru('uBalls'), 0);
    gl.bindVertexArray(this.emptyVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 2: composite onto the canvas with the chromatic fringe
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.blitProg);
    const bu = (name: string) => gl.getUniformLocation(this.blitProg, name);
    gl.uniform1i(bu('uTex'), 0);
    gl.uniform2f(bu('uInvRes'), 1 / this.canvas.width, 1 / this.canvas.height);
    gl.uniform1f(bu('uFringe'), 0.0035);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindVertexArray(null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
}
