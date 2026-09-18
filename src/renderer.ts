// Sculptural renderer: WebGL2.
//
// The body of the piece is a raymarched implicit surface: token positions
// (interpolated between the two layers bracketing the focal plane, then put
// through the same 4D -> 3D transform as everything else) become metaballs
// fused with a smooth-min into one glossy, iridescent form. Scrubbing the
// layer axis morphs the body continuously. It is rendered at half resolution
// into an FBO and composited, then association ribbons and token glints are
// drawn additively on top in three chromatic-fringe passes — nothing
// occludes, the filaments read as the form's internal structure.
import { EDGE_INTER, type Sculpture } from './sculpture.ts';

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
    float w = exp(-dd * dd * 1.5);
    c += texelFetch(uBalls, ivec2(i, 1), 0).rgb * w;
    wsum += w;
  }
  return c / max(wsum, 1e-4);
}

void main() {
  vec2 uv = (gl_FragCoord.xy / uRes) * 2.0 - 1.0;
  vec3 rd = normalize(uFwd + uRight * (uv.x * uTanFov * uAspect) + uUp * (uv.y * uTanFov));
  vec3 ro = uEye;
  vec3 bg = vec3(0.016, 0.012, 0.024);
  vec3 hdr = vec3(0.0);

  // bounding sphere (r=3 at origin) early-out
  float b = dot(ro, rd);
  float c = dot(ro, ro) - 9.0;
  float disc = b * b - c;
  if (disc > 0.0) {
    float sq = sqrt(disc);
    float t = max(-b - sq, 0.0);
    float t1 = -b + sq;
    float dmin = 1e9;
    bool hit = false;
    for (int s = 0; s < 56 && t < t1; s++) {
      float d = map(ro + rd * t);
      dmin = min(dmin, d);
      if (d < 0.004) {
        hit = true;
        break;
      }
      t += d * 0.9;
    }
    if (hit) {
      vec3 p = ro + rd * t;
      vec2 e = vec2(0.003, -0.003);
      vec3 nrm = normalize(
        e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) +
        e.yxy * map(p + e.yxy) + e.xxx * map(p + e.xxx)
      );
      vec3 base = bodyColor(p);
      float fres = pow(1.0 - max(0.0, dot(nrm, -rd)), 3.0);
      vec3 l1 = normalize(vec3(0.6, 0.8, 0.5));
      vec3 l2 = normalize(vec3(-0.7, -0.2, -0.4));
      float dif = max(0.0, dot(nrm, l1)) * 0.85 + max(0.0, dot(nrm, l2)) * 0.35;
      float spec = pow(max(0.0, dot(reflect(rd, nrm), l1)), 40.0);
      // iridescent rim: hue cycles with the fresnel angle
      vec3 irid = 0.5 + 0.5 * cos(6.2832 * (fres * 1.3 + vec3(0.0, 0.33, 0.67)));
      hdr = base * (0.22 + 0.85 * dif) + base * fres * 0.7 + irid * fres * 0.4 + vec3(1.0) * spec * 0.6;
    } else {
      // proximity aura around the form
      hdr = vec3(0.32, 0.28, 0.44) * exp(-dmin * dmin * 2.5) * 0.22;
    }
  }
  outColor = vec4(bg + 1.0 - exp(-hdr * 1.5), 1.0);
}
`;

const FRAG_BLIT = `#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform vec2 uInvRes;
out vec4 outColor;
void main() {
  outColor = texture(uTex, gl_FragCoord.xy * uInvRes);
}
`;

const VERT_POINTS = `#version 300 es
in vec3 aPos;
in float aW;
in float aBright;
in vec3 aColor;
uniform mat4 uVP;
uniform float uRotXW;
uniform float uFocus;
uniform float uSigma;
uniform float uFringe;
uniform float uPointScale;
uniform float uPersp4;
out float vAlpha;
out vec3 vColor;
void main() {
  float cs = cos(uRotXW);
  float sn = sin(uRotXW);
  float x = aPos.x * cs - aW * sn;
  float w = aPos.x * sn + aW * cs;
  vec3 p = vec3(x, aPos.y, aPos.z);
  p *= uPersp4 / (uPersp4 - w * 0.35);
  vec4 clip = uVP * vec4(p, 1.0);
  clip.xy *= 1.0 + uFringe;
  gl_Position = clip;
  float dw = w - uFocus;
  float depth = exp(-dw * dw / (2.0 * uSigma * uSigma));
  vAlpha = aBright * (0.06 + 0.94 * depth);
  vColor = mix(aColor, vec3(1.0), depth * 0.10);
  gl_PointSize = clamp(uPointScale * (0.8 + 1.2 * aBright) * (0.5 + 0.5 * depth) / clip.w, 1.0, 160.0);
}
`;

const FRAG_POINTS = `#version 300 es
precision mediump float;
in float vAlpha;
in vec3 vColor;
uniform vec3 uMask;
out vec4 outColor;
void main() {
  vec2 d2 = gl_PointCoord - 0.5;
  float d = length(d2) * 2.0;
  float core = exp(-d * d * 14.0);
  float mid = exp(-d * d * 4.0) * 0.5;
  float halo = exp(-d * d * 1.2) * 0.3;
  float window = smoothstep(1.0, 0.6, d); // kill the sprite-quad edge
  vec3 col = (vColor * (mid + halo) + vec3(1.0) * core * 0.45) * window * vAlpha;
  outColor = vec4(uMask * col, 1.0);
}
`;

const VERT_RIBBON = `#version 300 es
in vec2 aCorner; // x: across (-1..1), y: along (0..1)
in vec3 aP0;
in float aW0;
in vec3 aC0;
in vec3 aP1;
in float aW1;
in vec3 aC1;
in vec2 aDim; // width (logical px), alpha
uniform mat4 uVP;
uniform float uRotXW;
uniform float uFocus;
uniform float uSigma;
uniform float uFringe;
uniform float uPersp4;
uniform vec2 uResolution;
uniform float uWidthScale;
out vec3 vColor;
out float vAcross;
out float vAlpha;

vec4 xform(vec3 p, float w, out float wT) {
  float cs = cos(uRotXW);
  float sn = sin(uRotXW);
  float x = p.x * cs - w * sn;
  wT = p.x * sn + w * cs;
  vec3 q = vec3(x, p.y, p.z);
  q *= uPersp4 / (uPersp4 - wT * 0.35);
  return uVP * vec4(q, 1.0);
}

void main() {
  float w0;
  float w1;
  vec4 c0 = xform(aP0, aW0, w0);
  vec4 c1 = xform(aP1, aW1, w1);
  vec2 n0 = c0.xy / c0.w;
  vec2 n1 = c1.xy / c1.w;
  vec2 d = n1 - n0;
  float len = length(d);
  vec2 perp = len > 1e-7 ? vec2(-d.y, d.x) / len : vec2(0.0, 1.0);
  float along = aCorner.y;
  float cw = mix(c0.w, c1.w, along);
  vec2 ndc = mix(n0, n1, along) + perp * aCorner.x * aDim.x * uWidthScale * 2.0 / uResolution;
  vec4 clip = vec4(ndc * cw, mix(c0.z, c1.z, along), cw);
  clip.xy *= 1.0 + uFringe;
  gl_Position = clip;
  float wT = mix(w0, w1, along);
  float dw = wT - uFocus;
  float depth = exp(-dw * dw / (2.0 * uSigma * uSigma));
  vAlpha = aDim.y * (0.12 + 0.88 * depth);
  vColor = mix(aC0, aC1, along);
  vAcross = aCorner.x;
}
`;

const FRAG_RIBBON = `#version 300 es
precision mediump float;
in vec3 vColor;
in float vAcross;
in float vAlpha;
uniform vec3 uMask;
out vec4 outColor;
void main() {
  float edge = 1.0 - vAcross * vAcross;
  float spine = pow(max(0.0, 1.0 - abs(vAcross)), 10.0);
  float i = (edge * edge * 0.5 + spine * 0.3) * vAlpha;
  outColor = vec4(uMask * vColor * i, 1.0);
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

// minimal mat4 (column-major, matching WebGL)
function perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

function lookAt(eye: [number, number, number], center: [number, number, number], up: [number, number, number]): Float32Array {
  const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = (v: number[]) => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  };
  const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const z = norm(sub(eye, center));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  const m = new Float32Array(16);
  m.set([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0]);
  m[12] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
  m[13] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
  m[14] = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);
  m[15] = 1;
  return m;
}

function mul4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    out[c * 4 + r] = s;
  }
  return out;
}

const FRINGE = 0.0045;
const PASSES: Array<[number, [number, number, number]]> = [
  [FRINGE, [1, 0, 0]],
  [0, [0, 1, 0]],
  [-FRINGE, [0, 0, 1]],
];

// interleaved ribbon instance layout (floats): p0 xyz, w0, c0 rgb, p1 xyz, w1, c1 rgb, dim width+alpha
const RIBBON_STRIDE = 16 * 4;
const RIBBON_ATTRIBS = [
  ['aP0', 3, 0],
  ['aW0', 1, 3],
  ['aC0', 3, 4],
  ['aP1', 3, 7],
  ['aW1', 1, 10],
  ['aC1', 3, 11],
  ['aDim', 2, 14],
] as const;

// interleaved point layout (floats): pos xyz, w, bright, color rgb
const POINT_STRIDE = 8 * 4;
const POINT_ATTRIBS = [
  ['aPos', 3, 0],
  ['aW', 1, 3],
  ['aBright', 1, 4],
  ['aColor', 3, 5],
] as const;

const FOV_Y = (50 * Math.PI) / 180;

export class Renderer {
  private gl: WebGL2RenderingContext;
  private pointProg: WebGLProgram;
  private ribbonProg: WebGLProgram;
  private rmProg: WebGLProgram;
  private blitProg: WebGLProgram;
  private pointVAO: WebGLVertexArrayObject | null = null;
  private ribbonVAO: WebGLVertexArrayObject | null = null;
  private emptyVAO: WebGLVertexArrayObject;
  private pointBuf: WebGLBuffer | null = null;
  private ribbonBuf: WebGLBuffer | null = null;
  private cornerBuf: WebGLBuffer;
  private ballTex: WebGLTexture;
  private fbo: WebGLFramebuffer | null = null;
  private fboTex: WebGLTexture | null = null;
  private fboW = 0;
  private fboH = 0;
  private vertexCount = 0;
  private edgeCount = 0;
  private sculpture: Sculpture | null = null;
  private ballSel: Uint16Array = new Uint16Array(0);
  private ballData = new Float32Array(MAX_BALLS * 2 * 4);

  yaw = 0.6;
  pitch = 0.25;
  distance = 5.4;
  focus = 0;
  rotXW = 0;
  sigma = 1.15;
  persp4 = 3.2;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.pointProg = link(gl, VERT_POINTS, FRAG_POINTS);
    this.ribbonProg = link(gl, VERT_RIBBON, FRAG_RIBBON);
    this.rmProg = link(gl, VERT_FULLSCREEN, FRAG_RAYMARCH);
    this.blitProg = link(gl, VERT_FULLSCREEN, FRAG_BLIT);
    this.emptyVAO = gl.createVertexArray()!;
    this.cornerBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, 0, 1, 0, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
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
    // half-res raymarch target
    const fw = Math.max(1, w >> 1);
    const fh = Math.max(1, h >> 1);
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
    const gl = this.gl;

    const vd = new Float32Array(s.vertexCount * 8);
    for (let v = 0; v < s.vertexCount; v++) {
      vd.set(
        [
          s.positions[v * 3],
          s.positions[v * 3 + 1],
          s.positions[v * 3 + 2],
          s.wCoords[v],
          s.brightness[v],
          s.colors[v * 3],
          s.colors[v * 3 + 1],
          s.colors[v * 3 + 2],
        ],
        v * 8,
      );
    }

    const rd = new Float32Array(s.edgeCount * 16);
    for (let e = 0; e < s.edgeCount; e++) {
      const v0 = s.edgeIndices[e * 2];
      const v1 = s.edgeIndices[e * 2 + 1];
      const w = s.edgeWeights[e];
      // intra-layer ribbons widen and brighten with association strength;
      // inter-layer strands stay thin and steady so they read as threads
      const width = s.edgeKind[e] === EDGE_INTER ? 2.4 : 2.0 + w * w * 10.0;
      const alpha = s.edgeKind[e] === EDGE_INTER ? 0.1 : 0.05 + 0.2 * w;
      rd.set(
        [
          s.positions[v0 * 3],
          s.positions[v0 * 3 + 1],
          s.positions[v0 * 3 + 2],
          s.wCoords[v0],
          s.colors[v0 * 3],
          s.colors[v0 * 3 + 1],
          s.colors[v0 * 3 + 2],
          s.positions[v1 * 3],
          s.positions[v1 * 3 + 1],
          s.positions[v1 * 3 + 2],
          s.wCoords[v1],
          s.colors[v1 * 3],
          s.colors[v1 * 3 + 1],
          s.colors[v1 * 3 + 2],
          width,
          alpha,
        ],
        e * 16,
      );
    }

    // tokens feeding the raymarched body, most salient first (capped)
    const sal = new Float64Array(s.nTokens);
    for (let v = 0; v < s.vertexCount; v++) sal[s.tokenOf[v]] += s.brightness[v];
    this.ballSel = Uint16Array.from(
      Array.from({ length: s.nTokens }, (_, i) => i)
        .sort((a, b) => sal[b] - sal[a] || a - b)
        .slice(0, MAX_BALLS),
    );

    // points
    if (this.pointVAO) gl.deleteVertexArray(this.pointVAO);
    if (this.pointBuf) gl.deleteBuffer(this.pointBuf);
    this.pointVAO = gl.createVertexArray()!;
    this.pointBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.pointVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuf);
    gl.bufferData(gl.ARRAY_BUFFER, vd, gl.STATIC_DRAW);
    for (const [name, size, offset] of POINT_ATTRIBS) {
      const loc = gl.getAttribLocation(this.pointProg, name);
      if (loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, POINT_STRIDE, offset * 4);
    }

    // ribbons: static corner strip (divisor 0) + per-instance data (divisor 1)
    if (this.ribbonVAO) gl.deleteVertexArray(this.ribbonVAO);
    if (this.ribbonBuf) gl.deleteBuffer(this.ribbonBuf);
    this.ribbonVAO = gl.createVertexArray()!;
    this.ribbonBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.ribbonVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    const cornerLoc = gl.getAttribLocation(this.ribbonProg, 'aCorner');
    gl.enableVertexAttribArray(cornerLoc);
    gl.vertexAttribPointer(cornerLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ribbonBuf);
    gl.bufferData(gl.ARRAY_BUFFER, rd, gl.STATIC_DRAW);
    for (const [name, size, offset] of RIBBON_ATTRIBS) {
      const loc = gl.getAttribLocation(this.ribbonProg, name);
      if (loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, RIBBON_STRIDE, offset * 4);
      gl.vertexAttribDivisor(loc, 1);
    }

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.vertexCount = s.vertexCount;
    this.edgeCount = s.edgeCount;
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

  private viewProj(): Float32Array {
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const proj = perspective(FOV_Y, aspect, 0.1, 100);
    const { eye } = this.camera();
    return mul4(proj, lookAt(eye, [0, 0, 0], [0, 1, 0]));
  }

  /** Mirror of the vertex transform, for hover picking. Returns [x, y, alpha] in CSS pixels. */
  projectVertices(): Float32Array {
    const s = this.sculpture;
    if (!s) return new Float32Array(0);
    const vp = this.viewProj();
    const out = new Float32Array(s.vertexCount * 3);
    const cs = Math.cos(this.rotXW);
    const sn = Math.sin(this.rotXW);
    for (let v = 0; v < s.vertexCount; v++) {
      const px = s.positions[v * 3];
      const py = s.positions[v * 3 + 1];
      const pz = s.positions[v * 3 + 2];
      const w0 = s.wCoords[v];
      const x = px * cs - w0 * sn;
      const w = px * sn + w0 * cs;
      const scale = this.persp4 / (this.persp4 - w * 0.35);
      const cx = x * scale;
      const cy = py * scale;
      const cz = pz * scale;
      const cxs = vp[0] * cx + vp[4] * cy + vp[8] * cz + vp[12];
      const cys = vp[1] * cx + vp[5] * cy + vp[9] * cz + vp[13];
      const cws = vp[3] * cx + vp[7] * cy + vp[11] * cz + vp[15];
      const dpr = this.canvas.width / Math.max(1, this.canvas.clientWidth);
      out[v * 3] = ((cxs / cws) * 0.5 + 0.5) * this.canvas.width / dpr;
      out[v * 3 + 1] = (0.5 - (cys / cws) * 0.5) * this.canvas.height / dpr;
      const dw = w - this.focus;
      out[v * 3 + 2] = Math.exp((-dw * dw) / (2 * this.sigma * this.sigma));
    }
    return out;
  }

  /** Metaballs for the body: tokens lerped between the layers bracketing the
   *  focal plane, through the same 4D transform as the ribbons so all three
   *  layers of the image agree. */
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
    for (let k = 0; k < MAX_BALLS; k++) {
      if (k < m) {
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
        const bright = s.brightness[v0] + (s.brightness[v1] - s.brightness[v0]) * fr;
        d[k * 4] = xr * sc;
        d[k * 4 + 1] = y * sc;
        d[k * 4 + 2] = z * sc;
        d[k * 4 + 3] = (0.2 + 0.32 * bright) * sc;
        const co = MAX_BALLS * 4 + k * 4;
        d[co] = s.colors[v0 * 3] + (s.colors[v1 * 3] - s.colors[v0 * 3]) * fr;
        d[co + 1] = s.colors[v0 * 3 + 1] + (s.colors[v1 * 3 + 1] - s.colors[v0 * 3 + 1]) * fr;
        d[co + 2] = s.colors[v0 * 3 + 2] + (s.colors[v1 * 3 + 2] - s.colors[v0 * 3 + 2]) * fr;
        d[co + 3] = 1;
      } else {
        d[k * 4 + 3] = 0;
      }
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.ballTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, MAX_BALLS, 2, gl.RGBA, gl.FLOAT, d);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  render(): void {
    const gl = this.gl;
    this.resize();
    if (!this.sculpture) {
      gl.clearColor(0.016, 0.012, 0.024, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    this.updateBalls();
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const cam = this.camera();

    // 1: raymarch the body into the half-res target (opaque)
    gl.disable(gl.BLEND);
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
    gl.uniform1f(ru('uSmin'), 0.55);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.ballTex);
    gl.uniform1i(ru('uBalls'), 0);
    gl.bindVertexArray(this.emptyVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 2: composite onto the canvas
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.blitProg);
    const bu = (name: string) => gl.getUniformLocation(this.blitProg, name);
    gl.uniform1i(bu('uTex'), 0);
    gl.uniform2f(bu('uInvRes'), 1 / this.canvas.width, 1 / this.canvas.height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 3: additive ribbons + glints with the chromatic fringe
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const vp = this.viewProj();
    const pointScale = this.canvas.height * 0.16;
    const widthScale = this.canvas.height / 800;
    for (const [fringe, mask] of PASSES) {
      gl.useProgram(this.ribbonProg);
      this.ribbonUniforms(vp, fringe, mask, widthScale, aspect);
      gl.bindVertexArray(this.ribbonVAO);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.edgeCount);

      gl.useProgram(this.pointProg);
      this.pointUniforms(vp, fringe, mask, pointScale, aspect);
      gl.bindVertexArray(this.pointVAO);
      gl.drawArrays(gl.POINTS, 0, this.vertexCount);
    }
    gl.bindVertexArray(null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  private pointUniforms(vp: Float32Array, fringe: number, mask: [number, number, number], pointScale: number, aspect: number): void {
    const gl = this.gl;
    const u = (name: string) => gl.getUniformLocation(this.pointProg, name);
    gl.uniformMatrix4fv(u('uVP'), false, vp);
    gl.uniform1f(u('uRotXW'), this.rotXW);
    gl.uniform1f(u('uFocus'), this.focus);
    gl.uniform1f(u('uSigma'), this.sigma);
    gl.uniform1f(u('uFringe'), fringe * (1 + 0.4 * (aspect - 1)));
    gl.uniform3f(u('uMask'), mask[0], mask[1], mask[2]);
    gl.uniform1f(u('uPointScale'), pointScale);
    gl.uniform1f(u('uPersp4'), this.persp4);
  }

  private ribbonUniforms(vp: Float32Array, fringe: number, mask: [number, number, number], widthScale: number, aspect: number): void {
    const gl = this.gl;
    const u = (name: string) => gl.getUniformLocation(this.ribbonProg, name);
    gl.uniformMatrix4fv(u('uVP'), false, vp);
    gl.uniform1f(u('uRotXW'), this.rotXW);
    gl.uniform1f(u('uFocus'), this.focus);
    gl.uniform1f(u('uSigma'), this.sigma);
    gl.uniform1f(u('uFringe'), fringe * (1 + 0.4 * (aspect - 1)));
    gl.uniform3f(u('uMask'), mask[0], mask[1], mask[2]);
    gl.uniform1f(u('uPersp4'), this.persp4);
    gl.uniform2f(u('uResolution'), this.canvas.width, this.canvas.height);
    gl.uniform1f(u('uWidthScale'), widthScale);
  }
}
