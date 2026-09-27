// SDF grids upload as R16F: half floats are filterable in core WebGL2, so the
// marcher gets hardware trilinear interpolation. The dream worker packs them,
// so the main thread only uploads.
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

function toHalf(v: number): number {
  _f32[0] = v;
  const x = _u32[0];
  const sign = (x >> 16) & 0x8000;
  const exp = ((x >> 23) & 0xff) - 127 + 15;
  const man = x & 0x7fffff;
  if (exp <= 0) return sign; // flush denormals; SDF values are never that small
  if (exp >= 31) return sign | 0x7bff; // clamp to max half
  // + (not |): a mantissa that rounds up carries into the exponent
  return sign | ((exp << 10) + ((man + 0x1000) >> 13));
}

export function f32ToF16(v: Float32Array): Uint16Array {
  const out = new Uint16Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = toHalf(v[i]);
  return out;
}
