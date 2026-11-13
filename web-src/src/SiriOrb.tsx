import { useEffect, useRef } from "react"

// A glassy sphere with soft, multicoloured light ribbons flowing inside it, swelling with the voice.
// Original shader, written for Jarvis.

type Phase = "idle" | "listening" | "thinking" | "talking" | "loading"

const VERT = `
attribute vec2 p;
varying vec2 uv;
void main() { uv = p; gl_Position = vec4(p, 0.0, 1.0); }
`

const FRAG = `
precision highp float;
varying vec2 uv;
uniform float t;
uniform float level;
uniform float energy;
uniform float spin;
uniform float soft;   // 0 = crisp ribbons, 1 = soft blended light
uniform vec3 c1; uniform vec3 c2; uniform vec3 c3; uniform vec3 c4;

mat3 rotY(float a) { float c = cos(a), s = sin(a); return mat3(c,0.,-s, 0.,1.,0., s,0.,c); }
mat3 rotX(float a) { float c = cos(a), s = sin(a); return mat3(1.,0.,0., 0.,c,s, 0.,-s,c); }
mat3 rotZ(float a) { float c = cos(a), s = sin(a); return mat3(c,s,0., -s,c,0., 0.,0.,1.); }

// A ribbon is a thin, wavy sheet of light curling through the ball.
float ribbon(vec3 p, mat3 m, float f, float sp, float amp, float w, float seed) {
  vec3 q = m * p;
  float wave = sin(q.x * f + t * sp + seed) * amp + sin(q.z * f * 1.3 - t * sp * 0.8 + seed * 1.7) * amp * 0.6
             + sin((q.x + q.z) * f * 0.7 + t * sp * 0.5 + seed * 3.1) * amp * 0.4;
  float d = q.y - wave;
  float band = exp(-d * d / (w * w));
  // fade the sheet into a ribbon (narrow across z), twisting over time
  float across = q.z * cos(q.x * 1.5 + t * 0.4 + seed) + q.y * 0.3;
  return band * exp(-across * across / (0.16 + soft * 0.55));
}

void main() {
  vec2 q = uv * 1.1;
  float R = 0.8 + level * 0.05;
  vec3 ro = vec3(0.0, 0.0, 2.6);
  vec3 rd = normalize(vec3(q, -2.6));
  // ray-sphere intersection
  float b = dot(ro, rd);
  float c = dot(ro, ro) - R * R;
  float h = b * b - c;
  float r2 = length(q);

  vec3 bg = vec3(0.0);
  float alpha = 0.0;
  // soft outer glow even when the ray misses
  float edge = length(q) / (R * 1.0);
  float glow = exp(-max(edge - 1.0, 0.0) * (14.0 - level * 6.0));
  vec3 glowCol = mix(c2, c3, 0.5 + 0.5 * sin(atan(q.y, q.x) - spin)) * (0.25 + 0.35 * energy + 0.6 * level);

  if (h < 0.0) {
    gl_FragColor = vec4(glowCol * glow, glow * (0.25 + 0.3 * energy + 0.5 * level));
    return;
  }
  h = sqrt(h);
  float tn = -b - h, tf = -b + h;
  vec3 pn = ro + rd * tn;
  vec3 nrm = normalize(pn);

  mat3 world = rotY(spin * 0.6) * rotX(sin(t * 0.13) * 0.5 + 0.3);
  float amp = 0.10 + energy * 0.12 + level * 0.22;
  float w = (0.055 + level * 0.035) * (1.0 + soft * 1.5);
  float sp = 0.5 + energy * 1.4;

  mat3 m1 = rotZ(0.3);
  mat3 m2 = rotZ(1.9) * rotY(1.2);
  mat3 m3 = rotX(1.2) * rotZ(-0.8);
  mat3 m4 = rotY(2.4) * rotX(-0.9);

  // march through the ball front to back, accumulating glowing ribbons
  vec3 acc = vec3(0.0);
  float trans = 1.0;
  const int N = 36;
  float dt = (tf - tn) / float(N);
  for (int i = 0; i < N; i++) {
    float tt = tn + (float(i) + 0.5) * dt;
    vec3 p = (ro + rd * tt) * (1.0 / R);
    p = world * p;
    float d1 = ribbon(p, m1, 2.6, sp, amp, w, 0.0);
    float d2 = ribbon(p, m2, 2.2, sp * 1.2, amp, w, 1.7);
    float d3 = ribbon(p, m3, 3.0, sp * 0.9, amp, w, 3.3);
    float d4 = ribbon(p, m4, 2.4, sp * 1.1, amp, w, 4.9);
    vec3 e = c1 * d1 + c2 * d2 + c3 * d3 + c4 * d4;
    float dens = d1 + d2 + d3 + d4;
    // drifting clouds of coloured light that melt into each other (soft style)
    if (soft > 0.0) {
      float bs = 0.06 + level * 0.08 + energy * 0.02;
      vec3 b1 = vec3(sin(t * 0.41) * 0.45, cos(t * 0.33) * 0.35, sin(t * 0.27 + 1.0) * 0.4);
      vec3 b2 = vec3(cos(t * 0.37 + 2.0) * 0.45, sin(t * 0.29 + 0.5) * 0.4, cos(t * 0.31) * 0.4);
      vec3 b3 = vec3(sin(t * 0.23 + 4.0) * 0.4, cos(t * 0.43 + 3.0) * 0.4, sin(t * 0.35 + 2.0) * 0.45);
      float g1 = exp(-dot(p - b1, p - b1) / bs), g2 = exp(-dot(p - b2, p - b2) / bs), g3 = exp(-dot(p - b3, p - b3) / bs);
      e = e * (1.0 - soft * 0.5) + soft * (c1 * g1 + c3 * g2 + c2 * g3) * 0.6;
      dens = dens * (1.0 - soft * 0.6) + soft * (g1 + g2 + g3) * 0.3;
    }
    // parts further back are dimmer and cooler: depth you can see
    float depth = float(i) / float(N);
    e *= mix(1.25, 0.45, depth);
    acc += trans * e * dt * (6.0 - energy * 2.0) * (1.0 + level * 0.7);
    trans *= exp(-dens * dt * 2.2);
  }
  // white-hot where ribbons overlap
  float lum = dot(acc, vec3(0.333));
  acc += vec3(1.0) * smoothstep(1.2, 2.6, lum) * 0.35 * (1.0 - soft * 0.7);
  acc = acc / (1.0 + acc * 0.75);
  // richer, more luminous colour in the soft style (less milky grey where colours overlap)
  float l2 = dot(acc, vec3(0.299, 0.587, 0.114));
  acc = max(mix(vec3(l2), acc, 1.0 + soft * 0.8), 0.0) * (1.0 + soft * 0.1);

  // glass shell: fresnel rim, a soft key light, and a sharp specular highlight
  float fres = pow(1.0 - max(dot(nrm, -rd), 0.0), 3.0);
  vec3 L = normalize(vec3(-0.5, 0.7, 0.6));
  float spec = pow(max(dot(reflect(rd, nrm), L), 0.0), 60.0);
  float spec2 = pow(max(dot(reflect(rd, nrm), normalize(vec3(0.6, -0.5, 0.5))), 0.0), 18.0);
  vec3 rimCol = mix(c2, c3, 0.5 + 0.5 * sin(atan(q.y, q.x) + spin)) * fres * 0.9;
  vec3 body = vec3(0.015, 0.02, 0.045) * (0.6 + 0.4 * nrm.y);
  vec3 col = body + acc + rimCol + vec3(spec) * (0.8 - soft * 0.35) + c3 * spec2 * 0.12;

  float aa = smoothstep(0.0, 0.004, h * h);
  gl_FragColor = vec4(mix(glowCol * glow, col, aa), max(aa, glow * 0.4));
}
`

function hex(c: string): [number, number, number] {
  const n = parseInt(c.replace("#", ""), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

const PALETTES: Record<string, string[]> = {
  gemma: ["#FF3D9A", "#7A5CFF", "#22C7FF", "#FF8A3D"],   // pink, violet, cyan, orange
  claude: ["#FF6A3D", "#FF3D7F", "#FFB547", "#B45CFF"],  // warmer for Claude
}

export function SiriOrb({ phase, brain, inVol, outVol, style = "glow" }: {
  style?: "glow" | "ribbons"
  phase: Phase
  brain: string
  inVol: React.MutableRefObject<number>
  outVol: React.MutableRefObject<number>
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const phaseRef = useRef(phase)
  const brainRef = useRef(brain)
  const styleRef = useRef(style)
  styleRef.current = style
  phaseRef.current = phase
  brainRef.current = brain

  useEffect(() => {
    const cv = canvas.current!
    const gl = cv.getContext("webgl", { premultipliedAlpha: false, alpha: true, antialias: true })
    if (!gl) return
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!
      gl.shaderSource(s, src)
      gl.compileShader(s)
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s))
      return s
    }
    const prog = gl.createProgram()!
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT))
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG))
    gl.linkProgram(prog)
    gl.useProgram(prog)
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
    const loc = gl.getAttribLocation(prog, "p")
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    const u = (n: string) => gl.getUniformLocation(prog, n)
    const U = { t: u("t"), level: u("level"), energy: u("energy"), spin: u("spin"), soft: u("soft"),
      c: [u("c1"), u("c2"), u("c3"), u("c4")] }

    let raf = 0, last = performance.now(), spin = 0, lvl = 0, energy = 0.2, soft = styleRef.current === "glow" ? 1 : 0
    const cols = PALETTES.gemma.map(hex)
    const start = last
    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5)
      const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr)
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h }
      gl.viewport(0, 0, w, h)

      const ph = phaseRef.current
      const target = ph === "talking" ? 0.85 : ph === "listening" ? 0.6 : ph === "thinking" || ph === "loading" ? 0.7 : 0.2
      energy += (target - energy) * Math.min(1, dt * 3)
      const want = ph === "talking" ? outVol.current : ph === "listening" ? inVol.current : 0
      lvl += (want - lvl) * Math.min(1, dt * (want > lvl ? 18 : 6))
      const spinRate = ph === "thinking" || ph === "loading" ? 2.4 : 0.35 + energy * 0.8
      spin += dt * spinRate

      // ease colours toward the current brain's palette
      const pal = (PALETTES[brainRef.current] || PALETTES.gemma).map(hex)
      for (let i = 0; i < 4; i++) for (let k = 0; k < 3; k++) cols[i][k] += (pal[i][k] - cols[i][k]) * Math.min(1, dt * 2)

      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.uniform1f(U.t, (now - start) / 1000)
      gl.uniform1f(U.level, lvl)
      gl.uniform1f(U.energy, energy)
      gl.uniform1f(U.spin, spin)
      soft += ((styleRef.current === "glow" ? 1 : 0) - soft) * Math.min(1, dt * 3)
      gl.uniform1f(U.soft, soft)
      U.c.forEach((l, i) => gl.uniform3fv(l, cols[i]))
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [inVol, outVol])

  return <canvas ref={canvas} style={{ width: "100%", height: "100%", display: "block" }} />
}
