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
uniform float t;       // time, seconds
uniform float level;   // 0..1 voice / mic loudness (smoothed)
uniform float energy;  // 0..1 how lively: idle low, talking high
uniform float spin;    // accumulated rotation
uniform vec3 c1; uniform vec3 c2; uniform vec3 c3; uniform vec3 c4;

// One flowing ribbon: a wavy band through the sphere, rotated by angle a.
float ribbon(vec2 q, float a, float freq, float speed, float amp, float width, float seed) {
  float ca = cos(a), sa = sin(a);
  vec2 r = vec2(ca * q.x - sa * q.y, sa * q.x + ca * q.y);
  float wave = sin(r.x * freq + t * speed + seed) * amp
             + sin(r.x * freq * 1.9 - t * speed * 0.7 + seed * 2.3) * amp * 0.45;
  float d = abs(r.y - wave);
  float w = width * (0.6 + 0.4 * (1.0 - r.x * r.x));     // thinner toward the edges
  return exp(-d * d / (w * w));
}

void main() {
  vec2 q = uv * 1.08;
  float r = length(q);
  float R = 0.82 + level * 0.06;                          // sphere breathes with the voice
  if (r > 1.05) { gl_FragColor = vec4(0.0); return; }

  float inside = smoothstep(R + 0.006, R - 0.012, r);
  // Bend coordinates like a lens so ribbons look like they're inside a ball.
  vec2 s = q / R;
  float z = sqrt(max(0.0, 1.0 - dot(s, s)));
  vec2 lens = s * (0.75 + 0.25 * z);

  float amp = 0.12 + energy * 0.18 + level * 0.35;
  float wid = 0.14 + energy * 0.06 + level * 0.12;
  float sp = 0.6 + energy * 1.6;

  vec3 col = vec3(0.0);
  col += c1 * ribbon(lens, spin + 0.0, 3.1, sp * 1.00, amp, wid, 0.0);
  col += c2 * ribbon(lens, spin * -1.3 + 1.1, 2.6, sp * 1.25, amp * 1.1, wid * 0.9, 1.7);
  col += c3 * ribbon(lens, spin * 0.8 + 2.2, 3.7, sp * 0.85, amp * 0.9, wid * 1.1, 3.1);
  col += c4 * ribbon(lens, spin * -0.6 + 3.0, 2.2, sp * 1.10, amp * 1.2, wid * 0.8, 4.6);
  // soft colour haze filling the ball
  col += (c1 + c2 + c3) * 0.05 * (0.6 + energy) * z;
  // bright white core where ribbons overlap
  float lum = dot(col, vec3(0.33));
  col += vec3(1.0) * smoothstep(0.9, 2.2, lum) * 0.6;
  col = col / (1.0 + col * 0.55);                         // tone map so overlaps glow, not clip

  // Glass: dark body, a rim light and a highlight at the top left.
  vec3 glass = vec3(0.02, 0.025, 0.05) * (1.0 - z * 0.5);
  float rim = pow(1.0 - z, 3.0) * 0.55;
  vec3 rimCol = mix(c2, c3, 0.5 + 0.5 * sin(atan(q.y, q.x) + spin)) * rim;
  float hl = exp(-pow(length(s - vec2(-0.35, 0.45)) / 0.28, 2.0)) * 0.18;

  vec3 ball = glass + col * z + rimCol + vec3(hl);
  // Outer glow around the sphere.
  float glow = exp(-max(r - R, 0.0) / (0.05 + level * 0.06)) * (1.0 - inside);
  vec3 glowCol = mix(c2, c3, 0.5 + 0.5 * sin(atan(q.y, q.x) - spin)) * glow;

  vec3 outc = ball * inside + glowCol;
  float alpha = max(inside, clamp(glow * (0.15 + 0.25 * energy + 0.5 * level), 0.0, 1.0));
  gl_FragColor = vec4(outc, alpha);
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

export function SiriOrb({ phase, brain, inVol, outVol }: {
  phase: Phase
  brain: string
  inVol: React.MutableRefObject<number>
  outVol: React.MutableRefObject<number>
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const phaseRef = useRef(phase)
  const brainRef = useRef(brain)
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
    const U = { t: u("t"), level: u("level"), energy: u("energy"), spin: u("spin"),
      c: [u("c1"), u("c2"), u("c3"), u("c4")] }

    let raf = 0, last = performance.now(), spin = 0, lvl = 0, energy = 0.2
    const cols = PALETTES.gemma.map(hex)
    const start = last
    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
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
      U.c.forEach((l, i) => gl.uniform3fv(l, cols[i]))
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [inVol, outVol])

  return <canvas ref={canvas} style={{ width: "100%", height: "100%", display: "block" }} />
}
