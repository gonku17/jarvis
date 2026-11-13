import { useCallback, useEffect, useRef, useState } from "react"
import { SiriOrb } from "./SiriOrb"

type Phase = "loading" | "idle" | "listening" | "thinking" | "talking"
type Msg = { role: "user" | "assistant" | "err"; text: string }

const STATUS: Record<Phase, string> = {
  loading: "Waking up…",
  idle: "Say “Hey Jarvis”",
  listening: "Listening…",
  thinking: "Thinking…",
  talking: "",
}

export default function App() {
  const [started, setStarted] = useState(false)
  const [connected, setConnected] = useState(false)
  const [phase, setPhase] = useState<Phase>("loading")
  const [log, setLog] = useState<Msg[]>([])
  const [draft, setDraft] = useState("")
  const [brain, setBrain] = useState<"gemma" | "claude">("gemma")
  const [hasKey, setHasKey] = useState(false)
  const [models, setModels] = useState({ local: "", claude: "" })
  const [showSettings, setShowSettings] = useState(false)
  const [keyDraft, setKeyDraft] = useState("")
  const [keyNote, setKeyNote] = useState("")
  const [voice, setVoice] = useState("piper")
  const [voices, setVoices] = useState<Record<string, string>>({})
  const [voiceNote, setVoiceNote] = useState("")
  const [allow, setAllow] = useState({ calendar: false, mail: false })
  const [alwaysOn, setAlwaysOn] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const inputVol = useRef(0)
  const outputVol = useRef(0)
  const queue = useRef<ArrayBuffer[]>([])
  const playing = useRef(false)
  const current = useRef<AudioBufferSourceNode | null>(null)
  const speechEnded = useRef(false)
  const logEnd = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "smooth" })
  }, [log])

  const send = (obj: unknown) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj))
  }

  const maybeDone = () => {
    if (!playing.current && queue.current.length === 0 && speechEnded.current) {
      speechEnded.current = false
      send({ type: "playback_done" })
    }
  }

  const playNext = useCallback(async () => {
    const ctx = ctxRef.current
    if (!ctx || playing.current) return
    const next = queue.current.shift()
    if (!next) return maybeDone()
    playing.current = true
    try {
      const buf = await ctx.decodeAudioData(next)
      const src = ctx.createBufferSource()
      src.buffer = buf
      src.connect(analyserRef.current!)
      src.onended = () => {
        if (current.current !== src) return
        current.current = null
        playing.current = false
        playNext()
      }
      current.current = src
      src.start()
    } catch {
      playing.current = false
      playNext()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const connect = useCallback(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws"
    const ws = new WebSocket(`${proto}://${location.host}/ws`)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => {
      setConnected(false)
      setTimeout(connect, 1500)
    }
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        queue.current.push(ev.data)
        playNext()
        return
      }
      const m = JSON.parse(ev.data)
      switch (m.type) {
        case "brain":
          setBrain(m.brain)
          setHasKey(m.has_key)
          setModels({ local: m.local_model, claude: m.claude_model })
          if (m.voice) setVoice(m.voice)
          if (m.voices) setVoices(m.voices)
          setAllow({ calendar: !!m.allow_calendar, mail: !!m.allow_mail })
          setAlwaysOn(!!m.always_on)
          setVoiceNote("")
          break
        case "voice_loading":
          setVoiceNote("Loading voice… (the first time downloads about 120 MB)")
          break
        case "need_key":
          setShowSettings(true)
          setKeyNote("Add an Anthropic API key to use Claude.")
          break
        case "key_saved":
          setKeyNote(m.ok ? (m.removed ? "Key removed." : "Saved in your Mac's Keychain.") : `Couldn't save: ${m.error}`)
          if (m.ok) setKeyDraft("")
          break
        case "state":
          setPhase(m.state)
          break
        case "user":
          setLog((l) => [...l, { role: "user", text: m.text }])
          break
        case "assistant_start":
          setLog((l) => [...l, { role: "assistant", text: "" }])
          break
        case "assistant_delta":
          setLog((l) => {
            const c = l.slice()
            const last = c[c.length - 1]
            if (last?.role === "assistant") c[c.length - 1] = { ...last, text: last.text + m.text }
            return c
          })
          break
        case "stop_audio": {
          // Interrupted: drop everything queued and cut the current sentence.
          queue.current = []
          speechEnded.current = false
          const src = current.current
          current.current = null
          playing.current = false
          try {
            src?.stop()
          } catch {
            /* already stopped */
          }
          break
        }
        case "speech_end":
          speechEnded.current = true
          maybeDone()
          break
        case "error":
          setLog((l) => [...l, { role: "err", text: m.message }])
          break
      }
    }
  }, [playNext])

  const start = async () => {
    if (ctxRef.current) return
    setStarted(true)
    const ctx = new AudioContext({ sampleRate: 16000 })
    ctxRef.current = ctx
    if (ctx.state === "suspended") {
      // Browsers may block audio until the first click; resume on any click.
      document.addEventListener("click", () => ctx.resume(), { once: true })
    }
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 512
    analyser.connect(ctx.destination)
    analyserRef.current = analyser

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
    await ctx.audioWorklet.addModule("./mic-worklet.js")
    const node = new AudioWorkletNode(ctx, "mic-frames")
    ctx.createMediaStreamSource(stream).connect(node)
    node.port.onmessage = (e) => {
      if (e.data instanceof ArrayBuffer) {
        if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(e.data)
      } else {
        inputVol.current = Math.min(1, e.data.rms * 10)
      }
    }

    const data = new Uint8Array(analyser.fftSize)
    const tick = () => {
      analyser.getByteTimeDomainData(data)
      let s = 0
      for (const v of data) s += ((v - 128) / 128) ** 2
      outputVol.current = Math.min(1, Math.sqrt(s / data.length) * 6)
      requestAnimationFrame(tick)
    }
    tick()
    connect()
  }

  // Launched from Jarvis.app: start straight away, no button.
  useEffect(() => {
    if (new URLSearchParams(location.search).has("app")) start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    const text = draft.trim()
    if (!text) return
    send({ type: "text", text })
    setDraft("")
  }

  const chooseBrain = (b: "gemma" | "claude") => {
    if (b === "claude" && !hasKey) {
      setShowSettings(true)
      setKeyNote("Add an Anthropic API key to use Claude.")
      return
    }
    send({ type: "set_brain", brain: b })
  }

  const saveKey = (e: React.FormEvent) => {
    e.preventDefault()
    if (keyDraft.trim()) send({ type: "set_key", key: keyDraft.trim() })
  }


  return (
    <div className="app">
      <div className="top">
        <span>
          <span className={`dot ${connected ? "on" : ""}`} />
          Jarvis {connected ? "online" : "offline"}
        </span>
        <span className="brains">
          <span className="seg">
            <button className={brain === "gemma" ? "on" : ""} onClick={() => chooseBrain("gemma")} title={models.local}>
              Gemma
            </button>
            <button className={brain === "claude" ? "on claude" : ""} onClick={() => chooseBrain("claude")} title={models.claude}>
              Claude
            </button>
          </span>
          <button className="gear" onClick={() => setShowSettings((v) => !v)} title="Settings" aria-label="Settings">
            ⚙
          </button>
        </span>
      </div>

      {showSettings && (
        <div className="settings">
          <div className="settings-title">Claude brain</div>
          <p className="muted">
            Gemma runs free and offline on this Mac. Claude is smarter but needs internet and an Anthropic API key
            (paid per use). Get one at console.anthropic.com. It's stored in your Mac's Keychain and never leaves this
            computer except to talk to Claude.
          </p>
          <form className="bar" onSubmit={saveKey}>
            <input
              type="password"
              autoComplete="off"
              value={keyDraft}
              onChange={(e) => setKeyDraft(e.target.value)}
              placeholder={hasKey ? "Key saved ✓ (paste a new one to replace)" : "sk-ant-…"}
            />
            <button type="submit">Save</button>
          </form>
          <div className="muted row">
            <span>{keyNote || (hasKey ? `Ready: ${models.claude}` : "No key yet")}</span>
            {hasKey && (
              <button className="link" onClick={() => send({ type: "set_key", key: "" })}>
                Remove key
              </button>
            )}
          </div>
          <p className="muted">Tip: you can also say “switch to Claude” or “switch to Gemma”.</p>
          <div className="settings-title">Voice</div>
          <div className="voices">
            {Object.entries(voices).map(([id, label]) => (
              <button
                key={id}
                className={id === voice ? "chip on" : "chip"}
                onClick={() => {
                  setVoice(id)
                  send({ type: "set_voice", voice: id })
                }}
              >
                {label}
              </button>
            ))}
          </div>
          {voiceNote && <p className="muted">{voiceNote}</p>}
          <div className="settings-title">Access</div>
          {(["calendar", "mail"] as const).map((what) => (
            <label key={what} className="toggle">
              <input
                type="checkbox"
                checked={allow[what]}
                onChange={(e) => send({ type: "set_allow", what, on: e.target.checked })}
              />
              Let Jarvis read your {what === "calendar" ? "Calendar" : "Mail"} app
            </label>
          ))}
          <div className="settings-title">Always on</div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={alwaysOn}
              onChange={(e) => send({ type: "set_always_on", on: e.target.checked })}
            />
            Listen for “Hey Jarvis” even with this window closed, show a menu bar icon, and start at login
          </label>
        </div>
      )}

      <div className="orb-wrap" onClick={() => send({ type: "ptt" })} title="Click to talk">
        <SiriOrb phase={phase} brain={brain} inVol={inputVol} outVol={outputVol} />
      </div>
      <div className="status">{STATUS[phase]}</div>

      <div className="log">
        {log.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.text}
          </div>
        ))}
        <div ref={logEnd} />
      </div>

      <form className="bar" onSubmit={submit}>
        <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Or type to Jarvis…" />
        <button type="submit">Send</button>
      </form>
      <div className="hint">Say “Hey Jarvis”, or click the orb to talk without the wake word.</div>

      {!started && (
        <button className="start" onClick={start}>
          Start Jarvis
        </button>
      )}
    </div>
  )
}
