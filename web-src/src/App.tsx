import { useCallback, useEffect, useRef, useState } from "react"
import { SiriOrb } from "./SiriOrb"
import { Hud } from "./Hud"

type Phase = "loading" | "idle" | "listening" | "thinking" | "talking"
type Msg = { role: "user" | "assistant" | "err"; text: string; old?: boolean }

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
  const [fast, setFast] = useState({ available: false, on: true, model: "" })
  const [personality, setPersonality] = useState("butler")
  const [orbStyle, setOrbStyle] = useState<"glow" | "ribbons">(() => {
    try { return localStorage.getItem("orbStyle") === "ribbons" ? "ribbons" : "glow" } catch { return "glow" }
  })
  const pickOrb = (v: "glow" | "ribbons") => {
    setOrbStyle(v)
    try { localStorage.setItem("orbStyle", v) } catch { /* private mode */ }
  }
  const [personalities, setPersonalities] = useState<Record<string, string>>({})

  const wsRef = useRef<WebSocket | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const inputVol = useRef(0)
  const micStream = useRef<MediaStream | null>(null)
  const micSource = useRef<MediaStreamAudioSourceNode | null>(null)
  const micNode = useRef<AudioWorkletNode | null>(null)
  const audioCtx = useRef<AudioContext | null>(null)
  const [mics, setMics] = useState<{ id: string; label: string }[]>([])
  const [micId, setMicId] = useState<string>(() => { try { return localStorage.getItem("micId") || "" } catch { return "" } })
  const [micErr, setMicErr] = useState("")
  const [micLevel, setMicLevel] = useState(0)
  const orbRef = useRef<HTMLDivElement>(null)
  const [lastUserText, setLastUserText] = useState("")
  const [hudReset, setHudReset] = useState(0)
  const [hudOn, setHudOn] = useState(() => { try { return localStorage.getItem("hudOn") !== "0" } catch { return true } })
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
          setFast({ available: !!m.fast_available, on: m.fast_answers !== false, model: m.fast_model || "" })
          if (m.personality) setPersonality(m.personality)
          if (m.personalities) setPersonalities(m.personalities)
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
          setLastUserText(m.text)
          break
        case "history":
          setLog((m.turns || []).map((x: { role: string; content: string }) => ({
            role: x.role === "user" ? "user" : "assistant", text: x.content, old: true }) as Msg))
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

    await ctx.audioWorklet.addModule("./mic-worklet.js")
    const node = new AudioWorkletNode(ctx, "mic-frames")
    audioCtx.current = ctx
    micNode.current = node
    await openMic(micId)
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

  // Microphone: list the Mac's inputs and switch between them without restarting.
  const listMics = async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices()
      setMics(all.filter((d) => d.kind === "audioinput" && d.deviceId !== "communications")
        .map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}` })))
    } catch { /* ignore */ }
  }
  const openMic = async (id: string) => {
    const ctx = audioCtx.current, node = micNode.current
    if (!ctx || !node) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
          ...(id && id !== "default" ? { deviceId: { exact: id } } : {}) },
      })
      micSource.current?.disconnect()
      micStream.current?.getTracks().forEach((t) => t.stop())
      micStream.current = stream
      micSource.current = ctx.createMediaStreamSource(stream)
      micSource.current.connect(node)
      setMicErr("")
      listMics()
    } catch (e) {
      const name = (e as DOMException)?.name
      setMicErr(name === "NotAllowedError"
        ? "Jarvis isn't allowed to use the microphone. In System Settings → Privacy & Security → Microphone, turn on Google Chrome, then reopen Jarvis."
        : name === "NotFoundError" || name === "OverconstrainedError"
          ? "That microphone isn't connected any more. Pick another one."
          : `Couldn't open the microphone: ${name || e}`)
      if (id) { setMicId(""); try { localStorage.removeItem("micId") } catch { /* ignore */ } }
    }
  }
  const pickMic = (id: string) => {
    setMicId(id)
    try { localStorage.setItem("micId", id) } catch { /* ignore */ }
    openMic(id)
  }
  useEffect(() => {
    navigator.mediaDevices?.addEventListener?.("devicechange", listMics)
    const t = setInterval(() => setMicLevel(inputVol.current), 150)
    return () => { navigator.mediaDevices?.removeEventListener?.("devicechange", listMics); clearInterval(t) }
  }, [])

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
          <label className="toggle">
            <input type="checkbox" checked={fast.on && fast.available} disabled={!fast.available}
              onChange={(e) => send({ type: "set_fast_answers", on: e.target.checked })} />
            Quick answers: use a small, fast model for simple questions
            {!fast.available && <span className="muted"> (not installed yet)</span>}
          </label>
          <div className="settings-title">Microphone</div>
          <select className="mic-select" value={micId || "default"} onFocus={listMics}
            onChange={(e) => pickMic(e.target.value)}>
            {mics.length === 0 && <option value="default">Default microphone</option>}
            {mics.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          <div className="mic-meter" title="Speak to test"><div style={{ width: `${Math.round(micLevel * 100)}%` }} /></div>
          {micErr ? <p className="err-note">{micErr}</p> : <p className="muted">Speak and the bar should move.</p>}
          <div className="settings-title">Voice</div>
          {(["Female", "Male"] as const).map((group) => (
            <div key={group}>
              <div className="voice-group">{group}</div>
              <div className="voices">
                {Object.entries(voices)
                  .filter(([id]) => (id === "piper" ? "m" : id[1]) === (group === "Female" ? "f" : "m"))
                  .map(([id, label]) => (
                    <button key={id} className={id === voice ? "chip on" : "chip"}
                      title={label}
                      onClick={() => { setVoice(id); send({ type: "set_voice", voice: id }) }}>
                      {label.split(" (")[0]}
                    </button>
                  ))}
              </div>
            </div>
          ))}
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
          <div className="settings-title">Orb style</div>
          <div className="voices">
            <button className={"chip" + (orbStyle === "glow" ? " on" : "")} onClick={() => pickOrb("glow")}>Soft glow</button>
            <button className={"chip" + (orbStyle === "ribbons" ? " on" : "")} onClick={() => pickOrb("ribbons")}>3D ribbons</button>
          </div>
          <div className="settings-title">Info panels</div>
          <label className="toggle">
            <input type="checkbox" checked={hudOn} onChange={(e) => {
              setHudOn(e.target.checked)
              try { localStorage.setItem("hudOn", e.target.checked ? "1" : "0") } catch { /* ignore */ }
            }} />
            Show floating info around the orb (drag to move, × to hide)
          </label>
          <button className="chip" onClick={() => setHudReset((n) => n + 1)}>Reset panel layout</button>
          <div className="settings-title">Personality</div>
          <div className="voices">
            {Object.entries(personalities).map(([id, label]) => (
              <button key={id} className={"chip" + (personality === id ? " on" : "")}
                onClick={() => send({ type: "set_personality", personality: id })}>{label}</button>
            ))}
          </div>
          <div className="settings-title">Conversation history</div>
          <button className="chip" onClick={() => send({ type: "clear_history" })}>Clear history</button>
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

      <div className={"stage" + (hudOn ? " with-hud" : "")}>
      {hudOn && <Hud orbRef={orbRef} phase={phase} lastUserText={lastUserText} send={send} resetKey={hudReset}
        ask={(text) => send({ type: "text", text })} />}
      <div ref={orbRef} className="orb-wrap" onClick={() => send({ type: "ptt" })} title="Click to talk">
        <SiriOrb phase={phase} brain={brain} inVol={inputVol} outVol={outputVol} style={orbStyle} />
      </div>
      </div>
      <div className="status">{STATUS[phase]}</div>

      <div className="log">
        {log.map((m, i) => (
          <div key={i} className={`msg ${m.role}${m.old ? " old" : ""}`}>
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
