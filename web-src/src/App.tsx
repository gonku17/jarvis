import { useCallback, useEffect, useRef, useState } from "react"
import { SiriOrb } from "./SiriOrb"
import { Gestures } from "./Gestures"
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
  const [boot, setBoot] = useState<{ name: string; ok: boolean; detail: string }[] | null>(null)
  const [booted, setBooted] = useState(false)
  const [gesturesOn, setGesturesOn] = useState(() => { try { return localStorage.getItem("gestures") === "1" } catch { return false } })
  const [gestureErr, setGestureErr] = useState("")
  const onGestureErr = useCallback((m: string) => { setGestureErr(m); setGesturesOn(false) }, [])
  const [snap, setSnap] = useState<string | null>(null)  // last webcam photo, shown while Jarvis looks
  const [draft, setDraft] = useState("")
  const [brain, setBrain] = useState<"gemma" | "claude">("gemma")
  const [hasKey, setHasKey] = useState(false)
  const [models, setModels] = useState({ local: "", claude: "" })
  const [localLabel, setLocalLabel] = useState("Gemma")
  const [localModels, setLocalModels] = useState<{ name: string; label: string }[]>([])
  const [offline, setOffline] = useState(false)
  const [secondBrain, setSecondBrain] = useState("")
  const [brainDraft, setBrainDraft] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [keyDraft, setKeyDraft] = useState("")
  const [keyNote, setKeyNote] = useState("")
  const [voice, setVoice] = useState("piper")
  const [voices, setVoices] = useState<Record<string, string>>({})
  const [voiceGender, setVoiceGender] = useState<Record<string, string>>({})
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
  const serverMic = useRef(false)  // native window: the server listens through the Mac's mic
  const [serverMicName, setServerMicName] = useState("")
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
    ws.onopen = () => {
      setConnected(true)
      ws.send(JSON.stringify({ type: "window_ready" }))
      if (serverMic.current) ws.send(JSON.stringify({ type: "server_mic", device: (() => { try { return localStorage.getItem("serverMic") || "" } catch { return "" } })() }))
    }
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
          if (m.local_label) setLocalLabel(m.local_label.replace(/ E\d+B$/, ""))
          if (m.local_models) setLocalModels(m.local_models)
          setOffline(!!m.offline)
          if (m.second_brain !== undefined) setSecondBrain(m.second_brain)
          if (m.voice) setVoice(m.voice)
          if (m.voices) setVoices(m.voices)
          if (m.voice_gender) setVoiceGender(m.voice_gender)
          setAllow({ calendar: !!m.allow_calendar, mail: !!m.allow_mail })
          setAlwaysOn(!!m.always_on)
          setFast({ available: !!m.fast_available, on: m.fast_answers !== false, model: m.fast_model || "" })
          if (m.personality) setPersonality(m.personality)
          if (m.personalities) setPersonalities(m.personalities)
          setVoiceNote("")
          break
        case "systems":
          setBoot((b) => b ?? m.checks)  // once per window
          break
        case "camera_snap":
          takePhoto().then(
            (data) => { ws.send(JSON.stringify({ type: "camera_frame", data })); setSnap(data); setTimeout(() => setSnap(null), 6000) },
            (e) => ws.send(JSON.stringify({ type: "camera_frame", data: null, error: String(e?.name || e) })))
          break
        case "mic_level":
          inputVol.current = m.level
          break
        case "server_mic":
          if (m.ok) { setServerMicName(m.device); setMicErr("") }
          else setMicErr(`Couldn't open the microphone: ${m.error}`)
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
    if (serverMic.current) {
      try {
        const r = await (await fetch("/api/mics")).json()
        setMics((r.devices || []).map((n: string) => ({ id: n, label: n })))
      } catch { /* ignore */ }
      return
    }
    try {
      const all = await navigator.mediaDevices.enumerateDevices()
      setMics(all.filter((d) => d.kind === "audioinput" && d.deviceId !== "communications")
        .map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}` })))
    } catch { /* ignore */ }
  }
  const openMic = async (id: string) => {
    const ctx = audioCtx.current, node = micNode.current
    if (!ctx || !node) return
    if (!navigator.mediaDevices?.getUserMedia || serverMic.current) {
      // The native window can't open the mic itself: Jarvis listens through the Mac directly.
      serverMic.current = true
      if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify({ type: "server_mic", device: id }))
      listMics()
      return
    }
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
      ;(window as unknown as { jarvisReport?: (m: string) => void }).jarvisReport?.(`mic ok: ${stream.getAudioTracks()[0]?.label || "default"}`)
      listMics()
    } catch (e) {
      const name = (e as DOMException)?.name
      ;(window as unknown as { jarvisReport?: (m: string) => void }).jarvisReport?.(`mic: ${name} ${(e as Error)?.message || e}`)
      setMicErr(name === "NotAllowedError"
        ? "Jarvis isn't allowed to use the microphone. In System Settings → Privacy & Security → Microphone, turn on Jarvis, then reopen it."
        : name === "NotFoundError" || name === "OverconstrainedError"
          ? "That microphone isn't connected any more. Pick another one."
          : `Couldn't open the microphone: ${name || e}`)
      if (id) { setMicId(""); try { localStorage.removeItem("micId") } catch { /* ignore */ } }
    }
  }
  const pickMic = (id: string) => {
    setMicId(id)
    try { localStorage.setItem(serverMic.current ? "serverMic" : "micId", id) } catch { /* ignore */ }
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
          {connected && offline && <span className="offline-badge" title="Offline mode: nothing leaves this Mac">offline mode</span>}
        </span>
        <span className="brains">
          <span className="seg">
            <button className={brain === "gemma" ? "on" : ""} onClick={() => chooseBrain("gemma")} title={models.local}>
              {localLabel || "Gemma"}
            </button>
            <button className={brain === "claude" ? "on claude" : ""} onClick={() => chooseBrain("claude")}
              title={offline ? "Off while offline mode is on" : models.claude} disabled={offline}>
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
          <div className="settings-title">Local brain</div>
          <select className="mic-select" value={models.local}
            onChange={(e) => send({ type: "set_local_model", model: e.target.value })}>
            {localModels.length === 0 && <option value={models.local}>{localLabel}</option>}
            {localModels.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}
          </select>
          <p className="muted">Any model installed in Ollama. Say “switch to Qwen” or “what models do you have?”.
            Add more in Terminal with <code>ollama pull</code>.</p>
          <label className="toggle">
            <input type="checkbox" checked={offline}
              onChange={(e) => send({ type: "set_offline", on: e.target.checked })} />
            Offline mode: nothing leaves this Mac (no Claude, weather, news or web search). Say “go offline” / “go online”.
          </label>
          <div className="settings-title">Second brain</div>
          <form className="bar" onSubmit={(e) => {
            e.preventDefault()
            if (brainDraft !== null) send({ type: "set_second_brain", folder: brainDraft })
            setBrainDraft(null)
          }}>
            <input value={brainDraft ?? secondBrain} onChange={(e) => setBrainDraft(e.target.value)}
              placeholder="~/Documents" spellCheck={false} />
            <button type="submit">Save</button>
          </form>
          <p className="muted">The folder Jarvis searches when you ask “what do my notes say about…?” or “pull up…”.
            Searched on this Mac; if Claude is the brain, the matching excerpts go to Claude with your question.</p>
          <label className="toggle">
            <input type="checkbox" checked={fast.on && fast.available} disabled={!fast.available}
              onChange={(e) => send({ type: "set_fast_answers", on: e.target.checked })} />
            Quick answers: use a small, fast model for simple questions
            {!fast.available && <span className="muted"> (not installed yet)</span>}
          </label>
          <label className="toggle">
            <input type="checkbox" checked={gesturesOn} onChange={(e) => {
              setGesturesOn(e.target.checked); setGestureErr("")
              try { localStorage.setItem("gestures", e.target.checked ? "1" : "0") } catch { /* ignore */ }
            }} />
            Hand gestures: ✋ hold up your hand to talk or stop, 👍 play/pause, swipe to skip (uses the camera while this window is open)
          </label>
          {gestureErr && <p className="err-note">{gestureErr}</p>}
          <div className="settings-title">Microphone</div>
          <select className="mic-select" value={(serverMic.current ? serverMicName : micId) || "default"} onFocus={listMics}
            onChange={(e) => pickMic(e.target.value)}>
            {mics.length === 0 && <option value="default">Default microphone</option>}
            {mics.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          <div className="mic-meter" title="Speak to test"><div style={{ width: `${Math.round(micLevel * 100)}%` }} /></div>
          {micErr ? <p className="err-note">{micErr}</p> : <p className="muted">Speak and the bar should move.</p>}
          <div className="settings-title">Voice</div>
          <p className="muted">⚡ = fast on this Mac. The others sound more natural but take longer to speak.</p>
          {(["Female", "Male"] as const).map((group) => (
            <div key={group}>
              <div className="voice-group">{group}</div>
              <div className="voices">
                {Object.entries(voices)
                  .filter(([id]) => (voiceGender[id] || id[1]) === (group === "Female" ? "f" : "m"))
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

      <div className={"stage" + (hudOn ? " with-hud" : "") + (boot && !booted ? " booting" : "")}>
      {hudOn && <Hud orbRef={orbRef} phase={phase} lastUserText={lastUserText} send={send} resetKey={hudReset}
        ask={(text) => send({ type: "text", text })} />}
      {gesturesOn && started && <Gestures send={send} onError={onGestureErr} />}
      {boot && <Boot checks={boot} onDone={() => setBooted(true)} />}
      {snap && <div className="snap"><img src={`data:image/jpeg;base64,${snap}`} alt="What Jarvis sees" /><div className="scanline" /></div>}
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

// Film-style start-up: a systems check that types itself out next to the orb, then fades away.
function Boot({ checks, onDone }: { checks: { name: string; ok: boolean; detail: string }[]; onDone: () => void }) {
  const [shown, setShown] = useState(0)
  const [gone, setGone] = useState(false)
  const fired = useRef(false)
  useEffect(() => {
    const t = setInterval(() => setShown((n) => Math.min(n + 1, checks.length + 2)), 260)
    return () => clearInterval(t)
  }, [checks.length])
  useEffect(() => {
    if (shown === checks.length + 2 && !fired.current) {
      fired.current = true
      setTimeout(onDone, 900)  // the floating panels come online
      setTimeout(() => setGone(true), 3500)
    }
  }, [shown, checks.length, onDone])
  if (gone) return null
  return (
    <div className={"boot" + (shown > checks.length + 1 ? " fade" : "")}>
      <div className="boot-title">J.A.R.V.I.S. · systems check</div>
      {checks.slice(0, shown).map((c) => (
        <div key={c.name} className={"boot-line" + (c.ok ? " ok" : "")}>
          <span>{c.name}</span><span className="dots" /><span>{c.ok ? "online" : c.detail}</span>
        </div>
      ))}
      {shown > checks.length && <div className="boot-line ok all">All core systems online</div>}
    </div>
  )
}

// One webcam photo for "what am I holding?": camera on for about a second, then off again.
async function takePhoto(): Promise<string> {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } })
  try {
    const v = document.createElement("video")
    v.muted = true
    v.playsInline = true
    v.srcObject = stream
    await v.play()
    await new Promise((r) => setTimeout(r, 700))  // let the camera adjust its exposure
    const w = 768, h = Math.round((v.videoHeight / v.videoWidth) * w) || 432
    const c = document.createElement("canvas")
    c.width = w; c.height = h
    const g = c.getContext("2d")!
    g.translate(w, 0); g.scale(-1, 1)  // mirror, like looking in a mirror
    g.drawImage(v, 0, 0, w, h)
    return c.toDataURL("image/jpeg", 0.8).split(",")[1]
  } finally {
    stream.getTracks().forEach((t) => t.stop())
  }
}
