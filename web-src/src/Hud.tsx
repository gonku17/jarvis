import { useEffect, useRef, useState } from "react"

// Floating info panels around the orb. They drift gently, light up when Jarvis talks about them,
// expand on hover, act on click, and can be dragged around or hidden (remembered per browser).

type HudData = {
  weather?: { place: string; temp: number; desc: string; hi: number; lo: number; rain: number | null } | null
  event?: { locked?: boolean; title?: string | null; when?: string; count?: number } | null
  mail?: { locked?: boolean; unread?: number; latest?: { from: string; subject: string }[] } | null
  news?: string[] | null
  music?: { app: string; playing: boolean; title: string; artist: string } | null
  battery?: { percent: number; charging: boolean } | null
  timers?: { label: string; ends: number }[]
  now?: number
}

type PanelId = "clock" | "weather" | "calendar" | "mail" | "music" | "timer" | "news" | "battery"

// Default spots, as offsets from the orb's centre in units of the orb's radius.
const HOME: Record<PanelId, [number, number]> = {
  clock: [-1.8, -0.72], weather: [1.8, -0.72],
  calendar: [-2.0, 0.12], mail: [2.0, 0.12],
  music: [-1.75, 0.95], timer: [1.75, 0.95],
  news: [0, -1.22], battery: [0, 1.3],
}

const TOPICS: [PanelId, RegExp][] = [
  ["weather", /\b(weather|forecast|rain|temperature|umbrella|degrees)\b/i],
  ["calendar", /\b(calendar|schedule|meeting|event|agenda|free)\b/i],
  ["mail", /\b(e-?mails?|inbox|mail)\b/i],
  ["music", /\b(play|music|song|track|pause|skip|spotify)\b/i],
  ["timer", /\b(timer|countdown)\b/i],
  ["news", /\b(news|headlines?)\b/i],
  ["clock", /\b(time|date|day is it|good morning)\b/i],
  ["battery", /\b(battery|charge|charging)\b/i],
]

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v ? (JSON.parse(v) as T) : fallback
  } catch {
    return fallback
  }
}
function save(key: string, v: unknown) {
  try { localStorage.setItem(key, JSON.stringify(v)) } catch { /* private mode */ }
}

function icon(code?: string) {
  if (!code) return "☁️"
  if (/thunder/.test(code)) return "⛈️"
  if (/snow/.test(code)) return "❄️"
  if (/rain|drizzle|shower/.test(code)) return "🌧️"
  if (/fog/.test(code)) return "🌫️"
  if (/clear|sun/.test(code)) return "☀️"
  if (/partly/.test(code)) return "⛅"
  return "☁️"
}

function left(ends: number) {
  const s = Math.max(0, Math.round(ends - Date.now() / 1000))
  const m = Math.floor(s / 60)
  return m >= 60 ? `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}` : `${m}:${String(s % 60).padStart(2, "0")}`
}

export function Hud({ orbRef, phase, lastUserText, send, ask, resetKey }: {
  orbRef: React.RefObject<HTMLDivElement | null>
  phase: string
  lastUserText: string
  send: (m: object) => void
  ask: (text: string) => void
  resetKey: number
}) {
  const [data, setData] = useState<HudData>({})
  const [now, setNow] = useState(() => new Date())
  const [offsets, setOffsets] = useState<Partial<Record<PanelId, [number, number]>>>(() => load("hudOffsets", {}))
  const [hidden, setHidden] = useState<PanelId[]>(() => load("hudHidden", []))
  const [box, setBox] = useState({ cx: 0, cy: 0, r: 150, w: 800 })
  const drag = useRef<{ id: PanelId; sx: number; sy: number; ox: number; oy: number; moved: boolean } | null>(null)
  const layer = useRef<HTMLDivElement>(null)

  // reset layout from settings
  useEffect(() => {
    if (!resetKey) return
    setOffsets({}); setHidden([]); save("hudOffsets", {}); save("hudHidden", [])
  }, [resetKey])

  // live data: poll the server every 5 s (it caches the slow bits)
  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const r = await fetch("/api/hud")
        if (r.ok && !stop) setData(await r.json())
      } catch { /* server restarting */ }
    }
    tick()
    const a = setInterval(tick, 5000)
    const b = setInterval(() => setNow(new Date()), 1000)
    return () => { stop = true; clearInterval(a); clearInterval(b) }
  }, [])

  // follow the orb's position and size
  useEffect(() => {
    const measure = () => {
      const o = orbRef.current, l = layer.current
      if (!o || !l) return
      const ob = o.getBoundingClientRect(), lb = l.getBoundingClientRect()
      setBox({ cx: ob.left - lb.left + ob.width / 2, cy: ob.top - lb.top + ob.height / 2, r: ob.width / 2, w: lb.width })
    }
    measure()
    const ro = new ResizeObserver(measure)
    if (orbRef.current) ro.observe(orbRef.current)
    if (layer.current) ro.observe(layer.current)
    window.addEventListener("resize", measure)
    return () => { ro.disconnect(); window.removeEventListener("resize", measure) }
  }, [orbRef])

  // which panel Jarvis is talking about right now
  const busy = phase === "thinking" || phase === "talking"
  const topic = busy ? TOPICS.find(([, re]) => re.test(lastUserText))?.[0] : undefined

  const w = data.weather, ev = data.event, mail = data.mail, mu = data.music, bat = data.battery
  const timers = (data.timers || []).filter((t) => t.ends * 1000 > Date.now())

  type Panel = { id: PanelId; title: string; main: string; sub?: string; more?: React.ReactNode; onClick?: () => void; show: boolean }
  const panels: Panel[] = [
    { id: "clock", title: now.toLocaleDateString(undefined, { weekday: "long" }),
      main: now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      sub: now.toLocaleDateString(undefined, { day: "numeric", month: "long" }),
      onClick: () => ask("what's the date today"), show: true },
    { id: "weather", title: w ? w.place : "Weather", main: w ? `${icon(w.desc)} ${w.temp}°` : "…",
      sub: w ? w.desc : "Looking outside…",
      more: w && <>High {w.hi}° · Low {w.lo}°{w.rain != null && w.rain >= 20 ? <><br />Rain {w.rain}%</> : null}<br /><em>Click for the forecast</em></>,
      onClick: () => ask("what's the weather"), show: true },
    { id: "calendar", title: "Next up", main: ev?.locked ? "🔒" : ev?.title ? ev.title : "Nothing planned",
      sub: ev?.locked ? "Allow Calendar in settings" : ev?.when,
      more: ev?.title ? <>{ev.count && ev.count > 1 ? `${ev.count} events in the next two days` : "Only event coming up"}<br /><em>Click to open Calendar</em></> : null,
      onClick: () => (ev?.locked ? ask("what's on my calendar today") : send({ type: "hud_action", action: "open_calendar" })), show: true },
    { id: "mail", title: "Mail", main: mail?.locked ? "🔒" : `✉️ ${mail?.unread ?? "…"}`,
      sub: mail?.locked ? "Allow Mail in settings" : mail?.unread === 1 ? "unread email" : "unread emails",
      more: mail?.latest?.length ? <>{mail.latest.map((m, i) => <div key={i} className="hud-line"><b>{m.from}</b> {m.subject}</div>)}<em>Click to open Mail</em></> : null,
      onClick: () => (mail?.locked ? ask("do I have any new emails") : send({ type: "hud_action", action: "open_mail" })), show: true },
    { id: "music", title: mu ? (mu.playing ? "Now playing" : "Paused") : "Music",
      main: mu ? mu.title : "♪ Nothing playing", sub: mu ? mu.artist : "Say “play …”",
      more: mu ? <div className="hud-buttons" onPointerDown={(e) => e.stopPropagation()}>
        <button onClick={(e) => { e.stopPropagation(); send({ type: "hud_action", action: "music_prev" }) }}>⏮</button>
        <button onClick={(e) => { e.stopPropagation(); send({ type: "hud_action", action: "music_toggle" }) }}>{mu.playing ? "⏸" : "▶️"}</button>
        <button onClick={(e) => { e.stopPropagation(); send({ type: "hud_action", action: "music_next" }) }}>⏭</button>
      </div> : null,
      onClick: () => mu && send({ type: "hud_action", action: "music_toggle" }), show: true },
    { id: "timer", title: timers.length > 1 ? `${timers.length} timers` : "Timer",
      main: timers.length ? `⏱ ${left(Math.min(...timers.map((t) => t.ends)))}` : "⏱ –",
      sub: timers.length ? timers[0].label : "Say “set a timer for…”",
      more: timers.length ? <em>Click to cancel</em> : null,
      onClick: () => timers.length && send({ type: "hud_action", action: "cancel_timers" }), show: true },
    { id: "news", title: "Headline", main: data.news?.[0] || "…", more: data.news && data.news.length > 1 ? <>{data.news.slice(1).map((n, i) => <div key={i} className="hud-line">{n}</div>)}<em>Click to hear the news</em></> : null,
      onClick: () => ask("what's the news"), show: !!data.news?.length },
    { id: "battery", title: "Battery", main: bat ? `${bat.charging ? "⚡" : "🔋"} ${bat.percent}%` : "…",
      sub: bat ? (bat.charging ? "Charging" : "On battery") : undefined, show: !!bat },
  ]

  const narrow = box.w < 640
  const onDown = (id: PanelId) => (e: React.PointerEvent) => {
    if (narrow) return
    const o = offsets[id] || [0, 0]
    drag.current = { id, sx: e.clientX, sy: e.clientY, ox: o[0], oy: o[1], moved: false }
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  }
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const dx = e.clientX - d.sx, dy = e.clientY - d.sy
    if (Math.abs(dx) + Math.abs(dy) > 4) d.moved = true
    if (d.moved) setOffsets((o) => ({ ...o, [d.id]: [d.ox + dx, d.oy + dy] }))
  }
  const onUp = (p: Panel) => () => {
    const d = drag.current
    drag.current = null
    if (d?.moved) { setOffsets((o) => { save("hudOffsets", o); return o }); return }
    p.onClick?.()
  }
  const hide = (id: PanelId) => (e: React.MouseEvent) => {
    e.stopPropagation()
    const h = [...hidden, id]
    setHidden(h); save("hudHidden", h)
  }

  return (
    <div ref={layer} className={"hud" + (narrow ? " narrow" : "")}>
      {panels.filter((p) => p.show && !hidden.includes(p.id)).map((p, i) => {
        const [hx, hy] = HOME[p.id]
        const off = offsets[p.id] || [0, 0]
        const style: React.CSSProperties = narrow ? {} : {
          left: box.cx + hx * box.r + off[0], top: box.cy + hy * box.r + off[1],
          animationDelay: `${-i * 1.7}s`,
        }
        return (
          <div key={p.id}
            className={`hud-panel hud-${p.id}` + (topic === p.id ? " active" : "") + (p.onClick ? " clickable" : "")}
            style={style} onPointerDown={onDown(p.id)} onPointerMove={onMove} onPointerUp={onUp(p)}>
            <button className="hud-x" title="Hide" onPointerDown={(e) => e.stopPropagation()} onClick={hide(p.id)}>×</button>
            <div className="hud-title">{p.title}</div>
            <div className="hud-main">{p.main}</div>
            {p.sub && <div className="hud-sub">{p.sub}</div>}
            {p.more && <div className="hud-more">{p.more}</div>}
          </div>
        )
      })}
    </div>
  )
}
