import { useEffect, useRef, useState } from "react"

// Hand gestures through the webcam, recognised on this Mac (MediaPipe, no cloud):
//   ✋ hold an open hand  -> talk to Jarvis, or stop it talking
//   👍 thumbs up          -> play / pause music
//   swipe left / right    -> next / previous track
// The model files are fetched once by the Jarvis server and kept in the Jarvis folder.

type Props = { send: (m: object) => void; onError: (msg: string) => void }

const HOLD_MS = 600
const COOLDOWN_MS = 1500
const FPS = 8

export function Gestures({ send, onError }: Props) {
  const [flash, setFlash] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const sendRef = useRef(send)
  sendRef.current = send

  useEffect(() => {
    let stop = false
    let stream: MediaStream | null = null
    let timer = 0
    ;(async () => {
      try {
        const { FilesetResolver, GestureRecognizer } = await import("@mediapipe/tasks-vision")
        const files = await FilesetResolver.forVisionTasks("/mp")
        const rec = await GestureRecognizer.createFromOptions(files, {
          baseOptions: { modelAssetPath: "/mp/gesture_recognizer.task", delegate: "CPU" },
          runningMode: "VIDEO", numHands: 1,
        })
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240, frameRate: 15 } })
        if (stop) return
        const v = document.createElement("video")
        v.muted = true; v.playsInline = true; v.srcObject = stream
        await v.play()
        setReady(true)

        let held = "", heldSince = 0, lastFire = 0
        const xs: { t: number; x: number }[] = []
        const fire = (what: string, icon: string, msg: object) => {
          lastFire = performance.now()
          setFlash(icon); setTimeout(() => setFlash(null), 900)
          sendRef.current(msg)
          console.log("gesture", what)
        }
        const tick = () => {
          if (stop) return
          const now = performance.now()
          if (v.readyState >= 2) {
            const r = rec.recognizeForVideo(v, now)
            const g = r.gestures?.[0]?.[0]
            const name = g && g.score > 0.6 ? g.categoryName : ""
            const wrist = r.landmarks?.[0]?.[0]
            // swipes: the wrist moving fast sideways (image is mirrored, so flip x)
            if (wrist) {
              xs.push({ t: now, x: 1 - wrist.x })
              while (xs.length && now - xs[0].t > 450) xs.shift()
              const dx = xs.length > 2 ? xs[xs.length - 1].x - xs[0].x : 0
              if (Math.abs(dx) > 0.35 && now - lastFire > COOLDOWN_MS) {
                xs.length = 0
                if (dx > 0) fire("swipe right", "⏭", { type: "hud_action", action: "music_next" })
                else fire("swipe left", "⏮", { type: "hud_action", action: "music_prev" })
              }
            } else xs.length = 0
            if (name !== held) { held = name; heldSince = now }
            if (now - lastFire > COOLDOWN_MS && now - heldSince > HOLD_MS) {
              if (held === "Open_Palm") fire("open palm", "✋", { type: "ptt" })
              else if (held === "Thumb_Up") fire("thumbs up", "👍", { type: "hud_action", action: "music_toggle" })
              if (held === "Open_Palm" || held === "Thumb_Up") heldSince = now + 10_000  // once per hold
            }
          }
          timer = window.setTimeout(tick, 1000 / FPS)
        }
        tick()
      } catch (e) {
        const name = (e as DOMException)?.name
        onError(name === "NotAllowedError" ? "Jarvis isn't allowed to use the camera." : `Hand gestures couldn't start: ${(e as Error)?.message || e}`)
      }
    })()
    return () => { stop = true; clearTimeout(timer); stream?.getTracks().forEach((t) => t.stop()) }
  }, [onError])

  return (
    <div className="gestures" title="Hand gestures are on: ✋ hold to talk or stop, 👍 play/pause, swipe to skip">
      <span className={"cam-dot" + (ready ? " on" : "")} />✋
      {flash && <span className="gesture-flash">{flash}</span>}
    </div>
  )
}
