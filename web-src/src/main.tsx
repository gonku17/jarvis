import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./app.css"
import App from "./App"

// Send window errors to the server log (the native window has no visible console).
const report = (message: string) => {
  try { fetch("/api/clientlog", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message }) }) } catch { /* offline */ }
}
window.addEventListener("error", (e) => report(`${e.message} @ ${e.filename}:${e.lineno}`))
window.addEventListener("unhandledrejection", (e) => report(`unhandled: ${e.reason?.message || e.reason}`))
;(window as unknown as { jarvisReport: typeof report }).jarvisReport = report

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
