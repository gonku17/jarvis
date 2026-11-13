import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "../web", emptyOutDir: true },
  server: { proxy: { "/ws": { target: "ws://127.0.0.1:8765", ws: true } } },
})
