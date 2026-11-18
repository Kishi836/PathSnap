import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// `npm run dev` / `npm run preview` listen on all interfaces so the UI laptop can also be
// opened from another device; the API address comes from VITE_API_BASE_URL (see .env.example).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true },
  preview: { port: 5173, strictPort: true },
  test: { environment: "node" },
});
