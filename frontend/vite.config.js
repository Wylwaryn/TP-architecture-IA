import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En dev (npm run dev), on proxy /api vers le backend FastAPI local.
// En prod (Docker), c'est nginx qui fait ce proxy (cf. nginx.conf).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
