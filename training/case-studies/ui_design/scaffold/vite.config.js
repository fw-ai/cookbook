import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Relative base so the built index.html opens from file:// and from any static host.
export default defineConfig({ plugins: [react()], base: "./", build: { outDir: "dist", emptyOutDir: true } });
