import type { NextConfig } from "next";

// ── Deploy en GitHub Pages ──
// Requiere output: "export" para generar archivos estáticos en /out
// basePath debe coincidir con el nombre del repo en GitHub

const repoName = process.env.GITHUB_REPOSITORY
  ? "/" + process.env.GITHUB_REPOSITORY.split("/")[1]
  : "";

const nextConfig: NextConfig = {
  output: "export",
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
  basePath: repoName,
};

export default nextConfig;
