import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ML HyperLab - Laboratorio Interactivo de Machine Learning",
  description:
    "Experimenta con algoritmos de Machine Learning ajustando hiperparametros en tiempo real. XGBoost, Random Forest, KNN, redes neuronales, arboles de decision y regresion logistica.",
  keywords: [
    "Machine Learning",
    "Hiperparametros",
    "XGBoost",
    "Random Forest",
    "KNN",
    "Redes Neuronales",
    "Ciencia de Datos",
  ],
  icons: {
    icon: "logo.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
