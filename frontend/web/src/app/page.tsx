"use client";

import Navbar from "@/components/Navbar";
import AnalysisDashboard from "@/components/AnalysisDashboard";
import { motion } from "framer-motion";
import { Shield, Brain, Zap, ArrowRight } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen selection:bg-blue-500/30">
      <Navbar />

      {/* Hero Section */}
      <section className="relative pt-24 pb-16 overflow-hidden">
        {/* Background Decorations */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[600px] bg-blue-600/5 blur-[120px] rounded-full -z-10" />
        <div className="absolute top-24 left-1/4 w-64 h-64 bg-cyan-500/5 blur-[100px] rounded-full -z-10 animate-pulse" />

        <div className="max-w-7xl mx-auto px-4 text-center space-y-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/50 border border-zinc-800 text-xs font-bold text-blue-400 tracking-widest uppercase mb-4"
          >
            <Zap className="w-3 h-3 fill-current" />
            Next-Gen Wound Diagnostics
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-5xl md:text-7xl font-black tracking-tight"
          >
            Wound Analysis <br />
            <span className="text-gradient">Redefined by AI</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="max-w-2xl mx-auto text-lg text-zinc-400 leading-relaxed"
          >
            Empowering clinicians with automated segmentation, precise metrics, and
            real-time clinical risk assessment using state-of-the-art U-Net models.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-wrap items-center justify-center gap-4 py-4"
          >
            <div className="flex items-center gap-2 px-4 py-2 bg-zinc-900/30 rounded-lg border border-zinc-800/50 text-xs text-zinc-500">
              <Shield className="w-4 h-4 text-emerald-500" />
              Privacy Preserving
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-zinc-900/30 rounded-lg border border-zinc-800/50 text-xs text-zinc-500">
              <Brain className="w-4 h-4 text-blue-500" />
              Advanced U-Net Engine
            </div>
          </motion.div>
        </div>
      </section>

      {/* Main Analysis Tool */}
      <section id="analyze" className="pb-32">
        <AnalysisDashboard />
      </section>

      {/* Footer Disclaimer */}
      <footer className="py-12 border-t border-zinc-900 bg-zinc-950/50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="p-8 rounded-2xl glass-card flex flex-col md:flex-row items-center justify-between gap-8 border-none bg-zinc-900/20">
            <div className="space-y-2">
              <h4 className="font-bold flex items-center gap-2 text-red-500">
                <Shield className="w-5 h-5" />
                Safety Disclaimer
              </h4>
              <p className="text-sm text-zinc-500 max-w-xl">
                Heal-Track is a research prototype developed for demonstration purposes.
                AI-generated results must be verified by clinical professionals before
                making any medical decisions.
              </p>
            </div>
            <div className="text-right">
              <p className="text-xs font-bold text-zinc-600 uppercase tracking-widest">
                Heal-Track AI © 2026
              </p>
              <p className="text-xs text-zinc-700 mt-1">
                Advanced Wound Monitoring System
              </p>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
