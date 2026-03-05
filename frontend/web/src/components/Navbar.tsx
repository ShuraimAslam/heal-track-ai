"use client";

import React from "react";
import { Activity } from "lucide-react";

export default function Navbar() {
    return (
        <nav className="sticky top-0 z-50 glass border-b border-white/10">
            <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                        <Activity className="w-5 h-5 text-white" />
                    </div>
                    <span className="text-xl font-black bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">
                        HEAL-TRACK
                    </span>
                </div>

                <div className="hidden md:flex items-center gap-8 text-sm font-medium text-zinc-400">
                    <a href="#" className="hover:text-white transition-colors">Documentation</a>
                    <a href="#" className="hover:text-white transition-colors">Research</a>
                    <a href="https://github.com/ShuraimAslam/heal-track-ai" className="hover:text-white transition-colors">GitHub</a>
                </div>

                <div className="flex items-center gap-4">
                    <span className="hidden sm:inline px-3 py-1 bg-zinc-800 rounded-full text-[10px] uppercase font-bold tracking-widest text-zinc-500 border border-zinc-700">
                        v2.0 Beta
                    </span>
                    <button className="text-xs font-bold text-white bg-blue-600/20 border border-blue-500/50 px-4 py-2 rounded-full hover:bg-blue-600/30 transition-all">
                        Secure Portal
                    </button>
                </div>
            </div>
        </nav>
    );
}
