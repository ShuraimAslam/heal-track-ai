"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Activity, ShieldAlert, FileText, CheckCircle2, Loader2, Image as ImageIcon } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface AnalysisResults {
    case_id: string;
    metrics: {
        wound_area: number;
        area_ratio: number;
        shape_complexity: number;
        num_regions: number;
        segmentation_confidence: string;
    };
    risk: {
        risk_level: string;
        risk_score: number;
        confidence: string;
    };
    report: string;
    visualizations: {
        overlay: string;
        mask: string;
    };
}

export default function AnalysisDashboard() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [results, setResults] = useState<AnalysisResults | null>(null);
    const [error, setError] = useState<string | null>(null);

    const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = e.target.files?.[0];
        if (selected) {
            setFile(selected);
            setPreview(URL.createObjectURL(selected));
            setResults(null);
            setError(null);
        }
    };

    const runAnalysis = async () => {
        if (!file) return;

        setIsAnalyzing(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://localhost:8000/analyze", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Failed to analyze image");
            }

            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An unexpected error occurred");
        } finally {
            setIsAnalyzing(false);
        }
    };

    return (
        <div className="max-w-7xl mx-auto px-4 py-12">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column: Upload & Control */}
                <div className="space-y-6">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card"
                    >
                        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                            <Upload className="w-5 h-5 text-blue-400" />
                            Upload Image
                        </h2>

                        <div
                            className={cn(
                                "border-2 border-dashed border-zinc-700 rounded-xl p-8 text-center cursor-pointer transition-colors hover:border-blue-500/50",
                                preview && "border-blue-500/30 bg-blue-500/5"
                            )}
                            onClick={() => document.getElementById("file-upload")?.click()}
                        >
                            <input
                                id="file-upload"
                                type="file"
                                className="hidden"
                                onChange={onFileChange}
                                accept="image/*"
                            />
                            {preview ? (
                                <div className="relative group">
                                    <img src={preview} alt="Preview" className="w-full h-48 object-cover rounded-lg shadow-lg" />
                                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity rounded-lg">
                                        <p className="text-white text-sm font-medium">Click to change</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="py-8 space-y-3">
                                    <ImageIcon className="w-12 h-12 text-zinc-600 mx-auto" />
                                    <p className="text-zinc-400 text-sm">Drag and drop or click to upload wound image</p>
                                </div>
                            )}
                        </div>

                        <button
                            onClick={runAnalysis}
                            disabled={!file || isAnalyzing}
                            className="premium-button w-full mt-6 gap-2 flex items-center justify-center disabled:bg-zinc-800 disabled:shadow-none"
                        >
                            {isAnalyzing ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <Activity className="w-5 h-5" />
                                    Run AI Analysis
                                </>
                            )}
                        </button>
                    </motion.div>

                    {results && (
                        <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="glass-card bg-zinc-900/40 p-4 border-l-4 border-blue-500"
                        >
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-zinc-400">Case ID:</span>
                                <span className="font-mono text-blue-400">{results.case_id}</span>
                            </div>
                            <div className="flex justify-between items-center text-sm mt-2">
                                <span className="text-zinc-400">Status:</span>
                                <span className="text-emerald-400 flex items-center gap-1">
                                    <CheckCircle2 className="w-4 h-4" /> Analyzed
                                </span>
                            </div>
                        </motion.div>
                    )}

                    {error && (
                        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                            {error}
                        </div>
                    )}
                </div>

                {/* Right Column: Results Dashboard */}
                <div className="lg:col-span-2 space-y-8">
                    <AnimatePresence mode="wait">
                        {!results && !isAnalyzing ? (
                            <motion.div
                                key="empty"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="h-full min-h-[400px] glass-card flex flex-col items-center justify-center text-zinc-500 space-y-4"
                            >
                                <div className="w-16 h-16 rounded-full bg-zinc-800/50 flex items-center justify-center">
                                    <Activity className="w-8 h-8 opacity-20" />
                                </div>
                                <div className="text-center">
                                    <p className="font-medium">No Analysis Results</p>
                                    <p className="text-sm">Upload an image and click analyze to see the AI output.</p>
                                </div>
                            </motion.div>
                        ) : isAnalyzing ? (
                            <motion.div
                                key="loading"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="h-full min-h-[400px] glass-card flex flex-col items-center justify-center space-y-6"
                            >
                                <div className="relative">
                                    <div className="w-20 h-20 rounded-full border-4 border-blue-500/20 border-t-blue-500 animate-spin" />
                                    <Activity className="absolute inset-0 m-auto w-8 h-8 text-blue-500 animate-pulse" />
                                </div>
                                <div className="text-center space-y-2">
                                    <p className="text-xl font-bold text-gradient">AI Processing...</p>
                                    <p className="text-zinc-400 max-w-xs">Performing segmentation and calculating metrics from wound image.</p>
                                </div>
                            </motion.div>
                        ) : results ? (
                            <motion.div
                                key="results"
                                initial={{ opacity: 0, scale: 0.98 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="space-y-8"
                            >
                                {/* Images Layer */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="space-y-3">
                                        <p className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Mask Overlay</p>
                                        <div className="rounded-2xl overflow-hidden border border-zinc-800 shadow-2xl relative group">
                                            <img src={results.visualizations.overlay} alt="Overlay" className="w-full aspect-video object-cover" />
                                            <div className="absolute top-4 left-4 glass px-3 py-1 rounded-full text-xs font-medium">AI SEGMENTATION</div>
                                        </div>
                                    </div>
                                    <div className="space-y-3">
                                        <p className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Raw Mask</p>
                                        <div className="rounded-2xl overflow-hidden border border-zinc-800 shadow-2xl bg-black">
                                            <img src={results.visualizations.mask} alt="Mask" className="w-full aspect-video object-cover opacity-80" />
                                        </div>
                                    </div>
                                </div>

                                {/* Metrics Grid */}
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                    <div className="glass-card text-center py-6">
                                        <p className="text-zinc-400 text-xs mb-1 uppercase tracking-tighter font-medium">Wound Area</p>
                                        <p className="text-2xl font-bold text-white">{results.metrics.wound_area} <span className="text-xs font-normal text-zinc-500">px</span></p>
                                    </div>
                                    <div className="glass-card text-center py-6">
                                        <p className="text-zinc-400 text-xs mb-1 uppercase tracking-tighter font-medium">Area Ratio</p>
                                        <p className="text-2xl font-bold text-blue-400">{(results.metrics.area_ratio * 100).toFixed(2)}%</p>
                                    </div>
                                    <div className="glass-card text-center py-6">
                                        <p className="text-zinc-400 text-xs mb-1 uppercase tracking-tighter font-medium">Complexity</p>
                                        <p className="text-2xl font-bold text-cyan-400">{results.metrics.shape_complexity}</p>
                                    </div>
                                    <div className="glass-card text-center py-6">
                                        <p className="text-zinc-400 text-xs mb-1 uppercase tracking-tighter font-medium">Regions</p>
                                        <p className="text-2xl font-bold text-purple-400">{results.metrics.num_regions}</p>
                                    </div>
                                </div>

                                {/* Risk & Confidence Layer */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className={cn(
                                        "glass-card flex items-center justify-between",
                                        results.risk.risk_level.toLowerCase() === 'high' ? "border-red-500/40 bg-red-500/5" :
                                            results.risk.risk_level.toLowerCase() === 'moderate' ? "border-amber-500/40 bg-amber-500/5" : "border-emerald-500/40 bg-emerald-500/5"
                                    )}>
                                        <div>
                                            <p className="text-zinc-400 text-xs mb-1 uppercase font-medium">Clinical Risk</p>
                                            <p className={cn(
                                                "text-3xl font-black uppercase",
                                                results.risk.risk_level?.toLowerCase() === 'high' ? "text-red-400" :
                                                    results.risk.risk_level?.toLowerCase() === 'moderate' ? "text-amber-400" : "text-emerald-400"
                                            )}>{results.risk.risk_level || 'N/A'}</p>
                                        </div>
                                        <div className="text-right">
                                            <p className="text-zinc-500 text-[10px] uppercase font-bold">Severity Score</p>
                                            <p className="text-xl font-mono font-bold text-white">{results.risk.risk_score}<span className="text-sm text-zinc-500">/100</span></p>
                                        </div>
                                    </div>

                                    <div className="glass-card flex items-center justify-between">
                                        <div>
                                            <p className="text-zinc-400 text-xs mb-1 uppercase font-medium">AI Confidence</p>
                                            <p className={cn(
                                                "text-3xl font-black uppercase",
                                                results.risk.confidence?.toLowerCase() === 'high' ? "text-emerald-400" :
                                                    results.risk.confidence?.toLowerCase() === 'medium' ? "text-amber-400" : "text-red-400"
                                            )}>{results.risk.confidence || 'N/A'}</p>
                                        </div>
                                        <div className="p-3 bg-white/5 rounded-full">
                                            <ShieldAlert className={cn(
                                                "w-8 h-8",
                                                results.risk.confidence?.toLowerCase() === 'high' ? "text-emerald-500" :
                                                    results.risk.confidence?.toLowerCase() === 'medium' ? "text-amber-500" : "text-red-500"
                                            )} />
                                        </div>
                                    </div>
                                </div>

                                {/* Clinical Summary */}
                                <div className="glass-card">
                                    <div className="flex items-center gap-3 mb-6">
                                        <div className="p-2 bg-blue-500/10 rounded-lg">
                                            <FileText className="w-6 h-6 text-blue-400" />
                                        </div>
                                        <h3 className="text-xl font-bold">Clinical Case Summary</h3>
                                    </div>
                                    <div className="bg-zinc-950/50 rounded-xl p-6 border border-zinc-800/50 overflow-hidden relative">
                                        <div className="absolute top-0 right-0 p-4 opacity-10">
                                            <Activity className="w-32 h-32" />
                                        </div>
                                        <pre className="text-zinc-300 font-sans whitespace-pre-wrap leading-relaxed relative z-10">
                                            {results.report}
                                        </pre>
                                    </div>
                                </div>
                            </motion.div>
                        ) : null}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}
