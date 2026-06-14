const STEPS = [
    {
        num: '01',
        title: 'Define your topology',
        desc: 'Write a topology.json v3 with blocks, shapes, and env. Or import from HuggingFace config.json.',
        code: `{ "type": "ScaledDotProductAttn",
  "params": { "in_features": 4096,
    "num_heads": 32, "head_dim": 128 } }`,
        color: 'hsl(199 90% 48%)',
    },
    {
        num: '02',
        title: 'Run the analysis',
        desc: 'NEURAX compiles your graph through 6 layers: parse → canonicalize → dialect → atoms → cost → report.',
        code: `neurax-cli analyze llama3_8b.json \\
  --hardware H100 --pretty`,
        color: 'hsl(225 85% 62%)',
    },
    {
        num: '03',
        title: 'Get the full report',
        desc: 'FLOPs, VRAM, latency, per-layer breakdown, confidence score, and P1/P2/P3 optimization recommendations.',
        code: `Confidence : 97.0% [reliable]
Total FLOPs: 139.6 TFLOPs
VRAM peak  : 16.5 GB
Latency    : 68.4 ms [memory-bound]`,
        color: 'hsl(14 82% 55%)',
    },
];

export const DemoSection = () => (
    <section id="workflow" className="relative">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-20">
            <div className="text-center mb-14">
                <div className="inline-flex items-center gap-2 mb-4 px-3 py-1 rounded-full border border-white/10 bg-white/[0.03] text-[11px] font-mono uppercase tracking-widest text-white/40">
                    Real workflow · Real performance
                </div>
                <h2 className="text-3xl sm:text-4xl font-bold text-white">See it working.</h2>
                <p className="mt-4 max-w-xl mx-auto text-sm text-white/40 leading-relaxed">
                    From topology.json to full cost report in milliseconds. No GPU required.
                </p>
            </div>

            <div className="relative">
                {/* Connector line */}
                <div className="absolute left-8 top-10 bottom-10 w-px bg-gradient-to-b from-[hsl(199,90%,48%)]/50 via-[hsl(225,85%,62%)]/50 to-[hsl(14,82%,55%)]/50 hidden lg:block" />

                <div className="space-y-6">
                    {STEPS.map((s) => (
                        <div key={s.num} className="group flex flex-col lg:flex-row gap-6 items-start">
                            {/* Step number */}
                            <div
                                className="flex-shrink-0 w-16 h-16 rounded-2xl flex items-center justify-center text-xl font-mono font-bold border transition-all duration-300 group-hover:scale-105"
                                style={{
                                    borderColor: `${s.color}40`,
                                    backgroundColor: `${s.color}10`,
                                    color: s.color,
                                    boxShadow: `0 0 20px ${s.color}20`,
                                }}
                            >
                                {s.num}
                            </div>

                            {/* Content */}
                            <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <h3 className="text-base font-semibold text-white/90 mb-2">{s.title}</h3>
                                    <p className="text-sm text-white/45 leading-relaxed">{s.desc}</p>
                                </div>

                                <div
                                    className="rounded-xl border overflow-hidden"
                                    style={{ borderColor: `${s.color}20`, backgroundColor: '#060611' }}
                                >
                                    <div
                                        className="h-px"
                                        style={{ background: `linear-gradient(90deg, ${s.color}50, transparent)` }}
                                    />
                                    <pre className="p-4 text-[11px] font-mono leading-relaxed text-white/60 overflow-x-auto">
                                        {s.code}
                                    </pre>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    </section>
);
