const STEPS = [
    {
        num: '01',
        title: 'Define your topology',
        desc: 'Write a topology.json with blocks, shapes, and hardware config. Or import from HuggingFace config.json. Or use the visual canvas.',
        code: `{
  "model_type": "transformer",
  "layers": [
    { "type": "Embedding", "params": { "vocab_size": 32000, "hidden_size": 4096 } },
    { "type": "Attention", "params": { "num_heads": 32, "head_dim": 128 } }
  ],
  "training_config": { "batch_size": 32, "sequence_length": 2048 }
}`,
        color: 'hsl(199 90% 48%)',
    },
    {
        num: '02',
        title: 'Run the 10-pass analysis',
        desc: 'NEURAX compiles your architecture through 10 IR passes: Architecture → Graph → Tensor → Operator → Compute → Memory → Parallelism → Hardware → Cost → Report.',
        code: `$ neurax analyze llama3_8b.json --hardware H100

▸ Architecture pass ........................ 2ms
▸ Graph pass ............................... 1ms
▸ Tensor pass .............................. 3ms
▸ Operator pass ............................ 4ms
▸ Compute pass ............................. 2ms
▸ Memory pass .............................. 5ms
▸ Parallelism + Hardware (parallel) ....... 8ms
▸ Cost pass ................................ 1ms
▸ Report pass .............................. 1ms
✓ Analysis complete (27ms)`,
        color: 'hsl(225 85% 62%)',
    },
    {
        num: '03',
        title: 'Get the full report',
        desc: '55+ metrics: FLOPs, VRAM, latency, per-layer breakdown, confidence score, OOM risk, cost projections, and P1/P2/P3 optimization recommendations.',
        code: `╔══════════════════════════════════════════╗
║  NEURAX ANALYSIS REPORT — LLaMA 3 8B   ║
╠══════════════════════════════════════════╣
║  Confidence    : 97.0% [RELIABLE]        ║
║  Total FLOPs   : 139.6 TFLOPs            ║
║  VRAM Peak     : 16.5 GB                 ║
║  Latency       : 68.4 ms [memory-bound]  ║
║  Training Cost : $42,800 (8×H100, 3ep)   ║
║  OOM Risk      : SAFE (68% utilization)  ║
║  CO₂ Emissions : 124 kg                  ║
╚══════════════════════════════════════════╝`,
        color: 'hsl(14 82% 55%)',
    },
];

export const DemoSection = () => (
    <section id="pipeline" className="relative">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-20">
            <div className="text-center mb-16">
                <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full border border-white/[0.06] bg-white/[0.02]">
                    <span className="text-[10px] font-mono uppercase tracking-[0.3em] text-white/30">10-Pass Pipeline</span>
                </div>
                <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
                    From JSON to full report
                </h2>
                <p className="mt-4 max-w-xl mx-auto text-sm text-white/35 leading-relaxed">
                    Three steps. No GPU required. Deterministic results every time.
                </p>
            </div>

            <div className="relative">
                {/* Connector line */}
                <div className="absolute left-8 top-10 bottom-10 w-px bg-gradient-to-b from-[hsl(199,90%,48%)]/30 via-[hsl(225,85%,62%)]/30 to-[hsl(14,82%,55%)]/30 hidden lg:block" />

                <div className="space-y-8">
                    {STEPS.map((s) => (
                        <div key={s.num} className="group flex flex-col lg:flex-row gap-6 items-start">
                            {/* Step number */}
                            <div
                                className="flex-shrink-0 w-16 h-16 rounded-2xl flex items-center justify-center text-xl font-mono font-bold border transition-all duration-300 group-hover:scale-105"
                                style={{
                                    borderColor: `${s.color}40`,
                                    backgroundColor: `${s.color}10`,
                                    color: s.color,
                                    boxShadow: `0 0 20px ${s.color}15`,
                                }}
                            >
                                {s.num}
                            </div>

                            {/* Content */}
                            <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <h3 className="text-lg font-semibold text-white/90 mb-2">{s.title}</h3>
                                    <p className="text-sm text-white/40 leading-relaxed">{s.desc}</p>
                                </div>

                                <div
                                    className="rounded-xl border overflow-hidden"
                                    style={{ borderColor: `${s.color}15`, backgroundColor: '#060611' }}
                                >
                                    <div
                                        className="h-px"
                                        style={{ background: `linear-gradient(90deg, ${s.color}40, transparent)` }}
                                    />
                                    <pre className="p-4 text-[10px] sm:text-[11px] font-mono leading-relaxed text-white/50 overflow-x-auto scrollbar-thin">
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