const RUST_POINTS = [
    { label: 'Memory Safety', desc: 'No runtime GC pauses. Zero overhead abstractions. Exactly what a compiler deserves.' },
    { label: 'Blazing Performance', desc: 'FLOPs analysis on an 8B parameter model in < 50ms. Benchmarked with criterion.' },
    { label: 'Deterministic Builds', desc: 'Same topology.json always produces the same report. No probabilistic surprises.' },
    { label: 'Concurrency by Design', desc: 'Per-layer analysis parallelized with rayon. DAG traversal via petgraph.' },
];

const CODE_SNIPPET = `// 10-pass analytical pipeline — pure Rust, zero variance
pub fn run_analysis(config: &ModelConfig) -> AnalysisResult {
    let ctx = NeuraxContext::new(config, gpu_db);
    
    // Sequential passes (data dependencies)
    let arch  = ArchitecturePass::build(&ctx)?;
    let graph = GraphPass::build(&ctx, &arch)?;
    let tensor = TensorPass::build(&ctx, &graph)?;
    let ops   = OperatorPass::build(&ctx, &tensor)?;
    let compute = ComputePass::build(&ctx, &ops)?;
    let memory  = MemoryPass::build(&ctx, &compute)?;
    
    // Parallel passes (independent)
    let (par, hw) = rayon::join(
        || ParallelismPass::build(&ctx, &memory),
        || HardwarePass::build(&ctx, &memory),
    );
    
    let cost   = CostPass::build(&ctx, &hw)?;
    let report = ReportPass::build_report(&ctx, &all_metrics);
    Ok(report)
}`;

export const RustSection = () => (
    <section className="relative overflow-hidden">
        {/* Dark gradient background */}
        <div className="absolute inset-0 bg-[#040409]" />
        <div
            className="absolute inset-0 opacity-[0.1]"
            style={{
                background: 'radial-gradient(ellipse 80% 60% at 60% 50%, hsl(14 82% 45%), transparent)',
            }}
        />
        {/* Rust accent borders */}
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[hsl(14,82%,52%)]/40 to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-[hsl(14,82%,52%)]/20 to-transparent" />

        <div className="relative z-10 mx-auto max-w-6xl px-4 sm:px-6 py-20 md:py-28">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">

                {/* Left — copy */}
                <div>
                    <div className="inline-flex items-center gap-2 mb-6 px-3 py-1.5 rounded-full border border-[hsl(14,82%,52%)]/30 bg-[hsl(14,82%,52%)]/10">
                        <div className="w-1.5 h-1.5 rounded-full bg-[hsl(14,82%,55%)]" />
                        <span className="text-[11px] uppercase tracking-widest font-mono text-[hsl(14,82%,60%)]">
                            Powered by Rust
                        </span>
                    </div>

                    <h2 className="text-4xl sm:text-5xl font-bold text-white leading-tight mb-4">
                        Built on <span className="text-[hsl(14,82%,60%)]">Rust</span>.
                    </h2>
                    <p className="text-lg text-white/40 leading-relaxed mb-10">
                        Neural engineering without compromise. The entire analytical compiler — parser, cost engine, dialect router — runs in pure Rust with zero runtime surprises.
                    </p>

                    <div className="space-y-5">
                        {RUST_POINTS.map((pt) => (
                            <div key={pt.label} className="flex gap-4">
                                <div className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 bg-[hsl(14,82%,55%)]" />
                                <div>
                                    <div className="text-sm font-semibold text-white/80">{pt.label}</div>
                                    <div className="text-sm text-white/35 mt-0.5 leading-relaxed">{pt.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Right — code window */}
                <div className="relative group">
                    <div className="absolute -inset-[1px] rounded-xl bg-gradient-to-br from-[hsl(14,82%,55%)]/20 via-transparent to-[hsl(225,85%,62%)]/10 blur-sm opacity-60 group-hover:opacity-90 transition-opacity duration-500" />
                    <div className="relative rounded-xl border border-white/[0.06] bg-[#0b0b17] overflow-hidden">
                        {/* Titlebar */}
                        <div className="flex items-center gap-2 px-4 h-9 border-b border-white/[0.04] bg-[#0d0d1e]">
                            <span className="w-2 h-2 rounded-full bg-red-500/60" />
                            <span className="w-2 h-2 rounded-full bg-amber-500/60" />
                            <span className="w-2 h-2 rounded-full bg-emerald-500/60" />
                            <span className="ml-3 text-[10px] text-white/20 font-mono">neurax-core/src/engine.rs</span>
                        </div>
                        {/* Code */}
                        <pre className="overflow-x-auto p-5 text-[10px] sm:text-[11px] leading-relaxed font-mono scrollbar-thin">
                            {CODE_SNIPPET.split('\n').map((line, i) => {
                                const isComment = line.trim().startsWith('//');
                                const isKeyword = /\b(pub|fn|let|mut|return|use|impl|for|if|else|match|Ok|Err)\b/.test(line);
                                return (
                                    <div key={i} className="flex gap-3">
                                        <span className="select-none text-white/10 w-5 text-right flex-shrink-0">{i + 1}</span>
                                        <span
                                            className={
                                                isComment
                                                    ? 'text-white/25'
                                                    : isKeyword
                                                        ? 'text-[hsl(14,82%,60%)]'
                                                        : 'text-white/60'
                                            }
                                        >
                                            {line || '\u00a0'}
                                        </span>
                                    </div>
                                );
                            })}
                        </pre>
                        {/* Bottom glow */}
                        <div className="h-px bg-gradient-to-r from-transparent via-[hsl(14,82%,52%)]/30 to-transparent" />
                    </div>
                </div>

            </div>
        </div>
    </section>
);