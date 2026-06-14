const RUST_POINTS = [
    { label: 'Memory Safety', desc: 'No runtime GC pauses. Zero overhead abstractions. Exactly what a compiler deserves.' },
    { label: 'Blazing Performance', desc: 'FLOPs analysis on an 8B parameter model in < 50ms. Benchmarked with criterion.' },
    { label: 'Deterministic Builds', desc: 'Same topology.json always produces the same report. No probabilistic surprises.' },
    { label: 'Concurrency by Design', desc: 'Per-layer analysis parallelized with rayon. DAG traversal via petgraph.' },
];

const CODE_SNIPPET = `// Any architecture = Σ AtomOps with exact formulas
pub fn analyze_graph(topology: &Topology) -> NeuraxReport {
    let graph   = parser::parse(topology)?;
    let dialect = ArchDialect::detect(&graph.blocks, &graph.env);
    let atoms   = dialect_router::expand(&graph, dialect)?;
    let metrics = cost_engine::compute(&atoms, &graph.env);
    report::build(metrics, &graph)
}`;

export const RustSection = () => (
    <section className="relative overflow-hidden">
        {/* Dark gradient background — darker than hero */}
        <div className="absolute inset-0 bg-[#040409]" />
        <div
            className="absolute inset-0 opacity-[0.12]"
            style={{
                background: 'radial-gradient(ellipse 80% 60% at 60% 50%, hsl(14 82% 45%), transparent)',
            }}
        />
        {/* Rust accent top border */}
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[hsl(14,82%,52%)]/60 to-transparent" />
        {/* Rust accent bottom border */}
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-[hsl(14,82%,52%)]/30 to-transparent" />

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
                    <p className="text-lg text-white/50 leading-relaxed mb-10">
                        Neural engineering without compromise. The entire analytic compiler — parser, cost engine, dialect router — runs in pure Rust with zero runtime surprises.
                    </p>

                    <div className="space-y-5">
                        {RUST_POINTS.map((pt) => (
                            <div key={pt.label} className="flex gap-4">
                                <div className="mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 bg-[hsl(14,82%,55%)]" />
                                <div>
                                    <div className="text-sm font-semibold text-white/85">{pt.label}</div>
                                    <div className="text-sm text-white/45 mt-0.5 leading-relaxed">{pt.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Right — code window */}
                <div className="relative group">
                    <div className="absolute -inset-[1px] rounded-xl bg-gradient-to-br from-[hsl(14,82%,55%)]/25 via-transparent to-[hsl(225,85%,62%)]/15 blur-sm opacity-70 group-hover:opacity-100 transition-opacity duration-500" />
                    <div className="relative rounded-xl border border-white/[0.07] bg-[#0b0b17] overflow-hidden">
                        {/* Titlebar */}
                        <div className="flex items-center gap-2 px-4 h-9 border-b border-white/[0.05] bg-[#0d0d1e]">
                            <span className="w-2 h-2 rounded-full bg-red-500/70" />
                            <span className="w-2 h-2 rounded-full bg-amber-500/70" />
                            <span className="w-2 h-2 rounded-full bg-emerald-500/70" />
                            <span className="ml-3 text-[10px] text-white/25 font-mono">neurax-core/src/lib.rs</span>
                        </div>
                        {/* Code */}
                        <pre className="overflow-x-auto p-5 text-[11px] sm:text-xs leading-relaxed font-mono scrollbar-thin">
                            {CODE_SNIPPET.split('\n').map((line, i) => {
                                const isComment = line.trim().startsWith('//');
                                const isKeyword = /\b(pub|fn|let)\b/.test(line);
                                return (
                                    <div key={i} className="flex gap-3">
                                        <span className="select-none text-white/15 w-4 text-right flex-shrink-0">{i + 1}</span>
                                        <span
                                            className={
                                                isComment
                                                    ? 'text-white/30'
                                                    : isKeyword
                                                        ? 'text-[hsl(14,82%,60%)]'
                                                        : 'text-white/70'
                                            }
                                        >
                                            {line || '\u00a0'}
                                        </span>
                                    </div>
                                );
                            })}
                        </pre>
                        {/* Bottom glow */}
                        <div className="h-px bg-gradient-to-r from-transparent via-[hsl(14,82%,52%)]/40 to-transparent" />
                    </div>
                </div>

            </div>
        </div>
    </section>
);
