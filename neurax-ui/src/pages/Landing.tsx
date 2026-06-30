import { Link } from 'react-router-dom';
import { ArrowRight, Activity, ChevronRight, Shield, TrendingUp, Clock, Cpu, Zap, BarChart3, Layers, GitBranch, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { AuthControl } from '@/components/auth/AuthControl.tsx';
import { ThemeToggle } from '@/components/layout/ThemeToggle.tsx';
import { HeroBackground } from '@/components/landing/HeroBackground.tsx';
import { CanvasShowcase } from '@/components/landing/CanvasShowcase.tsx';
import { FeatureCards } from '@/components/landing/FeatureCards.tsx';
import { RustSection } from '@/components/landing/RustSection.tsx';
import { ArchitectureDialects } from '@/components/landing/ArchitectureDialects.tsx';
import { DemoSection } from '@/components/landing/DemoSection.tsx';
import { PageBackground } from '@/components/landing/PageBackground.tsx';
import { NeuraxAtomLogo } from '@/components/landing/NeuraxAtomLogo.tsx';

/* ─── Navbar ────────────────────────────────────────────────── */
const Navbar = () => (
  <header className="sticky top-0 z-50 border-b border-white/[0.06] bg-[#05050d]/80 backdrop-blur-2xl">
    <div className="mx-auto max-w-7xl px-4 sm:px-6 h-16 flex items-center justify-between">
      {/* Logo */}
      <Link to="/" className="flex items-center gap-3 group">
        <NeuraxAtomLogo size={32} />
        <div className="flex flex-col">
          <span className="text-sm font-bold tracking-tight text-white leading-none group-hover:text-blue-400 transition-colors">NEURAX</span>
          <span className="text-[8px] font-mono text-white/20 mt-0.5 tracking-[0.25em] uppercase">Analytic Compiler</span>
        </div>
      </Link>

      {/* Nav links */}
      <nav className="hidden md:flex items-center gap-8 text-[11px] font-mono uppercase tracking-[0.2em] text-white/30">
        <a href="#problem" className="hover:text-white transition-colors duration-300 relative group">
          Problem
          <span className="absolute -bottom-1 left-0 w-0 h-px bg-blue-500/50 group-hover:w-full transition-all" />
        </a>
        <a href="#vision" className="hover:text-white transition-colors duration-300 relative group">
          Vision
          <span className="absolute -bottom-1 left-0 w-0 h-px bg-violet-500/50 group-hover:w-full transition-all" />
        </a>
        <a href="#features" className="hover:text-white transition-colors duration-300 relative group">
          Features
          <span className="absolute -bottom-1 left-0 w-0 h-px bg-cyan-500/50 group-hover:w-full transition-all" />
        </a>
        <a href="#architectures" className="hover:text-white transition-colors duration-300 relative group">
          Architectures
          <span className="absolute -bottom-1 left-0 w-0 h-px bg-emerald-500/50 group-hover:w-full transition-all" />
        </a>
        <a href="#pipeline" className="hover:text-white transition-colors duration-300 relative group">
          Pipeline
          <span className="absolute -bottom-1 left-0 w-0 h-px bg-orange-500/50 group-hover:w-full transition-all" />
        </a>
      </nav>

      {/* Actions */}
      <div className="flex items-center gap-3">
        <div className="hidden lg:flex items-center gap-1.5 text-[10px] text-white/15 font-mono mr-2">
          <Activity size={10} className="text-emerald-500/50 animate-pulse" />
          <span>COMPILER: ONLINE</span>
        </div>
        <ThemeToggle />
        <AuthControl triggerLabel="Sign In" triggerSize="sm" triggerVariant="ghost" />
        <Button
          asChild
          size="sm"
          className="bg-white text-[#05050d] hover:bg-white/90 font-bold shadow-[0_0_25px_rgba(255,255,255,0.1)] px-5"
        >
          <Link to="/app">
            Launch Studio
            <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
          </Link>
        </Button>
      </div>
    </div>
  </header>
);

/* ─── Hero ──────────────────────────────────────────────────── */
const Hero = () => (
  <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
    <HeroBackground />

    <div className="relative z-10 mx-auto max-w-6xl px-4 sm:px-6 pt-32 pb-16 text-center">
      {/* Eyebrow */}
      <div className="inline-flex items-center gap-3 mb-8 px-5 py-2.5 rounded-full border border-white/[0.06] bg-white/[0.02] backdrop-blur-sm">
        <div className="relative flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          <div className="absolute w-2 h-2 rounded-full bg-blue-500 animate-ping opacity-50" />
        </div>
        <span className="text-[11px] font-mono uppercase tracking-[0.3em] text-white/40">
          Analytic Compiler for Neural Architectures
        </span>
      </div>

      {/* Main headline */}
      <h1 className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold text-white leading-[1.05] tracking-tight mb-8">
        Know your model's cost
        <br />
        <span className="bg-clip-text text-transparent" style={{ backgroundImage: 'linear-gradient(135deg, hsl(225 85% 75%), hsl(280 72% 70%), hsl(199 90% 68%))' }}>
          before you train.
        </span>
      </h1>

      {/* Sub-headline */}
      <p className="max-w-2xl mx-auto text-lg sm:text-xl text-white/40 leading-relaxed mb-14">
        NEURAX is the analytical compiler that predicts cost, memory, speed, and feasibility
        of any neural architecture — <span className="text-white/60">before a single GPU hour is spent.</span>
      </p>

      {/* CTAs */}
      <div className="flex flex-wrap items-center justify-center gap-4 mb-20">
        <Button
          asChild
          size="lg"
          className="bg-white text-[#05050d] hover:bg-white/90 font-bold px-10 py-7 text-base shadow-[0_0_40px_rgba(255,255,255,0.12)] transition-all transform hover:scale-105"
        >
          <Link to="/app">
            Start Analyzing
            <ArrowRight className="w-4 h-4 ml-2" />
          </Link>
        </Button>
        <Button
          asChild
          variant="outline"
          size="lg"
          className="border-white/10 bg-white/[0.02] text-white/60 hover:bg-white/[0.05] hover:text-white hover:border-white/20 px-10 py-7 text-base backdrop-blur-md"
        >
          <a href="#pipeline">See How It Works</a>
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-3xl mx-auto">
        {[
          { value: '±3–8%', label: 'PREDICTION ACCURACY', sub: 'vs real hardware' },
          { value: '55+', label: 'METRICS', sub: 'per analysis' },
          { value: '80+', label: 'ARCHITECTURES', sub: 'supported' },
          { value: '<50ms', label: 'ANALYSIS TIME', sub: 'for 8B params' },
        ].map((s) => (
          <div key={s.label} className="group border border-white/[0.04] bg-white/[0.01] p-5 rounded-2xl hover:border-white/[0.1] hover:bg-white/[0.03] transition-all duration-300">
            <div className="text-2xl sm:text-3xl font-bold text-white group-hover:scale-105 transition-transform origin-left">{s.value}</div>
            <div className="text-[9px] font-mono text-white/25 tracking-[0.2em] mt-1">{s.label}</div>
            <div className="text-[10px] text-white/15 mt-0.5">{s.sub}</div>
          </div>
        ))}
      </div>
    </div>

    {/* Product showcase */}
    <div className="relative z-10 w-full animate-fade-in" style={{ animationDelay: '0.8s' }}>
      <CanvasShowcase />
    </div>

    {/* Scroll fade */}
    <div className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-[#05050d] to-transparent pointer-events-none" />
  </section>
);

/* ─── Problem Section ────────────────────────────────────────── */
const PROBLEM_ITEMS = [
  {
    icon: Clock,
    title: 'Months of GPU time wasted',
    desc: 'Teams spend weeks training architectures that OOM at step 10,000 or converge poorly. No way to predict before committing resources.',
    stat: '$2M+',
    statLabel: 'avg. cost of a failed training run',
    color: 'hsl(0 72% 55%)',
  },
  {
    icon: BarChart3,
    title: 'Blind architecture decisions',
    desc: 'Choosing between GQA, MQA, Flash Attention, or MoE? Without cost estimates, you\'re guessing. Memory, latency, and FLOPs remain unknowns.',
    stat: '73%',
    statLabel: 'of models need redesign after first training',
    color: 'hsl(38 92% 55%)',
  },
  {
    icon: Shield,
    title: 'Production surprises',
    desc: 'Inference latency spikes. VRAM overflow under load. Cost per token exceeds budget. These are discovered in production, not in design.',
    stat: '5×',
    statLabel: 'cost overrun when issues surface late',
    color: 'hsl(280 72% 60%)',
  },
];

const ProblemSection = () => (
  <section id="problem" className="py-32 relative">
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      {/* Section header */}
      <div className="text-center mb-20">
        <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full border border-red-500/20 bg-red-500/[0.05]">
          <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
          <span className="text-[10px] font-mono uppercase tracking-[0.3em] text-red-400/80">The Problem</span>
        </div>
        <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
          Building AI models is
          <br />
          <span className="text-red-400/80">expensive guesswork.</span>
        </h2>
        <p className="text-lg text-white/35 max-w-2xl mx-auto leading-relaxed">
          Every year, organizations waste millions on neural network architectures that fail.
          Not because the ideas are bad — but because there's no way to predict how they'll perform
          before committing GPU hours.
        </p>
      </div>

      {/* Problem cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {PROBLEM_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.title}
              className="group relative rounded-2xl p-7 border border-white/[0.04] bg-white/[0.01] hover:border-white/[0.08] hover:bg-white/[0.02] transition-all duration-300"
            >
              {/* Top accent */}
              <div
                className="absolute inset-x-0 top-0 h-px rounded-t-2xl opacity-40 group-hover:opacity-80 transition-opacity"
                style={{ background: `linear-gradient(90deg, transparent, ${item.color}, transparent)` }}
              />

              <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-5" style={{ backgroundColor: `${item.color}15`, border: `1px solid ${item.color}30` }}>
                <Icon className="w-5 h-5" style={{ color: item.color }} />
              </div>

              <h3 className="text-base font-semibold text-white/90 mb-3">{item.title}</h3>
              <p className="text-sm text-white/40 leading-relaxed mb-6">{item.desc}</p>

              <div className="pt-4 border-t border-white/[0.04]">
                <div className="text-2xl font-bold" style={{ color: item.color }}>{item.stat}</div>
                <div className="text-[10px] font-mono text-white/25 tracking-wider mt-1">{item.statLabel}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  </section>
);

/* ─── Vision Section ─────────────────────────────────────────── */
const VISION_POINTS = [
  {
    icon: Eye,
    title: 'Predict, don\'t guess',
    desc: 'NEURAX compiles your architecture through 10 deterministic IR passes — each one refining the analysis from abstract structure to concrete cost numbers.',
    color: 'hsl(199 90% 55%)',
  },
  {
    icon: Layers,
    title: 'Universal coverage',
    desc: 'From Transformers to Mamba, from CNNs to Diffusion models — 80+ architecture families analyzed with specialized formulas for every operator.',
    color: 'hsl(280 72% 65%)',
  },
  {
    icon: TrendingUp,
    title: 'The Lovable of models',
    desc: 'Just as Lovable lets you build apps without writing code, NEURAX lets you design and validate neural architectures without burning GPU hours. Design → Analyze → Deploy.',
    color: 'hsl(160 65% 52%)',
  },
  {
    icon: GitBranch,
    title: 'Deterministic by design',
    desc: 'Same topology, same result. Every time. No stochastic simulations, no Monte Carlo — pure analytical formulas with ±3–8% accuracy against real hardware.',
    color: 'hsl(38 92% 55%)',
  },
];

const VisionSection = () => (
  <section id="vision" className="py-32 relative overflow-hidden">
    {/* Background accent */}
    <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-500/[0.02] to-transparent pointer-events-none" />

    <div className="mx-auto max-w-6xl px-4 sm:px-6 relative z-10">
      {/* Section header */}
      <div className="text-center mb-20">
        <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full border border-blue-500/20 bg-blue-500/[0.05]">
          <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
          <span className="text-[10px] font-mono uppercase tracking-[0.3em] text-blue-400/80">The Vision</span>
        </div>
        <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
          An analytical compiler
          <br />
          <span className="bg-clip-text text-transparent" style={{ backgroundImage: 'linear-gradient(135deg, hsl(199 90% 65%), hsl(160 65% 55%))' }}>
            for neural architectures.
          </span>
        </h2>
        <p className="text-lg text-white/35 max-w-2xl mx-auto leading-relaxed">
          NEURAX treats neural network design the way compilers treat source code: 
          analyze, optimize, and predict behavior before execution. 
          No training required. No GPU needed. Pure math.
        </p>
      </div>

      {/* Vision cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {VISION_POINTS.map((point) => {
          const Icon = point.icon;
          return (
            <div
              key={point.title}
              className="group relative rounded-2xl p-8 border border-white/[0.04] bg-white/[0.01] hover:border-white/[0.08] hover:bg-white/[0.02] transition-all duration-300"
            >
              <div className="flex items-start gap-5">
                <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0" style={{ backgroundColor: `${point.color}12`, border: `1px solid ${point.color}25` }}>
                  <Icon className="w-6 h-6" style={{ color: point.color }} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white/90 mb-2">{point.title}</h3>
                  <p className="text-sm text-white/40 leading-relaxed">{point.desc}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparison callout */}
      <div className="mt-12 rounded-2xl border border-white/[0.06] bg-white/[0.02] p-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Cpu className="w-5 h-5 text-blue-400/60" />
          <span className="text-[11px] font-mono uppercase tracking-[0.3em] text-white/30">The Analogy</span>
        </div>
        <p className="text-xl sm:text-2xl text-white/70 font-medium leading-relaxed">
          <span className="text-white font-bold">GCC</span> compiles C before running it.
          <br />
          <span className="text-white font-bold">NEURAX</span> compiles architectures before training them.
        </p>
        <p className="text-sm text-white/30 mt-4 max-w-xl mx-auto">
          Same principle, different domain. Static analysis, deterministic formulas, and IR passes 
          replace stochastic training runs. You get cost, memory, latency, and feasibility — in milliseconds.
        </p>
      </div>
    </div>
  </section>
);

/* ─── Metrics Section ────────────────────────────────────────── */
const METRICS = [
  { category: 'Structure', items: ['Total Parameters', 'Layers by Type', 'Param Distribution', 'Model Size', 'Architecture Family'], color: 'hsl(199 90% 55%)' },
  { category: 'Compute', items: ['Total FLOPs', 'FLOPs/Token', 'FLOPs/Batch', 'Arithmetic Intensity', 'Complexity Class', 'Forward/Backward Split'], color: 'hsl(225 85% 65%)' },
  { category: 'Memory', items: ['Peak VRAM', 'Activation Memory', 'Gradient Memory', 'Optimizer State', 'OOM Risk Level', 'Max Batch Size'], color: 'hsl(280 72% 65%)' },
  { category: 'Hardware', items: ['Latency (ms)', 'Throughput (tok/s)', 'GPU Utilization', 'Roofline Position', 'Bottleneck Type', 'Tensor Core Usage'], color: 'hsl(14 82% 55%)' },
  { category: 'Cost', items: ['Training Cost ($)', 'GPU Hours', 'Energy (kWh)', 'CO₂ (kg)', 'Cost/Token', 'Inference Cost/mo'], color: 'hsl(160 65% 52%)' },
  { category: 'Inference', items: ['Stability Index', 'Hallucination Risk', 'Entropy Evolution', 'Context Degradation', 'Sampling Volatility', 'Risk Overview'], color: 'hsl(345 82% 60%)' },
];

const MetricsSection = () => (
  <section className="py-32 relative border-t border-white/[0.03]">
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full border border-white/[0.06] bg-white/[0.02]">
          <BarChart3 className="w-3.5 h-3.5 text-white/30" />
          <span className="text-[10px] font-mono uppercase tracking-[0.3em] text-white/30">55+ Metrics</span>
        </div>
        <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
          Every dimension. Every metric.
        </h2>
        <p className="text-white/35 max-w-xl mx-auto">
          From parameter count to inference stability — a complete analytical picture of your architecture.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {METRICS.map((m) => (
          <div key={m.category} className="group rounded-2xl border border-white/[0.04] bg-white/[0.01] p-6 hover:border-white/[0.08] hover:bg-white/[0.02] transition-all duration-300">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
              <h3 className="text-sm font-semibold" style={{ color: m.color }}>{m.category}</h3>
            </div>
            <div className="space-y-2">
              {m.items.map((item) => (
                <div key={item} className="flex items-center gap-2 text-[12px] text-white/40 group-hover:text-white/50 transition-colors">
                  <ChevronRight className="w-3 h-3 opacity-30" />
                  {item}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ─── Page ─────────────────────────────────────────────────────── */
export default function Landing() {
  return (
    <div className="min-h-screen bg-[#05050d] text-white selection:bg-blue-500/30">
      <PageBackground />
      <Navbar />

      <main className="relative z-10">
        <Hero />

        <div className="mx-auto max-w-[1600px] px-6">
          {/* Problem */}
          <ProblemSection />

          {/* Vision */}
          <VisionSection />

          {/* Features */}
          <section id="features" className="py-32 border-t border-white/[0.03]">
            <div className="text-center mb-20">
              <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full border border-white/[0.06] bg-white/[0.02]">
                <Zap className="w-3.5 h-3.5 text-emerald-400/60" />
                <span className="text-[10px] font-mono uppercase tracking-[0.3em] text-white/30">Core Capabilities</span>
              </div>
              <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
                Engineered for every stage
              </h2>
              <p className="text-white/35 max-w-xl mx-auto">
                From architecture design to production deployment — one unified analytical platform.
              </p>
            </div>
            <FeatureCards />
          </section>

          {/* Architectures */}
          <section id="architectures" className="py-32 border-t border-white/[0.03]">
            <ArchitectureDialects />
          </section>

          {/* Metrics */}
          <MetricsSection />

          {/* Pipeline / Demo */}
          <section id="pipeline" className="py-32 border-t border-white/[0.03]">
            <DemoSection />
          </section>

          {/* Rust */}
          <section id="rust" className="py-32 border-t border-white/[0.03]">
            <RustSection />
          </section>
        </div>

        {/* Closing CTA */}
        <section className="relative py-48 overflow-hidden border-t border-white/[0.03]">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-500/[0.03] to-transparent" />
          <div className="relative z-10 mx-auto max-w-4xl px-4 text-center">
            <NeuraxAtomLogo size={48} className="mx-auto mb-8 opacity-60" />
            <h2 className="text-5xl sm:text-6xl font-bold text-white mb-6 tracking-tight">
              Stop guessing.
              <br />
              <span className="bg-clip-text text-transparent" style={{ backgroundImage: 'linear-gradient(135deg, hsl(225 85% 75%), hsl(280 72% 70%), hsl(199 90% 68%))' }}>
                Start compiling.
              </span>
            </h2>
            <p className="text-lg text-white/35 mb-10 max-w-xl mx-auto">
              The future of neural architecture design is analytical, deterministic, and instant. 
              NEURAX makes it possible today.
            </p>
            <Button
              asChild
              size="lg"
              className="bg-white text-[#05050d] hover:bg-white/90 font-bold px-12 py-8 text-lg rounded-full shadow-[0_0_50px_rgba(255,255,255,0.15)]"
            >
              <Link to="/app">
                Launch Studio
                <ArrowRight className="w-5 h-5 ml-2" />
              </Link>
            </Button>
            <div className="mt-12 flex items-center justify-center gap-12 text-[10px] font-mono text-white/20 uppercase tracking-[0.3em]">
              <span>MIT LICENSED</span>
              <span>RUST POWERED</span>
              <span>±3–8% ACCURACY</span>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.04] bg-[#040409]/80 backdrop-blur-md">
        <div className="mx-auto max-w-[1600px] px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-2.5 mb-4">
                <NeuraxAtomLogo size={24} />
                <span className="text-sm font-bold text-white">NEURAX</span>
              </div>
              <p className="text-xs text-white/25 leading-relaxed">
                Analytical compiler for neural architectures. Predict cost, memory, speed, and feasibility before training.
              </p>
            </div>
            <div>
              <h4 className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/30 mb-3">Product</h4>
              <div className="space-y-2">
                <a href="#features" className="block text-xs text-white/25 hover:text-white/60 transition-colors">Features</a>
                <a href="#architectures" className="block text-xs text-white/25 hover:text-white/60 transition-colors">Architectures</a>
                <a href="#pipeline" className="block text-xs text-white/25 hover:text-white/60 transition-colors">Pipeline</a>
                <Link to="/app" className="block text-xs text-white/25 hover:text-white/60 transition-colors">Studio</Link>
              </div>
            </div>
            <div>
              <h4 className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/30 mb-3">Technical</h4>
              <div className="space-y-2">
                <span className="block text-xs text-white/25">10 IR Passes</span>
                <span className="block text-xs text-white/25">55+ Metrics</span>
                <span className="block text-xs text-white/25">Rust Core</span>
                <span className="block text-xs text-white/25">MLIR Backend</span>
              </div>
            </div>
            <div>
              <h4 className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/30 mb-3">Resources</h4>
              <div className="space-y-2">
                <a href="https://github.com/Martial-Christian/Universal_Neurax" className="block text-xs text-white/25 hover:text-white/60 transition-colors">GitHub</a>
                <span className="block text-xs text-white/25">Documentation</span>
                <span className="block text-xs text-white/25">API Reference</span>
              </div>
            </div>
          </div>
          <div className="pt-6 border-t border-white/[0.04] flex flex-col sm:flex-row items-center justify-between gap-4 text-[10px] font-mono text-white/15 uppercase tracking-[0.2em]">
            <span>© {new Date().getFullYear()} NEURAX — DEEP ANALYTICS PLATFORM</span>
            <a href="https://github.com/Martial-Christian/Universal_Neurax" className="hover:text-blue-400 transition-colors">GITHUB</a>
          </div>
        </div>
      </footer>
    </div>
  );
}