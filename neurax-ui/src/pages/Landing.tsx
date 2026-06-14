import { Link } from 'react-router-dom';
import { ArrowRight, Cpu, GitBranch, Zap, Activity } from 'lucide-react';
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

/* ─── Navbar ────────────────────────────────────────────────── */
const Navbar = () => (
  <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-[#05050d]/70 backdrop-blur-xl">
    <div className="mx-auto max-w-6xl px-4 sm:px-6 h-14 flex items-center justify-between">
      {/* Logo */}
      <div className="flex items-center gap-2.5 group cursor-default">
        <div className="flex flex-col">
          <span className="text-sm font-bold tracking-tight text-white leading-none">NEURAX</span>
          <span className="text-[9px] font-mono text-white/20 mt-0.5 tracking-widest uppercase">Universal Analytic</span>
        </div>
      </div>

      {/* Nav links */}
      <nav className="hidden md:flex items-center gap-10 text-xs font-mono uppercase tracking-[0.2em] text-white/30">
        <a href="#features" className="hover:text-white transition-colors duration-300 relative group">
          Features
          <span className="absolute -bottom-1 left-0 w-0 h-px bg-blue-500/50 group-hover:w-full transition-all" />
        </a>
        <a href="#architectures" className="hover:text-white transition-colors duration-300 relative group">
          Arch
          <span className="absolute -bottom-1 left-0 w-0 h-px bg-violet-500/50 group-hover:w-full transition-all" />
        </a>
        <a href="#workflow" className="hover:text-white transition-colors duration-300 relative group">
          Workflow
          <span className="absolute -bottom-1 left-0 w-0 h-px bg-cyan-500/50 group-hover:w-full transition-all" />
        </a>
      </nav>

      {/* Actions */}
      <div className="flex items-center gap-4">
        <div className="hidden lg:flex items-center gap-1.5 text-[10px] text-white/15 font-mono mr-4">
          <Activity size={10} className="text-emerald-500/50 animate-pulse" />
          <span>NET_LNK: READY</span>
        </div>
        <ThemeToggle />
        <Button
          asChild
          size="sm"
          className="bg-white text-[#05050d] hover:bg-white/90 font-bold shadow-[0_0_25px_rgba(255,255,255,0.1)] px-5"
        >
          <Link to="/app">Start Designing</Link>
        </Button>
      </div>
    </div>
  </header>
);

/* ─── Hero ──────────────────────────────────────────────────── */
const Hero = () => (
  <section className="relative min-h-[calc(100vh-56px)] flex flex-col items-center justify-center overflow-hidden">
    <HeroBackground />

    <div className="relative z-10 mx-auto max-w-5xl px-4 sm:px-6 pt-24 pb-12 text-center">
      {/* Eyebrow badge */}
      <div className="inline-flex items-center gap-3 mb-10 px-4 py-2 rounded-full border border-white/5 bg-white/[0.02] text-[10px] font-mono uppercase tracking-[0.3em] text-white/30 backdrop-blur-sm animate-fade-in">
        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
        Analytic Compiler v0.1.0-STABLE
      </div>

      {/* Main headline */}
      <h1 className="text-6xl sm:text-7xl lg:text-8xl font-bold text-white leading-[0.95] tracking-tighter mb-8 animate-fade-in transition-all duration-700">
        Design. Simulate.
        <br />
        <span
          className="bg-clip-text text-transparent"
          style={{
            backgroundImage:
              'linear-gradient(135deg, hsl(225 85% 75%), hsl(280 72% 70%), hsl(199 90% 68%))',
          }}
        >
          Deploy.
        </span>
      </h1>

      {/* Sub-headline */}
      <p className="max-w-2xl mx-auto text-xl text-white/40 leading-relaxed mb-12 animate-fade-in" style={{ animationDelay: '0.2s' }}>
        Universal static analysis for neural architectures.
        <br />
        Deterministic costs. Zero variance.
      </p>

      {/* CTAs */}
      <div className="flex flex-wrap items-center justify-center gap-4 animate-fade-in" style={{ animationDelay: '0.4s' }}>
        <Button
          asChild
          size="lg"
          className="bg-white text-[#05050d] hover:bg-white/90 font-bold px-10 py-7 text-base shadow-[0_0_40px_rgba(255,255,255,0.15)] transition-all transform hover:scale-105"
        >
          <Link to="/app">
            Launch Studio
            <ArrowRight className="w-4 h-4 ml-2" />
          </Link>
        </Button>
        <Button
          asChild
          variant="outline"
          size="lg"
          className="border-white/10 bg-white/[0.02] text-white/60 hover:bg-white/[0.05] hover:text-white hover:border-white/20 px-10 py-7 text-base backdrop-blur-md"
        >
          <a href="#workflow">Technical Report</a>
        </Button>
      </div>

      {/* Quick stats with hover motion */}
      <div className="mt-20 grid grid-cols-3 gap-6 max-w-3xl mx-auto animate-fade-in" style={{ animationDelay: '0.6s' }}>
        {[
          { value: '±3–8%', label: 'ACCURACY' },
          { value: '80+', label: 'ARCHS' },
          { value: 'ZERO', label: 'DRIFT' },
        ].map((s) => (
          <div key={s.label} className="group border border-white/[0.03] bg-white/[0.01] p-5 rounded-2xl hover:border-white/[0.08] hover:bg-white/[0.03] transition-all duration-300">
            <div className="text-[9px] font-mono text-white/20 tracking-widest mb-1 group-hover:text-blue-400/50 transition-colors">{s.label}</div>
            <div className="text-3xl font-bold text-white group-hover:scale-110 transition-transform">{s.value}</div>
          </div>
        ))}
      </div>
    </div>

    {/* Product showcase - full width */}
    <div className="relative z-10 w-full animate-fade-in" style={{ animationDelay: '0.8s' }}>
      <CanvasShowcase />
    </div>

    {/* Scroll fade into next section */}
    <div className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-[#05050d] to-transparent pointer-events-none" />
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

        {/* Scroll Content Block */}
        <div className="mx-auto max-w-[1600px] px-6">
          <section id="features" className="py-32">
            <div className="text-center mb-20 animate-fade-in">
              <h2 className="text-4xl sm:text-5xl font-bold text-white mb-6">Designed for Determinism</h2>
              <p className="text-white/30 text-lg max-w-2xl mx-auto">Skip the noise. Get exact hardware formulas for every operator.</p>
            </div>
            <FeatureCards />
          </section>

          <section id="architectures" className="py-32 border-t border-white/[0.03]">
            <ArchitectureDialects />
          </section>

          <section id="workflow" className="py-32 border-t border-white/[0.03]">
            <DemoSection />
          </section>

          <section id="rust" className="py-32 border-t border-white/[0.03]">
            <RustSection />
          </section>
        </div>

        {/* Closing CTA */}
        <section className="relative py-48 overflow-hidden border-t border-white/[0.03]">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-500/[0.02] to-transparent" />
          <div className="relative z-10 mx-auto max-w-4xl px-4 text-center">
            <h2 className="text-6xl font-bold text-white mb-8 tracking-tight">The Science of Prediction.</h2>
            <Button
              asChild
              size="lg"
              className="bg-white text-[#05050d] hover:bg-white/90 font-bold px-12 py-8 text-lg rounded-full shadow-[0_0_50px_rgba(255,255,255,0.15)]"
            >
              <Link to="/app">Start Compiling Now</Link>
            </Button>
            <div className="mt-12 flex items-center justify-center gap-12 text-[10px] font-mono text-white/20 uppercase tracking-[0.3em]">
              <span>MIT LICENSED</span>
              <span>CARGO_READY</span>
              <span>EST_ACC: 97.4%</span>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.04] bg-[#040409]/80 backdrop-blur-md">
        <div className="mx-auto max-w-[1600px] px-8 py-10 flex flex-col md:flex-row items-center justify-between gap-6 text-[10px] font-mono text-white/20 uppercase tracking-[0.2em]">
          <div className="flex items-center gap-4">
            <span>NEURAX — DEEP ANALYTICS PLATFORM</span>
          </div>
          <div className="flex items-center gap-10">
            <span>© {new Date().getFullYear()} CONTRIBUTORS</span>
            <a href="https://github.com/Martial-Christian/Universal_Neurax" className="hover:text-blue-400 transition-colors">GITHUB_REPO</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
