export const HeroBackground = () => (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Style tag for animations */}
        <style dangerouslySetInnerHTML={{
            __html: `
      @keyframes grid-pulse {
        0%, 100% { opacity: 0.12; }
        50% { opacity: 0.22; }
      }
      @keyframes vertical-scan {
        0% { transform: translateY(-300%) rotateX(72deg); opacity: 0; }
        50% { opacity: 0.25; }
        100% { transform: translateY(300%) rotateX(72deg); opacity: 0; }
      }
      @keyframes float-orb {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-20px) scale(1.05); }
      }
      @keyframes pulse-ring {
        0% { transform: scale(0.8); opacity: 0.4; }
        50% { transform: scale(1.2); opacity: 0.1; }
        100% { transform: scale(0.8); opacity: 0.4; }
      }
      .animate-grid-pulse { animation: grid-pulse 4s ease-in-out infinite; }
      .animate-vertical-scan { animation: vertical-scan 10s linear infinite; }
      .animate-float-orb { animation: float-orb 8s ease-in-out infinite; }
      .animate-pulse-ring { animation: pulse-ring 6s ease-in-out infinite; }
    `}} />

        {/* ── 3D PERSPECTIVE GRID ── */}
        <div
            className="absolute inset-x-0 bottom-[-10%] h-[120%] animate-grid-pulse"
            style={{ perspective: '800px', perspectiveOrigin: '50% 0%' }}
        >
            <div
                className="absolute inset-0 w-full h-full"
                style={{ transform: 'rotateX(72deg)', transformOrigin: '50% 0%' }}
            >
                <svg className="w-full h-full opacity-50" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <defs>
                        <linearGradient id="grid-grad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="hsl(225 85% 65%)" stopOpacity="0" />
                            <stop offset="20%" stopColor="hsl(225 85% 65%)" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="hsl(225 85% 65%)" stopOpacity="0.6" />
                        </linearGradient>
                    </defs>
                    <g stroke="url(#grid-grad)" strokeWidth="0.08">
                        {Array.from({ length: 21 }).map((_, i) => (
                            <line key={`h-${i}`} x1="0" y1={i * 5} x2="100" y2={i * 5} />
                        ))}
                        {Array.from({ length: 21 }).map((_, i) => (
                            <line key={`v-${i}`} x1={i * 5} y1="0" x2={i * 5} y2="100" />
                        ))}
                    </g>
                </svg>
            </div>

            {/* Grid Scan Line */}
            <div className="absolute inset-0 w-full h-[20%] animate-vertical-scan" style={{ background: 'linear-gradient(to bottom, transparent, rgba(147, 197, 253, 0.15), transparent)' }} />
        </div>

        {/* ── Floating orbs ── */}
        <div
            className="absolute top-[20%] left-[15%] w-[400px] h-[400px] rounded-full blur-[120px] animate-float-orb"
            style={{ background: 'radial-gradient(circle, hsl(225 85% 55%) 0%, transparent 70%)', opacity: 0.08 }}
        />
        <div
            className="absolute top-[40%] right-[10%] w-[350px] h-[350px] rounded-full blur-[100px] animate-float-orb"
            style={{ background: 'radial-gradient(circle, hsl(280 72% 55%) 0%, transparent 70%)', opacity: 0.06, animationDelay: '3s' }}
        />
        <div
            className="absolute bottom-[30%] left-[40%] w-[300px] h-[300px] rounded-full blur-[80px] animate-float-orb"
            style={{ background: 'radial-gradient(circle, hsl(199 90% 50%) 0%, transparent 70%)', opacity: 0.05, animationDelay: '5s' }}
        />

        {/* ── Pulse rings ── */}
        <div className="absolute top-[45%] left-[50%] -translate-x-1/2 -translate-y-1/2">
            <div className="w-[600px] h-[600px] rounded-full border border-blue-500/[0.04] animate-pulse-ring" />
            <div className="absolute inset-0 w-[400px] h-[400px] m-auto rounded-full border border-violet-500/[0.04] animate-pulse-ring" style={{ animationDelay: '2s' }} />
        </div>

        {/* Section Fades */}
        <div className="absolute inset-x-0 top-0 h-48 bg-gradient-to-b from-[#05050d] to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-[60%] bg-gradient-to-t from-[#05050d] via-[#05050d]/80 to-transparent" />
    </div>
);