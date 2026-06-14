export const HeroBackground = () => (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Style tag for grid pulse */}
        <style dangerouslySetInnerHTML={{
            __html: `
      @keyframes grid-pulse {
        0%, 100% { opacity: 0.15; }
        50% { opacity: 0.25; }
      }
      @keyframes vertical-scan {
        0% { transform: translateY(-300%) rotateX(72deg); opacity: 0; }
        50% { opacity: 0.3; }
        100% { transform: translateY(300%) rotateX(72deg); opacity: 0; }
      }
      .animate-grid-pulse { animation: grid-pulse 4s ease-in-out infinite; }
      .animate-vertical-scan { animation: vertical-scan 10s linear infinite; }
    `}} />

        {/* ── 3D PERSPECTIVE GRID (burn.dev style) ── */}
        <div
            className="absolute inset-x-0 bottom-[-10%] h-[120%] animate-grid-pulse"
            style={{ perspective: '800px', perspectiveOrigin: '50% 0%' }}
        >
            <div
                className="absolute inset-0 w-full h-full"
                style={{ transform: 'rotateX(72deg)', transformOrigin: '50% 0%' }}
            >
                <svg className="w-full h-full opacity-60" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <defs>
                        <linearGradient id="grid-grad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="hsl(225 85% 65%)" stopOpacity="0" />
                            <stop offset="20%" stopColor="hsl(225 85% 65%)" stopOpacity="0.4" />
                            <stop offset="100%" stopColor="hsl(225 85% 65%)" stopOpacity="0.8" />
                        </linearGradient>
                    </defs>
                    <g stroke="url(#grid-grad)" strokeWidth="0.1">
                        {/* Horizontal lines */}
                        {Array.from({ length: 21 }).map((_, i) => (
                            <line key={`h-${i}`} x1="0" y1={i * 5} x2="100" y2={i * 5} />
                        ))}
                        {/* Vertical lines */}
                        {Array.from({ length: 21 }).map((_, i) => (
                            <line key={`v-${i}`} x1={i * 5} y1="0" x2={i * 5} y2="100" />
                        ))}
                    </g>
                </svg>
            </div>

            {/* Grid Scan Line */}
            <div className="absolute inset-0 w-full h-[20%] animate-vertical-scan" style={{ background: 'linear-gradient(to bottom, transparent, rgba(147, 197, 253, 0.2), transparent)' }} />
        </div>

        {/* Section Fades — ensure smooth transition into global background */}
        <div className="absolute inset-x-0 top-0 h-48 bg-gradient-to-b from-[#05050d] to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-[60%] bg-gradient-to-t from-[#05050d] via-[#05050d]/80 to-transparent" />
    </div>
);
