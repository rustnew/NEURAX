// Full-page fixed background — covers the ENTIRE viewport
// Creates the "background occupies the whole screen" effect
// Enhanced with scanning beams, ambient particles, and edge HUD elements for total immersion

export const PageBackground = () => (
    <div
        className="fixed inset-0 pointer-events-none overflow-hidden"
        style={{ zIndex: 0 }}
        aria-hidden="true"
    >
        {/* Deep void base */}
        <div className="absolute inset-0 bg-[#05050d]" />

        {/* Style tag for floating animations & beams */}
        <style dangerouslySetInnerHTML={{
            __html: `
      @keyframes drift {
        0% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(30px, -20px) scale(1.05); }
        66% { transform: translate(-20px, 40px) scale(0.95); }
        100% { transform: translate(0, 0) scale(1); }
      }
      @keyframes drift-reverse {
        0% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(-40px, 30px) scale(0.97); }
        66% { transform: translate(25px, -35px) scale(1.03); }
        100% { transform: translate(0, 0) scale(1); }
      }
      @keyframes scan-beam {
        0% { transform: translateY(-100%) translateX(-100%) rotate(45deg); opacity: 0; }
        20% { opacity: 0.15; }
        80% { opacity: 0.15; }
        100% { transform: translateY(200%) translateX(200%) rotate(45deg); opacity: 0; }
      }
      @keyframes float-particle {
        0% { transform: translateY(0) translateX(0); opacity: 0; }
        10% { opacity: 0.3; }
        90% { opacity: 0.3; }
        100% { transform: translateY(-100vh) translateX(50px); opacity: 0; }
      }
      @keyframes hud-flicker {
        0%, 100% { opacity: 0.2; }
        50% { opacity: 0.4; }
        55% { opacity: 0.1; }
        60% { opacity: 0.4; }
      }
      .animate-drift { animation: drift 20s ease-in-out infinite; }
      .animate-drift-slow { animation: drift 35s ease-in-out infinite; }
      .animate-drift-reverse { animation: drift-reverse 25s ease-in-out infinite; }
      .animate-scan { animation: scan-beam 12s linear infinite; }
      .animate-hud { animation: hud-flicker 8s ease-in-out infinite; }
      
      .particle {
        position: absolute;
        width: 1.5px;
        height: 1.5px;
        background: white;
        border-radius: 50%;
        pointer-events: none;
      }
    `}} />

        {/* DATA BEAMS — Cross the whole screen */}
        <div className="absolute top-0 left-0 w-full h-[500%] animate-scan pointer-events-none" style={{ background: 'linear-gradient(to right, transparent, rgba(255,255,255,0.03), transparent)', width: '2px', left: '10%' }} />
        <div className="absolute top-0 left-0 w-full h-[500%] animate-scan pointer-events-none" style={{ background: 'linear-gradient(to right, transparent, rgba(255,255,255,0.02), transparent)', width: '1px', left: '30%', animationDelay: '4s' }} />
        <div className="absolute top-0 left-0 w-full h-[500%] animate-scan pointer-events-none" style={{ background: 'linear-gradient(to right, transparent, rgba(255,100,250,0.02), transparent)', width: '1px', left: '70%', animationDelay: '8s' }} />

        {/* Ambient Particles */}
        {[...Array(30)].map((_, i) => (
            <div
                key={i}
                className="particle"
                style={{
                    left: `${Math.random() * 100}%`,
                    top: `${Math.random() * 100 + 100}%`,
                    animation: `float-particle ${15 + Math.random() * 25}s linear infinite`,
                    animationDelay: `${Math.random() * 20}s`,
                    opacity: 0.2
                }}
            />
        ))}

        {/* PERIPHERAL HUD ELEMENTS — Fill the corners */}
        <div className="absolute top-20 left-6 flex flex-col gap-1 font-mono text-[9px] text-white/20 animate-hud">
            <div>SYS_COORDS: [40.7128, -74.0060]</div>
            <div>NEURAL_LOAD: STABLE</div>
            <div className="w-16 h-px bg-white/10 mt-1" />
        </div>

        <div className="absolute top-20 right-6 flex flex-col items-end gap-1 font-mono text-[9px] text-white/20 animate-hud" style={{ animationDelay: '2s' }}>
            <div>REVISION: v0.1.0-BETA</div>
            <div>UPTIME: 14:02:55:01</div>
            <div className="w-16 h-px bg-white/10 mt-1" />
        </div>

        <div className="absolute bottom-10 left-6 flex flex-col gap-1 font-mono text-[9px] text-white/20 animate-hud" style={{ animationDelay: '4s' }}>
            <div>COMPILER_STATUS: READY</div>
            <div>TARGET: WASM/X86_64</div>
        </div>

        {/* PRIMARY glow — large blue */}
        <div
            className="absolute top-[-15%] left-[-10%] w-[1300px] h-[1100px] rounded-full blur-[220px] animate-drift-slow"
            style={{
                background: 'radial-gradient(circle, hsl(225 85% 52%) 0%, transparent 75%)',
                opacity: 0.16,
            }}
        />

        {/* Top-right violet nebula */}
        <div
            className="absolute top-[5%] right-[-20%] w-[1400px] h-[1200px] rounded-full blur-[220px] animate-drift-reverse"
            style={{
                background: 'radial-gradient(circle, hsl(280 72% 50%), transparent 75%)',
                opacity: 0.13,
            }}
        />

        {/* Mid-left cyan accent */}
        <div
            className="absolute top-[30%] left-[-25%] w-[1200px] h-[1000px] rounded-full blur-[200px] animate-drift"
            style={{
                background: 'radial-gradient(circle, hsl(199 90% 45%), transparent 75%)',
                opacity: 0.11,
            }}
        />

        {/* Bottom-right rust accent */}
        <div
            className="absolute bottom-[-20%] right-[-15%] w-[1300px] h-[900px] rounded-full blur-[200px] animate-drift-reverse"
            style={{
                background: 'radial-gradient(circle, hsl(14 82% 48%), transparent 75%)',
                opacity: 0.09,
            }}
        />

        {/* Fine dot mesh */}
        <svg
            className="absolute inset-0 w-full h-full opacity-[0.05]"
            xmlns="http://www.w3.org/2000/svg"
        >
            <defs>
                <pattern id="page-grid" width="80" height="80" patternUnits="userSpaceOnUse">
                    <circle cx="40" cy="40" r="1.2" fill="white" />
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#page-grid)" />
        </svg>
    </div>
);
