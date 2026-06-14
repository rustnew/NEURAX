// Inline SVG atom logo — matches the generated V2 concept
// 3 orbital rings (blue, violet, rust) around a glowing nucleus
// Represents: 3 core metrics (FLOPs/VRAM/Latency) + 24 AtomOps

interface NeuraxAtomLogoProps {
    size?: number;
    className?: string;
}

export const NeuraxAtomLogo = ({ size = 28, className = '' }: NeuraxAtomLogoProps) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
    >
        <defs>
            <radialGradient id="nucleus-glow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="hsl(225 85% 75%)" stopOpacity="1" />
                <stop offset="60%" stopColor="hsl(225 85% 60%)" stopOpacity="0.9" />
                <stop offset="100%" stopColor="hsl(225 85% 55%)" stopOpacity="0" />
            </radialGradient>
            <filter id="atom-glow">
                <feGaussianBlur stdDeviation="1.2" result="blur" />
                <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
        </defs>

        {/* Outer glow halo */}
        <circle cx="20" cy="20" r="19" fill="hsl(225 85% 55%)" fillOpacity="0.04" />

        {/* Orbital ring 1 — electric blue (horizontal ellipse) */}
        <ellipse
            cx="20" cy="20"
            rx="17" ry="8"
            stroke="hsl(199 90% 58%)"
            strokeWidth="1.2"
            fill="none"
            opacity="0.85"
            filter="url(#atom-glow)"
        />

        {/* Orbital ring 2 — violet (rotated 60°) */}
        <ellipse
            cx="20" cy="20"
            rx="17" ry="8"
            stroke="hsl(280 72% 65%)"
            strokeWidth="1.2"
            fill="none"
            opacity="0.75"
            transform="rotate(60 20 20)"
            filter="url(#atom-glow)"
        />

        {/* Orbital ring 3 — rust-orange (rotated 120°) */}
        <ellipse
            cx="20" cy="20"
            rx="17" ry="8"
            stroke="hsl(14 82% 60%)"
            strokeWidth="1.2"
            fill="none"
            opacity="0.65"
            transform="rotate(120 20 20)"
            filter="url(#atom-glow)"
        />

        {/* Nucleus outer glow */}
        <circle cx="20" cy="20" r="5.5" fill="url(#nucleus-glow)" opacity="0.4" />

        {/* Nucleus core */}
        <circle
            cx="20" cy="20" r="3.5"
            fill="#0a0f1e"
            stroke="hsl(225 85% 65%)"
            strokeWidth="1"
            filter="url(#atom-glow)"
        />

        {/* Tiny waveform / pulse in nucleus */}
        <path
            d="M17.5 20 L18.5 18 L19.5 22 L20.5 19 L21.5 20 L22.5 20"
            stroke="hsl(199 90% 70%)"
            strokeWidth="0.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
        />
    </svg>
);
