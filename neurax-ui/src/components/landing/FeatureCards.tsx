import { Brain, Rocket, Clock, Zap } from 'lucide-react';
import { useState } from 'react';

const CARDS = [
    {
        icon: Brain,
        title: 'Simulation',
        desc: 'Real-time neural system modeling.',
        iconColor: 'hsl(199 90% 55%)',
        iconBg: 'hsl(199 90% 55% / 0.15)',
        glow: false,
    },
    {
        icon: Rocket,
        title: 'Production',
        desc: 'Compile-ready architecture built on Rust.',
        iconColor: 'hsl(14 82% 60%)',
        iconBg: 'hsl(14 82% 55% / 0.15)',
        glow: false,
    },
    {
        icon: Zap,
        title: 'Inference Intelligence',
        desc: 'Optimized inference and deployment insights.',
        iconColor: 'hsl(160 65% 52%)',
        iconBg: 'hsl(160 65% 45% / 0.15)',
        glow: true, // active / highlighted card
    },
    {
        icon: Clock,
        title: 'Time Machine',
        desc: 'Rollback states. Compare runs. Version intelligence.',
        iconColor: 'hsl(280 72% 68%)',
        iconBg: 'hsl(280 72% 60% / 0.15)',
        glow: false,
    },
];

export const FeatureCards = () => (
    <div>
        {/* Section header */}
        <div className="text-center mb-12">
            <h2 className="text-4xl sm:text-5xl font-bold text-white mb-3">
                Engineered for every stage
            </h2>
            <p className="text-base text-white/40">
                From concept to deployment, one unified platform.
            </p>
        </div>

        {/* 4-card grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {CARDS.map((card) => {
                const Icon = card.icon;
                return (
                    <FeatureCard key={card.title} card={card} Icon={Icon} />
                );
            })}
        </div>
    </div>
);

type CardProps = {
    card: typeof CARDS[number];
    Icon: typeof Brain;
};

const FeatureCard = ({ card, Icon }: CardProps) => {
    const [hovered, setHovered] = useState(false);

    return (
        <div
            className="relative rounded-2xl p-6 flex flex-col gap-5 transition-all duration-300 cursor-default"
            style={{
                background: card.glow
                    ? `linear-gradient(135deg, #0f1a14 0%, #0b1510 100%)`
                    : '#0e0e1a',
                border: `1px solid ${card.glow
                    ? 'hsl(160 65% 45% / 0.35)'
                    : hovered ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.06)'}`,
                boxShadow: card.glow
                    ? `0 0 40px hsl(160 65% 45% / 0.12), inset 0 1px 0 hsl(160 65% 45% / 0.15)`
                    : hovered
                        ? `0 8px 32px rgba(0,0,0,0.4)`
                        : `0 2px 12px rgba(0,0,0,0.3)`,
            }}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
        >
            {/* Top accent line on glow card */}
            {card.glow && (
                <div
                    className="absolute inset-x-0 top-0 h-px rounded-t-2xl"
                    style={{ background: `linear-gradient(90deg, transparent, ${card.iconColor}80, transparent)` }}
                />
            )}

            {/* Icon */}
            <div
                className="w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300"
                style={{
                    backgroundColor: card.iconBg,
                    border: `1px solid ${card.iconColor}30`,
                    boxShadow: (hovered || card.glow) ? `0 0 16px ${card.iconColor}30` : 'none',
                }}
            >
                <Icon className="w-5 h-5" style={{ color: card.iconColor }} />
            </div>

            {/* Text */}
            <div>
                <h3
                    className="font-bold text-base mb-1.5 transition-colors duration-200"
                    style={{ color: card.glow ? 'white' : hovered ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.85)' }}
                >
                    {card.title}
                </h3>
                <p className="text-sm leading-relaxed text-white/40">{card.desc}</p>
            </div>
        </div>
    );
};
