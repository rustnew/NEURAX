import { Brain, Rocket, Clock, Zap } from 'lucide-react';
import { useState } from 'react';

const CARDS = [
    {
        icon: Brain,
        title: 'Architecture Simulation',
        desc: 'Real-time neural system modeling. Define any topology and get instant FLOPs, memory, and latency predictions — no GPU required.',
        iconColor: 'hsl(199 90% 55%)',
        iconBg: 'hsl(199 90% 55% / 0.12)',
        glow: false,
    },
    {
        icon: Rocket,
        title: 'Production Readiness',
        desc: 'Compile-ready architecture analysis on Rust. From design to deployment with deterministic cost estimates and OOM risk assessment.',
        iconColor: 'hsl(14 82% 60%)',
        iconBg: 'hsl(14 82% 55% / 0.12)',
        glow: false,
    },
    {
        icon: Zap,
        title: 'Inference Intelligence',
        desc: 'Predict inference stability, hallucination risk, and sampling volatility before serving a single request. 20-point entropy evolution analysis.',
        iconColor: 'hsl(160 65% 52%)',
        iconBg: 'hsl(160 65% 45% / 0.12)',
        glow: true,
    },
    {
        icon: Clock,
        title: 'Time Machine',
        desc: 'Rollback architecture states. Compare runs across hardware. Version your intelligence with full analytical history.',
        iconColor: 'hsl(280 72% 68%)',
        iconBg: 'hsl(280 72% 60% / 0.12)',
        glow: false,
    },
];

export const FeatureCards = () => (
    <div>
        {/* 4-card grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
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
            className="relative rounded-2xl p-7 flex flex-col gap-5 transition-all duration-300 cursor-default group"
            style={{
                background: card.glow
                    ? `linear-gradient(135deg, #0f1a14 0%, #0b1510 100%)`
                    : '#0e0e1a',
                border: `1px solid ${card.glow
                    ? 'hsl(160 65% 45% / 0.3)'
                    : hovered ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.04)'}`,
                boxShadow: card.glow
                    ? `0 0 40px hsl(160 65% 45% / 0.1), inset 0 1px 0 hsl(160 65% 45% / 0.12)`
                    : hovered
                        ? `0 8px 32px rgba(0,0,0,0.4)`
                        : `0 2px 12px rgba(0,0,0,0.2)`,
            }}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
        >
            {/* Top accent line on glow card */}
            {card.glow && (
                <div
                    className="absolute inset-x-0 top-0 h-px rounded-t-2xl"
                    style={{ background: `linear-gradient(90deg, transparent, ${card.iconColor}60, transparent)` }}
                />
            )}

            {/* Icon */}
            <div
                className="w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300"
                style={{
                    backgroundColor: card.iconBg,
                    border: `1px solid ${card.iconColor}25`,
                    boxShadow: (hovered || card.glow) ? `0 0 20px ${card.iconColor}20` : 'none',
                }}
            >
                <Icon className="w-6 h-6" style={{ color: card.iconColor }} />
            </div>

            {/* Text */}
            <div>
                <h3
                    className="font-bold text-lg mb-2 transition-colors duration-200"
                    style={{ color: card.glow ? 'white' : hovered ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.85)' }}
                >
                    {card.title}
                </h3>
                <p className="text-sm leading-relaxed text-white/35">{card.desc}</p>
            </div>
        </div>
    );
};