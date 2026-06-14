const FAMILIES = [
    {
        title: 'Transformer / LLM',
        color: 'hsl(199 90% 55%)',
        ops: ['Multi-Head Attention', 'Feed Forward', 'LayerNorm', 'Positional Encoding'],
        models: ['GPT-4', 'LLaMA 3', 'BERT', 'ViT', 'Whisper'],
    },
    {
        title: 'CNN / Vision',
        color: 'hsl(199 90% 55%)',
        ops: ['Conv2D', 'MaxPool', 'BatchNorm', 'Depthwise Conv', 'Skip Connection'],
        models: ['YOLOv8', 'ResNet-152', 'EfficientNet', 'U-Net', 'MobileNet'],
    },
    {
        title: 'State Space (Mamba)',
        color: 'hsl(160 65% 52%)',
        ops: ['SSM Block', 'Linear Recurrence', 'Selective Scan', 'Gated MLP'],
        models: ['Mamba-2', 'S4', 'H3', 'Hyena', 'RWKV'],
    },
    {
        title: 'Mixture of Experts',
        color: 'hsl(280 72% 68%)',
        ops: ['Expert Router', 'Top-K Gate', 'Expert FFN', 'Load Balancer'],
        models: ['Mixtral 8×7B', 'Switch Transformer', 'GShard', 'DeepSeek MoE'],
    },
    {
        title: 'Diffusion',
        color: 'hsl(38 92% 58%)',
        ops: ['U-Net', 'Cross-Attention', 'Noise Scheduler', 'VAE Decoder'],
        models: ['Stable Diffusion 3', 'DALL-E 3', 'Imagen', 'DiT'],
    },
    {
        title: 'Graph Neural Nets',
        color: 'hsl(345 82% 62%)',
        ops: ['Message Passing', 'Graph Attention', 'Readout', 'Edge Conv'],
        models: ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'SchNet'],
    },
];

export const ArchitectureDialects = () => (
    <section id="architectures">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-20">
            {/* Header */}
            <div className="text-center mb-12">
                <h2 className="text-4xl sm:text-5xl font-bold text-white mb-3">
                    Every architecture family, one canvas
                </h2>
                <p className="text-base text-white/40">
                    Design any model type with specialized blocks and real model presets.
                </p>
            </div>

            {/* 3×2 grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {FAMILIES.map((f) => (
                    <FamilyCard key={f.title} family={f} />
                ))}
            </div>
        </div>
    </section>
);

type FamilyCardProps = { family: typeof FAMILIES[number] };

const FamilyCard = ({ family }: FamilyCardProps) => (
    <div
        className="relative rounded-2xl p-6 flex flex-col gap-5 transition-all duration-300 group"
        style={{
            background: '#0e0e1a',
            border: `1px solid ${family.color}30`,
        }}
    >
        {/* Colored top accent line */}
        <div
            className="absolute inset-x-0 top-0 h-px rounded-t-2xl"
            style={{ background: `linear-gradient(90deg, ${family.color}80, transparent 70%)` }}
        />

        {/* Colored left accent bar */}
        <div
            className="absolute left-0 inset-y-0 w-[2px] rounded-l-2xl"
            style={{ background: `linear-gradient(to bottom, ${family.color}90, transparent)` }}
        />

        {/* Title */}
        <h3 className="font-bold text-base" style={{ color: family.color }}>
            {family.title}
        </h3>

        {/* Op badges */}
        <div className="flex flex-wrap gap-1.5">
            {family.ops.map((op) => (
                <span
                    key={op}
                    className="text-[11px] px-2.5 py-1 rounded-md font-mono"
                    style={{
                        color: family.color,
                        border: `1px solid ${family.color}40`,
                        backgroundColor: `${family.color}0c`,
                    }}
                >
                    {op}
                </span>
            ))}
        </div>

        {/* Divider */}
        <div className="h-px" style={{ background: `${family.color}20` }} />

        {/* Model names */}
        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
            {family.models.map((m) => (
                <span key={m} className="text-sm text-white/60 font-medium">
                    {m}
                </span>
            ))}
        </div>
    </div>
);
