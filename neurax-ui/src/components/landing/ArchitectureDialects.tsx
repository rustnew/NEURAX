const FAMILIES = [
    {
        title: 'Transformer / LLM',
        color: 'hsl(199 90% 55%)',
        ops: ['Multi-Head Attention', 'Feed Forward', 'LayerNorm', 'Positional Encoding', 'GQA', 'MQA'],
        models: ['GPT-4', 'LLaMA 3', 'BERT', 'ViT', 'Whisper', 'Mistral'],
        desc: 'The dominant architecture family. Full support for attention variants, positional encodings, and decoder-only configurations.',
    },
    {
        title: 'CNN / Vision',
        color: 'hsl(225 85% 62%)',
        ops: ['Conv2D', 'MaxPool', 'BatchNorm', 'Depthwise Conv', 'Skip Connection', 'Inception'],
        models: ['YOLOv8', 'ResNet-152', 'EfficientNet', 'U-Net', 'MobileNet', 'DINOv2'],
        desc: 'From ResNet to modern vision transformers. Convolution, pooling, and normalization layers fully modeled.',
    },
    {
        title: 'State Space (Mamba)',
        color: 'hsl(160 65% 52%)',
        ops: ['SSM Block', 'Linear Recurrence', 'Selective Scan', 'Gated MLP', 'Mamba Conv1d'],
        models: ['Mamba-2', 'S4', 'H3', 'Hyena', 'RWKV', 'Mamba Vision'],
        desc: 'Next-gen sequence models. Selective state spaces, linear attention, and hybrid architectures.',
    },
    {
        title: 'Mixture of Experts',
        color: 'hsl(280 72% 68%)',
        ops: ['Expert Router', 'Top-K Gate', 'Expert FFN', 'Load Balancer', 'GQA + MoE'],
        models: ['Mixtral 8×7B', 'Switch Transformer', 'GShard', 'DeepSeek-V3', 'DBRX'],
        desc: 'Sparse architectures with expert routing. Full load balancing and communication cost analysis.',
    },
    {
        title: 'Diffusion',
        color: 'hsl(38 92% 58%)',
        ops: ['U-Net', 'Cross-Attention', 'Noise Scheduler', 'VAE Decoder', 'DiT Block'],
        models: ['Stable Diffusion 3', 'DALL-E 3', 'Imagen', 'DiT', 'SDXL', 'Flux'],
        desc: 'Generative image models with timestep conditioning, noise scheduling, and VAE cost modeling.',
    },
    {
        title: 'Graph Neural Nets',
        color: 'hsl(345 82% 62%)',
        ops: ['Message Passing', 'Graph Attention', 'Readout', 'Edge Conv', 'GIN', 'SchNet'],
        models: ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'SchNet', 'PNA'],
        desc: 'Molecular, social, and knowledge graph architectures. Full message-passing cost analysis.',
    },
];

export const ArchitectureDialects = () => (
    <section id="architectures">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-20">
            {/* Header */}
            <div className="text-center mb-16">
                <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full border border-white/[0.06] bg-white/[0.02]">
                    <span className="text-[10px] font-mono uppercase tracking-[0.3em] text-white/30">80+ Architectures</span>
                </div>
                <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
                    Every architecture family, one canvas
                </h2>
                <p className="text-base text-white/35 max-w-xl mx-auto">
                    Design any model type with specialized blocks, real model presets, and family-specific analytical formulas.
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
        className="relative rounded-2xl p-6 flex flex-col gap-4 transition-all duration-300 group hover:border-white/[0.1]"
        style={{
            background: '#0e0e1a',
            border: `1px solid ${family.color}20`,
        }}
    >
        {/* Colored top accent line */}
        <div
            className="absolute inset-x-0 top-0 h-px rounded-t-2xl"
            style={{ background: `linear-gradient(90deg, ${family.color}60, transparent 70%)` }}
        />

        {/* Colored left accent bar */}
        <div
            className="absolute left-0 inset-y-0 w-[2px] rounded-l-2xl"
            style={{ background: `linear-gradient(to bottom, ${family.color}80, transparent)` }}
        />

        {/* Title */}
        <h3 className="font-bold text-base" style={{ color: family.color }}>
            {family.title}
        </h3>

        {/* Description */}
        <p className="text-xs text-white/30 leading-relaxed">{family.desc}</p>

        {/* Op badges */}
        <div className="flex flex-wrap gap-1.5">
            {family.ops.map((op) => (
                <span
                    key={op}
                    className="text-[10px] px-2 py-0.5 rounded-md font-mono"
                    style={{
                        color: family.color,
                        border: `1px solid ${family.color}30`,
                        backgroundColor: `${family.color}08`,
                    }}
                >
                    {op}
                </span>
            ))}
        </div>

        {/* Divider */}
        <div className="h-px" style={{ background: `${family.color}15` }} />

        {/* Model names */}
        <div className="flex flex-wrap gap-x-4 gap-y-1">
            {family.models.map((m) => (
                <span key={m} className="text-xs text-white/45 font-medium">{m}</span>
            ))}
        </div>
    </div>
);