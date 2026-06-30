import { useRef, useState, useEffect } from 'react';
import {
    SquareArrowRight,
    Layers,
    Focus,
    AlignJustify,
    SquareArrowOutUpRight,
} from 'lucide-react';

const NODE_DATA = [
    {
        id: 'input',
        title: 'Input',
        subtitle: 'INPUT',
        icon: SquareArrowRight,
        initialX: 100,
        initialY: 40,
        rows: [
            { label: 'shape', val: '[B, seq_len]', isOutput: false, isGreen: false },
            { label: 'in', val: '[B, seq_len]', isOutput: true, isGreen: true },
        ]
    },
    {
        id: 'embedding',
        title: 'Embedding',
        subtitle: 'EMBEDDING',
        icon: Layers,
        initialX: 620,
        initialY: 40,
        rows: [
            { label: 'vocab', val: '32000', isOutput: false, isGreen: false },
            { label: 'dim', val: '768', isOutput: false, isGreen: false },
            { label: 'out', val: '[B, S, 768]', isOutput: true, isGreen: false },
        ]
    },
    {
        id: 'attention',
        title: 'Attention',
        subtitle: 'ATTN_SCORE',
        icon: Focus,
        initialX: 100,
        initialY: 230,
        rows: [
            { label: 'heads', val: '12', isOutput: false, isGreen: false },
            { label: 'type', val: 'sdpa', isOutput: false, isGreen: false },
            { label: 'out', val: '[B, S, 768]', isOutput: true, isGreen: true },
        ]
    },
    {
        id: 'layernorm',
        title: 'LayerNorm',
        subtitle: 'NORM',
        icon: AlignJustify,
        initialX: 620,
        initialY: 230,
        rows: [
            { label: 'eps', val: '1e-6', isOutput: false, isGreen: false },
            { label: 'out', val: '[B, S, 768]', isOutput: true, isGreen: false },
        ]
    },
    {
        id: 'ffn',
        title: 'FFN',
        subtitle: 'LINEAR',
        icon: Layers,
        initialX: 100,
        initialY: 420,
        rows: [
            { label: 'dim', val: '3072', isOutput: false, isGreen: false },
            { label: 'out', val: '[B, S, 3072]', isOutput: true, isGreen: true },
        ]
    },
    {
        id: 'lm_head',
        title: 'LM Head',
        subtitle: 'OUTPUT',
        icon: SquareArrowOutUpRight,
        initialX: 620,
        initialY: 420,
        rows: [
            { label: 'vocab', val: '32k', isOutput: false, isGreen: false },
            { label: 'out', val: '[B, S, V]', isOutput: true, isGreen: false },
        ]
    },
];

export const CanvasShowcase = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [nodes, setNodes] = useState(
        NODE_DATA.reduce((acc, node) => ({
            ...acc,
            [node.id]: { x: node.initialX, y: node.initialY }
        }), {} as Record<string, { x: number, y: number }>)
    );

    const [draggingNode, setDraggingNode] = useState<string | null>(null);
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
    const [tilt, setTilt] = useState({ rotX: 0, rotY: 0 });

    const handleMouseMove = (e: React.MouseEvent) => {
        // Tilt effect
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const dx = (e.clientX - cx) / (rect.width / 2);
        const dy = (e.clientY - cy) / (rect.height / 2);
        setTilt({ rotX: -dy * 1.5, rotY: dx * 1.5 });

        // Drag logic
        if (draggingNode) {
            const x = e.clientX - rect.left - dragOffset.x;
            const y = e.clientY - rect.top - dragOffset.y;
            setNodes(prev => ({
                ...prev,
                [draggingNode]: { x, y }
            }));
        }
    };

    const startDrag = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const containerRect = containerRef.current?.getBoundingClientRect();
        if (!containerRect) return;

        setDraggingNode(id);
        setDragOffset({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        });
    };

    const stopDrag = () => setDraggingNode(null);

    useEffect(() => {
        if (draggingNode) {
            window.addEventListener('mouseup', stopDrag);
            return () => window.removeEventListener('mouseup', stopDrag);
        }
    }, [draggingNode]);

    return (
        <div className="mt-12 px-4 sm:px-6 w-full max-w-[1000px] mx-auto select-none">
            <div
                ref={containerRef}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => { setTilt({ rotX: 0, rotY: 0 }); }}
                className="relative perspective-2000 group"
            >
                {/* Glow */}
                <div className="absolute -inset-20 bg-blue-600/5 blur-[120px] pointer-events-none" />

                {/* Window */}
                <div
                    className="relative rounded-xl border border-white/[0.06] bg-[#09090b]/90 backdrop-blur-2xl shadow-2xl overflow-hidden transition-transform duration-500 ease-out"
                    style={{ transform: `rotateX(${tilt.rotX}deg) rotateY(${tilt.rotY}deg)` }}
                >
                    {/* Title Bar */}
                    <div className="flex items-center gap-2 px-4 h-10 border-b border-white/[0.05] bg-[#0a0a0f]">
                        <div className="flex gap-1.2">
                            <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/30" />
                            <div className="w-2.5 h-2.5 rounded-full bg-amber-500/20 border border-amber-500/30" />
                            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/20 border border-emerald-500/30" />
                        </div>
                        <span className="ml-4 text-[10px] font-mono text-white/20 tracking-tight lowercase">neurax_studio — system_graph_v2</span>
                        <div className="ml-auto text-[9px] text-emerald-400/80 font-mono bg-emerald-400/5 px-2 py-0.5 rounded border border-emerald-400/10">
                            LIVE
                        </div>
                    </div>

                    {/* Canvas */}
                    <div className="relative h-[600px] overflow-hidden bg-[radial-gradient(#ffffff05_1px,transparent_1px)] [background-size:24px_24px]">
                        {/* SVG Connections */}
                        <svg className="absolute inset-0 w-full h-full pointer-events-none">
                            <defs>
                                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                    <polygon points="0 0, 10 3.5, 0 7" fill="white" opacity="0.1" />
                                </marker>
                            </defs>
                            {NODE_DATA.map((node, i) => {
                                if (i === NODE_DATA.length - 1) return null;
                                const nextNode = NODE_DATA[i + 1];
                                const start = nodes[node.id];
                                const end = nodes[nextNode.id];
                                if (!start || !end) return null;

                                // Connection points logic
                                // Horizontal connection
                                const isHorizontal = (i === 0 || i === 2 || i === 4);

                                let sx, sy, ex, ey;
                                if (isHorizontal) {
                                    sx = start.x + 224;
                                    sy = start.y + 55;
                                    ex = end.x;
                                    ey = end.y + 55;
                                } else {
                                    // Downward diagonal connection
                                    sx = start.x + 112;
                                    sy = start.y + 110;
                                    ex = end.x + 112;
                                    ey = end.y;
                                }

                                return (
                                    <path
                                        key={`${node.id}-${nextNode.id}`}
                                        d={isHorizontal
                                            ? `M ${sx} ${sy} C ${sx + 40} ${sy}, ${ex - 40} ${ey}, ${ex} ${ey}`
                                            : `M ${sx} ${sy} C ${sx} ${sy + 40}, ${ex} ${ey - 40}, ${ex} ${ey}`
                                        }
                                        fill="none"
                                        stroke="white"
                                        strokeOpacity="0.08"
                                        strokeWidth="1.5"
                                        strokeDasharray="4 4"
                                        markerEnd="url(#arrowhead)"
                                    />
                                );
                            })}
                        </svg>

                        {/* Nodes */}
                        {NODE_DATA.map((node) => {
                            const pos = nodes[node.id];
                            const Icon = node.icon;
                            return (
                                <div
                                    key={node.id}
                                    onMouseDown={(e) => startDrag(e, node.id)}
                                    className={`absolute group/node cursor-grab active:cursor-grabbing transition-shadow duration-200 ${draggingNode === node.id ? 'z-50' : 'z-10'}`}
                                    style={{
                                        left: `${pos.x}px`,
                                        top: `${pos.y}px`,
                                    }}
                                >
                                    <div className={`relative w-56 rounded-lg border bg-[#111116]/95 p-3 flex flex-col gap-2.5 transition-all duration-300 ${draggingNode === node.id ? 'border-blue-500/50 shadow-[0_0_30px_rgba(59,130,246,0.15)] bg-[#14141e]' : 'border-white/[0.08]'
                                        } hover:border-white/20`}>

                                        {/* Header */}
                                        <div className="flex items-center gap-2.5">
                                            <div className="w-7 h-7 rounded bg-white/[0.03] border border-white/[0.05] flex items-center justify-center text-white/50">
                                                <Icon size={14} />
                                            </div>
                                            <div>
                                                <div className="text-[11px] font-bold text-white tracking-tight">{node.title}</div>
                                                <div className="text-[8px] font-mono text-white/20 uppercase tracking-widest">{node.subtitle}</div>
                                            </div>
                                        </div>

                                        {/* Rows */}
                                        <div className="space-y-1.5">
                                            {node.rows.map((row, idx) => (
                                                <div key={idx} className="flex items-center justify-between gap-2 border-b border-white/[0.02] pb-1 last:border-0 last:pb-0">
                                                    <span className="text-[9px] font-mono text-white/20">{row.label}</span>
                                                    <span className={`text-[9px] font-mono px-1.5 py-0.2 rounded border ${row.isOutput
                                                        ? (row.isGreen ? 'bg-emerald-500/5 text-emerald-400/70 border-emerald-500/10' : 'bg-white/5 text-white/40 border-white/5')
                                                        : 'text-white/70 border-transparent'
                                                        }`}>
                                                        {row.val}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>

                                        {/* Port Indicators */}
                                        <div className="absolute left-0 top-1/2 -translate-x-1 -translate-y-1/2 w-2 h-2 rounded-full bg-[#1a1a24] border border-white/20" />
                                        <div className="absolute right-0 top-1/2 translate-x-1 -translate-y-1/2 w-2 h-2 rounded-full bg-[#1a1a24] border border-white/20" />
                                        <div className="absolute left-1/2 bottom-0 -translate-x-1/2 translate-y-1 w-2 h-2 rounded-full bg-[#1a1a24] border border-white/20 opacity-30" />
                                        <div className="absolute left-1/2 top-0 -translate-x-1/2 -translate-y-1 w-2 h-2 rounded-full bg-[#1a1a24] border border-white/20 opacity-30" />
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Stats Bar */}
                    <div className="grid grid-cols-5 divide-x divide-white/[0.03] border-t border-white/[0.05] bg-[#0a0a0f]">
                        {[
                            { label: 'TFLOP/S', val: '312' },
                            { label: 'VRAM', val: '24GB' },
                            { label: 'LATENCY', val: '42ms' },
                            { label: 'NODES', val: '24' },
                            { label: 'CONF.', val: '99%' },
                        ].map((s, i) => (
                            <div key={i} className="px-4 py-2 text-center">
                                <div className="text-[8px] font-mono text-white/20 mb-0.5">{s.label}</div>
                                <div className="text-[10px] font-bold text-white/70 uppercase">{s.val}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};
