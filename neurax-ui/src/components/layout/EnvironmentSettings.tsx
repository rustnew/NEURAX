import {
    Settings2,
    Maximize,
    Minimize,
    Sliders,
    TrendingUp,
    Database,
    Server
} from 'lucide-react';
import { useHardware, HardwareConfig, validateHardwareConfig, MANDATORY_FIELDS, ArchitectureFamily as HwFamily } from '@/contexts/HardwareContext.tsx';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { AlertCircle } from 'lucide-react';
import { Input } from '@/components/ui/input.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { cn } from '@/lib/utils.ts';
import { useState, useEffect } from 'react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip.tsx';

interface EnvironmentSettingsProps {
    family: ArchitectureFamily;
    isSearchActive?: boolean;
}

export function EnvironmentSettings({ family, isSearchActive }: EnvironmentSettingsProps) {
    const { config, updateConfig, lastAttemptTime } = useHardware();
    const [isExpanded, setIsExpanded] = useState(true);
    const [isTrainingExpanded, setIsTrainingExpanded] = useState(true);
    const [isDataExpanded, setIsDataExpanded] = useState(true);
    const [isJiggling, setIsJiggling] = useState(false);
    const [autoFields, setAutoFields] = useState<Record<string, boolean>>({
        headDim: true,
        kvHeads: true,
        maxSeqLen: true,
    });

    const validation = validateHardwareConfig(config, family as HwFamily);
    const missingSet = new Set(validation.missingFields);
    const requiredSet = new Set<keyof HardwareConfig>([
        ...(MANDATORY_FIELDS.common ?? []),
        ...((MANDATORY_FIELDS[family as string] ?? []) as (keyof HardwareConfig)[]),
    ]);

    useEffect(() => {
        if (lastAttemptTime > 0 && !validation.isValid) {
            setIsJiggling(true);
            const timer = setTimeout(() => setIsJiggling(false), 500);
            return () => clearTimeout(timer);
        }
    }, [lastAttemptTime]);

    if (isSearchActive) return null;

    const renderField = (label: string, value: any, key: keyof HardwareConfig, type: 'number' | 'text' | 'switch' | 'select' = 'number', options?: string[]) => {
        const isMandatory = missingSet.has(key);
        const isMissing = isMandatory && (!value || (typeof value === 'number' && value <= 0));
        const isRequired = requiredSet.has(key);

        return (
            <div key={key} className={cn(
                "space-y-1 rounded-sm transition-all duration-300",
                isJiggling && isMissing ? "animate-jiggle" : ""
            )}>
                <div className="flex items-center justify-between">
                    <label className={cn(
                        "text-[9px] font-medium uppercase tracking-tighter",
                        "text-muted-foreground/80"
                    )}>
                        {label}
                    </label>
                    <span className={cn(
                        "text-[8px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border",
                        isRequired
                            ? "border-primary/30 bg-primary/10 text-primary"
                            : "border-border/40 bg-background/30 text-muted-foreground"
                    )}>
                        {isRequired ? 'Required' : 'Optional'}
                    </span>
                    {isMissing && (
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger>
                                    <AlertCircle className="w-2.5 h-2.5 text-primary/70 stroke-[3]" />
                                </TooltipTrigger>
                                <TooltipContent side="top" className="text-[10px]">
                                    Mandatory field
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    )}
                </div>
                {type === 'number' && (
                    <Input
                        type="number"
                        value={value ?? ''}
                        onChange={(e) => updateConfig({ [key]: parseFloat(e.target.value) || 0 })}
                        className="h-6 text-[10px] px-1.5 bg-background/50 border-border/40"
                    />
                )}
                {type === 'text' && (
                    <Input
                        type="text"
                        value={value ?? ''}
                        onChange={(e) => updateConfig({ [key]: e.target.value })}
                        className="h-6 text-[10px] px-1.5 bg-background/50 border-border/40"
                    />
                )}
                {type === 'switch' && (
                    <div className="flex items-center justify-between h-6 px-1 bg-background/30 rounded border border-border/20">
                        <span className="text-[9px] text-muted-foreground px-1">Active</span>
                        <Switch
                            checked={!!value}
                            onCheckedChange={(checked) => updateConfig({ [key]: checked })}
                            className="scale-75"
                        />
                    </div>
                )}
                {type === 'select' && options && (
                    <Select value={value as string} onValueChange={(v) => updateConfig({ [key]: v })}>
                        <SelectTrigger className="h-6 text-[10px] px-1.5 bg-background/50 border-border/40">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            {options.map(opt => (
                                <SelectItem key={opt} value={opt} className="text-xs">{opt}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                )}
            </div>
        );
    };

    const renderAutoNumberField = (label: string, value: any, key: keyof HardwareConfig, autoKey: string) => {
        const isMandatory = missingSet.has(key);
        const isAuto = !!autoFields[autoKey];
        const isMissing = !isAuto && isMandatory && (!value || (typeof value === 'number' && value <= 0));
        const isRequired = requiredSet.has(key);

        return (
            <div key={key} className={cn(
                "space-y-1 rounded-sm transition-all duration-300",
                isJiggling && isMissing ? "animate-jiggle" : ""
            )}>
                <div className="flex items-center justify-between gap-2">
                    <label className={cn(
                        "text-[9px] font-medium uppercase tracking-tighter",
                        "text-muted-foreground/80"
                    )}>
                        {label}
                    </label>
                    <div className="flex items-center gap-2">
                        <span className={cn(
                            "text-[8px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border",
                            isRequired
                                ? "border-primary/30 bg-primary/10 text-primary"
                                : "border-border/40 bg-background/30 text-muted-foreground"
                        )}>
                            {isRequired ? 'Required' : 'Optional'}
                        </span>
                        <div className="flex items-center gap-1">
                            <span className="text-[8px] font-mono uppercase tracking-wider text-muted-foreground">Auto</span>
                            <Switch
                                checked={isAuto}
                                onCheckedChange={(checked) => {
                                    setAutoFields(prev => ({ ...prev, [autoKey]: checked }));
                                    updateConfig({ [key]: checked ? undefined : 0 } as Partial<HardwareConfig>);
                                }}
                                className="scale-75"
                            />
                        </div>
                        {isMissing && (
                            <TooltipProvider>
                                <Tooltip>
                                    <TooltipTrigger>
                                        <AlertCircle className="w-2.5 h-2.5 text-primary/70 stroke-[3]" />
                                    </TooltipTrigger>
                                    <TooltipContent side="top" className="text-[10px]">
                                        Mandatory field
                                    </TooltipContent>
                                </Tooltip>
                            </TooltipProvider>
                        )}
                    </div>
                </div>
                <Input
                    type="number"
                    disabled={isAuto}
                    value={isAuto ? '' : (value ?? '')}
                    onChange={(e) => updateConfig({ [key]: parseFloat(e.target.value) || 0 })}
                    className="h-6 text-[10px] px-1.5 bg-background/50 border-border/40 disabled:opacity-60"
                />
            </div>
        );
    };

    const getFamilySpecificFields = () => {
        switch (family) {
            case 'transformer':
                return (
                    <>
                        {renderField('seq_len (seq)', config.seqLen, 'seqLen')}
                        {renderField('num_layers (L)', config.numLayers, 'numLayers')}
                        {renderField('vocab_size (vocab)', config.vocabSize, 'vocabSize')}
                        {renderField('d_model (d)', config.hiddenDim, 'hiddenDim')}
                        {renderField('num_heads (h)', config.numHeads, 'numHeads')}
                        {renderAutoNumberField('head_dim (hd)', config.headDim, 'headDim', 'headDim')}
                        {renderField('ff_dim (ff)', config.ffnDim, 'ffnDim')}
                        {renderAutoNumberField('kv_heads (kv)', config.kvHeads, 'kvHeads', 'kvHeads')}
                        {renderField('bias', config.useBias, 'useBias', 'switch')}
                        {renderField('dropout (drop)', config.dropout, 'dropout')}
                        {renderField('flash', config.useFlash, 'useFlash', 'switch')}
                        {renderField('rope_theta', config.ropeTheta, 'ropeTheta')}
                        {renderAutoNumberField('max_seq_len', config.maxSeqLen, 'maxSeqLen', 'maxSeqLen')}
                        {renderField('alibi', config.useAlibi, 'useAlibi', 'switch')}
                        {renderField('relative_bias', config.useRelativeBias, 'useRelativeBias', 'switch')}
                        {renderField('use_cache', config.useCache, 'useCache', 'switch')}
                        {renderField('act_default (act)', config.activation, 'activation', 'select', ['gelu', 'relu', 'silu'])}
                    </>
                );
            case 'cnn':
                return (
                    <>
                        {renderField('Height', config.imgHeight, 'imgHeight')}
                        {renderField('Width', config.imgWidth, 'imgWidth')}
                        {renderField('In Ch', config.inChannels, 'inChannels')}
                        {renderField('Out Cls', config.numClasses, 'numClasses')}
                        {renderField('Layers', config.numLayers, 'numLayers')}
                        {renderField('Norm', config.normType, 'normType', 'select', ['batch_norm', 'layer_norm', 'none'])}
                        {renderField('Act', config.convActivation, 'convActivation', 'select', ['relu', 'silu', 'gelu'])}
                        {renderField('Pool', config.poolType, 'poolType', 'select', ['avg', 'max'])}
                    </>
                );
            case 'diffusion':
                return (
                    <>
                        {renderField('Latent H', config.imgHeight, 'imgHeight')}
                        {renderField('Latent W', config.imgWidth, 'imgWidth')}
                        {renderField('in_channels (cin)', config.inChannels, 'inChannels')}
                        {renderField('num_denoising_steps (steps)', config.numDenoisingSteps, 'numDenoisingSteps')}
                        {renderField('cfg', config.guidanceScale, 'guidanceScale')}
                        {renderField('Model Ch', config.modelChannels, 'modelChannels')}
                        {renderField('Res Blocks', config.numResBlocks, 'numResBlocks')}
                        {renderField('channel_mult', config.channelMult, 'channelMult', 'text')}
                        {renderField('attn_resolutions', config.attnResolutions, 'attnResolutions', 'text')}
                        {renderField('Drop', config.dropout, 'dropout')}
                        {renderField('Checkpoint', config.useCheckpoint, 'useCheckpoint', 'switch')}
                        {renderField('out_channels (cout)', config.outChannels, 'outChannels')}
                    </>
                );
            case 'gnn':
                return (
                    <>
                        {renderField('num_nodes', config.numNodes, 'numNodes')}
                        {renderField('num_edges', config.numEdges, 'numEdges')}
                        {renderField('node_feat_dim (in_feat)', config.nodeFeatDim, 'nodeFeatDim')}
                        {renderField('gnn_out_dim (out_dim)', config.outDim, 'outDim')}
                        {renderField('Layers', config.numLayers, 'numLayers')}
                        {renderField('edge_dim', config.edgeFeatDim, 'edgeFeatDim')}
                        {renderField('aggr', config.aggrType, 'aggrType', 'select', ['mean', 'sum', 'max'])}
                        {renderField('normalize_gnn', config.useNormalize, 'useNormalize', 'switch')}
                        {renderField('add_self_loops', config.addSelfLoops, 'addSelfLoops', 'switch')}
                        {renderField('bias', config.useBias, 'useBias', 'switch')}
                        {renderField('Drop', config.dropout, 'dropout')}
                    </>
                );
            case 'ssm':
            case 'snn':
            case 'rnn':
                return (
                    <>
                        {renderField('Seq Len', config.seqLen, 'seqLen')}
                        {renderField('Vocab', config.vocabSize, 'vocabSize')}
                        {renderField('Dim (d)', config.hiddenDim, 'hiddenDim')}
                        {renderField('Hidden (hid)', config.hiddenSize, 'hiddenSize')}
                        {renderField('Layers (L)', config.numLayers, 'numLayers')}
                        {renderField('Bidir', config.isBidirectional, 'isBidirectional', 'switch')}
                        {renderField('Bias', config.useBias, 'useBias', 'switch')}
                        {renderField('Drop', config.dropout, 'dropout')}
                        {renderField('State (d_state)', config.dState, 'dState')}
                        {renderField('DT Rank', config.dtRank, 'dtRank')}
                        {renderField('Conv Kernel', config.convKernel, 'convKernel')}
                        {renderField('Expand', config.expandFactor, 'expandFactor')}
                        {renderField('Fast Path', config.useFastPath, 'useFastPath', 'switch')}
                        {renderField('Proj Size', config.projSize, 'projSize')}
                    </>
                );
            case 'moe':
                return (
                    <>
                        {renderField('seq_len (seq)', config.seqLen, 'seqLen')}
                        {renderField('num_layers (L)', config.numLayers, 'numLayers')}
                        {renderField('vocab_size (vocab)', config.vocabSize, 'vocabSize')}
                        {renderField('d_model (d)', config.hiddenDim, 'hiddenDim')}
                        {renderField('num_heads (h)', config.numHeads, 'numHeads')}
                        {renderField('ff_dim (ff)', config.ffnDim, 'ffnDim')}
                        {renderField('num_experts (exp)', config.numExperts, 'numExperts')}
                        {renderField('top_k (topk)', config.topK, 'topK')}
                        {renderField('expert_capacity', config.expertCapacity, 'expertCapacity')}
                        {renderField('shared_expert', config.useSharedExpert, 'useSharedExpert', 'switch')}
                        {renderField('bias', config.useBias, 'useBias', 'switch')}
                        {renderField('dropout (drop)', config.dropout, 'dropout')}
                    </>
                );
            default:
                return (
                    <div className="text-[10px] text-muted-foreground/50 py-4 text-center">
                        No specific parameters for this family
                    </div>
                );
        }
    };

    return (
        <div className="overflow-hidden flex flex-col min-h-full">
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full px-3 py-2 flex items-center justify-between hover:bg-secondary/30 transition-colors group"
            >
                <div className="flex items-center gap-2">
                    <Settings2 className="w-3.5 h-3.5 text-primary" />
                    <span className="text-[10px] font-bold uppercase tracking-widest text-foreground/80">
                        Hyperparameters
                    </span>
                </div>
                {isExpanded ? <Minimize className="w-3 h-3 text-muted-foreground" /> : <Maximize className="w-3 h-3 text-muted-foreground" />}
            </button>

            {isExpanded && (
                <div className="p-3 pt-0 space-y-4 animate-in fade-in slide-in-from-top-1 duration-200">
                    {/* ─── General Grid ───────────────────────────────────────── */}
                    <div className="space-y-2">
                        <p className="text-[11px] font-extrabold tracking-[0.18em] text-foreground uppercase">Global</p>
                        <div
                            className="grid gap-2"
                            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}
                        >
                            <div className="space-y-1">
                                <div className="flex items-center justify-between">
                                    <label className="text-[9px] font-medium text-muted-foreground/80 uppercase">hardware (hw)</label>
                                    <span className="text-[8px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border border-primary/30 bg-primary/10 text-primary">Required</span>
                                </div>
                                <Select value={config.hardware} onValueChange={(v) => updateConfig({ hardware: v })}>
                                    <SelectTrigger className="h-6 text-[10px] px-1 bg-background/50 border-border/40">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="H100" className="text-xs">H100</SelectItem>
                                        <SelectItem value="A100" className="text-xs">A100</SelectItem>
                                        <SelectItem value="L40" className="text-xs">L40</SelectItem>
                                        <SelectItem value="V100" className="text-xs">V100</SelectItem>
                                        <SelectItem value="T4" className="text-xs">T4</SelectItem>
                                        <SelectItem value="RTX4090" className="text-xs">RTX4090</SelectItem>
                                        <SelectItem value="RTX4080" className="text-xs">RTX4080</SelectItem>
                                        <SelectItem value="RTX3090" className="text-xs">RTX3090</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            {renderField('batch_size (batch)', config.batchSize, 'batchSize')}
                            <div className="space-y-1">
                                <div className="flex items-center justify-between">
                                    <label className="text-[9px] font-medium text-muted-foreground/80 uppercase">precision (prec)</label>
                                    <span className="text-[8px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border border-primary/30 bg-primary/10 text-primary">Required</span>
                                </div>
                                <Select value={config.precision} onValueChange={(v) => updateConfig({ precision: v as any })}>
                                    <SelectTrigger className="h-6 text-[10px] px-1 bg-background/50 border-border/40 font-mono">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="fp32" className="text-xs font-mono">FP32</SelectItem>
                                        <SelectItem value="fp16" className="text-xs font-mono">FP16</SelectItem>
                                        <SelectItem value="bf16" className="text-xs font-mono">BF16</SelectItem>
                                        <SelectItem value="int8" className="text-xs font-mono">INT8</SelectItem>
                                        <SelectItem value="int4" className="text-xs font-mono">INT4</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            {renderField('Seed', config.seed, 'seed')}
                        </div>
                    </div>

                    {/* ─── Training Config ────────────────────────────────── */}
                    <div className="space-y-2">
                        <button
                            type="button"
                            onClick={() => setIsTrainingExpanded(!isTrainingExpanded)}
                            className="w-full flex items-center justify-between hover:bg-secondary/20 transition-colors rounded px-0.5 py-0.5"
                        >
                            <div className="flex items-center gap-1.5">
                                <TrendingUp className="w-3 h-3 text-primary/70" />
                                <p className="text-[11px] font-extrabold tracking-[0.18em] text-foreground uppercase">Training</p>
                            </div>
                            {isTrainingExpanded
                                ? <Minimize className="w-3 h-3 text-muted-foreground" />
                                : <Maximize className="w-3 h-3 text-muted-foreground" />}
                        </button>
                        {isTrainingExpanded && (
                            <div
                                className="grid gap-2 animate-in fade-in slide-in-from-top-1 duration-200"
                                style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}
                            >
                                {renderField('learning_rate', config.learningRate, 'learningRate')}
                                {renderField('num_epochs', config.numEpochs, 'numEpochs')}
                                {renderField('sequence_length', config.seqLen > 0 ? config.seqLen : 1024, 'seqLen')}
                            </div>
                        )}
                    </div>

                    {/* ─── Hardware (GPU) Config ──────────────────────────── */}
                    <div className="space-y-2">
                        <div className="flex items-center gap-1.5">
                            <Server className="w-3 h-3 text-primary/70" />
                            <p className="text-[11px] font-extrabold tracking-[0.18em] text-foreground uppercase">Hardware GPUs</p>
                        </div>
                        <div
                            className="grid gap-2"
                            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}
                        >
                            {renderField('gpu_count', config.gpuCount, 'gpuCount')}
                            {renderField('gpu_memory_gb', config.gpuMemoryGb, 'gpuMemoryGb')}
                        </div>
                    </div>

                    {/* ─── Data Config ───────────────────────────────────── */}
                    <div className="space-y-2">
                        <button
                            type="button"
                            onClick={() => setIsDataExpanded(!isDataExpanded)}
                            className="w-full flex items-center justify-between hover:bg-secondary/20 transition-colors rounded px-0.5 py-0.5"
                        >
                            <div className="flex items-center gap-1.5">
                                <Database className="w-3 h-3 text-primary/70" />
                                <p className="text-[11px] font-extrabold tracking-[0.18em] text-foreground uppercase">Data</p>
                            </div>
                            {isDataExpanded
                                ? <Minimize className="w-3 h-3 text-muted-foreground" />
                                : <Maximize className="w-3 h-3 text-muted-foreground" />}
                        </button>
                        {isDataExpanded && (
                            <div
                                className="grid gap-2 animate-in fade-in slide-in-from-top-1 duration-200"
                                style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}
                            >
                                {renderField('dataset_size', config.datasetSize, 'datasetSize')}
                                {renderField('vocab_size', config.vocabSize, 'vocabSize')}
                                {renderField('num_classes', config.numClasses ?? 0, 'numClasses')}
                            </div>
                        )}
                    </div>

                    {/* Family Dynamic Section */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <p className="text-[11px] font-extrabold tracking-[0.18em] text-foreground uppercase">Env</p>
                            <TooltipProvider>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Sliders className="w-3 h-3 text-muted-foreground/50 cursor-help" />
                                    </TooltipTrigger>
                                    <TooltipContent side="right" className="text-[10px]">
                                        Parameters only active for {family} family
                                    </TooltipContent>
                                </Tooltip>
                            </TooltipProvider>
                        </div>
                        <div
                            className="grid gap-2"
                            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}
                        >
                            {getFamilySpecificFields()}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
