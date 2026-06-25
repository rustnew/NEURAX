import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import {
  X,
  ChevronUp,
  ChevronDown,
  GripHorizontal,
  Activity,
  AlertTriangle,
  Binary,
  Clock3,
  HardDrive,
  Layers3,
} from 'lucide-react';
import { AnalysisResult, CanvasNode, ACTIVATION_OPTIONS, NodeGroup, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';
import { getPluginLayers } from '@/plugins/registry.ts';
import { getBlockDefaults } from '@/utils/blockDefaults.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select.tsx';

interface InspectorPanelProps {
  node: CanvasNode | null;
  group: NodeGroup | null;
  nodes?: CanvasNode[];
  selectionRevision?: number;
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  warnings?: Warning[];
  onJumpToWarnings?: () => void;
  onUpdateNode: (id: string, updates: Partial<CanvasNode>) => void;
  onUpdateGroup: (id: string, updates: Partial<NodeGroup>) => void;
  onClose: () => void;
  onDelete: (id: string) => void;
  onDeleteGroup: (id: string) => void;
  selectedArchitecture?: ArchitectureFamily;
}

// Params that should render as dropdowns
const DROPDOWN_PARAMS: Record<string, { label: string; options: { value: string; label: string }[] }> = {
  activation: {
    label: 'Activation',
    options: ACTIVATION_OPTIONS.map(a => ({ value: a.value, label: a.label })),
  },
  attentionType: {
    label: 'Attention Type',
    options: [
      { value: 'sdpa', label: 'Scaled Dot-Product' },
      { value: 'flash', label: 'FlashAttention' },
      { value: 'linear', label: 'Linear Attention' },
      { value: 'mqa', label: 'Multi-Query (MQA)' },
      { value: 'gqa', label: 'Grouped-Query (GQA)' },
    ],
  },
  routingMode: {
    label: 'Routing Mode',
    options: [
      { value: 'softmax', label: 'Softmax' },
      { value: 'noisy_top_k', label: 'Noisy Top-K' },
      { value: 'expert_choice', label: 'Expert Choice' },
    ],
  },
  aggregation: {
    label: 'Aggregation',
    options: [
      { value: 'sum', label: 'Sum' },
      { value: 'mean', label: 'Mean' },
      { value: 'max', label: 'Max' },
    ],
  },
  distribution: {
    label: 'Distribution',
    options: [
      { value: 'categorical', label: 'Categorical' },
      { value: 'gaussian', label: 'Gaussian' },
    ],
  },
  encoding: {
    label: 'Encoding',
    options: [
      { value: 'rate', label: 'Rate Coding' },
      { value: 'temporal', label: 'Temporal Coding' },
      { value: 'latency', label: 'Latency Coding' },
    ],
  },
  schedule: {
    label: 'Schedule',
    options: [
      { value: 'linear', label: 'Linear' },
      { value: 'cosine', label: 'Cosine' },
      { value: 'sigmoid', label: 'Sigmoid' },
    ],
  },
  method: {
    label: 'Method',
    options: [
      { value: 'zoh', label: 'Zero-Order Hold' },
      { value: 'bilinear', label: 'Bilinear' },
    ],
  },
  padding: {
    label: 'Padding',
    options: [
      { value: 'same', label: 'Same' },
      { value: 'valid', label: 'Valid' },
    ],
  },
  type: {
    label: 'Type',
    options: [
      { value: 'avg', label: 'Average' },
      { value: 'max', label: 'Max' },
      { value: 'sum', label: 'Sum' },
      { value: 'mean', label: 'Mean' },
    ],
  },
};

// Params that are boolean toggles
const BOOLEAN_PARAMS = ['causal', 'bidirectional', 'normalize', 'flash'];

// Params that are read-only display
const READONLY_PARAMS = ['shape'];

function formatMetricNumber(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0';
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(value >= 100 ? 0 : 2);
}

function formatBytesCompact(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${Math.round(bytes)} B`;
}

function formatLatencyCompact(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms) || ms <= 0) return 'n/a';
  if (ms < 1) return `${(ms * 1000).toFixed(1)} us`;
  return `${ms.toFixed(2)} ms`;
}

function MetricTile({
  icon: Icon,
  label,
  value,
  tone = 'default',
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  tone?: 'default' | 'accent' | 'warning';
}) {
  const toneClass =
    tone === 'accent'
      ? 'border-primary/25 bg-primary/10'
      : tone === 'warning'
        ? 'border-warning/30 bg-warning/10'
        : 'border-border/60 bg-secondary/25';

  return (
    <div className={`rounded-xl border p-3 ${toneClass}`}>
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
          <div className="mt-1 text-sm font-semibold text-foreground">{value}</div>
        </div>
        <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-border/60 bg-background/60">
          <Icon className="h-4 w-4 text-primary" />
        </div>
      </div>
    </div>
  );
}

export function InspectorPanel({
  node,
  group,
  nodes = [],
  selectionRevision,
  analysis,
  perLayer = [],
  warnings = [],
  onJumpToWarnings,
  onUpdateNode,
  onUpdateGroup,
  onClose,
  onDelete,
  onDeleteGroup,
  selectedArchitecture = 'transformer'
}: InspectorPanelProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [panelHeight, setPanelHeight] = useState(192);
  const [compilerPaneWidth, setCompilerPaneWidth] = useState(420);
  const isResizing = useRef(false);
  const isWidthResizing = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);
  const startX = useRef(0);
  const startWidth = useRef(420);

  // Auto-expand when a node or group is selected (even the same one)
  useEffect(() => {
    if (node || group) {
      setIsMinimized(false);
    }
  }, [node?.id, group?.id, selectionRevision]);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    startY.current = e.clientY;
    startHeight.current = panelHeight;

    const handleMouseMove = (ev: MouseEvent) => {
      if (!isResizing.current) return;
      const delta = startY.current - ev.clientY;
      const newHeight = Math.max(100, Math.min(600, startHeight.current + delta));
      setPanelHeight(newHeight);
    };

    const handleMouseUp = () => {
      isResizing.current = false;
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [panelHeight]);

  const clampCompilerPaneWidth = useCallback((width: number) => {
    return Math.max(300, Math.min(720, width));
  }, []);

  const handleWidthResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isWidthResizing.current = true;
    startX.current = e.clientX;
    startWidth.current = compilerPaneWidth;

    const handleMouseMove = (ev: MouseEvent) => {
      if (!isWidthResizing.current) return;
      const delta = startX.current - ev.clientX;
      setCompilerPaneWidth(clampCompilerPaneWidth(startWidth.current + delta));
    };

    const handleMouseUp = () => {
      isWidthResizing.current = false;
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [clampCompilerPaneWidth, compilerPaneWidth]);

  const allLayers = getPluginLayers(selectedArchitecture);
  const config = node ? allLayers.find(l => l.type === node.type) : undefined;

  const selectedLayerMetrics = useMemo(() => {
    if (!node) return null;

    const row = perLayer.find((entry) => entry.id === node.id || entry.name === node.id || entry.name === node.name) ?? null;
    const paramsCount = row?.params ?? 0;
    const latencyMs = analysis?.perLayerLatency?.[node.id]
      ?? analysis?.perLayerLatency?.[node.name]
      ?? null;
    const vramBytes = analysis?.perLayerVram?.[node.id]
      ?? analysis?.perLayerVram?.[node.name]
      ?? null;
    const nodeWarnings = warnings.filter((warning) => warning.nodeId === node.id);

    return {
      row,
      paramsCount,
      latencyMs,
      vramBytes,
      nodeWarnings,
    };
  }, [node, perLayer, analysis, warnings]);

  const compilerWarnings = useMemo(() => (
    warnings.filter((warning) => /custom operations|estimated flops|unsupported/i.test(warning.message))
  ), [warnings]);

  if (!node && !group) {
    if (isMinimized) return null;
    return (
      <div className="bg-card border-t border-border flex items-center justify-center h-48 transition-all duration-200">
        <span className="text-sm text-muted-foreground/50">
          Select a layer or group to inspect
        </span>
      </div>
    );
  }

  const handleParamChange = (key: string, value: string | boolean) => {
    if (!node) return;
    if (typeof value === 'boolean') {
      onUpdateNode(node.id, { params: { ...node.params, [key]: value } });
      return;
    }
    const numValue = parseFloat(value);
    onUpdateNode(node.id, {
      params: {
        ...node.params,
        [key]: isNaN(numValue) ? value : numValue,
      },
    });
  };

  // Check if block supports activation (from config or has activation param)
  const supportsActivation = node ? (config?.hasActivation || 'activation' in node.params) : false;

  // Ensure activation param exists if block supports it
  const params = node ? { ...getBlockDefaults(node.type), ...(config?.defaultParams ?? {}), ...node.params } : {};
  if (supportsActivation && !('activation' in params)) {
    params.activation = 'none';
  }

  return (
    <div
      className={`bg-card border-t border-border flex flex-col ${isMinimized ? 'h-0 overflow-hidden' : ''
        }`}
      style={isMinimized ? undefined : { height: panelHeight }}
    >
      {/* Resize Handle */}
      {!isMinimized && (
        <div
          className="h-2 cursor-row-resize flex items-center justify-center hover:bg-primary/10 transition-colors shrink-0 group"
          onMouseDown={handleResizeStart}
        >
          <GripHorizontal className="w-4 h-3 text-muted-foreground/40 group-hover:text-primary/60" />
        </div>
      )}
      {/* Header */}
      <div className="h-10 px-4 flex items-center justify-between border-b border-border bg-panel-header shrink-0">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="w-6 h-6 text-muted-foreground"
            onClick={() => setIsMinimized(!isMinimized)}
          >
            {isMinimized ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </Button>
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Inspector
          </span>
          <span className="text-xs text-foreground font-medium">
            — {node?.name || group?.name}
          </span>
          {config?.category && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary/80 border border-primary/20 uppercase tracking-wider">
              {config.category}
            </span>
          )}
          {group && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary/80 border border-primary/20 uppercase tracking-wider">
              Group ({group.nodeIds.length} blocks)
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 text-destructive hover:text-destructive hover:bg-destructive/10"
            onClick={() => node ? onDelete(node.id) : group && onDeleteGroup(group.id)}
          >
            Delete
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="w-6 h-6 text-muted-foreground"
            onClick={onClose}
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Content */}
      {!isMinimized && (
        <div className="flex-1 overflow-hidden p-4">
          {group ? (
            <div className="h-full overflow-auto scrollbar-thin pr-1 space-y-4">
              {/* Group-level controls */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                {/* Group Name */}
                <div className="space-y-1.5">
                  <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Group Name
                  </Label>
                  <Input
                    value={group.name}
                    onChange={(e) => onUpdateGroup(group.id, { name: e.target.value })}
                    className="h-8 text-xs bg-secondary border-border"
                  />
                </div>

                {/* Repeat Count */}
                <div className="space-y-1.5">
                  <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Repeat Count (×N)
                  </Label>
                  <Input
                    type="number"
                    value={group.repeatCount}
                    onChange={(e) => onUpdateGroup(group.id, { repeatCount: parseInt(e.target.value) || 1 })}
                    className="h-8 text-xs font-mono bg-secondary border-border"
                    min={1}
                  />
                </div>
              </div>

              {/* Per-block parameter editors */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Block Parameters — shared across all ×{group.repeatCount}
                  </Label>
                  <div className="flex-1 h-px bg-border/60" />
                </div>

                {nodes
                  .filter(n => group.nodeIds.includes(n.id))
                  .map((blockNode) => {
                    const blockConfig = getPluginLayers(selectedArchitecture).find(l => l.type === blockNode.type);
                    // Merge catalog default params first, then stored params override — 
                    // this ensures ALL catalog-defined params are shown even if not yet persisted
                    const blockParams: Record<string, unknown> = {
                      ...getBlockDefaults(blockNode.type),
                      ...(blockConfig?.defaultParams ?? {}),
                      ...blockNode.params,
                    };
                    if (blockConfig?.hasActivation && !('activation' in blockParams)) {
                      blockParams.activation = 'none';
                    }

                    return (
                      <div
                        key={blockNode.id}
                        className="rounded-lg border border-border/60 bg-secondary/30 overflow-hidden"
                      >
                        {/* Block header */}
                        <div className="flex items-center gap-2 px-3 py-2 bg-secondary/60 border-b border-border/40">
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/15 text-primary/80 border border-primary/20 uppercase tracking-wider font-medium">
                            {blockConfig?.name || blockNode.type}
                          </span>
                          <span className="text-xs font-medium text-foreground truncate">{blockNode.name}</span>
                        </div>

                        {/* Block params */}
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 p-3">
                          {/* Name */}
                          <div className="space-y-1">
                            <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">Name</Label>
                            <Input
                              value={blockNode.name}
                              onChange={(e) => onUpdateNode(blockNode.id, { name: e.target.value })}
                              className="h-7 text-xs bg-secondary border-border"
                            />
                          </div>

                          {Object.entries(blockParams).map(([key, value]) => {
                            const dropdownConfig = DROPDOWN_PARAMS[key];
                            if (dropdownConfig) {
                              return (
                                <div key={key} className="space-y-1">
                                  <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                                    {dropdownConfig.label}
                                  </Label>
                                  <Select
                                    value={String(value)}
                                    onValueChange={(v) => {
                                      const numV = parseFloat(v);
                                      onUpdateNode(blockNode.id, { params: { ...blockNode.params, [key]: isNaN(numV) ? v : numV } });
                                    }}
                                  >
                                    <SelectTrigger className="h-7 text-xs bg-secondary border-border">
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      {dropdownConfig.options.map((opt) => (
                                        <SelectItem key={opt.value} value={opt.value} className="text-xs">
                                          {opt.label}
                                        </SelectItem>
                                      ))}
                                    </SelectContent>
                                  </Select>
                                </div>
                              );
                            }

                            if (BOOLEAN_PARAMS.includes(key)) {
                              return (
                                <div key={key} className="space-y-1">
                                  <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">{key}</Label>
                                  <Button
                                    variant={value ? 'default' : 'outline'}
                                    size="sm"
                                    className="h-7 w-full text-xs"
                                    onClick={() => onUpdateNode(blockNode.id, { params: { ...blockNode.params, [key]: !value } })}
                                  >
                                    {value ? 'Yes' : 'No'}
                                  </Button>
                                </div>
                              );
                            }

                            if (READONLY_PARAMS.includes(key)) {
                              return (
                                <div key={key} className="space-y-1">
                                  <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">{key}</Label>
                                  <div className="h-7 px-2 flex items-center bg-secondary rounded-md border border-border text-xs font-mono text-muted-foreground">
                                    {String(value)}
                                  </div>
                                </div>
                              );
                            }

                            return (
                              <div key={key} className="space-y-1">
                                <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">{key}</Label>
                                <Input
                                  value={String(value)}
                                  onChange={(e) => {
                                    const numV = parseFloat(e.target.value);
                                    onUpdateNode(blockNode.id, { params: { ...blockNode.params, [key]: isNaN(numV) ? e.target.value : numV } });
                                  }}
                                  className="h-7 text-xs font-mono bg-secondary border-border"
                                />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          ) : node ? (
            <div className="h-full min-h-0 overflow-hidden rounded-xl border border-border/50 bg-background/20">
              <div
                className="grid h-full min-h-0 grid-cols-1 gap-0 xl:grid-cols-[minmax(0,1fr)_10px_minmax(300px,var(--compiler-pane-width))]"
                style={{
                  ['--compiler-pane-width' as string]: `${compilerPaneWidth}px`,
                }}
              >
                <div className="min-w-0 min-h-0 overflow-auto scrollbar-thin p-4 pr-5">
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                  {/* Name */}
                  <div className="space-y-1.5">
                    <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Name
                    </Label>
                    <Input
                      value={node.name}
                      onChange={(e) => onUpdateNode(node.id, { name: e.target.value })}
                      className="h-8 text-xs bg-secondary border-border"
                    />
                  </div>

                  {/* Type */}
                  <div className="space-y-1.5">
                    <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Type
                    </Label>
                    <div className="h-8 px-2 flex items-center bg-secondary rounded-md border border-border text-xs text-muted-foreground">
                      {config?.name || node.type}
                    </div>
                  </div>

                  {/* Input Shape */}
                  <div className="space-y-1.5">
                    <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Input Shape
                    </Label>
                    <div className="h-8 px-2 flex items-center bg-secondary rounded-md border border-border text-xs font-mono text-success/80">
                      {node.inputShape || 'auto'}
                    </div>
                  </div>

                  {/* Output Shape */}
                  <div className="space-y-1.5">
                    <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Output Shape
                    </Label>
                    <div className="h-8 px-2 flex items-center bg-secondary rounded-md border border-border text-xs font-mono text-primary/80">
                      {node.outputShape || 'auto'}
                    </div>
                  </div>

                  {/* Dynamic Parameters */}
                  {Object.entries(params).map(([key, value]) => {
                    const dropdownConfig = DROPDOWN_PARAMS[key];
                    if (dropdownConfig) {
                      return (
                        <div key={key} className="space-y-1.5">
                          <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            {dropdownConfig.label}
                          </Label>
                          <Select
                            value={String(value)}
                            onValueChange={(v) => handleParamChange(key, v)}
                          >
                            <SelectTrigger className="h-8 text-xs bg-secondary border-border">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {dropdownConfig.options.map((opt) => (
                                <SelectItem key={opt.value} value={opt.value} className="text-xs">
                                  {opt.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      );
                    }

                    if (BOOLEAN_PARAMS.includes(key)) {
                      return (
                        <div key={key} className="space-y-1.5">
                          <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            {key}
                          </Label>
                          <Button
                            variant={value ? 'default' : 'outline'}
                            size="sm"
                            className="h-8 w-full text-xs"
                            onClick={() => handleParamChange(key, !value)}
                          >
                            {value ? 'Yes' : 'No'}
                          </Button>
                        </div>
                      );
                    }

                    if (READONLY_PARAMS.includes(key)) {
                      return (
                        <div key={key} className="space-y-1.5">
                          <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            {key}
                          </Label>
                          <div className="h-8 px-2 flex items-center bg-secondary rounded-md border border-border text-xs font-mono text-muted-foreground">
                            {String(value)}
                          </div>
                        </div>
                      );
                    }

                    return (
                      <div key={key} className="space-y-1.5">
                        <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                          {key}
                        </Label>
                        <Input
                          value={String(value)}
                          onChange={(e) => handleParamChange(key, e.target.value)}
                          className="h-8 text-xs font-mono bg-secondary border-border"
                        />
                      </div>
                    );
                  })}
                </div>
                </div>

                <div
                  className="hidden xl:flex min-h-0 cursor-col-resize items-center justify-center border-l border-r border-border/50 bg-background/30 hover:bg-primary/5 transition-colors"
                  onMouseDown={handleWidthResizeStart}
                  title="Resize compiler pane"
                >
                  <div className="h-12 w-px rounded-full bg-muted-foreground/40" />
                </div>

                <aside className="min-w-0 min-h-0 overflow-auto border-t border-border/50 bg-gradient-to-b from-secondary/35 to-secondary/10 scrollbar-thin xl:border-t-0">
                <div className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b border-border/60 bg-background/80 px-4 py-3 backdrop-blur-sm">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-primary" />
                    <div>
                      <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Compiler View</div>
                      <div className="text-xs font-medium text-foreground">Layer-local telemetry</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {compilerWarnings.length > 0 && (
                      <button
                        type="button"
                        onClick={onJumpToWarnings}
                        className="rounded-full border border-amber-400/30 bg-amber-500/10 px-2.5 py-1 text-[10px] font-medium uppercase tracking-wide text-amber-200 transition-colors hover:bg-amber-500/15"
                      >
                        {compilerWarnings.length} warning{compilerWarnings.length === 1 ? '' : 's'}
                      </button>
                    )}
                    <div className="rounded-full border border-primary/20 bg-primary/10 px-2 py-0.5 text-[10px] uppercase tracking-wide text-primary">
                      live
                    </div>
                  </div>
                </div>

                <div className="space-y-4 p-4">
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                    <MetricTile
                      icon={Binary}
                      label="Compiler FLOPs"
                      value={selectedLayerMetrics?.row?.flops || 'n/a'}
                      tone="accent"
                    />
                    <MetricTile
                      icon={Clock3}
                      label="Estimated Latency"
                      value={formatLatencyCompact(selectedLayerMetrics?.latencyMs)}
                    />
                    <MetricTile
                      icon={HardDrive}
                      label="Activation VRAM"
                      value={formatBytesCompact(selectedLayerMetrics?.vramBytes ?? 0)}
                    />
                    <MetricTile
                      icon={Layers3}
                      label="Parameter Count"
                      value={formatMetricNumber(selectedLayerMetrics?.paramsCount ?? 0)}
                    />
                  </div>

                  <div className="rounded-xl border border-border/60 bg-background/40 p-3">
                    <div className="mb-3 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Compiler Signals</div>
                    <div className="space-y-2 text-xs">
                      <div className="flex items-center justify-between gap-4">
                        <span className="text-muted-foreground">Tensor route</span>
                        <span className="font-mono text-right text-foreground">{`${node.inputShape || 'auto'} -> ${node.outputShape || 'auto'}`}</span>
                      </div>
                      <div className="flex items-center justify-between gap-4">
                        <span className="text-muted-foreground">Model phase</span>
                        <span className="text-right text-foreground">{analysis?.compilation?.current_phase || 'Idle'}</span>
                      </div>
                      <div className="flex items-center justify-between gap-4">
                        <span className="text-muted-foreground">Arithmetic intensity</span>
                        <span className="font-mono text-right text-foreground">
                          {analysis && Number.isFinite(analysis.arithmeticIntensity)
                            ? analysis.arithmeticIntensity.toFixed(2)
                            : 'n/a'}
                        </span>
                      </div>
                      <div className="flex items-center justify-between gap-4">
                        <span className="text-muted-foreground">Precision</span>
                        <span className="text-right text-foreground uppercase">{analysis?.selectedPrecision || 'n/a'}</span>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-border/60 bg-background/30 p-3">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Diagnostics</div>
                      <div className="text-[10px] text-muted-foreground">
                        {selectedLayerMetrics?.nodeWarnings.length || 0} bound to node
                      </div>
                    </div>
                    {selectedLayerMetrics?.nodeWarnings.length ? (
                      <div className="space-y-2">
                        {selectedLayerMetrics.nodeWarnings.slice(0, 3).map((warning) => (
                          <div
                            key={warning.id}
                            className="rounded-lg border border-warning/20 bg-warning/10 px-3 py-2"
                          >
                            <div className="flex items-start gap-2">
                              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 text-warning" />
                              <div className="min-w-0">
                                <div className="text-xs leading-snug text-foreground">{warning.message}</div>
                                {warning.code ? (
                                  <div className="mt-1 font-mono text-[10px] text-muted-foreground">{warning.code}</div>
                                ) : null}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="rounded-lg border border-dashed border-border/70 bg-secondary/20 px-3 py-4 text-xs text-muted-foreground">
                        No compiler diagnostics are currently attached to this block.
                      </div>
                    )}
                  </div>
                </div>
	              </aside>
	            </div>
	          </div>
	          ) : null}
        </div>
      )}
    </div>
  );
}
