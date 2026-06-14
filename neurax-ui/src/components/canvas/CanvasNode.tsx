import {
  ArrowRightToLine,
  ArrowRightFromLine,
  ArrowLeftRight,
  ArrowRight,
  Layers,
  Grid3X3,
  Zap,
  Activity,
  Focus,
  AlignCenter,
  AlignJustify,
  GitBranch,
  GitMerge,
  Box,
  RotateCw,
  RotateCcw,
  Clock,
  Share2,
  Wand2,
  Shield,
  Workflow,
  Network,
  Gamepad2,
  Brain,
  Repeat,
  BarChart3,
  Shuffle,
  Move3d,
  Target,
  Waves,
  PlayCircle,
  Sparkles,
  Search,
  Scale,
  MessageSquare,
  Users,
  Database,
  Eye,
  TrendingUp,
  Maximize,
  Blend,
  FlaskConical,
  Sigma,
  X,
  Link,
  Copy,
  Trash2,
  Plus,
  Minus,
  Pencil,
  Ungroup,
  Filter,
  Tag,
  Fingerprint,
  Code,
  ShieldCheck,
  ShieldAlert,
  Lock,
  Scissors,
  Image,
  Calculator,
  Video,
  Stethoscope,
  Globe,
  LucideIcon
} from 'lucide-react';
import { CanvasNode as CanvasNodeType } from '@/types/architecture.ts';
import { PLUGINS } from '@/plugins/registry.ts';
import { getBlockDefaults } from '@/utils/blockDefaults.ts';
import { cn } from '@/lib/utils.ts';

const iconMap: Record<string, LucideIcon> = {
  ArrowRightToLine,
  ArrowRightFromLine,
  ArrowLeftRight,
  ArrowRight,
  Layers,
  Grid3X3,
  Zap,
  Activity,
  Focus,
  AlignCenter,
  AlignJustify,
  GitBranch,
  GitMerge,
  Box,
  RotateCw,
  RotateCcw,
  Clock,
  Share2,
  Wand2,
  Shield,
  Workflow,
  Network,
  Gamepad2,
  Brain,
  Repeat,
  BarChart3,
  Shuffle,
  Move3d,
  Target,
  Waves,
  PlayCircle,
  Sparkles,
  Search,
  Scale,
  MessageSquare,
  Users,
  Database,
  Eye,
  TrendingUp,
  Maximize,
  Blend,
  FlaskConical,
  Sigma,
  X,
  Link,
  Copy,
  Trash2,
  Plus,
  Minus,
  Pencil,
  Ungroup,
  Filter,
  Tag,
  Fingerprint,
  Code,
  ShieldCheck,
  ShieldAlert,
  Lock,
  Scissors,
  Image,
  Calculator,
  Video,
  Stethoscope,
  Globe,
};

const ALL_LAYER_CONFIGS = Object.values(PLUGINS).flatMap((plugin) => plugin.layers);

interface CanvasNodeProps {
  node: CanvasNodeType;
  isSelected: boolean;
  isMultiSelected?: boolean;
  onSelect: () => void;
  onClick?: (e: React.MouseEvent) => void;
  onDragStart: (e: React.MouseEvent) => void;
  onConnectionStart?: (isOutput: boolean, e: React.MouseEvent) => void;
  onConnectionEnd?: (e: React.MouseEvent) => void;
  isConnecting?: boolean;
}

export function CanvasNode({
  node,
  isSelected,
  isMultiSelected = false,
  onSelect,
  onClick,
  onDragStart,
  onConnectionStart,
  onConnectionEnd,
  isConnecting
}: CanvasNodeProps) {
  const config = ALL_LAYER_CONFIGS.find((layer) => layer.type === node.type);
  const mergedParams: Record<string, unknown> = {
    ...getBlockDefaults(node.type),
    ...(config?.defaultParams ?? {}),
    ...(node.params ?? {}),
  };

  if ((config?.hasActivation || 'activation' in mergedParams) && !('activation' in mergedParams)) {
    mergedParams.activation = 'none';
  }

  const getIcon = (iconName: string) => {
    const Icon = iconMap[iconName];
    return Icon ? <Icon className="w-5 h-5" /> : null;
  };

  const paramEntries = Object.entries(mergedParams);
  const visibleParamEntries = (() => {
    if (paramEntries.length <= 4) return paramEntries;

    const activationEntry = paramEntries.find(([key]) => key === 'activation');
    const leadingEntries = paramEntries
      .filter(([key]) => key !== 'activation')
      .slice(0, activationEntry ? 3 : 4);

    return activationEntry ? [...leadingEntries, activationEntry] : leadingEntries;
  })();
  const shapeEntries = [
    node.inputShape ? { label: 'in', value: node.inputShape, tone: 'success' as const } : null,
    node.outputShape ? { label: 'out', value: node.outputShape, tone: 'primary' as const } : null,
  ].filter(Boolean) as Array<{ label: 'in' | 'out'; value: string; tone: 'success' | 'primary' }>;
  const hasBody = paramEntries.length > 0 || shapeEntries.length > 0;

  const formatParams = () => {
    if (visibleParamEntries.length === 0) return null;
    return visibleParamEntries.map(([key, value]) => (
      <div key={key} className="flex items-center justify-between gap-2 text-xs">
        <span className="text-muted-foreground/70">{key}</span>
        <span className="font-mono text-foreground/90">{String(value)}</span>
      </div>
    ));
  };

  return (
    <div
      className={cn(
        "absolute select-none transition-all duration-200 ease-out",
        "bg-card",
        "border-2 border-border rounded-2xl",
        "shadow-lg",
        "hover:shadow-xl hover:border-primary/40",
        "cursor-move",
        "min-w-[200px]",
        isSelected && "border-primary shadow-xl ring-4 ring-primary/20",
        isMultiSelected && "ring-4 ring-primary/30 ring-offset-2 ring-offset-background"
      )}
      style={{
        left: node.x,
        top: node.y,
      }}
      onClick={(e) => {
        e.stopPropagation();
        if (onClick) {
          onClick(e);
        } else {
          onSelect();
        }
      }}
      onMouseDown={onDragStart}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-center gap-3 px-4 py-3 bg-secondary/30",
          hasBody ? "border-b border-border rounded-t-2xl" : "rounded-2xl",
        )}
      >
        <div
          className={cn(
            "w-10 h-10 rounded-xl flex items-center justify-center",
            !config?.color && "bg-primary/10 border-primary/30 text-primary",
            "border"
          )}
          style={config?.color ? {
            backgroundColor: `${config.color}20`, // 12% opacity
            borderColor: `${config.color}50`,
            color: config.color
          } : {}}
        >
          {config && getIcon(config.icon)}
        </div>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-semibold truncate block text-foreground">{node.name}</span>
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{node.type}</span>
        </div>
      </div>

      {/* Parameters */}
      {hasBody && (
        <div className="px-4 py-3 space-y-1.5 bg-card rounded-b-2xl">
          {formatParams()}

          {shapeEntries.length > 0 && (
            <div className={cn("flex flex-col gap-1", paramEntries.length > 0 && "pt-2 mt-2 border-t border-border")}>
              {shapeEntries.map((shape) => (
                <div key={shape.label} className="flex items-center gap-2 text-[11px]">
                  <span className="text-muted-foreground/60 w-8">{shape.label}</span>
                  <code
                    className={cn(
                      "font-mono px-2 py-0.5 rounded-md",
                      shape.tone === 'success'
                        ? "text-success bg-success/10"
                        : "text-primary bg-primary/10",
                    )}
                  >
                    {shape.value}
                  </code>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Input connection point */}
      <div
        className={cn(
          "absolute -left-3 top-1/2 -translate-y-1/2",
          "w-6 h-6 rounded-full",
          "bg-card",
          "border-2 border-border",
          "cursor-pointer transition-all duration-200",
          "flex items-center justify-center",
          "hover:scale-125 hover:border-primary hover:bg-secondary",
          isConnecting && "animate-pulse border-primary scale-110"
        )}
        onMouseDown={(e) => {
          e.stopPropagation();
          onConnectionStart?.(false, e);
        }}
        onMouseUp={(e) => {
          e.stopPropagation();
          onConnectionEnd?.(e);
        }}
      >
        <div className="w-2 h-2 rounded-full bg-primary" />
      </div>

      {/* Output connection point */}
      <div
        className={cn(
          "absolute -right-3 top-1/2 -translate-y-1/2",
          "w-6 h-6 rounded-full",
          "bg-card",
          "border-2 border-border",
          "cursor-pointer transition-all duration-200",
          "flex items-center justify-center",
          "hover:scale-125 hover:border-primary hover:bg-secondary",
          isConnecting && "animate-pulse border-primary scale-110"
        )}
        onMouseDown={(e) => {
          e.stopPropagation();
          onConnectionStart?.(true, e);
        }}
        onMouseUp={(e) => {
          e.stopPropagation();
          onConnectionEnd?.(e);
        }}
      >
        <div className="w-2 h-2 rounded-full bg-primary" />
      </div>
    </div>
  );
}
