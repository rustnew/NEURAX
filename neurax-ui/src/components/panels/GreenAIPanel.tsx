import { useState, useMemo } from 'react';
import { 
  Leaf, 
  Clock, 
  Database, 
  Download,
  Info,
  Check,
  Sparkles,
  Cpu,
  Activity,
  ChevronDown,
  ChevronRight,
  Copy
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Slider } from '@/components/ui/slider.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Progress } from '@/components/ui/progress.tsx';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group.tsx';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible.tsx';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { cn } from '@/lib/utils.ts';
import { CanvasNode, Connection } from '@/types/architecture.ts';
import {
  InitializationMethod,
  InitializationConfig,
  INITIALIZATION_METHODS,
  initializeArchitecture,
  generateGreenAIONNX,
  getRecommendedInit,
} from '@/utils/weightInitialization.ts';

interface GreenAIPanelProps {
  nodes: CanvasNode[];
  connections: Connection[];
  modelName: string;
}

export function GreenAIPanel({ nodes, connections, modelName }: GreenAIPanelProps) {
  const { toast } = useToast();
  const [selectedMethod, setSelectedMethod] = useState<InitializationMethod>('xavier_normal');
  const [gain, setGain] = useState([1.0]);
  const [sparsity, setSparsity] = useState([0.9]);
  const [copied, setCopied] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Get recommended method based on architecture
  const recommendedMethod = useMemo(() => getRecommendedInit(nodes), [nodes]);

  // Calculate initialization and metrics
  const architecture = useMemo(() => {
    if (nodes.length === 0) return null;
    
    const config: InitializationConfig = {
      method: selectedMethod,
      gain: gain[0],
      sparsity: selectedMethod === 'sparse' ? sparsity[0] : undefined,
    };
    
    return initializeArchitecture(nodes, connections, config, modelName);
  }, [nodes, connections, selectedMethod, gain, sparsity, modelName]);

  const handleExportONNX = () => {
    if (!architecture) {
      toast({
        title: "No Architecture",
        description: "Add layers to the canvas first",
        variant: "destructive",
      });
      return;
    }

    const code = generateGreenAIONNX(architecture);
    
    // Create and download file
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelName.toLowerCase()}_green_ai.py`;
    a.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Green AI Export Complete",
      description: "Pre-initialized model exported successfully",
    });
  };

  const handleCopyCode = () => {
    if (!architecture) return;
    const code = generateGreenAIONNX(architecture);
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast({
      title: "Copied",
      description: "Green AI code copied to clipboard",
    });
  };

  const trainableLayers = nodes.filter(n => 
    ['dense', 'conv2d', 'attention', 'transformer', 'layernorm', 'batchnorm'].includes(n.type)
  );

  return (
    <div className="space-y-6">
      {/* Header with Green Badge */}
      <div className="flex items-center gap-3 pb-4 border-b border-border">
        <div className="w-12 h-12 rounded-xl bg-success/10 flex items-center justify-center">
          <Leaf className="w-6 h-6 text-success" />
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            Green AI Initialization
            <Badge className="bg-success/20 text-success border-0 text-[10px]">
              ECO-FRIENDLY
            </Badge>
          </h3>
          <p className="text-xs text-muted-foreground">
            Smart weight initialization to reduce training time, energy & data requirements
          </p>
        </div>
      </div>

      {/* Quick Stats */}
      {architecture && (
        <div className="grid grid-cols-3 gap-3">
          <MetricCard
            icon={Clock}
            label="Epochs Saved"
            value={`~${architecture.metrics.estimatedEpochsSaved}`}
            color="text-success"
          />
          <MetricCard
            icon={Cpu}
            label="Hours Saved"
            value={`~${architecture.metrics.computeHoursSaved}h`}
            color="text-info"
          />
          <MetricCard
            icon={Database}
            label="Data Efficiency"
            value={`+${architecture.metrics.datasetEfficiency}%`}
            color="text-warning"
          />
        </div>
      )}

      {/* Initialization Method Selection */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label className="text-sm font-medium">Initialization Method</Label>
          {recommendedMethod && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-[10px] text-success hover:text-success"
              onClick={() => setSelectedMethod(recommendedMethod)}
            >
              <Sparkles className="w-3 h-3 mr-1" />
              Use Recommended
            </Button>
          )}
        </div>

        <RadioGroup
          value={selectedMethod}
          onValueChange={(v) => setSelectedMethod(v as InitializationMethod)}
          className="grid grid-cols-2 gap-2"
        >
          {INITIALIZATION_METHODS.slice(0, 6).map((method) => (
            <Tooltip key={method.id}>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "relative flex items-start space-x-2 rounded-lg border p-3 cursor-pointer transition-all",
                    selectedMethod === method.id
                      ? "border-success bg-success/5"
                      : "border-border hover:border-success/50"
                  )}
                  onClick={() => setSelectedMethod(method.id)}
                >
                  <RadioGroupItem value={method.id} id={method.id} className="mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <Label
                      htmlFor={method.id}
                      className="text-xs font-medium cursor-pointer flex items-center gap-1"
                    >
                      {method.name}
                      {method.id === recommendedMethod && (
                        <Badge variant="outline" className="text-[8px] px-1 py-0 h-4 border-success/50 text-success">
                          BEST
                        </Badge>
                      )}
                    </Label>
                    <p className="text-[10px] text-muted-foreground truncate">
                      {method.bestFor}
                    </p>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-[250px]">
                <div className="space-y-1">
                  <p className="text-xs font-medium">{method.name}</p>
                  <p className="text-[10px] text-muted-foreground">{method.description}</p>
                  <code className="text-[9px] bg-secondary px-1 py-0.5 rounded">
                    {method.formula}
                  </code>
                </div>
              </TooltipContent>
            </Tooltip>
          ))}
        </RadioGroup>
      </div>

      {/* Advanced Options */}
      <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
        <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors">
          {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          Advanced Options
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-muted-foreground">Gain Factor</Label>
              <span className="text-xs font-mono">{gain[0].toFixed(2)}</span>
            </div>
            <Slider
              value={gain}
              onValueChange={setGain}
              min={0.1}
              max={3.0}
              step={0.1}
              className="w-full"
            />
          </div>

          {selectedMethod === 'sparse' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs text-muted-foreground">Sparsity</Label>
                <span className="text-xs font-mono">{Math.round(sparsity[0] * 100)}%</span>
              </div>
              <Slider
                value={sparsity}
                onValueChange={setSparsity}
                min={0.5}
                max={0.99}
                step={0.01}
                className="w-full"
              />
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>

      {/* Sustainability Dashboard */}
      {architecture && (
        <div className="space-y-4 p-4 rounded-lg bg-secondary/30 border border-border">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-success" />
            <span className="text-sm font-medium">Sustainability Impact</span>
          </div>

          <div className="space-y-3">
            <SustainabilityMeter
              label="Gradient Flow Score"
              value={architecture.metrics.gradientFlowScore}
              max={100}
              color="success"
            />
            <SustainabilityMeter
              label="Convergence Boost"
              value={Math.round((architecture.metrics.convergenceSpeedBoost - 1) * 100)}
              max={100}
              suffix="%"
              color="info"
            />
            {architecture.metrics.memoryOptimization > 0 && (
              <SustainabilityMeter
                label="Memory Saved (Sparse)"
                value={architecture.metrics.memoryOptimization}
                max={100}
                suffix="%"
                color="warning"
              />
            )}
          </div>

          <div className="text-[10px] text-muted-foreground flex items-start gap-1.5 pt-2 border-t border-border">
            <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
            <span>
              Pre-computed weights eliminate random initialization overhead and provide 
              better gradient flow from the start, reducing training iterations needed.
            </span>
          </div>
        </div>
      )}

      {/* Layer Summary */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground">
          Layers to Initialize ({trainableLayers.length})
        </Label>
        <div className="flex flex-wrap gap-1.5">
          {trainableLayers.slice(0, 8).map((layer) => (
            <Badge
              key={layer.id}
              variant="outline"
              className="text-[9px] font-mono"
            >
              {layer.name}
            </Badge>
          ))}
          {trainableLayers.length > 8 && (
            <Badge variant="outline" className="text-[9px]">
              +{trainableLayers.length - 8} more
            </Badge>
          )}
        </div>
      </div>

      {/* Export Actions */}
      <div className="flex gap-2 pt-2">
        <Button
          variant="outline"
          className="flex-1"
          onClick={handleCopyCode}
          disabled={!architecture}
        >
          {copied ? <Check className="w-4 h-4 mr-2" /> : <Copy className="w-4 h-4 mr-2" />}
          {copied ? 'Copied!' : 'Copy Code'}
        </Button>
        <Button
          className="flex-1 bg-success hover:bg-success/90 text-success-foreground"
          onClick={handleExportONNX}
          disabled={!architecture}
        >
          <Download className="w-4 h-4 mr-2" />
          Export ONNX
        </Button>
      </div>
    </div>
  );
}

interface MetricCardProps {
  icon: React.ElementType;
  label: string;
  value: string;
  color?: string;
}

function MetricCard({ icon: Icon, label, value, color = "text-primary" }: MetricCardProps) {
  return (
    <div className="p-3 rounded-lg bg-secondary/50 border border-border text-center">
      <Icon className={cn("w-5 h-5 mx-auto mb-1", color)} />
      <div className={cn("text-lg font-bold", color)}>{value}</div>
      <div className="text-[9px] text-muted-foreground uppercase tracking-wide">{label}</div>
    </div>
  );
}

interface SustainabilityMeterProps {
  label: string;
  value: number;
  max: number;
  suffix?: string;
  color?: 'success' | 'info' | 'warning';
}

function SustainabilityMeter({ label, value, max, suffix = '', color = 'success' }: SustainabilityMeterProps) {
  const percentage = Math.min(100, (value / max) * 100);
  
  const colorClasses = {
    success: '[&>div]:bg-success',
    info: '[&>div]:bg-info',
    warning: '[&>div]:bg-warning',
  };

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className={cn("font-medium", `text-${color}`)}>
          {value}{suffix}
        </span>
      </div>
      <Progress value={percentage} className={cn("h-1.5", colorClasses[color])} />
    </div>
  );
}
