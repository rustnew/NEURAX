import { useState, useMemo } from 'react';
import {
  Leaf, Zap, Clock, Database, Cpu, Activity, Info,
  Check, Copy, Download, Sparkles, ChevronDown, ChevronRight,
  Settings2, GraduationCap
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Slider } from '@/components/ui/slider.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Progress } from '@/components/ui/progress.tsx';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group.tsx';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip.tsx';
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
  getRecommendedHyperparams,
  HyperparameterConfig,
} from '@/utils/weightInitialization.ts';

interface ProductionWorkspaceProps {
  nodes: CanvasNode[];
  connections: Connection[];
  modelName: string;
}

export function ProductionWorkspace({ nodes, connections, modelName }: ProductionWorkspaceProps) {
  const { toast } = useToast();
  const [selectedMethod, setSelectedMethod] = useState<InitializationMethod>('xavier_normal');
  const [gain, setGain] = useState([1.0]);
  const [sparsity, setSparsity] = useState([0.9]);
  const [copied, setCopied] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const recommendedMethod = useMemo(() => getRecommendedInit(nodes), [nodes]);
  const recommendedHyperparams = useMemo(() => getRecommendedHyperparams(nodes, connections), [nodes, connections]);

  // Editable hyperparams initialized from recommendations
  const [hyperparams, setHyperparams] = useState<HyperparameterConfig>(recommendedHyperparams);

  // Sync when nodes change
  useMemo(() => {
    setHyperparams(recommendedHyperparams);
  }, [recommendedHyperparams]);

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
      toast({ title: "No Architecture", description: "Add layers to the canvas first", variant: "destructive" });
      return;
    }
    const code = generateGreenAIONNX(architecture);
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelName.toLowerCase()}_green_ai.py`;
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Green AI Export Complete", description: "Pre-initialized model exported successfully" });
  };

  const handleCopyCode = () => {
    if (!architecture) return;
    const code = generateGreenAIONNX(architecture);
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast({ title: "Copied", description: "Green AI code copied to clipboard" });
  };

  const trainableLayers = nodes.filter(n =>
    ['dense', 'conv2d', 'attention', 'transformer', 'layernorm', 'batchnorm'].includes(n.type)
  );

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      {/* Top Toolbar */}
      <div className="border-b border-border bg-card flex flex-col sm:flex-row items-start sm:items-center justify-between px-3 sm:px-6 py-2 sm:py-0 sm:h-14 gap-2 sm:gap-0">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-success/10 flex items-center justify-center shrink-0">
            <Leaf className="w-4 h-4 text-success" />
          </div>
          <div className="min-w-0">
            <h2 className="text-xs sm:text-sm font-semibold flex items-center gap-2">
              <span className="truncate">Training Config Optimizer</span>
              <Badge className="bg-success/20 text-success border-0 text-[10px] shrink-0">GREEN AI</Badge>
            </h2>
            <p className="text-[10px] text-muted-foreground hidden sm:block">Optimal weights, biases & hyperparameters for efficient training</p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button variant="outline" size="sm" className="text-xs h-7 sm:h-8" onClick={handleCopyCode} disabled={!architecture}>
            {copied ? <Check className="w-3.5 h-3.5 sm:mr-1.5" /> : <Copy className="w-3.5 h-3.5 sm:mr-1.5" />}
            <span className="hidden sm:inline">{copied ? 'Copied!' : 'Copy Code'}</span>
          </Button>
          <Button size="sm" className="bg-success hover:bg-success/90 text-success-foreground text-xs h-7 sm:h-8" onClick={handleExportONNX} disabled={!architecture}>
            <Download className="w-3.5 h-3.5 sm:mr-1.5" />
            <span className="hidden sm:inline">Export ONNX</span>
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-3 sm:p-4 lg:p-6">
        <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">

          {/* Left Column: Weights & Biases */}
          <div className="space-y-6">
            <SectionHeader icon={Settings2} title="Optimal Weights & Biases" />

            {/* Quick Stats */}
            {architecture && (
              <div className="grid grid-cols-3 gap-3">
                <MetricCard icon={Clock} label="Epochs Saved" value={`~${architecture.metrics.estimatedEpochsSaved}`} color="text-success" />
                <MetricCard icon={Cpu} label="Hours Saved" value={`~${architecture.metrics.computeHoursSaved}h`} color="text-info" />
                <MetricCard icon={Database} label="Data Efficiency" value={`+${architecture.metrics.datasetEfficiency}%`} color="text-warning" />
              </div>
            )}

            {/* Init Method Selection */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Initialization Method</Label>
                {recommendedMethod && (
                  <Button variant="ghost" size="sm" className="h-6 text-[10px] text-success hover:text-success" onClick={() => setSelectedMethod(recommendedMethod)}>
                    <Sparkles className="w-3 h-3 mr-1" /> Use Recommended
                  </Button>
                )}
              </div>
              <RadioGroup value={selectedMethod} onValueChange={(v) => setSelectedMethod(v as InitializationMethod)} className="grid grid-cols-2 gap-2">
                {INITIALIZATION_METHODS.slice(0, 6).map((method) => (
                  <Tooltip key={method.id}>
                    <TooltipTrigger asChild>
                      <div
                        className={cn(
                          "relative flex items-start space-x-2 rounded-lg border p-3 cursor-pointer transition-all",
                          selectedMethod === method.id ? "border-success bg-success/5" : "border-border hover:border-success/50"
                        )}
                        onClick={() => setSelectedMethod(method.id)}
                      >
                        <RadioGroupItem value={method.id} id={`prod-${method.id}`} className="mt-0.5" />
                        <div className="flex-1 min-w-0">
                          <Label htmlFor={`prod-${method.id}`} className="text-xs font-medium cursor-pointer flex items-center gap-1">
                            {method.name}
                            {method.id === recommendedMethod && (
                              <Badge variant="outline" className="text-[8px] px-1 py-0 h-4 border-success/50 text-success">BEST</Badge>
                            )}
                          </Label>
                          <p className="text-[10px] text-muted-foreground truncate">{method.bestFor}</p>
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-[250px]">
                      <p className="text-xs font-medium">{method.name}</p>
                      <p className="text-[10px] text-muted-foreground">{method.description}</p>
                      <code className="text-[9px] bg-secondary px-1 py-0.5 rounded">{method.formula}</code>
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
                <SliderField label="Gain Factor" value={gain} onChange={setGain} min={0.1} max={3.0} step={0.1} format={(v) => v.toFixed(2)} />
                {selectedMethod === 'sparse' && (
                  <SliderField label="Sparsity" value={sparsity} onChange={setSparsity} min={0.5} max={0.99} step={0.01} format={(v) => `${Math.round(v * 100)}%`} />
                )}
              </CollapsibleContent>
            </Collapsible>

            {/* Layer Summary */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Layers to Initialize ({trainableLayers.length})</Label>
              <div className="flex flex-wrap gap-1.5">
                {trainableLayers.slice(0, 8).map((layer) => (
                  <Badge key={layer.id} variant="outline" className="text-[9px] font-mono">{layer.name}</Badge>
                ))}
                {trainableLayers.length > 8 && <Badge variant="outline" className="text-[9px]">+{trainableLayers.length - 8} more</Badge>}
              </div>
            </div>
          </div>

          {/* Right Column: Hyperparameters & Sustainability */}
          <div className="space-y-6">
            <SectionHeader icon={GraduationCap} title="Training Hyperparameters" />

            <div className="space-y-4 p-4 rounded-lg bg-card border border-border">
              <SliderField label="Learning Rate" value={[hyperparams.learningRate]} onChange={([v]) => setHyperparams(h => ({ ...h, learningRate: v }))} min={0.00001} max={0.01} step={0.00001} format={(v) => v.toExponential(1)} />
              <SliderField label="Dropout" value={[hyperparams.dropout]} onChange={([v]) => setHyperparams(h => ({ ...h, dropout: v }))} min={0} max={0.5} step={0.01} format={(v) => v.toFixed(2)} />
              <SliderField label="Weight Decay" value={[hyperparams.weightDecay]} onChange={([v]) => setHyperparams(h => ({ ...h, weightDecay: v }))} min={0} max={0.1} step={0.001} format={(v) => v.toFixed(3)} />
              <SliderField label="Warmup Steps" value={[hyperparams.warmupSteps]} onChange={([v]) => setHyperparams(h => ({ ...h, warmupSteps: Math.round(v) }))} min={0} max={5000} step={50} format={(v) => String(Math.round(v))} />
              <SliderField label="Gradient Clipping" value={[hyperparams.gradientClipping]} onChange={([v]) => setHyperparams(h => ({ ...h, gradientClipping: v }))} min={0.1} max={5.0} step={0.1} format={(v) => v.toFixed(1)} />

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Optimizer</Label>
                </div>
                <Select value={hyperparams.optimizer} onValueChange={(v) => setHyperparams(h => ({ ...h, optimizer: v as HyperparameterConfig['optimizer'] }))}>
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Adam">Adam</SelectItem>
                    <SelectItem value="AdamW">AdamW</SelectItem>
                    <SelectItem value="SGD">SGD</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button variant="ghost" size="sm" className="w-full text-[10px] text-success hover:text-success" onClick={() => setHyperparams(recommendedHyperparams)}>
                <Sparkles className="w-3 h-3 mr-1" /> Reset to Recommended
              </Button>
            </div>

            {/* Sustainability Metrics */}
            {architecture && (
              <div className="space-y-4 p-4 rounded-lg bg-secondary/30 border border-border">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-success" />
                  <span className="text-sm font-medium">Sustainability Impact</span>
                </div>
                <div className="space-y-3">
                  <SustainabilityMeter label="Gradient Flow Score" value={architecture.metrics.gradientFlowScore} max={100} color="success" />
                  <SustainabilityMeter label="Convergence Boost" value={Math.round((architecture.metrics.convergenceSpeedBoost - 1) * 100)} max={100} suffix="%" color="info" />
                  {architecture.metrics.memoryOptimization > 0 && (
                    <SustainabilityMeter label="Memory Saved (Sparse)" value={architecture.metrics.memoryOptimization} max={100} suffix="%" color="warning" />
                  )}
                </div>
                <div className="text-[10px] text-muted-foreground flex items-start gap-1.5 pt-2 border-t border-border">
                  <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <span>Pre-computed weights eliminate random initialization overhead and provide better gradient flow from the start.</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Sub-components ---

function SectionHeader({ icon: Icon, title }: { icon: React.ElementType; title: string }) {
  return (
    <div className="flex items-center gap-2 pb-2 border-b border-border">
      <Icon className="w-4 h-4 text-primary" />
      <h3 className="text-sm font-semibold">{title}</h3>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, color = "text-primary" }: { icon: React.ElementType; label: string; value: string; color?: string }) {
  return (
    <div className="p-3 rounded-lg bg-secondary/50 border border-border text-center">
      <Icon className={cn("w-5 h-5 mx-auto mb-1", color)} />
      <div className={cn("text-lg font-bold", color)}>{value}</div>
      <div className="text-[9px] text-muted-foreground uppercase tracking-wide">{label}</div>
    </div>
  );
}

function SliderField({ label, value, onChange, min, max, step, format }: {
  label: string; value: number[]; onChange: (v: number[]) => void;
  min: number; max: number; step: number; format: (v: number) => string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <span className="text-xs font-mono">{format(value[0])}</span>
      </div>
      <Slider value={value} onValueChange={onChange} min={min} max={max} step={step} className="w-full" />
    </div>
  );
}

function SustainabilityMeter({ label, value, max, suffix = '', color = 'success' }: {
  label: string; value: number; max: number; suffix?: string; color?: 'success' | 'info' | 'warning';
}) {
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
        <span className={cn("font-medium", `text-${color}`)}>{value}{suffix}</span>
      </div>
      <Progress value={percentage} className={cn("h-1.5", colorClasses[color])} />
    </div>
  );
}
