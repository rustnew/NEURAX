import { useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Package,
  Cpu,
  Settings,
  Layers,
  Zap,
  Box,
} from 'lucide-react';
import { cn } from '@/lib/utils.ts';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Separator } from '@/components/ui/separator.tsx';
import { CanvasNode, Connection } from '@/types/architecture.ts';

interface ExportAssistantProps {
  isOpen: boolean;
  onClose: () => void;
  format: string;
  nodes: CanvasNode[];
  connections: Connection[];
  architectureName: string;
  selectedArchitecture: string;
}

type PackageStructure = 'single' | 'modular';
type DeviceTarget = 'cpu' | 'cuda' | 'auto';
type Precision = 'fp32' | 'fp16' | 'bf16';
type TrainingLoop = 'none' | 'minimal' | 'structured';

const STEPS = [
  { id: 1, label: 'Project', icon: Package },
  { id: 2, label: 'Model', icon: Cpu },
  { id: 3, label: 'Export', icon: Settings },
] as const;

function OptionCard({
  selected,
  onClick,
  label,
  description,
}: {
  selected: boolean;
  onClick: () => void;
  label: string;
  description?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex items-center gap-3 px-4 py-3 rounded-lg border text-left transition-all text-sm',
        selected
          ? 'border-primary bg-primary/5 text-foreground'
          : 'border-border bg-secondary/20 text-muted-foreground hover:border-muted-foreground/40'
      )}
    >
      <div
        className={cn(
          'w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0 transition-colors',
          selected ? 'border-primary bg-primary' : 'border-muted-foreground/40'
        )}
      >
        {selected && <Check className="w-2.5 h-2.5 text-primary-foreground" />}
      </div>
      <div>
        <div className="font-medium">{label}</div>
        {description && (
          <div className="text-xs text-muted-foreground mt-0.5">{description}</div>
        )}
      </div>
    </button>
  );
}

function ToggleOption({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <Label className="text-sm text-foreground">{label}</Label>
      <div className="flex gap-1">
        <button
          onClick={() => onChange(true)}
          className={cn(
            'px-3 py-1 rounded text-xs font-medium transition-colors',
            value
              ? 'bg-primary text-primary-foreground'
              : 'bg-secondary text-muted-foreground hover:text-foreground'
          )}
        >
          Yes
        </button>
        <button
          onClick={() => onChange(false)}
          className={cn(
            'px-3 py-1 rounded text-xs font-medium transition-colors',
            !value
              ? 'bg-primary text-primary-foreground'
              : 'bg-secondary text-muted-foreground hover:text-foreground'
          )}
        >
          No
        </button>
      </div>
    </div>
  );
}

export function ExportAssistant({
  isOpen,
  onClose,
  format,
  nodes,
  connections: _connections,
  architectureName,
  selectedArchitecture,
}: ExportAssistantProps) {
  const [step, setStep] = useState(1);

  // Step 1 state
  const [projectName, setProjectName] = useState(architectureName.toLowerCase().replace(/\s+/g, '_'));
  const [packageStructure, setPackageStructure] = useState<PackageStructure>('modular');
  const [includeTrainingScript, setIncludeTrainingScript] = useState(true);
  const [includeDatasetLoader, setIncludeDatasetLoader] = useState(false);

  // Step 2 state
  const [device, setDevice] = useState<DeviceTarget>('auto');
  const [precision, setPrecision] = useState<Precision>('fp32');

  // Step 3 state
  const [includeOnnx, setIncludeOnnx] = useState(false);
  const [trainingLoop, setTrainingLoop] = useState<TrainingLoop>('minimal');
  const [includeTests, setIncludeTests] = useState(false);

  // Mock model stats from nodes
  const activations = new Set<string>();
  let hasAttention = false;
  nodes.forEach((n) => {
    if (n.params?.activation) activations.add(String(n.params.activation));
    if (n.type === 'attention' || n.type === 'cross_attention' || n.type === 'attention_aggregation') hasAttention = true;
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div className="w-full max-w-[560px] bg-card border border-border rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Progress bar */}
        <div className="h-1 bg-secondary">
          <div
            className="h-full bg-primary transition-all duration-500 ease-out"
            style={{ width: `${(step / 3) * 100}%` }}
          />
        </div>

        {/* Header */}
        <div className="px-6 pt-5 pb-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-semibold text-foreground">
                Export to {format}
              </h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                Configure your project scaffold
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground text-xs px-2 py-1 rounded hover:bg-secondary transition-colors"
            >
              Cancel
            </button>
          </div>

          {/* Stepper */}
          <div className="flex items-center gap-1">
            {STEPS.map((s, i) => {
              const Icon = s.icon;
              const isActive = step === s.id;
              const isDone = step > s.id;
              return (
                <div key={s.id} className="flex items-center flex-1">
                  <div className="flex items-center gap-2 flex-1">
                    <div
                      className={cn(
                        'w-7 h-7 rounded-md flex items-center justify-center transition-colors shrink-0',
                        isDone
                          ? 'bg-primary text-primary-foreground'
                          : isActive
                          ? 'bg-primary/10 text-primary border border-primary/30'
                          : 'bg-secondary text-muted-foreground'
                      )}
                    >
                      {isDone ? (
                        <Check className="w-3.5 h-3.5" />
                      ) : (
                        <Icon className="w-3.5 h-3.5" />
                      )}
                    </div>
                    <span
                      className={cn(
                        'text-xs font-medium transition-colors',
                        isActive ? 'text-foreground' : 'text-muted-foreground'
                      )}
                    >
                      {s.label}
                    </span>
                  </div>
                  {i < STEPS.length - 1 && (
                    <div
                      className={cn(
                        'h-px flex-1 mx-2 transition-colors',
                        isDone ? 'bg-primary' : 'bg-border'
                      )}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <Separator />

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-5 scrollbar-thin">
          {step === 1 && (
            <div className="space-y-5">
              <div>
                <h3 className="text-sm font-semibold text-foreground mb-1">
                  Project Configuration
                </h3>
                <p className="text-xs text-muted-foreground">
                  Define your scaffold structure
                </p>
              </div>

              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Project Name</Label>
                <Input
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  className="h-9 text-sm bg-secondary/30 border-border"
                  placeholder="my_model"
                />
              </div>

              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">Package Structure</Label>
                <div className="grid grid-cols-2 gap-2">
                  <OptionCard
                    selected={packageStructure === 'single'}
                    onClick={() => setPackageStructure('single')}
                    label="Single File"
                    description="All code in one module"
                  />
                  <OptionCard
                    selected={packageStructure === 'modular'}
                    onClick={() => setPackageStructure('modular')}
                    label="Modular"
                    description="layers/ models/ train/"
                  />
                </div>
              </div>

              <Separator className="my-2" />

              <div className="space-y-1">
                <ToggleOption
                  label="Include Training Script?"
                  value={includeTrainingScript}
                  onChange={setIncludeTrainingScript}
                />
                <ToggleOption
                  label="Include Example Dataset Loader?"
                  value={includeDatasetLoader}
                  onChange={setIncludeDatasetLoader}
                />
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-5">
              <div>
                <h3 className="text-sm font-semibold text-foreground mb-1">
                  Model Integration Settings
                </h3>
                <p className="text-xs text-muted-foreground">
                  Architecture summary & device configuration
                </p>
              </div>

              {/* Read-only model info */}
              <div className="rounded-lg border border-border bg-secondary/20 p-4 space-y-3">
                <div className="flex items-center gap-2 mb-2">
                  <Layers className="w-3.5 h-3.5 text-primary" />
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Graph Summary
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <InfoRow label="Total Parameters" value="25.6M" />
                  <InfoRow label="Total Blocks" value={String(nodes.length)} />
                  <InfoRow
                    label="Model Family"
                    value={selectedArchitecture.charAt(0).toUpperCase() + selectedArchitecture.slice(1)}
                  />
                  <InfoRow
                    label="Activations"
                    value={activations.size > 0 ? Array.from(activations).join(', ') : 'None'}
                  />
                </div>
                <div className="pt-1">
                  <InfoRow
                    label="Attention"
                    value={hasAttention ? 'Multi-Head Self-Attention' : 'None'}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">Default Device</Label>
                <div className="grid grid-cols-3 gap-2">
                  {(['cpu', 'cuda', 'auto'] as DeviceTarget[]).map((d) => (
                    <OptionCard
                      key={d}
                      selected={device === d}
                      onClick={() => setDevice(d)}
                      label={d.toUpperCase()}
                    />
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">Precision</Label>
                <div className="grid grid-cols-3 gap-2">
                  {(['fp32', 'fp16', 'bf16'] as Precision[]).map((p) => (
                    <OptionCard
                      key={p}
                      selected={precision === p}
                      onClick={() => setPrecision(p)}
                      label={p.toUpperCase()}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-5">
              <div>
                <h3 className="text-sm font-semibold text-foreground mb-1">
                  Export Preferences
                </h3>
                <p className="text-xs text-muted-foreground">
                  Final scaffold options
                </p>
              </div>

              <ToggleOption
                label="Include ONNX Export Script?"
                value={includeOnnx}
                onChange={setIncludeOnnx}
              />

              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">
                  Training Loop Template
                </Label>
                <div className="grid grid-cols-3 gap-2">
                  {([
                    { key: 'none', label: 'None' },
                    { key: 'minimal', label: 'Minimal' },
                    { key: 'structured', label: 'Structured' },
                  ] as const).map((opt) => (
                    <OptionCard
                      key={opt.key}
                      selected={trainingLoop === opt.key}
                      onClick={() => setTrainingLoop(opt.key)}
                      label={opt.label}
                    />
                  ))}
                </div>
              </div>

              <ToggleOption
                label="Include Unit Test Template?"
                value={includeTests}
                onChange={setIncludeTests}
              />

              <Separator />

              {/* Summary */}
              <div className="rounded-lg border border-border bg-secondary/20 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Box className="w-3.5 h-3.5 text-primary" />
                  <span className="text-xs font-semibold text-foreground">
                    Your project scaffold will include:
                  </span>
                </div>
                <ul className="space-y-1.5 text-xs text-muted-foreground">
                  <SummaryItem label="Model definition" sub={`${projectName}.py`} />
                  <SummaryItem label="Config file" sub="config.yaml" />
                  {packageStructure === 'modular' && (
                    <SummaryItem label="Package structure" sub="layers/ models/ train/" />
                  )}
                  {includeTrainingScript && (
                    <SummaryItem label="Training script" sub="train.py" />
                  )}
                  {includeDatasetLoader && (
                    <SummaryItem label="Dataset loader" sub="data/loader.py" />
                  )}
                  {includeOnnx && (
                    <SummaryItem label="ONNX export script" sub="export_onnx.py" />
                  )}
                  {trainingLoop !== 'none' && (
                    <SummaryItem
                      label="Training loop"
                      sub={`${trainingLoop} template`}
                    />
                  )}
                  {includeTests && (
                    <SummaryItem label="Unit tests" sub="tests/" />
                  )}
                </ul>
                <div className="mt-3 flex gap-2 flex-wrap">
                  <Badge variant="outline" className="text-[10px]">
                    {device.toUpperCase()}
                  </Badge>
                  <Badge variant="outline" className="text-[10px]">
                    {precision.toUpperCase()}
                  </Badge>
                  <Badge variant="outline" className="text-[10px]">
                    {selectedArchitecture}
                  </Badge>
                </div>
              </div>
            </div>
          )}
        </div>

        <Separator />

        {/* Footer */}
        <div className="px-6 py-4 flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => (step === 1 ? onClose() : setStep(step - 1))}
            className="text-xs"
          >
            <ArrowLeft className="w-3.5 h-3.5 mr-1" />
            {step === 1 ? 'Cancel' : 'Back'}
          </Button>

          <div className="flex items-center gap-1.5">
            {STEPS.map((s) => (
              <div
                key={s.id}
                className={cn(
                  'w-1.5 h-1.5 rounded-full transition-colors',
                  step === s.id ? 'bg-primary' : step > s.id ? 'bg-primary/50' : 'bg-border'
                )}
              />
            ))}
          </div>

          {step < 3 ? (
            <Button size="sm" onClick={() => setStep(step + 1)} className="text-xs">
              Continue
              <ArrowRight className="w-3.5 h-3.5 ml-1" />
            </Button>
          ) : (
            <Button size="sm" onClick={onClose} className="text-xs">
              <Zap className="w-3.5 h-3.5 mr-1" />
              Generate Scaffold
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] text-muted-foreground">{label}</span>
      <span className="text-xs font-medium text-foreground">{value}</span>
    </div>
  );
}

function SummaryItem({ label, sub }: { label: string; sub: string }) {
  return (
    <li className="flex items-center gap-2">
      <Check className="w-3 h-3 text-primary shrink-0" />
      <span className="text-foreground font-medium">{label}</span>
      <span className="text-muted-foreground font-mono text-[10px]">{sub}</span>
    </li>
  );
}
