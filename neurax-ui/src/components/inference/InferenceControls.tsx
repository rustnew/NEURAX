import { useState } from 'react';
import { ChevronDown, ChevronRight, Thermometer, Filter, Repeat, MessageSquare, Settings2, AlertTriangle } from 'lucide-react';
import { Slider } from '@/components/ui/slider.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible.tsx';
import { InferenceParams } from '@/services/neuraxApi.ts';

export function buildDefaultInferenceParams(architectureType: ArchitectureFamily): InferenceParams {
  return {
    temperature: 0.7,
    top_k: 40,
    top_p: 0.9,
    beam_width: 1,
    repetition_penalty: 1.1,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    prompt_length: 2048,
    max_output_tokens: 1024,
    sliding_window: true,
    kv_cache_reuse: true,
    architecture_family: architectureType,
    attention_type: 'standard',
    moe_router_mode: architectureType === 'moe' ? 'top-k' : undefined,
    quantization_level: 'fp16',
    long_context_simulation: false,
    adversarial_prompt: false,
    high_temperature_mode: false,
    low_temperature_mode: false,
  };
}

interface InferenceControlsProps {
  architectureType: ArchitectureFamily;
  params: InferenceParams;
  onParamsChange: (params: InferenceParams) => void;
}

interface CollapsibleSectionProps {
  title: string;
  icon: React.ElementType;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

function CollapsibleSection({ title, icon: Icon, defaultOpen = true, children }: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="border border-border rounded-lg overflow-hidden">
      <CollapsibleTrigger className="w-full flex items-center gap-2 px-3 py-2.5 bg-panel-header hover:bg-secondary/50 transition-colors">
        {isOpen ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
        <Icon className="w-4 h-4 text-primary" />
        <span className="text-sm font-medium">{title}</span>
      </CollapsibleTrigger>
      <CollapsibleContent className="p-3 space-y-4 bg-card">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
}

export function InferenceControls({ architectureType, params, onParamsChange }: InferenceControlsProps) {
  const set = <K extends keyof InferenceParams>(key: K, value: InferenceParams[K]) =>
    onParamsChange({ ...params, [key]: value });

  return (
    <div className="w-full md:w-72 lg:w-80 max-h-64 md:max-h-none md:h-full bg-sidebar border-b md:border-b-0 md:border-r border-border flex flex-col overflow-y-auto md:overflow-y-hidden">
      <div className="h-10 md:h-12 px-3 md:px-4 flex items-center border-b border-border bg-panel-header shrink-0">
        <h2 className="text-xs md:text-sm font-semibold">Inference Controls</h2>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-3">
        {/* Sampling Strategy */}
        <CollapsibleSection title="Sampling Strategy" icon={Thermometer}>
          {([
            { label: 'Temperature', key: 'temperature', min: 0, max: 2, step: 0.01 },
            { label: 'Top-k', key: 'top_k', min: 1, max: 100, step: 1 },
            { label: 'Top-p', key: 'top_p', min: 0, max: 1, step: 0.01 },
            { label: 'Beam Width', key: 'beam_width', min: 1, max: 10, step: 1 },
            { label: 'Repetition Penalty', key: 'repetition_penalty', min: 1, max: 2, step: 0.01 },
            { label: 'Presence Penalty', key: 'presence_penalty', min: 0, max: 2, step: 0.01 },
            { label: 'Frequency Penalty', key: 'frequency_penalty', min: 0, max: 2, step: 0.01 },
          ] as const).map(({ label, key, min, max, step }) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs text-muted-foreground">{label}</Label>
                <span className="text-xs font-mono text-foreground">{(params[key] as number).toFixed(step < 1 ? 2 : 0)}</span>
              </div>
              <Slider
                min={min} max={max} step={step}
                value={[params[key] as number]}
                onValueChange={([v]) => set(key, v as InferenceParams[typeof key])}
                className="w-full"
              />
            </div>
          ))}
        </CollapsibleSection>

        {/* Context Configuration */}
        <CollapsibleSection title="Context Configuration" icon={MessageSquare}>
          {([
            { label: 'Prompt Length', key: 'prompt_length', min: 128, max: 32768, step: 128, unit: ' tokens' },
            { label: 'Max Output Tokens', key: 'max_output_tokens', min: 64, max: 8192, step: 64, unit: ' tokens' },
          ] as const).map(({ label, key, min, max, step, unit }) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs text-muted-foreground">{label}</Label>
                <span className="text-xs font-mono text-foreground">{params[key]}{unit}</span>
              </div>
              <Slider
                min={min} max={max} step={step}
                value={[params[key] as number]}
                onValueChange={([v]) => set(key, v as InferenceParams[typeof key])}
                className="w-full"
              />
            </div>
          ))}
          {([
            { label: 'Sliding Window', key: 'sliding_window', description: 'Enable sliding window attention' },
            { label: 'KV Cache Reuse', key: 'kv_cache_reuse', description: 'Reuse key-value cache across generations' },
          ] as const).map(({ label, key, description }) => (
            <div key={key} className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <Label className="text-xs font-medium">{label}</Label>
                <p className="text-[10px] text-muted-foreground">{description}</p>
              </div>
              <Switch checked={params[key] as boolean} onCheckedChange={(v) => set(key, v as InferenceParams[typeof key])} />
            </div>
          ))}
        </CollapsibleSection>

        {/* Model Behavior */}
        <CollapsibleSection title="Model Behavior" icon={Settings2}>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Architecture Type</Label>
            <Input value={architectureType.charAt(0).toUpperCase() + architectureType.slice(1)} disabled className="text-xs h-8 bg-secondary/50" />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Attention Type</Label>
            <Select value={params.attention_type} onValueChange={(v) => set('attention_type', v)}>
              <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="standard">Standard</SelectItem>
                <SelectItem value="flash">Flash Attention</SelectItem>
                <SelectItem value="linear">Linear Attention</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {architectureType === 'moe' && (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">MoE Router Mode</Label>
              <Select value={params.moe_router_mode ?? 'top-k'} onValueChange={(v) => set('moe_router_mode', v)}>
                <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="top-k">Top-k Routing</SelectItem>
                  <SelectItem value="expert-choice">Expert Choice</SelectItem>
                  <SelectItem value="soft">Soft Routing</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Quantization Level</Label>
            <Select value={params.quantization_level} onValueChange={(v) => set('quantization_level', v)}>
              <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="fp32">FP32 (Full Precision)</SelectItem>
                <SelectItem value="fp16">FP16 (Half Precision)</SelectItem>
                <SelectItem value="bf16">BF16 (Brain Float)</SelectItem>
                <SelectItem value="int8">INT8 (8-bit)</SelectItem>
                <SelectItem value="int4">INT4 (4-bit)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CollapsibleSection>

        {/* Stability Stress Test */}
        <CollapsibleSection title="Stability Stress Test" icon={AlertTriangle} defaultOpen={false}>
          {([
            { label: 'Long-context Simulation', key: 'long_context_simulation', description: 'Test behavior with extended context windows' },
            { label: 'Adversarial Prompt', key: 'adversarial_prompt', description: 'Simulate adversarial input patterns' },
            { label: 'High-temperature Mode', key: 'high_temperature_mode', description: 'Force high temperature sampling' },
            { label: 'Low-temperature Mode', key: 'low_temperature_mode', description: 'Force deterministic sampling' },
          ] as const).map(({ label, key, description }) => (
            <div key={key} className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <Label className="text-xs font-medium">{label}</Label>
                <p className="text-[10px] text-muted-foreground">{description}</p>
              </div>
              <Switch checked={params[key] as boolean} onCheckedChange={(v) => set(key, v as InferenceParams[typeof key])} />
            </div>
          ))}
        </CollapsibleSection>
      </div>
    </div>
  );
}
