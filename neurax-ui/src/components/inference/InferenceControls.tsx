import { useState } from 'react';
import { ChevronDown, ChevronRight, Thermometer, Filter, Repeat, MessageSquare, Settings2, AlertTriangle } from 'lucide-react';
import { Slider } from '@/components/ui/slider.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { cn } from '@/lib/utils.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible.tsx';

interface InferenceControlsProps {
  architectureType: ArchitectureFamily;
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

function SliderControl({ label, min, max, step, defaultValue, unit }: { 
  label: string; 
  min: number; 
  max: number; 
  step: number; 
  defaultValue: number;
  unit?: string;
}) {
  const [value, setValue] = useState([defaultValue]);
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <span className="text-xs font-mono text-foreground">{value[0]}{unit}</span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={value}
        onValueChange={setValue}
        className="w-full"
      />
    </div>
  );
}

function ToggleControl({ label, description, defaultChecked = false }: {
  label: string;
  description?: string;
  defaultChecked?: boolean;
}) {
  const [checked, setChecked] = useState(defaultChecked);
  
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="space-y-0.5">
        <Label className="text-xs font-medium">{label}</Label>
        {description && <p className="text-[10px] text-muted-foreground">{description}</p>}
      </div>
      <Switch checked={checked} onCheckedChange={setChecked} />
    </div>
  );
}

export function InferenceControls({ architectureType }: InferenceControlsProps) {
  return (
    <div className="w-full md:w-72 lg:w-80 max-h-64 md:max-h-none md:h-full bg-sidebar border-b md:border-b-0 md:border-r border-border flex flex-col overflow-y-auto md:overflow-y-hidden">
      {/* Header */}
      <div className="h-10 md:h-12 px-3 md:px-4 flex items-center border-b border-border bg-panel-header shrink-0">
        <h2 className="text-xs md:text-sm font-semibold">Inference Controls</h2>
      </div>
      
      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-3">
        {/* Sampling Strategy */}
        <CollapsibleSection title="Sampling Strategy" icon={Thermometer}>
          <SliderControl label="Temperature" min={0} max={2} step={0.01} defaultValue={0.7} />
          <SliderControl label="Top-k" min={1} max={100} step={1} defaultValue={40} />
          <SliderControl label="Top-p" min={0} max={1} step={0.01} defaultValue={0.9} />
          <SliderControl label="Beam Width" min={1} max={10} step={1} defaultValue={1} />
          <SliderControl label="Repetition Penalty" min={1} max={2} step={0.01} defaultValue={1.1} />
          <SliderControl label="Presence Penalty" min={0} max={2} step={0.01} defaultValue={0} />
          <SliderControl label="Frequency Penalty" min={0} max={2} step={0.01} defaultValue={0} />
        </CollapsibleSection>

        {/* Context Configuration */}
        <CollapsibleSection title="Context Configuration" icon={MessageSquare}>
          <SliderControl label="Prompt Length" min={128} max={32768} step={128} defaultValue={2048} unit=" tokens" />
          <SliderControl label="Max Output Tokens" min={64} max={8192} step={64} defaultValue={1024} unit=" tokens" />
          <ToggleControl label="Sliding Window" description="Enable sliding window attention" defaultChecked />
          <ToggleControl label="KV Cache Reuse" description="Reuse key-value cache across generations" defaultChecked />
        </CollapsibleSection>

        {/* Model Behavior Assumptions */}
        <CollapsibleSection title="Model Behavior" icon={Settings2}>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Architecture Type</Label>
            <Input 
              value={architectureType.charAt(0).toUpperCase() + architectureType.slice(1)} 
              disabled 
              className="text-xs h-8 bg-secondary/50"
            />
          </div>
          
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Attention Type</Label>
            <Select defaultValue="standard">
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
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
              <Select defaultValue="top-k">
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
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
            <Select defaultValue="fp16">
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
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
          <ToggleControl 
            label="Long-context Simulation" 
            description="Test behavior with extended context windows"
          />
          <ToggleControl 
            label="Adversarial Prompt" 
            description="Simulate adversarial input patterns"
          />
          <ToggleControl 
            label="High-temperature Mode" 
            description="Force high temperature sampling"
          />
          <ToggleControl 
            label="Low-temperature Mode" 
            description="Force deterministic sampling"
          />
        </CollapsibleSection>
      </div>
    </div>
  );
}
