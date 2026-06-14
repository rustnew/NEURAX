import { useState } from 'react';
import {
  Repeat,
  ArrowRight,
  Calculator,
  TrendingUp,
  AlertTriangle,
  Cpu,
  Layers,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Progress } from '@/components/ui/progress.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { cn } from '@/lib/utils.ts';

interface RNNPluginProps {
  className?: string;
}

export function RNNPlugin({ className }: RNNPluginProps) {
  const [activeTab, setActiveTab] = useState('gates');

  const gates = [
    { name: 'Forget Gate', desc: 'σ(Wf·[h_{t-1}, x_t] + bf)', activation: 'sigmoid', role: 'Controls memory retention' },
    { name: 'Input Gate', desc: 'σ(Wi·[h_{t-1}, x_t] + bi)', activation: 'sigmoid', role: 'Controls new information' },
    { name: 'Cell Candidate', desc: 'tanh(Wc·[h_{t-1}, x_t] + bc)', activation: 'tanh', role: 'Candidate cell state' },
    { name: 'Output Gate', desc: 'σ(Wo·[h_{t-1}, x_t] + bo)', activation: 'sigmoid', role: 'Controls hidden output' },
  ];

  const gradientFlow = [
    { depth: 10, vanishing: 12, exploding: 3, status: 'normal' },
    { depth: 50, vanishing: 35, exploding: 8, status: 'warning' },
    { depth: 100, vanishing: 65, exploding: 15, status: 'critical' },
    { depth: 200, vanishing: 88, exploding: 25, status: 'critical' },
  ];

  return (
    <div className={cn('space-y-4 animate-fade-in', className)}>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-4 bg-secondary/50">
          <TabsTrigger value="gates" className="text-xs gap-1.5">
            <Repeat className="w-3.5 h-3.5" />
            Gates
          </TabsTrigger>
          <TabsTrigger value="sequence" className="text-xs gap-1.5">
            <ArrowRight className="w-3.5 h-3.5" />
            Sequence
          </TabsTrigger>
          <TabsTrigger value="gradient" className="text-xs gap-1.5">
            <TrendingUp className="w-3.5 h-3.5" />
            Gradient
          </TabsTrigger>
          <TabsTrigger value="params" className="text-xs gap-1.5">
            <Calculator className="w-3.5 h-3.5" />
            Params
          </TabsTrigger>
        </TabsList>

        <TabsContent value="gates" className="mt-4 space-y-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Repeat className="w-4 h-4" style={{ color: 'hsl(25, 80%, 52%)' }} />
                LSTM Gate Structure
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {gates.map((gate) => (
                <div key={gate.name} className="p-3 rounded-lg bg-secondary/30 border border-border/30">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium">{gate.name}</span>
                    <Badge variant="outline" className="text-[9px] px-1.5 h-4">
                      {gate.activation}
                    </Badge>
                  </div>
                  <div className="text-[10px] font-mono text-muted-foreground">{gate.desc}</div>
                  <div className="text-[10px] text-muted-foreground mt-1">{gate.role}</div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sequence" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Layers className="w-4 h-4" style={{ color: 'hsl(25, 80%, 52%)' }} />
                Sequence Processing
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 overflow-x-auto pb-2">
                {['x_t', 'h_{t-1}', 'Gates', 'c_t', 'h_t', 'y_t'].map((stage, i) => (
                  <div key={stage} className="flex items-center gap-2">
                    <div className="px-3 py-2 rounded-lg bg-secondary/50 border border-border/50 text-xs font-mono whitespace-nowrap">
                      {stage}
                    </div>
                    {i < 5 && <ArrowRight className="w-4 h-4 text-muted-foreground shrink-0" />}
                  </div>
                ))}
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                <div className="p-2 rounded bg-secondary/30">
                  <div className="text-muted-foreground">Hidden Size</div>
                  <div className="font-mono">512</div>
                </div>
                <div className="p-2 rounded bg-secondary/30">
                  <div className="text-muted-foreground">Num Layers</div>
                  <div className="font-mono">3</div>
                </div>
                <div className="p-2 rounded bg-secondary/30">
                  <div className="text-muted-foreground">Bidirectional</div>
                  <div className="font-mono">No</div>
                </div>
                <div className="p-2 rounded bg-secondary/30">
                  <div className="text-muted-foreground">Dropout</div>
                  <div className="font-mono">0.1</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="gradient" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-warning" />
                Gradient Flow Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {gradientFlow.map((item) => (
                <div
                  key={item.depth}
                  className={cn(
                    'p-2.5 rounded-lg border text-xs',
                    item.status === 'normal' && 'bg-success/10 border-success/30',
                    item.status === 'warning' && 'bg-warning/10 border-warning/30',
                    item.status === 'critical' && 'bg-destructive/10 border-destructive/30'
                  )}
                >
                  <div className="flex justify-between mb-1">
                    <span className="font-medium">Depth {item.depth} steps</span>
                  </div>
                  <div className="flex gap-4">
                    <div>
                      <span className="text-muted-foreground">Vanishing risk: </span>
                      <span className="font-mono">{item.vanishing}%</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Exploding risk: </span>
                      <span className="font-mono">{item.exploding}%</span>
                    </div>
                  </div>
                </div>
              ))}
              <div className="p-2 rounded bg-secondary/30 text-xs text-muted-foreground">
                💡 Use LSTM or GRU to mitigate vanishing gradients for sequences &gt; 50 steps
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="params" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Cpu className="w-4 h-4" style={{ color: 'hsl(25, 80%, 52%)' }} />
                Parameter Breakdown (LSTM)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { name: 'Input→Hidden (Wi, Wf, Wc, Wo)', params: '4.2M', pct: 50 },
                { name: 'Hidden→Hidden (Ui, Uf, Uc, Uo)', params: '4.2M', pct: 50 },
                { name: 'Biases (4 gates)', params: '8.2K', pct: 0.1 },
                { name: 'Embedding', params: '2.6M', pct: 0 },
              ].map((item) => (
                <div key={item.name} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">{item.name}</span>
                    <span className="font-mono">{item.params}</span>
                  </div>
                  {item.pct > 0 && <Progress value={item.pct} className="h-1.5" />}
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
