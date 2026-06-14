import { useState } from 'react';
import { 
  Workflow, 
  TrendingUp, 
  Shield, 
  Repeat, 
  Combine,
  Zap,
  CheckCircle2
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Progress } from '@/components/ui/progress.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { cn } from '@/lib/utils.ts';

interface SSMPluginProps {
  className?: string;
}

export function SSMPlugin({ className }: SSMPluginProps) {
  const [activeTab, setActiveTab] = useState('state');

  const stateParams = [
    { name: 'State Dimension (N)', value: 16, optimal: true },
    { name: 'Expansion Factor (E)', value: 2, optimal: true },
    { name: 'Conv Dimension', value: 4, optimal: true },
    { name: 'Discretization', value: 'ZOH', optimal: true },
  ];

  const scalingData = [
    { length: '1K', ssm: '0.5ms', transformer: '2ms', memory: '16MB' },
    { length: '4K', ssm: '2ms', transformer: '32ms', memory: '64MB' },
    { length: '16K', ssm: '8ms', transformer: '512ms', memory: '256MB' },
    { length: '64K', ssm: '32ms', transformer: '8.2s', memory: '1GB' },
    { length: '128K', ssm: '64ms', transformer: 'OOM', memory: '2GB' },
  ];

  return (
    <div className={cn("space-y-4 animate-fade-in", className)}>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-5 bg-secondary/50">
          <TabsTrigger value="state" className="text-xs gap-1.5">
            <Workflow className="w-3.5 h-3.5" />
            State
          </TabsTrigger>
          <TabsTrigger value="scaling" className="text-xs gap-1.5">
            <TrendingUp className="w-3.5 h-3.5" />
            Scaling
          </TabsTrigger>
          <TabsTrigger value="stability" className="text-xs gap-1.5">
            <Shield className="w-3.5 h-3.5" />
            Stability
          </TabsTrigger>
          <TabsTrigger value="recurrence" className="text-xs gap-1.5">
            <Repeat className="w-3.5 h-3.5" />
            Recurrence
          </TabsTrigger>
          <TabsTrigger value="hybrid" className="text-xs gap-1.5">
            <Combine className="w-3.5 h-3.5" />
            Hybrid
          </TabsTrigger>
        </TabsList>

        <TabsContent value="state" className="mt-4 space-y-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Workflow className="w-4 h-4 text-success" />
                State Transition Configuration
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 mb-4">
                {stateParams.map((param) => (
                  <div key={param.name} className="flex items-center justify-between p-2 rounded-lg bg-secondary/30">
                    <span className="text-xs">{param.name}</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm">{param.value}</span>
                      {param.optimal && (
                        <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="p-3 rounded-lg bg-success/10 border border-success/30">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-success" />
                  <span className="text-sm font-medium">Linear Complexity</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  SSM provides O(L) time and memory complexity vs O(L²) for attention.
                </p>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">State Space Equations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="font-mono text-xs space-y-2 p-3 rounded bg-secondary/30">
                <div>h'(t) = <span className="text-primary">A</span>h(t) + <span className="text-success">B</span>x(t)</div>
                <div>y(t) = <span className="text-warning">C</span>h(t) + <span className="text-destructive">D</span>x(t)</div>
              </div>
              <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
                <div className="p-2 rounded bg-primary/10 border border-primary/30 text-center">
                  <div className="font-mono text-primary">A</div>
                  <div className="text-[10px] text-muted-foreground">State</div>
                </div>
                <div className="p-2 rounded bg-success/10 border border-success/30 text-center">
                  <div className="font-mono text-success">B</div>
                  <div className="text-[10px] text-muted-foreground">Input</div>
                </div>
                <div className="p-2 rounded bg-warning/10 border border-warning/30 text-center">
                  <div className="font-mono text-warning">C</div>
                  <div className="text-[10px] text-muted-foreground">Output</div>
                </div>
                <div className="p-2 rounded bg-destructive/10 border border-destructive/30 text-center">
                  <div className="font-mono text-destructive">D</div>
                  <div className="text-[10px] text-muted-foreground">Skip</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scaling" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-success" />
                Sequence Length Scaling
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border/50">
                      <th className="text-left py-2 text-muted-foreground">Length</th>
                      <th className="text-left py-2 text-muted-foreground">SSM</th>
                      <th className="text-left py-2 text-muted-foreground">Transformer</th>
                      <th className="text-left py-2 text-muted-foreground">Memory</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scalingData.map((row) => (
                      <tr key={row.length} className="border-b border-border/30">
                        <td className="py-2 font-mono">{row.length}</td>
                        <td className="py-2 font-mono text-success">{row.ssm}</td>
                        <td className={cn(
                          "py-2 font-mono",
                          row.transformer === 'OOM' ? "text-destructive" : "text-muted-foreground"
                        )}>{row.transformer}</td>
                        <td className="py-2 font-mono">{row.memory}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 flex items-center gap-4">
                <Badge variant="outline" className="bg-success/10 text-success border-success/30">
                  O(L) Time
                </Badge>
                <Badge variant="outline" className="bg-success/10 text-success border-success/30">
                  O(L) Memory
                </Badge>
                <Badge variant="outline" className="bg-primary/10 text-primary border-primary/30">
                  Parallel Scan
                </Badge>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="stability" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Shield className="w-4 h-4 text-success" />
                Numerical Stability
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-lg bg-success/10 border border-success/30">
                  <div className="flex items-center gap-2 mb-1">
                    <CheckCircle2 className="w-4 h-4 text-success" />
                    <span className="text-xs font-medium">HiPPO Init</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground">Optimal A matrix initialization</p>
                </div>
                <div className="p-3 rounded-lg bg-success/10 border border-success/30">
                  <div className="flex items-center gap-2 mb-1">
                    <CheckCircle2 className="w-4 h-4 text-success" />
                    <span className="text-xs font-medium">Stable Discretization</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground">ZOH ensures bounded states</p>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Eigenvalue Range</span>
                  <Badge variant="outline" className="text-success border-success/30">Stable</Badge>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Gradient Norm</span>
                  <span className="font-mono">1.24</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">State Magnitude</span>
                  <span className="font-mono">Bounded</span>
                </div>
              </div>
              
              <Progress value={95} className="h-2" />
              <div className="text-xs text-muted-foreground text-center">Overall Stability Score: 95%</div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="recurrence" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Repeat className="w-4 h-4 text-success" />
                Recurrence & Convolution Mode
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="p-3 rounded-lg bg-secondary/30 border border-border/30">
                  <div className="text-xs font-medium mb-2">Recurrence (Inference)</div>
                  <div className="text-[10px] text-muted-foreground space-y-1">
                    <div>• O(1) per step</div>
                    <div>• Stateful generation</div>
                    <div>• Cache-efficient</div>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-secondary/30 border border-border/30">
                  <div className="text-xs font-medium mb-2">Convolution (Training)</div>
                  <div className="text-[10px] text-muted-foreground space-y-1">
                    <div>• O(L log L) FFT</div>
                    <div>• Fully parallel</div>
                    <div>• Hardware optimized</div>
                  </div>
                </div>
              </div>
              
              <div className="p-2 rounded bg-success/10 border border-success/30 text-xs">
                <span className="text-success font-medium">Mode: </span>
                <span className="text-muted-foreground">Auto-select based on batch size and sequence length</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hybrid" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Combine className="w-4 h-4 text-success" />
                Hybrid Attention-SSM Architecture
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs">
                  <Badge variant="outline">Jamba Style</Badge>
                  <span className="text-muted-foreground">SSM + Attention interleaving</span>
                </div>
                
                <div className="p-3 rounded-lg bg-secondary/30 space-y-2">
                  <div className="text-xs font-medium">Layer Configuration</div>
                  <div className="flex flex-wrap gap-1">
                    {['SSM', 'SSM', 'SSM', 'Attn', 'SSM', 'SSM', 'SSM', 'Attn'].map((type, i) => (
                      <Badge 
                        key={i}
                        variant="outline"
                        className={cn(
                          type === 'SSM' ? "bg-success/10 border-success/30 text-success" : "bg-primary/10 border-primary/30 text-primary"
                        )}
                      >
                        L{i+1}: {type}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2 rounded bg-secondary/30">
                    <span className="text-muted-foreground">SSM Layers:</span>
                    <span className="font-mono ml-2">6</span>
                  </div>
                  <div className="p-2 rounded bg-secondary/30">
                    <span className="text-muted-foreground">Attention Layers:</span>
                    <span className="font-mono ml-2">2</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
