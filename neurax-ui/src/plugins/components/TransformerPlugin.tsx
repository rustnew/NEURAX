import { useState } from 'react';
import { 
  Focus, 
  ArrowRight, 
  Calculator, 
  TrendingUp, 
  GitBranch,
  Eye,
  Layers,
  Cpu
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Progress } from '@/components/ui/progress.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { cn } from '@/lib/utils.ts';

interface TransformerPluginProps {
  className?: string;
}

export function TransformerPlugin({ className }: TransformerPluginProps) {
  const [activeTab, setActiveTab] = useState('attention');

  const attentionHeads = [
    { id: 1, pattern: 'Local', coverage: 85 },
    { id: 2, pattern: 'Global', coverage: 72 },
    { id: 3, pattern: 'Positional', coverage: 90 },
    { id: 4, pattern: 'Semantic', coverage: 68 },
    { id: 5, pattern: 'Syntactic', coverage: 78 },
    { id: 6, pattern: 'Copy', coverage: 45 },
    { id: 7, pattern: 'Induction', coverage: 82 },
    { id: 8, pattern: 'Mixed', coverage: 55 },
  ];

  const memoryScaling = [
    { seqLen: '512', memory: '128MB', status: 'normal' },
    { seqLen: '1024', memory: '512MB', status: 'normal' },
    { seqLen: '2048', memory: '2GB', status: 'warning' },
    { seqLen: '4096', memory: '8GB', status: 'critical' },
    { seqLen: '8192', memory: '32GB', status: 'critical' },
  ];

  return (
    <div className={cn("space-y-4 animate-fade-in", className)}>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-5 bg-secondary/50">
          <TabsTrigger value="attention" className="text-xs gap-1.5">
            <Focus className="w-3.5 h-3.5" />
            Attention
          </TabsTrigger>
          <TabsTrigger value="tokens" className="text-xs gap-1.5">
            <ArrowRight className="w-3.5 h-3.5" />
            Token Flow
          </TabsTrigger>
          <TabsTrigger value="params" className="text-xs gap-1.5">
            <Calculator className="w-3.5 h-3.5" />
            Params
          </TabsTrigger>
          <TabsTrigger value="memory" className="text-xs gap-1.5">
            <TrendingUp className="w-3.5 h-3.5" />
            Memory
          </TabsTrigger>
          <TabsTrigger value="residual" className="text-xs gap-1.5">
            <GitBranch className="w-3.5 h-3.5" />
            Residual
          </TabsTrigger>
        </TabsList>

        <TabsContent value="attention" className="mt-4 space-y-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Eye className="w-4 h-4 text-primary" />
                Attention Head Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-2">
                {attentionHeads.map((head) => (
                  <div 
                    key={head.id}
                    className="p-2 rounded-lg bg-secondary/30 border border-border/30 hover:border-primary/50 transition-colors cursor-pointer"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-muted-foreground">Head {head.id}</span>
                      <Badge variant="outline" className="text-[9px] px-1 py-0">
                        {head.pattern}
                      </Badge>
                    </div>
                    <Progress value={head.coverage} className="h-1" />
                    <span className="text-[9px] text-muted-foreground">{head.coverage}%</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Attention Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Attention Type</span>
                <Badge>Multi-Head (MHA)</Badge>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Num Heads</span>
                <span className="font-mono">8</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Head Dimension</span>
                <span className="font-mono">64</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Total Dimension</span>
                <span className="font-mono">512</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tokens" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Layers className="w-4 h-4 text-primary" />
                Token Processing Pipeline
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 overflow-x-auto pb-2">
                {['Input', 'Tokenize', 'Embed', 'Position', 'Attention', 'FFN', 'Output'].map((stage, i) => (
                  <div key={stage} className="flex items-center gap-2">
                    <div className="px-3 py-2 rounded-lg bg-secondary/50 border border-border/50 text-xs font-medium whitespace-nowrap">
                      {stage}
                    </div>
                    {i < 6 && <ArrowRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />}
                  </div>
                ))}
              </div>
              <div className="mt-4 p-3 rounded-lg bg-secondary/30 border border-border/30">
                <div className="text-xs text-muted-foreground mb-2">Current Configuration</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>Vocab Size: <span className="font-mono text-primary">50,257</span></div>
                  <div>Embed Dim: <span className="font-mono text-primary">768</span></div>
                  <div>Max Length: <span className="font-mono text-primary">2,048</span></div>
                  <div>Position: <span className="font-mono text-primary">RoPE</span></div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="params" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Cpu className="w-4 h-4 text-primary" />
                Parameter Breakdown
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { name: 'Embedding', params: '38.6M', pct: 25 },
                { name: 'Attention (QKV)', params: '28.3M', pct: 18 },
                { name: 'Attention (Output)', params: '9.4M', pct: 6 },
                { name: 'FFN', params: '75.5M', pct: 49 },
                { name: 'LayerNorm', params: '147K', pct: 0.1 },
                { name: 'LM Head', params: '2.8M', pct: 2 },
              ].map((item) => (
                <div key={item.name} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">{item.name}</span>
                    <span className="font-mono">{item.params}</span>
                  </div>
                  <Progress value={item.pct} className="h-1.5" />
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="memory" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-primary" />
                Memory Scaling (KV Cache)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {memoryScaling.map((item) => (
                  <div 
                    key={item.seqLen}
                    className={cn(
                      "flex items-center justify-between p-2 rounded-lg border text-xs",
                      item.status === 'normal' && "bg-success/10 border-success/30",
                      item.status === 'warning' && "bg-warning/10 border-warning/30",
                      item.status === 'critical' && "bg-destructive/10 border-destructive/30"
                    )}
                  >
                    <span className="font-mono">{item.seqLen} tokens</span>
                    <span className="font-medium">{item.memory}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 p-2 rounded bg-secondary/30 text-xs text-muted-foreground">
                💡 Consider GQA or MQA to reduce KV cache by 4-8x
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="residual" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <GitBranch className="w-4 h-4 text-primary" />
                Residual & Normalization Flow
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className="bg-success/10 border-success/50 text-success">Pre-LN</Badge>
                  <span className="text-xs text-muted-foreground">LayerNorm before attention/FFN (stable training)</span>
                </div>
                <div className="p-3 rounded-lg bg-secondary/30 border border-border/30">
                  <div className="text-xs font-medium mb-2">Residual Connections</div>
                  <div className="flex flex-col gap-1 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-success"></div>
                      <span>x + Attention(LN(x)) → stable gradient flow</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-success"></div>
                      <span>x + FFN(LN(x)) → prevents vanishing gradients</span>
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2 rounded bg-secondary/30">
                    <div className="text-muted-foreground">Norm Type</div>
                    <div className="font-mono">RMSNorm</div>
                  </div>
                  <div className="p-2 rounded bg-secondary/30">
                    <div className="text-muted-foreground">Epsilon</div>
                    <div className="font-mono">1e-6</div>
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
