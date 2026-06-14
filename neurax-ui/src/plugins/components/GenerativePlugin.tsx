import { useState } from 'react';
import { 
  ArrowLeftRight, 
  Waves, 
  LineChart, 
  AlertTriangle, 
  PlayCircle,
  Sparkles,
  TrendingDown
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Progress } from '@/components/ui/progress.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { cn } from '@/lib/utils.ts';

interface GenerativePluginProps {
  className?: string;
}

export function GenerativePlugin({ className }: GenerativePluginProps) {
  const [activeTab, setActiveTab] = useState('interaction');

  const noiseSchedule = [
    { step: 0, noise: 1.0, signal: 0.0 },
    { step: 200, noise: 0.8, signal: 0.2 },
    { step: 400, noise: 0.5, signal: 0.5 },
    { step: 600, noise: 0.25, signal: 0.75 },
    { step: 800, noise: 0.1, signal: 0.9 },
    { step: 1000, noise: 0.0, signal: 1.0 },
  ];

  return (
    <div className={cn("space-y-4 animate-fade-in", className)}>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-5 bg-secondary/50">
          <TabsTrigger value="interaction" className="text-xs gap-1.5">
            <ArrowLeftRight className="w-3.5 h-3.5" />
            Gen↔Disc
          </TabsTrigger>
          <TabsTrigger value="noise" className="text-xs gap-1.5">
            <Waves className="w-3.5 h-3.5" />
            Noise
          </TabsTrigger>
          <TabsTrigger value="loss" className="text-xs gap-1.5">
            <LineChart className="w-3.5 h-3.5" />
            Loss
          </TabsTrigger>
          <TabsTrigger value="collapse" className="text-xs gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5" />
            Collapse
          </TabsTrigger>
          <TabsTrigger value="sampling" className="text-xs gap-1.5">
            <PlayCircle className="w-3.5 h-3.5" />
            Sampling
          </TabsTrigger>
        </TabsList>

        <TabsContent value="interaction" className="mt-4 space-y-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <ArrowLeftRight className="w-4 h-4 text-warning" />
                Generator ↔ Discriminator Flow
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between gap-4 mb-4">
                <div className="flex-1 p-4 rounded-lg bg-primary/10 border border-primary/30 text-center">
                  <Sparkles className="w-8 h-8 mx-auto mb-2 text-primary" />
                  <div className="text-sm font-medium">Generator</div>
                  <div className="text-[10px] text-muted-foreground mt-1">Creates fake samples</div>
                </div>
                <div className="flex flex-col items-center gap-1">
                  <ArrowLeftRight className="w-6 h-6 text-muted-foreground" />
                  <span className="text-[10px] text-muted-foreground">Adversarial</span>
                </div>
                <div className="flex-1 p-4 rounded-lg bg-warning/10 border border-warning/30 text-center">
                  <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-warning" />
                  <div className="text-sm font-medium">Discriminator</div>
                  <div className="text-[10px] text-muted-foreground mt-1">Real vs Fake classifier</div>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-2 rounded bg-secondary/30">
                  <span className="text-muted-foreground">G Learning Rate:</span>
                  <span className="font-mono ml-2">1e-4</span>
                </div>
                <div className="p-2 rounded bg-secondary/30">
                  <span className="text-muted-foreground">D Learning Rate:</span>
                  <span className="font-mono ml-2">4e-4</span>
                </div>
                <div className="p-2 rounded bg-secondary/30">
                  <span className="text-muted-foreground">G Steps:</span>
                  <span className="font-mono ml-2">1</span>
                </div>
                <div className="p-2 rounded bg-secondary/30">
                  <span className="text-muted-foreground">D Steps:</span>
                  <span className="font-mono ml-2">5</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="noise" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Waves className="w-4 h-4 text-warning" />
                Diffusion Noise Schedule
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 mb-4">
                {noiseSchedule.map((point, i) => (
                  <div key={point.step} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Step {point.step}</span>
                      <div className="flex gap-4">
                        <span className="text-destructive/80">Noise: {(point.noise * 100).toFixed(0)}%</span>
                        <span className="text-success">Signal: {(point.signal * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <div className="flex h-3 rounded overflow-hidden">
                      <div 
                        className="bg-gradient-to-r from-destructive/80 to-destructive/60"
                        style={{ width: `${point.noise * 100}%` }}
                      />
                      <div 
                        className="bg-gradient-to-r from-success/60 to-success"
                        style={{ width: `${point.signal * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="flex items-center gap-4 text-xs">
                <Badge variant="outline">Cosine Schedule</Badge>
                <span className="text-muted-foreground">β_start: 0.0001</span>
                <span className="text-muted-foreground">β_end: 0.02</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="loss" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <LineChart className="w-4 h-4 text-warning" />
                Training Stability & Loss
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-lg bg-primary/10 border border-primary/30">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingDown className="w-4 h-4 text-primary" />
                    <span className="text-xs font-medium">Generator Loss</span>
                  </div>
                  <div className="text-2xl font-mono font-bold">0.82</div>
                  <div className="text-[10px] text-success">↓ 12% from epoch 10</div>
                </div>
                <div className="p-3 rounded-lg bg-warning/10 border border-warning/30">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingDown className="w-4 h-4 text-warning" />
                    <span className="text-xs font-medium">Discriminator Loss</span>
                  </div>
                  <div className="text-2xl font-mono font-bold">0.45</div>
                  <div className="text-[10px] text-muted-foreground">Stable</div>
                </div>
              </div>
              
              <div className="p-3 rounded-lg bg-success/10 border border-success/30">
                <div className="text-xs font-medium mb-2">Stability Indicators</div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">G/D Loss Ratio</span>
                    <Badge variant="outline" className="text-success border-success/30">1.82 (Healthy)</Badge>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Gradient Variance</span>
                    <Badge variant="outline" className="text-success border-success/30">Low</Badge>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">D Accuracy on Real</span>
                    <span className="font-mono">68%</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="collapse" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-warning" />
                Mode Collapse Indicators
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-3 rounded-lg bg-success/10 border border-success/30">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full bg-success animate-pulse" />
                  <span className="text-sm font-medium text-success">No Mode Collapse Detected</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Generator output diversity is within healthy bounds.
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Output Diversity Score</span>
                  <span className="font-mono text-success">0.89</span>
                </div>
                <Progress value={89} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Inception Score</span>
                  <span className="font-mono">8.4</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">FID Score</span>
                  <span className="font-mono">12.4</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Unique Modes</span>
                  <span className="font-mono">~1000</span>
                </div>
              </div>
              
              <div className="p-2 rounded bg-secondary/30 text-xs text-muted-foreground">
                💡 Tip: Monitor D accuracy on fake samples. If it drops to 0%, mode collapse may be occurring.
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sampling" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <PlayCircle className="w-4 h-4 text-warning" />
                Sampling Pipeline
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-2 overflow-x-auto pb-2">
                  {['Noise', 'Denoise×T', 'Decode', 'Post-Process', 'Output'].map((stage, i) => (
                    <div key={stage} className="flex items-center gap-2">
                      <div className={cn(
                        "px-3 py-2 rounded-lg border text-xs font-medium whitespace-nowrap",
                        i === 1 ? "bg-warning/10 border-warning/30" : "bg-secondary/50 border-border/50"
                      )}>
                        {stage}
                      </div>
                      {i < 4 && <Sparkles className="w-3 h-3 text-muted-foreground flex-shrink-0" />}
                    </div>
                  ))}
                </div>
                
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-2 rounded bg-secondary/30">
                    <span className="text-muted-foreground">Sampler:</span>
                    <span className="font-mono ml-2">DDPM</span>
                  </div>
                  <div className="p-2 rounded bg-secondary/30">
                    <span className="text-muted-foreground">Steps:</span>
                    <span className="font-mono ml-2">50</span>
                  </div>
                  <div className="p-2 rounded bg-secondary/30">
                    <span className="text-muted-foreground">CFG Scale:</span>
                    <span className="font-mono ml-2">7.5</span>
                  </div>
                  <div className="p-2 rounded bg-secondary/30">
                    <span className="text-muted-foreground">ETA:</span>
                    <span className="font-mono ml-2">0.0</span>
                  </div>
                </div>
                
                <div className="p-3 rounded-lg bg-primary/10 border border-primary/30">
                  <div className="text-xs font-medium mb-1">Inference Speed</div>
                  <div className="flex items-center gap-4">
                    <span className="text-lg font-mono font-bold text-primary">2.4s</span>
                    <span className="text-xs text-muted-foreground">per 512×512 image @ 50 steps</span>
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
