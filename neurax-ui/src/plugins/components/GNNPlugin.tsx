import { useState } from 'react';
import { 
  Share2, 
  MessageSquare, 
  GitMerge, 
  TrendingUp, 
  Layers,
  Circle,
  Database
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { cn } from '@/lib/utils.ts';

interface GNNPluginProps {
  className?: string;
}

export function GNNPlugin({ className }: GNNPluginProps) {
  const [activeTab, setActiveTab] = useState('graph');

  const aggregationMethods = [
    { name: 'Sum', complexity: 'O(E)', memory: 'Low', selected: true },
    { name: 'Mean', complexity: 'O(E)', memory: 'Low', selected: false },
    { name: 'Max', complexity: 'O(E)', memory: 'Low', selected: false },
    { name: 'Attention', complexity: 'O(E×d)', memory: 'Medium', selected: false },
    { name: 'LSTM', complexity: 'O(E×d²)', memory: 'High', selected: false },
  ];

  const nodeTypes = [
    { type: 'User', count: 5000, color: 'hsl(199, 89%, 48%)' },
    { type: 'Item', count: 12000, color: 'hsl(340, 75%, 55%)' },
    { type: 'Category', count: 50, color: 'hsl(142, 71%, 45%)' },
  ];

  const edgeTypes = [
    { type: 'purchased', count: 45000 },
    { type: 'viewed', count: 120000 },
    { type: 'belongs_to', count: 12000 },
  ];

  return (
    <div className={cn("space-y-4 animate-fade-in", className)}>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-5 bg-secondary/50">
          <TabsTrigger value="graph" className="text-xs gap-1.5">
            <Share2 className="w-3.5 h-3.5" />
            Structure
          </TabsTrigger>
          <TabsTrigger value="message" className="text-xs gap-1.5">
            <MessageSquare className="w-3.5 h-3.5" />
            Messages
          </TabsTrigger>
          <TabsTrigger value="aggregation" className="text-xs gap-1.5">
            <GitMerge className="w-3.5 h-3.5" />
            Aggregate
          </TabsTrigger>
          <TabsTrigger value="scaling" className="text-xs gap-1.5">
            <TrendingUp className="w-3.5 h-3.5" />
            Scaling
          </TabsTrigger>
          <TabsTrigger value="hetero" className="text-xs gap-1.5">
            <Layers className="w-3.5 h-3.5" />
            Hetero
          </TabsTrigger>
        </TabsList>

        <TabsContent value="graph" className="mt-4 space-y-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Share2 className="w-4 h-4 text-[hsl(340,75%,55%)]" />
                Graph Structure Configuration
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-3 mb-4">
                <div className="p-3 rounded-lg bg-primary/10 border border-primary/30 text-center">
                  <Database className="w-6 h-6 mx-auto mb-1 text-primary" />
                  <div className="text-lg font-mono font-bold">17,050</div>
                  <div className="text-[10px] text-muted-foreground">Total Nodes</div>
                </div>
                <div className="p-3 rounded-lg bg-[hsl(340,75%,55%)]/10 border border-[hsl(340,75%,55%)]/30 text-center">
                  <Share2 className="w-6 h-6 mx-auto mb-1 text-[hsl(340,75%,55%)]" />
                  <div className="text-lg font-mono font-bold">177,000</div>
                  <div className="text-[10px] text-muted-foreground">Total Edges</div>
                </div>
                <div className="p-3 rounded-lg bg-success/10 border border-success/30 text-center">
                  <Circle className="w-6 h-6 mx-auto mb-1 text-success" />
                  <div className="text-lg font-mono font-bold">10.4</div>
                  <div className="text-[10px] text-muted-foreground">Avg Degree</div>
                </div>
              </div>
              
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Graph Type</span>
                  <Badge variant="outline">Heterogeneous</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Directed</span>
                  <span className="font-mono">Yes</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Self-loops</span>
                  <span className="font-mono">Allowed</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Feature Dim</span>
                  <span className="font-mono">128</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="message" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-[hsl(340,75%,55%)]" />
                Message Passing Visualization
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="p-4 rounded-lg bg-secondary/30 border border-border/30 mb-4">
                <div className="text-xs font-medium mb-3">Message Function</div>
                <div className="font-mono text-xs p-2 rounded bg-background/50">
                  m_ij = W_msg × concat(h_i, h_j, e_ij)
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between p-2 rounded bg-secondary/30">
                  <span className="text-xs">Message Layers</span>
                  <span className="font-mono text-sm">3</span>
                </div>
                <div className="flex items-center justify-between p-2 rounded bg-secondary/30">
                  <span className="text-xs">Hidden Channels</span>
                  <span className="font-mono text-sm">256</span>
                </div>
                <div className="flex items-center justify-between p-2 rounded bg-secondary/30">
                  <span className="text-xs">Message Dropout</span>
                  <span className="font-mono text-sm">0.1</span>
                </div>
                <div className="flex items-center justify-between p-2 rounded bg-secondary/30">
                  <span className="text-xs">Edge Features</span>
                  <Badge variant="outline" className="text-success border-success/30">Enabled</Badge>
                </div>
              </div>
              
              <div className="mt-4 p-2 rounded bg-primary/10 border border-primary/30 text-xs">
                <span className="text-primary font-medium">Architecture: </span>
                <span className="text-muted-foreground">GraphSAGE with edge features</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="aggregation" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <GitMerge className="w-4 h-4 text-[hsl(340,75%,55%)]" />
                Aggregation Functions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 mb-4">
                {aggregationMethods.map((method) => (
                  <div 
                    key={method.name}
                    className={cn(
                      "flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-colors",
                      method.selected 
                        ? "bg-[hsl(340,75%,55%)]/10 border-[hsl(340,75%,55%)]/50" 
                        : "bg-secondary/30 border-border/30 hover:border-border"
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "w-3 h-3 rounded-full",
                        method.selected ? "bg-[hsl(340,75%,55%)]" : "bg-muted-foreground/30"
                      )} />
                      <span className="text-xs font-medium">{method.name}</span>
                    </div>
                    <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                      <span className="font-mono">{method.complexity}</span>
                      <Badge variant="outline" className="text-[9px] px-1 py-0">{method.memory}</Badge>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="p-3 rounded-lg bg-secondary/30">
                <div className="text-xs font-medium mb-2">Update Function</div>
                <div className="font-mono text-[10px] p-2 rounded bg-background/50">
                  h_i' = ReLU(W × concat(h_i, AGG(m_j)))
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scaling" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-[hsl(340,75%,55%)]" />
                Per-Node Memory Scaling
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                {[
                  { nodes: '1K', edges: '10K', memory: '12MB', time: '15ms' },
                  { nodes: '10K', edges: '100K', memory: '120MB', time: '150ms' },
                  { nodes: '100K', edges: '1M', memory: '1.2GB', time: '1.5s' },
                  { nodes: '1M', edges: '10M', memory: '12GB', time: '15s' },
                ].map((row) => (
                  <div key={row.nodes} className="flex items-center justify-between p-2 rounded bg-secondary/30 text-xs">
                    <div className="flex items-center gap-4">
                      <span className="font-mono w-12">{row.nodes}</span>
                      <span className="text-muted-foreground w-12">{row.edges}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-mono">{row.memory}</span>
                      <span className="font-mono text-muted-foreground">{row.time}</span>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Memory per Node</span>
                  <span className="font-mono">~12KB</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Memory per Edge</span>
                  <span className="font-mono">~1.2KB</span>
                </div>
              </div>
              
              <div className="p-2 rounded bg-success/10 border border-success/30 text-xs">
                💡 Use mini-batch sampling (e.g., GraphSAINT) for graphs with 1M+ nodes
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hetero" className="mt-4">
          <Card className="bg-card/50 border-border/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Layers className="w-4 h-4 text-[hsl(340,75%,55%)]" />
                Heterogeneous Graph Support
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="text-xs font-medium mb-2">Node Types</div>
                <div className="space-y-2">
                  {nodeTypes.map((node) => (
                    <div key={node.type} className="flex items-center justify-between p-2 rounded bg-secondary/30">
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: node.color }}
                        />
                        <span className="text-xs">{node.type}</span>
                      </div>
                      <span className="font-mono text-xs">{node.count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <div className="text-xs font-medium mb-2">Edge Types (Relations)</div>
                <div className="space-y-2">
                  {edgeTypes.map((edge) => (
                    <div key={edge.type} className="flex items-center justify-between p-2 rounded bg-secondary/30">
                      <span className="text-xs font-mono">{edge.type}</span>
                      <span className="font-mono text-xs">{edge.count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="bg-[hsl(340,75%,55%)]/10 text-[hsl(340,75%,55%)] border-[hsl(340,75%,55%)]/30">
                  HeteroConv
                </Badge>
                <span className="text-xs text-muted-foreground">Type-specific message passing</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
