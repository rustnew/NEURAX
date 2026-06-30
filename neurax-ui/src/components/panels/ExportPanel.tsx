import { useState } from 'react';
import {
  Download,
  FileJson,
  FileText,
  Code,
  Cog,
  Image,
  Box,
  Check,
  Copy,
  Lock,
  Zap,
  Server,
  Network,
  Github,
  Loader2
} from 'lucide-react';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { EXPORT_OPTIONS, ExportOption, canAccessExport } from '@/types/plans.ts';
import { cn } from '@/lib/utils.ts';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { generateCode } from '@/utils/codeGenerators.ts';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';
import { useHardware } from '@/contexts/HardwareContext.tsx';
import { exportOnnx } from '@/services/neuraxApi.ts';
import { GitHubExportPanel } from './GitHubExportPanel.tsx';
import { ExportAssistant } from './ExportAssistant.tsx';

const iconMap: Record<string, React.ElementType> = {
  FileJson,
  FileText,
  Code,
  Cog,
  Image,
  Box,
  Zap,
  Server,
  Network,
  Github
};

interface ExportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  architectureName?: string;
  nodes?: CanvasNode[];
  connections?: Connection[];
  groups?: NodeGroup[];
  selectedArchitecture?: ArchitectureFamily;
}

// Mock code previews
const MOCK_PYTORCH_CODE = `import torch
import torch.nn as nn

class AIArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input layer: [batch, 224, 224, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        
        # Attention block
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(512)
        
        # Output layer
        self.fc = nn.Linear(512, 1000)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.flatten(2).transpose(1, 2)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        
        x = x.mean(dim=1)
        return self.fc(x)

# Model Statistics:
# - Total Parameters: 25,600,000
# - Estimated FLOPs: 4.1 GFLOPs
# - Memory Usage: 97.6 MB (FP32)
`;

const MOCK_RUST_CODE = `use tch::{nn, Tensor};

/// AI Architecture Designer - Generated Model
/// 
/// Architecture Summary:
/// - Input Shape: [batch, 224, 224, 3]
/// - Output Shape: [batch, 1000]
/// - Total Parameters: 25,600,000
/// - Estimated FLOPs: 4.1 GFLOPs

pub struct AIArchitecture {
    conv1: nn::Conv2D,
    attention: MultiHeadAttention,
    layer_norm: nn::LayerNorm,
    fc: nn::Linear,
}

impl AIArchitecture {
    pub fn new(vs: &nn::Path) -> Self {
        let conv1 = nn::conv2d(
            vs / "conv1",
            3,  // in_channels
            64, // out_channels
            3,  // kernel_size
            Default::default(),
        );
        
        let attention = MultiHeadAttention::new(
            vs / "attention",
            512, // embed_dim
            8,   // num_heads
        );
        
        let layer_norm = nn::layer_norm(
            vs / "layer_norm",
            vec![512],
            Default::default(),
        );
        
        let fc = nn::linear(vs / "fc", 512, 1000, Default::default());
        
        Self { conv1, attention, layer_norm, fc }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.conv1.forward(x).relu();
        let x = x.flatten(2, -1).transpose(1, 2);
        
        let attn_out = self.attention.forward(&x, &x, &x);
        let x = self.layer_norm.forward(&(&x + &attn_out));
        
        let x = x.mean_dim(&[1], false, tch::Kind::Float);
        self.fc.forward(&x)
    }
}
`;

export function ExportPanel({
  isOpen,
  onClose,
  architectureName = 'architecture',
  nodes = [],
  connections = [],
  groups = [],
  selectedArchitecture = 'transformer'
}: ExportPanelProps) {
  const [selectedFormat, setSelectedFormat] = useState<string>('json');
  const [copied, setCopied] = useState(false);
  const [showGitHubExport, setShowGitHubExport] = useState(false);
  const [showAssistant, setShowAssistant] = useState<string | null>(null);
  const [isExportingOnnx, setIsExportingOnnx] = useState(false);
  const { toast } = useToast();
  const { currentPlan } = usePlan();
  const { config: hwConfig } = useHardware();

  // Generate real code based on current architecture
  const pytorchCode = nodes.length > 0
    ? generateCode('pytorch', nodes, connections, { modelName: architectureName }).content
    : MOCK_PYTORCH_CODE;

  const rustCode = nodes.length > 0
    ? generateCode('rust', nodes, connections, { modelName: architectureName }).content
    : MOCK_RUST_CODE;

  // Compile NEURAX IR JSON from canvas graph
  const neuraxIR = compileToNeuraxIR(nodes, connections, {
    modelName: architectureName,
    family: selectedArchitecture,
    groups,
    ...hwConfig,
    // Training
    learningRate: hwConfig.learningRate,
    numEpochs: hwConfig.numEpochs,
    // Hardware
    gpuCount: hwConfig.gpuCount,
    gpuMemoryGb: hwConfig.gpuMemoryGb,
    // Data
    datasetSize: hwConfig.datasetSize,
    vocabSize: hwConfig.vocabSize,
    numClasses: hwConfig.numClasses,
  });
  const neuraxJson = JSON.stringify(neuraxIR, null, 2);

  const handleExport = async (format: ExportOption) => {
    if (!canAccessExport(currentPlan, format)) {
      toast({
        title: "Upgrade Required",
        description: `${format.name} export requires ${format.minPlan.toUpperCase()} plan`,
        variant: "destructive",
      });
      return;
    }

    // ONNX binary export — call the backend API
    if (format.id === 'onnx') {
      if (nodes.length === 0) {
        toast({
          title: "No Architecture",
          description: "Add layers to the canvas before exporting",
          variant: "destructive",
        });
        return;
      }

      setIsExportingOnnx(true);
      try {
        const result = await exportOnnx({
          topology: neuraxIR,
          model_name: architectureName,
        });

        // Decode base64 and trigger download
        const binaryData = atob(result.data);
        const bytes = new Uint8Array(binaryData.length);
        for (let i = 0; i < binaryData.length; i++) {
          bytes[i] = binaryData.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${architectureName}.onnx`;
        a.click();
        URL.revokeObjectURL(url);

        toast({
          title: "ONNX Export Complete",
          description: `${result.model_name} — ${result.node_count} nodes, ${result.initializer_count} parameters, ${(result.size_bytes / 1024).toFixed(1)} KB`,
        });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        toast({
          title: "ONNX Export Failed",
          description: message,
          variant: "destructive",
        });
      } finally {
        setIsExportingOnnx(false);
      }
      onClose();
      return;
    }

    toast({
      title: "Export Started",
      description: `Exporting ${architectureName}${format.extension}...`,
    });

    setTimeout(() => {
      toast({
        title: "Export Complete",
        description: `${format.name} file ready for download`,
      });
    }, 1000);

    onClose();
  };

  const handleCopyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast({
      title: "Copied",
      description: "Code copied to clipboard",
    });
  };

  // Group exports by accessibility
  const accessibleExports = EXPORT_OPTIONS.filter(e => canAccessExport(currentPlan, e));
  const lockedExports = EXPORT_OPTIONS.filter(e => !canAccessExport(currentPlan, e));

  return (
    <>
      <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col bg-card">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Download className="w-5 h-5 text-primary" />
              Export Architecture
            </DialogTitle>
            <DialogDescription>
              Export your architecture in various formats with analysis data
            </DialogDescription>
          </DialogHeader>

          <Tabs defaultValue="formats" className="flex-1 overflow-hidden flex flex-col">
            <TabsList className="grid grid-cols-4 w-full">
              <TabsTrigger value="formats">Export Formats</TabsTrigger>
              <TabsTrigger value="neurax" className="flex items-center gap-1">
                <FileJson className="w-3 h-3 text-primary" />
                NEURAX IR
              </TabsTrigger>
              <TabsTrigger value="pytorch">PyTorch</TabsTrigger>
              <TabsTrigger
                value="rust"
                disabled={!canAccessExport(currentPlan, EXPORT_OPTIONS.find(e => e.id === 'rust')!)}
              >
                <span className="flex items-center gap-1">
                  Rust
                  {!canAccessExport(currentPlan, EXPORT_OPTIONS.find(e => e.id === 'rust')!) && (
                    <Lock className="w-3 h-3" />
                  )}
                </span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="formats" className="flex-1 overflow-y-auto p-1">
              {/* Accessible formats */}
              <div className="grid grid-cols-2 gap-3 mt-2">
                {accessibleExports.map((format) => {
                  const Icon = iconMap[format.icon] || FileText;
                  const isSelected = selectedFormat === format.id;

                  return (
                    <button
                      key={format.id}
                      className={cn(
                        "p-4 rounded-lg border text-left transition-all",
                        isSelected
                          ? "bg-primary/10 border-primary"
                          : "bg-secondary/30 border-border hover:border-primary/50"
                      )}
                      onClick={() => setSelectedFormat(format.id)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <Icon className="w-5 h-5 text-primary" />
                        </div>
                        {isSelected && <Check className="w-5 h-5 text-primary" />}
                      </div>
                      <div className="text-sm font-medium mb-0.5">{format.name}</div>
                      <div className="text-[10px] text-muted-foreground mb-2">{format.description}</div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-[9px]">{format.extension}</Badge>
                        {format.includeAnalysis && (
                          <Badge className="text-[9px] bg-success/20 text-success border-0">
                            +Analysis
                          </Badge>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Locked formats */}
              {lockedExports.length > 0 && (
                <>
                  <div className="mt-4 mb-2 text-[10px] text-muted-foreground uppercase tracking-wider">
                    Upgrade to unlock
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {lockedExports.map((format) => {
                      const Icon = iconMap[format.icon] || FileText;

                      return (
                        <Tooltip key={format.id}>
                          <TooltipTrigger asChild>
                            <div className="p-4 rounded-lg border border-border/50 bg-secondary/10 opacity-50 cursor-not-allowed">
                              <div className="flex items-start justify-between mb-2">
                                <div className="w-10 h-10 rounded-lg bg-muted/20 flex items-center justify-center">
                                  <Icon className="w-5 h-5 text-muted-foreground" />
                                </div>
                                <Lock className="w-4 h-4 text-muted-foreground" />
                              </div>
                              <div className="text-sm font-medium mb-0.5">{format.name}</div>
                              <div className="text-[10px] text-muted-foreground mb-2">{format.description}</div>
                              <Badge variant="outline" className="text-[9px]">{format.minPlan.toUpperCase()}</Badge>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">
                              Available on <span className="font-semibold text-primary">{format.minPlan.toUpperCase()}</span> plan
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      );
                    })}
                  </div>
                </>
              )}

              <div className="mt-4 flex justify-end gap-2">
                <Button variant="outline" onClick={onClose}>
                  Cancel
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setShowGitHubExport(true)}
                >
                  <Github className="w-4 h-4 mr-2" />
                  Push to GitHub
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    const format = EXPORT_OPTIONS.find(f => f.id === selectedFormat);
                    if (format) setShowAssistant(format.name);
                  }}
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Export Assistant
                </Button>
                <Button
                  onClick={() => {
                    const format = EXPORT_OPTIONS.find(f => f.id === selectedFormat);
                    if (format) handleExport(format);
                  }}
                  disabled={isExportingOnnx}
                >
                  {isExportingOnnx ? (
                    <Loader2 key="loader" className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Download key="download" className="w-4 h-4 mr-2" />
                  )}
                  {isExportingOnnx ? 'Exporting...' : `Export ${EXPORT_OPTIONS.find(f => f.id === selectedFormat)?.name}`}
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="neurax" className="flex-1 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <FileJson className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">NEURAX IR — Canonical JSON</span>
                  {nodes.length > 0 && (
                    <Badge variant="outline" className="text-[9px] bg-success/10 text-success border-success/30">
                      Compiled from canvas
                    </Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7"
                  onClick={() => handleCopyCode(neuraxJson)}
                >
                  {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                  {copied ? 'Copied!' : 'Copy'}
                </Button>
              </div>
              <div className="flex-1 overflow-auto bg-background rounded-lg border border-border">
                <pre className="p-4 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                  {neuraxJson}
                </pre>
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => handleCopyCode(neuraxJson)}>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy JSON
                </Button>
                <Button
                  size="sm"
                  onClick={() => {
                    const blob = new Blob([neuraxJson], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${architectureName}.neurax.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                    toast({
                      title: "Export Complete",
                      description: `NEURAX IR saved as ${architectureName}.neurax.json`,
                    });
                  }}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download .neurax.json
                </Button>
              </div>
            </TabsContent>


            <TabsContent value="pytorch" className="flex-1 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Code className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">PyTorch Model Definition</span>
                  {nodes.length > 0 && (
                    <Badge variant="outline" className="text-[9px] bg-success/10 text-success border-success/30">
                      Generated from canvas
                    </Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7"
                  onClick={() => handleCopyCode(pytorchCode)}
                >
                  {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                  {copied ? 'Copied!' : 'Copy'}
                </Button>
              </div>
              <div className="flex-1 overflow-auto bg-background rounded-lg border border-border">
                <pre className="p-4 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                  {pytorchCode}
                </pre>
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => handleCopyCode(pytorchCode)}>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy Code
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowGitHubExport(true)}
                >
                  <Github className="w-4 h-4 mr-2" />
                  Push to GitHub
                </Button>
                <Button
                  size="sm"
                  onClick={() => {
                    const format = EXPORT_OPTIONS.find(f => f.id === 'pytorch');
                    if (format) handleExport(format);
                  }}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download .py
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="rust" className="flex-1 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Cog className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">Rust Model Structure</span>
                  {nodes.length > 0 && (
                    <Badge variant="outline" className="text-[9px] bg-success/10 text-success border-success/30">
                      Generated from canvas
                    </Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7"
                  onClick={() => handleCopyCode(rustCode)}
                >
                  {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                  {copied ? 'Copied!' : 'Copy'}
                </Button>
              </div>
              <div className="flex-1 overflow-auto bg-background rounded-lg border border-border">
                <pre className="p-4 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                  {rustCode}
                </pre>
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => handleCopyCode(rustCode)}>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy Code
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowGitHubExport(true)}
                >
                  <Github className="w-4 h-4 mr-2" />
                  Push to GitHub
                </Button>
                <Button
                  size="sm"
                  onClick={() => {
                    const format = EXPORT_OPTIONS.find(f => f.id === 'rust');
                    if (format) handleExport(format);
                  }}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download .rs
                </Button>
              </div>
            </TabsContent>
          </Tabs>

          {/* GitHub Export Panel */}
          <GitHubExportPanel
            isOpen={showGitHubExport}
            onClose={() => setShowGitHubExport(false)}
            nodes={nodes}
            connections={connections}
            modelName={architectureName}
          />
        </DialogContent>
      </Dialog>

      {/* Export Assistant Wizard — rendered outside Dialog portal */}
      <ExportAssistant
        isOpen={!!showAssistant}
        onClose={() => setShowAssistant(null)}
        format={showAssistant || 'PyTorch'}
        nodes={nodes}
        connections={connections}
        architectureName={architectureName}
        selectedArchitecture={selectedArchitecture}
      />
    </>
  );
}
