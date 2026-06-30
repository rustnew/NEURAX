import { useState } from 'react';
import { 
  Github, 
  FolderGit2, 
  GitBranch, 
  FileCode,
  Check,
  X,
  Loader2,
  ExternalLink,
  Plus,
  Lock
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Checkbox } from '@/components/ui/checkbox.tsx';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog.tsx';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { canAccessExport, EXPORT_OPTIONS } from '@/types/plans.ts';
import { CanvasNode, Connection } from '@/types/architecture.ts';
import { generateCode, GeneratedCode } from '@/utils/codeGenerators.ts';
import { cn } from '@/lib/utils.ts';

interface GitHubExportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  nodes: CanvasNode[];
  connections: Connection[];
  modelName?: string;
}

type ExportFormat = 'pytorch' | 'onnx' | 'rust' | 'triton' | 'json';

interface ExportFormatOption {
  id: ExportFormat;
  name: string;
  extension: string;
  description: string;
  planRequired: 'essential' | 'architect' | 'elite';
}

const EXPORT_FORMATS: ExportFormatOption[] = [
  { id: 'pytorch', name: 'PyTorch', extension: '.py', description: 'Python model definition', planRequired: 'essential' },
  { id: 'onnx', name: 'ONNX Export', extension: '.py', description: 'ONNX export script', planRequired: 'essential' },
  { id: 'json', name: 'JSON Schema', extension: '.json', description: 'Architecture schema', planRequired: 'essential' },
  { id: 'rust', name: 'Rust / Burn', extension: '.rs', description: 'Rust model structure', planRequired: 'architect' },
  { id: 'triton', name: 'Triton Kernels', extension: '.py', description: 'Optimized GPU kernels', planRequired: 'architect' },
];

export function GitHubExportPanel({ 
  isOpen, 
  onClose, 
  nodes, 
  connections,
  modelName = 'GeneratedModel'
}: GitHubExportPanelProps) {
  const { toast } = useToast();
  const { currentPlan } = usePlan();
  
  // GitHub connection state
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting] = useState(false);
  
  // Export configuration
  const [selectedRepo, setSelectedRepo] = useState<string>('');
  const [branch, setBranch] = useState('main');
  const [directory, setDirectory] = useState('models/');
  const [commitMessage, setCommitMessage] = useState(`Add ${modelName} architecture from NEURAX`);
  const [createPR, setCreatePR] = useState(false);
  const [prBranch, setPrBranch] = useState(`neurax/${modelName.toLowerCase()}`);
  
  // Selected formats
  const [selectedFormats, setSelectedFormats] = useState<ExportFormat[]>(['pytorch', 'json']);
  
  // Export state
  const [isExporting] = useState(false);
  const [exportResult, setExportResult] = useState<{ success: boolean; url?: string } | null>(null);
  
  // Preview state
  const [previewCode, setPreviewCode] = useState<GeneratedCode | null>(null);

  const repos: Array<{ id: string; name: string; fullName: string; private: boolean }> = [];

  const handleConnect = async () => {
    toast({
      title: "GitHub Integration Not Available",
      description: "GitHub OAuth/repo listing is not implemented yet.",
      variant: "destructive",
    });
  };

  const handleDisconnect = () => {
    setIsConnected(false);
    setSelectedRepo('');
    toast({
      title: "GitHub Disconnected",
      description: "Your GitHub account has been unlinked",
    });
  };

  const toggleFormat = (format: ExportFormat) => {
    const exportOption = EXPORT_OPTIONS.find(e => e.id === format);
    if (exportOption && !canAccessExport(currentPlan, exportOption)) {
      toast({
        title: "Upgrade Required",
        description: `${format.toUpperCase()} export requires ${exportOption.minPlan.toUpperCase()} plan`,
        variant: "destructive",
      });
      return;
    }
    
    setSelectedFormats(prev => 
      prev.includes(format) 
        ? prev.filter(f => f !== format)
        : [...prev, format]
    );
  };

  const handlePreview = (format: ExportFormat) => {
    const code = generateCode(format, nodes, connections, { modelName });
    setPreviewCode(code);
  };

  const handleExport = async () => {
    if (!selectedRepo || selectedFormats.length === 0) {
      toast({
        title: "Missing Configuration",
        description: "Please select a repository and at least one format",
        variant: "destructive",
      });
      return;
    }

    toast({
      title: "Export Not Available",
      description: "GitHub export is disabled until GitHub integration is implemented.",
      variant: "destructive",
    });
    return;
  };

  const handleClose = () => {
    setExportResult(null);
    setPreviewCode(null);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col bg-card">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Github className="w-5 h-5" />
            Export to GitHub
          </DialogTitle>
          <DialogDescription>
            Push your generated code directly to your GitHub repository
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-6 py-4">
          {/* GitHub Connection Status */}
          <div className="p-4 rounded-lg border border-border bg-secondary/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "w-10 h-10 rounded-lg flex items-center justify-center",
                  isConnected ? "bg-success/20" : "bg-muted"
                )}>
                  <Github className={cn(
                    "w-5 h-5",
                    isConnected ? "text-success" : "text-muted-foreground"
                  )} />
                </div>
                <div>
                  <div className="font-medium text-sm">
                    {isConnected ? 'GitHub Connected' : 'Connect GitHub'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {isConnected 
                      ? 'Ready to push code to your repositories' 
                      : 'Link your GitHub account to export code'
                    }
                  </div>
                </div>
              </div>
              
              {isConnected ? (
                <Button variant="outline" size="sm" onClick={handleDisconnect}>
                  Disconnect
                </Button>
              ) : (
                <Button 
                  size="sm" 
                  onClick={handleConnect}
                  disabled={isConnecting}
                >
                  {isConnecting ? (
                    <>
                      <Loader2 key="loader" className="w-4 h-4 mr-2 animate-spin" />
                      Connecting...
                    </>
                  ) : (
                    <>
                      <Github key="github" className="w-4 h-4 mr-2" />
                      Connect
                    </>
                  )}
                </Button>
              )}
            </div>
          </div>

          {isConnected && (
            <>
              {/* Export Formats Selection */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Export Formats</Label>
                <div className="grid grid-cols-2 gap-2">
                  {EXPORT_FORMATS.map((format) => {
                    const exportOption = EXPORT_OPTIONS.find(e => e.id === format.id);
                    const hasAccess = exportOption ? canAccessExport(currentPlan, exportOption) : true;
                    const isSelected = selectedFormats.includes(format.id);
                    
                    return (
                      <div
                        key={format.id}
                        className={cn(
                          "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all",
                          isSelected && hasAccess
                            ? "border-primary bg-primary/5"
                            : hasAccess
                            ? "border-border hover:border-primary/50"
                            : "border-border/50 opacity-50 cursor-not-allowed"
                        )}
                        onClick={() => hasAccess && toggleFormat(format.id)}
                      >
                        <Checkbox 
                          checked={isSelected && hasAccess}
                          disabled={!hasAccess}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">{format.name}</span>
                            <Badge variant="outline" className="text-[9px]">
                              {format.extension}
                            </Badge>
                            {!hasAccess && (
                              <Lock className="w-3 h-3 text-muted-foreground" />
                            )}
                          </div>
                          <div className="text-[10px] text-muted-foreground truncate">
                            {format.description}
                          </div>
                        </div>
                        {hasAccess && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 text-[10px]"
                            onClick={(e) => {
                              e.stopPropagation();
                              handlePreview(format.id);
                            }}
                          >
                            Preview
                          </Button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Repository Selection */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Repository</Label>
                <Select value={selectedRepo} onValueChange={setSelectedRepo}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a repository" />
                  </SelectTrigger>
                  <SelectContent>
                    {repos.map((repo) => (
                      <SelectItem key={repo.id} value={repo.id}>
                        <div className="flex items-center gap-2">
                          <FolderGit2 className="w-4 h-4" />
                          <span className="truncate">{repo.fullName}</span>
                          {repo.private && <Lock className="w-3 h-3 text-muted-foreground" />}
                        </div>
                      </SelectItem>
                    ))}
                    <div className="p-2 border-t border-border">
                      <Button variant="ghost" size="sm" className="w-full justify-start text-muted-foreground">
                        <Plus className="w-4 h-4 mr-2" />
                        Create new repository
                      </Button>
                    </div>
                  </SelectContent>
                </Select>
              </div>

              {/* Branch & Directory */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm">Branch</Label>
                  <div className="relative">
                    <GitBranch className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input 
                      value={branch}
                      onChange={(e) => setBranch(e.target.value)}
                      className="pl-9"
                      placeholder="main"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm">Directory</Label>
                  <div className="relative">
                    <FileCode className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input 
                      value={directory}
                      onChange={(e) => setDirectory(e.target.value)}
                      className="pl-9"
                      placeholder="models/"
                    />
                  </div>
                </div>
              </div>

              {/* Commit Message */}
              <div className="space-y-2">
                <Label className="text-sm">Commit Message</Label>
                <Input 
                  value={commitMessage}
                  onChange={(e) => setCommitMessage(e.target.value)}
                  placeholder="Add model architecture"
                />
              </div>

              {/* Create PR Option */}
              <div className="flex items-center justify-between p-3 rounded-lg border border-border">
                <div className="space-y-0.5">
                  <Label className="text-sm font-medium">Create Pull Request</Label>
                  <p className="text-[10px] text-muted-foreground">
                    Create a new branch and open a PR instead of pushing directly
                  </p>
                </div>
                <Switch checked={createPR} onCheckedChange={setCreatePR} />
              </div>

              {createPR && (
                <div className="space-y-2 pl-4 border-l-2 border-primary/30">
                  <Label className="text-sm">PR Branch Name</Label>
                  <Input 
                    value={prBranch}
                    onChange={(e) => setPrBranch(e.target.value)}
                    placeholder="neurax/model-name"
                  />
                </div>
              )}

              {/* Code Preview */}
              {previewCode && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">
                      Preview: {previewCode.filename}
                    </Label>
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={() => setPreviewCode(null)}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="max-h-48 overflow-auto bg-background rounded-lg border border-border">
                    <pre className="p-3 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                      {previewCode.content}
                    </pre>
                  </div>
                </div>
              )}

              {/* Export Result */}
              {exportResult && (
                <div className={cn(
                  "p-4 rounded-lg border",
                  exportResult.success 
                    ? "bg-success/10 border-success/30" 
                    : "bg-destructive/10 border-destructive/30"
                )}>
                  <div className="flex items-center gap-2">
                    {exportResult.success ? (
                      <>
                        <Check className="w-5 h-5 text-success" />
                        <div className="flex-1">
                          <div className="font-medium text-sm text-success">Export Successful!</div>
                          <div className="text-xs text-muted-foreground">
                            Your code has been pushed to GitHub
                          </div>
                        </div>
                        {exportResult.url && (
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => window.open(exportResult.url, '_blank')}
                          >
                            <ExternalLink className="w-4 h-4 mr-1" />
                            View
                          </Button>
                        )}
                      </>
                    ) : (
                      <>
                        <X className="w-5 h-5 text-destructive" />
                        <div className="flex-1">
                          <div className="font-medium text-sm text-destructive">Export Failed</div>
                          <div className="text-xs text-muted-foreground">
                            Please check your connection and try again
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button 
            onClick={handleExport}
            disabled={!isConnected || !selectedRepo || selectedFormats.length === 0 || isExporting}
          >
            {isExporting ? (
              <>
                <Loader2 key="loader" className="w-4 h-4 mr-2 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Github key="github" className="w-4 h-4 mr-2" />
                Push to GitHub
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
