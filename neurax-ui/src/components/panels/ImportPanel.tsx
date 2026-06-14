import { useState } from 'react';
import { Upload, FileJson, AlertCircle, CheckCircle2, Copy } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Textarea } from '@/components/ui/textarea.tsx';
import { Label } from '@/components/ui/label.tsx';
import { parseArchitectureJSON, sampleTransformerJSON, ImportResult } from '@/utils/architectureImporter.ts';
import { useToast } from '@/hooks/use-toast.ts';

interface ImportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onImport: (result: ImportResult) => void;
}

export function ImportPanel({ isOpen, onClose, onImport }: ImportPanelProps) {
  const [jsonInput, setJsonInput] = useState('');
  const [previewResult, setPreviewResult] = useState<ImportResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const { toast } = useToast();

  const handleValidate = () => {
    setIsValidating(true);

    setTimeout(() => {
      const result = parseArchitectureJSON(jsonInput);
      setPreviewResult(result);
      setIsValidating(false);

      if (result.error) {
        toast({
          title: 'Validation Failed',
          description: result.error,
          variant: 'destructive',
        });
      } else {
        toast({
          title: 'Validation Successful',
          description: `Found ${result.nodes.length} nodes and ${result.connections.length} connections`,
        });
      }
    }, 300);
  };

  const handleImport = () => {
    if (!previewResult || previewResult.error) {
      toast({
        title: 'Cannot Import',
        description: 'Please validate the JSON first',
        variant: 'destructive',
      });
      return;
    }

    onImport(previewResult);
    handleClose();

    toast({
      title: 'Architecture Imported',
      description: `Imported "${previewResult.modelName}" with ${previewResult.nodes.length} nodes`,
    });
  };

  const handleClose = () => {
    setJsonInput('');
    setPreviewResult(null);
    onClose();
  };

  const handleLoadSample = () => {
    setJsonInput(sampleTransformerJSON);
    setPreviewResult(null);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setJsonInput(content);
      setPreviewResult(null);
    };
    reader.readAsText(file);
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileJson className="w-5 h-5 text-primary" />
            Import Architecture JSON
          </DialogTitle>
          <DialogDescription>
            Paste your architecture JSON or upload a file to create a graph on the canvas.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-4 py-4">
          {/* File Upload */}
          <div className="flex items-center gap-2">
            <Label
              htmlFor="json-file"
              className="flex items-center gap-2 px-4 py-2 bg-secondary hover:bg-secondary/80 rounded-md cursor-pointer transition-colors"
            >
              <Upload className="w-4 h-4" />
              Upload JSON File
            </Label>
            <input
              id="json-file"
              type="file"
              accept=".json"
              className="hidden"
              onChange={handleFileUpload}
            />

            <Button variant="outline" size="sm" onClick={handleLoadSample}>
              <Copy className="w-4 h-4 mr-1.5" />
              Load Sample
            </Button>
          </div>

          {/* JSON Input */}
          <div className="space-y-2">
            <Label htmlFor="json-input">Architecture JSON</Label>
            <Textarea
              id="json-input"
              value={jsonInput}
              onChange={(e) => {
                setJsonInput(e.target.value);
                setPreviewResult(null);
              }}
              placeholder={`Paste your architecture JSON here...\n\nExample structure:\n{\n  "model": { "name": "my-model", "type": "transformer", "layers": [...] },\n  "training": { "batch_size": 64, "num_epochs": 100 },\n  "hardware": { "gpus": [{"name": "A100", "count": 8}] },\n  "data": { "dataset_size": 1000000000 }\n}`}
              className="min-h-[250px] font-mono text-xs"
            />
          </div>

          {/* Preview Result */}
          {previewResult && (
            <div
              className={`p-4 rounded-lg border ${previewResult.error
                  ? 'bg-destructive/10 border-destructive/30'
                  : 'bg-primary/10 border-primary/30'
                }`}
            >
              <div className="flex items-start gap-3">
                {previewResult.error ? (
                  <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1 min-w-0">
                  {previewResult.error ? (
                    <>
                      <p className="font-medium text-destructive">Validation Error</p>
                      <p className="text-sm text-muted-foreground mt-1">{previewResult.error}</p>
                    </>
                  ) : (
                    <>
                      <p className="font-medium text-foreground">
                        Model: {previewResult.modelName}
                      </p>
                      <div className="text-sm text-muted-foreground mt-1 space-y-1">
                        <p>• {previewResult.nodes.length} nodes will be created</p>
                        <p>• {previewResult.connections.length} connections will be created</p>
                        <p className="text-xs mt-2">
                          Layers: {previewResult.nodes.map(n => n.name).join(' → ')}
                        </p>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            variant="secondary"
            onClick={handleValidate}
            disabled={!jsonInput.trim() || isValidating}
          >
            {isValidating ? 'Validating...' : 'Validate'}
          </Button>
          <Button
            onClick={handleImport}
            disabled={!previewResult || !!previewResult.error}
          >
            Import Architecture
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
