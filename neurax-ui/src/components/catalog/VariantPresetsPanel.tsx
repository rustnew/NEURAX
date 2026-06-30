import { useState, useCallback, useEffect } from 'react';
import {
  BookOpen,
  Copy,
  ChevronRight,
  Tag,
  Sparkles,
  AlertTriangle,
  Check,
  Plus,
  Trash2,
  Pencil,
  User,
  Loader2,
} from 'lucide-react';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { VariantPreset } from '@/types/catalog.ts';
import { getPresets, getPreset, PresetMetadata } from '@/services/neuraxApi.ts';
import { getBlockDefaults } from '@/utils/blockDefaults.ts';
import { useToast } from '@/hooks/use-toast.ts';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Textarea } from '@/components/ui/textarea.tsx';
import { cn } from '@/lib/utils.ts';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from '@/components/ui/dialog.tsx';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { LockedFeature } from '@/components/ui/locked-feature.tsx';
import { CanvasNode, Connection, ParameterValue } from '@/types/architecture.ts';
import { getPluginLayers } from '@/plugins/registry.ts';

interface VariantPresetsPanelProps {
  family: ArchitectureFamily;
  onLoadPreset: (preset: VariantPreset) => void;
  onClonePreset: (preset: VariantPreset) => void;
  currentPresetId?: string | null;
  hasDeviations?: boolean;
  currentNodes?: CanvasNode[];
  currentConnections?: Connection[];
}

export function VariantPresetsPanel({
  family,
  onLoadPreset,
  onClonePreset,
  currentPresetId,
  hasDeviations = false,
  currentNodes = [],
  currentConnections = [],
}: VariantPresetsPanelProps) {
  const [builtInPresets, setBuiltInPresets] = useState<PresetMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingIds, setLoadingIds] = useState<Record<string, boolean>>({});
  const { toast } = useToast();

  useEffect(() => {
    const fetchPresets = async () => {
      try {
        setLoading(true);
        const allMetadata = await getPresets();
        // Filter by family
        const familyMetadata = allMetadata.filter(m => m.family === family);
        setBuiltInPresets(familyMetadata);
      } catch (error) {
        console.error('Failed to fetch presets:', error);
        toast({
          title: "Failed to fetch presets",
          description: "Could not load architecture templates from backend.",
          variant: "destructive",
        });
      } finally {
        setLoading(false);
      }
    };
    fetchPresets();
  }, [family, toast]);

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [customTemplates, setCustomTemplates] = useState<VariantPreset[]>([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<VariantPreset | null>(null);
  const [templateName, setTemplateName] = useState('');
  const [templateDescription, setTemplateDescription] = useState('');
  const [templateTags, setTemplateTags] = useState('');
  const { canAccess } = usePlan();

  const canCreateTemplates = canAccess('architect');

  const familyCustomTemplates = customTemplates.filter(t => t.family === family);

  // Combine for total count
  const totalCount = builtInPresets.length + familyCustomTemplates.length;

  const normalizeTemplateNodes = useCallback((nodes: CanvasNode[]): CanvasNode[] => {
    const layerMap = new Map(getPluginLayers(family).map((layer) => [layer.type, layer]));

    return nodes.map((node) => {
      const config = layerMap.get(node.type);
      if (!config) return node;

      const params: Record<string, unknown> = {
        ...getBlockDefaults(node.type),
        ...(config.defaultParams ?? {}),
        ...(node.params ?? {}),
      };

      if (config.hasActivation && !('activation' in params)) {
        params.activation = 'none';
      }

      return {
        ...node,
        params: params as Record<string, ParameterValue>,
      };
    });
  }, [family]);

  const handleSaveTemplate = useCallback(() => {
    if (!templateName.trim()) return;

    const tags = templateTags
      .split(',')
      .map(t => t.trim())
      .filter(Boolean);

    if (editingTemplate) {
      setCustomTemplates(prev =>
        prev.map(t =>
          t.id === editingTemplate.id
            ? { ...t, name: templateName.trim(), description: templateDescription.trim(), tags }
            : t
        )
      );
    } else {
      const newTemplate: VariantPreset = {
        id: `custom-${Date.now()}`,
        name: templateName.trim(),
        family,
        description: templateDescription.trim(),
        tags: ['custom', ...tags],
        nodes: normalizeTemplateNodes(currentNodes).map(n => ({ ...n })),
        connections: currentConnections.map(c => ({ ...c })),
        defaultParams: {},
      };
      setCustomTemplates(prev => [...prev, newTemplate]);
    }

    setShowSaveDialog(false);
    setEditingTemplate(null);
    setTemplateName('');
    setTemplateDescription('');
    setTemplateTags('');
  }, [templateName, templateDescription, templateTags, editingTemplate, family, currentNodes, currentConnections, normalizeTemplateNodes]);

  const handleDeleteCustomTemplate = useCallback((id: string) => {
    setCustomTemplates(prev => prev.filter(t => t.id !== id));
    if (expandedId === id) setExpandedId(null);
  }, [expandedId]);

  const openEditDialog = useCallback((template: VariantPreset) => {
    setEditingTemplate(template);
    setTemplateName(template.name);
    setTemplateDescription(template.description);
    setTemplateTags(template.tags.filter(t => t !== 'custom').join(', '));
    setShowSaveDialog(true);
  }, []);

  const openSaveDialog = useCallback(() => {
    setEditingTemplate(null);
    setTemplateName('');
    setTemplateDescription('');
    setTemplateTags('');
    setShowSaveDialog(true);
  }, []);

  const isCustom = (id: string) => id.startsWith('custom-');

  const handleAction = async (id: string, action: 'load' | 'clone', presetData?: VariantPreset) => {
    if (presetData && isCustom(id)) {
      if (action === 'load') onLoadPreset(presetData);
      else onClonePreset(presetData);
      return;
    }

    try {
      setLoadingIds(prev => ({ ...prev, [id]: true }));
      const fullPreset = await getPreset(id);
      // Backend returns nodes/connections as any[] but they match CanvasNode/Connection
      if (action === 'load') onLoadPreset(fullPreset as unknown as VariantPreset);
      else onClonePreset(fullPreset as unknown as VariantPreset);
    } catch (error) {
      toast({
        title: `Failed to ${action} template`,
        description: "Communication with the backend failed.",
        variant: "destructive",
      });
    } finally {
      setLoadingIds(prev => ({ ...prev, [id]: false }));
    }
  };

  const renderPresetCard = (preset: PresetMetadata | VariantPreset) => {
    const isActive = currentPresetId === preset.id;
    const isExpanded = expandedId === preset.id;
    const custom = 'id' in preset && preset.id.startsWith('custom-');
    const isLoading = loadingIds[preset.id];

    // Type guard/helper for counts
    let nodesCount = 0;
    let connsCount = 0;

    if ('node_count' in preset) {
      nodesCount = preset.node_count;
      connsCount = preset.connection_count;
    } else if ('nodes' in preset) {
      nodesCount = (preset as VariantPreset).nodes.length;
      connsCount = (preset as VariantPreset).connections.length;
    }

    return (
      <div
        key={preset.id}
        className={cn(
          'rounded-lg border transition-all',
          isActive
            ? 'bg-primary/10 border-primary/50'
            : 'bg-secondary/30 border-border hover:border-primary/30'
        )}
      >
        {/* Preset Header */}
        <button
          className="w-full px-3 py-2.5 flex items-start gap-2.5 text-left"
          onClick={() => setExpandedId(isExpanded ? null : preset.id)}
        >
          <ChevronRight
            className={cn(
              'w-3.5 h-3.5 mt-0.5 text-muted-foreground transition-transform shrink-0',
              isExpanded && 'rotate-90'
            )}
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium truncate">
                {preset.name}
              </span>
              {custom && (
                <Badge variant="outline" className="text-[8px] px-1 py-0 h-3.5 border-primary/40 text-primary">
                  <User className="w-2 h-2 mr-0.5" />
                  Custom
                </Badge>
              )}
              {isActive && (
                <Check className="w-3.5 h-3.5 text-primary shrink-0" />
              )}
            </div>
            <p className="text-[10px] text-muted-foreground mt-0.5 line-clamp-2">
              {preset.description}
            </p>
          </div>
        </button>

        {/* Expanded Details */}
        {isExpanded && (
          <div className="px-3 pb-3 space-y-2.5 animate-fade-in">
            {/* Tags */}
            <div className="flex flex-wrap gap-1">
              {(preset as any).tags?.map((tag: string) => (
                <Badge
                  key={tag}
                  variant="outline"
                  className="text-[9px] px-1.5 py-0 h-4 bg-secondary/50"
                >
                  <Tag className="w-2.5 h-2.5 mr-0.5" />
                  {tag}
                </Badge>
              ))}
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <div className="p-1.5 rounded bg-background/50 border border-border/50">
                <span className="text-muted-foreground">Nodes: </span>
                <span className="font-mono font-medium">{nodesCount}</span>
              </div>
              <div className="p-1.5 rounded bg-background/50 border border-border/50">
                <span className="text-muted-foreground">Connections: </span>
                <span className="font-mono font-medium">{connsCount}</span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <Button
                size="sm"
                className="flex-1 h-7 text-[10px]"
                disabled={isLoading}
                onClick={(e) => {
                  e.stopPropagation();
                  handleAction(preset.id, 'load', custom ? (preset as VariantPreset) : undefined);
                }}
              >
                {isLoading ? (
                  <Loader2 key="loader" className="w-3 h-3 mr-1 animate-spin" />
                ) : (
                  <Sparkles key="sparkles" className="w-3 h-3 mr-1" />
                )}
                {isActive ? 'Reload' : 'Load Template'}
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-[10px]"
                disabled={isLoading}
                onClick={(e) => {
                  e.stopPropagation();
                  handleAction(preset.id, 'clone', custom ? (preset as VariantPreset) : undefined);
                }}
              >
                <Copy className="w-3 h-3 mr-1" />
                Clone
              </Button>
            </div>

            {/* Custom template actions */}
            {custom && (
              <div className="flex gap-2 pt-1 border-t border-border/50">
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex-1 h-6 text-[10px] text-muted-foreground hover:text-foreground"
                  onClick={(e) => {
                    e.stopPropagation();
                    openEditDialog(preset as VariantPreset);
                  }}
                >
                  <Pencil className="w-3 h-3 mr-1" />
                  Edit
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex-1 h-6 text-[10px] text-destructive hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteCustomTemplate(preset.id);
                  }}
                >
                  <Trash2 className="w-3 h-3 mr-1" />
                  Delete
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col bg-card overflow-hidden">
      {/* Header */}
      <div className="h-10 px-4 flex items-center justify-between border-b border-border bg-panel-header shrink-0">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-primary" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Templates
          </span>
          <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4">
            {totalCount}
          </Badge>
        </div>
        <div className="flex items-center gap-1.5">
          {currentPresetId && hasDeviations && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1 text-warning">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span className="text-[10px] font-medium">Modified</span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="left" className="text-xs">
                Design deviates from the original preset
              </TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>

      {/* Save as Template Button */}
      <div className="px-3 pt-3 pb-1 shrink-0">
        {canCreateTemplates ? (
          <Button
            variant="outline"
            size="sm"
            className="w-full h-8 text-xs border-dashed border-primary/30 text-primary hover:bg-primary/5 hover:border-primary/50"
            onClick={openSaveDialog}
            disabled={currentNodes.length === 0}
          >
            <Plus className="w-3.5 h-3.5 mr-1.5" />
            Save Current as Template
          </Button>
        ) : (
          <LockedFeature minPlan="architect" tooltipSide="bottom">
            <Button
              variant="outline"
              size="sm"
              className="w-full h-8 text-xs border-dashed"
              disabled
            >
              <Plus className="w-3.5 h-3.5 mr-1.5" />
              Save Current as Template
            </Button>
          </LockedFeature>
        )}
      </div>

      {/* Preset List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-2">
        {/* Loading state */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <Loader2 className="w-6 h-6 animate-spin mb-2" />
            <span className="text-xs">Fetching templates...</span>
          </div>
        )}

        {/* Custom templates section */}
        {!loading && familyCustomTemplates.length > 0 && (
          <>
            <div className="flex items-center gap-2 pb-1">
            </div>
          </>
        )}

        {/* Built-in section header (if there are other types) */}
        {!loading && familyCustomTemplates.length > 0 && builtInPresets.length > 0 && (
          <div className="flex items-center gap-2 pb-1">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Reference Architectures
            </span>
            <Badge variant="secondary" className="text-[8px] px-1 py-0 h-3.5">
              {builtInPresets.length}
            </Badge>
          </div>
        )}

        {!loading && builtInPresets.map(renderPresetCard)}
        {!loading && totalCount === 0 && (
          <div className="p-4 text-center text-sm text-muted-foreground">
            No presets available for this architecture family.
          </div>
        )}
      </div>

      {/* Save Template Dialog */}
      <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-base">
              {editingTemplate ? 'Edit Template' : 'Save as Template'}
            </DialogTitle>
            <DialogDescription className="text-xs text-muted-foreground">
              {editingTemplate
                ? 'Update the name, description, or tags for this template.'
                : `Save the current ${family} architecture (${currentNodes.length} nodes, ${currentConnections.length} connections) as a reusable template.`}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Name</label>
              <Input
                value={templateName}
                onChange={(e) => setTemplateName(e.target.value)}
                placeholder="e.g. My LLaMA Variant"
                className="h-8 text-sm"
                autoFocus
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Description</label>
              <Textarea
                value={templateDescription}
                onChange={(e) => setTemplateDescription(e.target.value)}
                placeholder="Brief description of this architecture variant..."
                className="min-h-[60px] text-sm resize-none"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Tags (comma-separated)</label>
              <Input
                value={templateTags}
                onChange={(e) => setTemplateTags(e.target.value)}
                placeholder="e.g. llm, causal, 7b"
                className="h-8 text-sm"
              />
            </div>
            {!editingTemplate && (
              <div className="rounded-md bg-secondary/50 border border-border/50 p-2.5 text-[10px] text-muted-foreground">
                <span className="font-medium text-foreground">Snapshot: </span>
                {currentNodes.length} nodes · {currentConnections.length} connections · {family}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="ghost" size="sm" onClick={() => setShowSaveDialog(false)}>
              Cancel
            </Button>
            <Button size="sm" onClick={handleSaveTemplate} disabled={!templateName.trim()}>
              {editingTemplate ? 'Update' : 'Save Template'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
