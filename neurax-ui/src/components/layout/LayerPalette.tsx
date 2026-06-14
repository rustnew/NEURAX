import { useState, useCallback, useRef, useMemo } from 'react';
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Search,
  X,
  ArrowRightToLine,
  ArrowRightFromLine,
  Layers,
  Grid3X3,
  Zap,
  Activity,
  Focus,
  AlignCenter,
  AlignJustify,
  GitBranch,
  Box,
  Network,
  Workflow,
  Wand2,
  Share2,
  Users,
  Route,
  Waves,
  MessageSquare,
  Repeat,
  Brain,
  Gamepad2,
  Shield,
  LucideIcon
} from 'lucide-react';
import { LayerConfig } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { getPluginLayers } from '@/plugins/registry.ts';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { cn } from '@/lib/utils.ts';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';

const iconMap: Record<string, LucideIcon> = {
  ArrowRightToLine,
  ArrowRightFromLine,
  Layers,
  Grid3X3,
  Zap,
  Activity,
  Focus,
  AlignCenter,
  AlignJustify,
  GitBranch,
  Box,
  Network,
  Workflow,
  Wand2,
  Share2,
  Users,
  Route,
  Waves,
  MessageSquare,
  Repeat,
  Brain,
  Gamepad2,
  Shield,
};

interface LayerPaletteProps {
  onDragStart: (config: LayerConfig) => void;
  selectedArchitecture?: ArchitectureFamily;
}

export function LayerPalette({ onDragStart, selectedArchitecture = 'transformer' }: LayerPaletteProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  const [height, setHeight] = useState<number | null>(null);
  const [width, setWidth] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const isResizing = useRef(false);
  const isResizingWidth = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    const startY = e.clientY;
    const startHeight = containerRef.current?.getBoundingClientRect().height ?? 400;

    const onMouseMove = (ev: MouseEvent) => {
      if (!isResizing.current) return;
      const delta = ev.clientY - startY;
      const newHeight = Math.max(80, startHeight + delta);
      setHeight(newHeight);
    };

    const onMouseUp = () => {
      isResizing.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, []);

  const handleWidthResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizingWidth.current = true;
    const startX = e.clientX;
    const startWidth = containerRef.current?.getBoundingClientRect().width ?? 256;

    const onMouseMove = (ev: MouseEvent) => {
      if (!isResizingWidth.current) return;
      const delta = ev.clientX - startX;
      const newWidth = Math.max(120, Math.min(480, startWidth + delta));
      setWidth(newWidth);
    };

    const onMouseUp = () => {
      isResizingWidth.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, []);

  const toggleGroup = useCallback((group: string) => {
    setCollapsedGroups(prev => ({ ...prev, [group]: !prev[group] }));
  }, []);

  const getIcon = (iconName: string) => {
    const Icon = iconMap[iconName];
    return Icon ? <Icon className="w-4 h-4" /> : null;
  };

  // Get architecture-specific layers from the plugin registry
  const architectureLayers = getPluginLayers(selectedArchitecture);

  // Group layers by their category field
  const groupedLayers: Record<string, LayerConfig[]> = {};
  architectureLayers.forEach(layer => {
    const cat = layer.category || 'Other';
    if (!groupedLayers[cat]) groupedLayers[cat] = [];
    groupedLayers[cat].push(layer);
  });
  const filteredGroups = useMemo(() => {
    const query = searchQuery.toLowerCase().trim();
    return Object.entries(groupedLayers)
      .map(([group, layers]) => {
        const filtered = query
          ? layers.filter(l => l.name.toLowerCase().includes(query) || l.description?.toLowerCase().includes(query))
          : layers;
        return [group, filtered] as [string, LayerConfig[]];
      })
      .filter(([_, layers]) => layers.length > 0);
  }, [groupedLayers, searchQuery]);

  return (
    <aside
      ref={containerRef}
      className={cn(
        "bg-sidebar border-r border-sidebar-border flex flex-col transition-[width] duration-300 relative",
        isCollapsed ? "w-12" : (width == null ? "w-48 md:w-64" : "")
      )}
      style={{
        height: height != null ? `${height}px` : '100%',
        ...(width != null && !isCollapsed ? { width: `${width}px` } : {})
      }}
    >
      {/* Header */}
      <div className="h-10 px-3 flex items-center justify-between border-b border-sidebar-border shrink-0">
        {!isCollapsed && (
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Blocks
          </span>
        )}
        <div className="flex items-center gap-1 ml-auto">
          {height != null && (
            <Button
              variant="ghost"
              size="icon"
              className="w-6 h-6 text-muted-foreground hover:text-foreground"
              onClick={() => setHeight(null)}
              title="Reset height"
            >
              <ChevronDown className="w-3 h-3" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="w-6 h-6 text-muted-foreground hover:text-foreground"
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </Button>
        </div>
      </div>

      {/* Search */}
      {!isCollapsed && (
        <div className="px-2 py-1.5 border-b border-sidebar-border shrink-0">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <Input
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search blocks…"
              className="h-7 pl-7 pr-7 text-xs bg-sidebar border-sidebar-border"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Block Groups */}
      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin py-2">
        {!isCollapsed ? (
          <>
            {filteredGroups.map(([group, layers]) => {
              const isGroupCollapsed = collapsedGroups[group];
              return (
                <div key={group} className="mb-1 animate-fade-in">
                  <button
                    onClick={() => toggleGroup(group)}
                    className="w-full px-3 py-1.5 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70 hover:text-muted-foreground transition-colors"
                  >
                    {isGroupCollapsed ? (
                      <ChevronRight className="w-3 h-3 shrink-0" />
                    ) : (
                      <ChevronDown className="w-3 h-3 shrink-0" />
                    )}
                    {group}
                    <span className="ml-auto text-[9px] font-normal tabular-nums">{layers.length}</span>
                  </button>
                  {!isGroupCollapsed && layers.map((layer) => (
                    <Tooltip key={layer.id}>
                      <TooltipTrigger asChild>
                        <div
                          className="layer-item mx-2"
                          draggable
                          onDragStart={(e) => {
                            e.dataTransfer.setData('layer', JSON.stringify(layer));
                            onDragStart(layer);
                          }}
                        >
                          <div className="layer-icon">
                            {getIcon(layer.icon)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium text-foreground truncate">
                              {layer.name}
                            </div>
                            <div className="text-[10px] text-muted-foreground truncate">
                              {layer.description}
                            </div>
                          </div>
                          {layer.hasActivation && (
                            <span className="text-[8px] px-1 py-0.5 rounded bg-accent/50 text-accent-foreground/60 border border-border/50 shrink-0">
                              act
                            </span>
                          )}
                        </div>
                      </TooltipTrigger>
                      {layer.tooltip && (
                        <TooltipContent side="right" className="max-w-[200px]">
                          <p className="text-xs">{layer.tooltip}</p>
                        </TooltipContent>
                      )}
                    </Tooltip>
                  ))}
                </div>
              );
            })}
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 py-2">
            {architectureLayers.map((layer) => (
              <Tooltip key={layer.id}>
                <TooltipTrigger asChild>
                  <div
                    className="w-8 h-8 rounded-md flex items-center justify-center text-primary bg-primary/10 border border-primary/20 cursor-grab hover:bg-primary/20 transition-colors"
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.setData('layer', JSON.stringify(layer));
                      onDragStart(layer);
                    }}
                  >
                    {getIcon(layer.icon)}
                  </div>
                </TooltipTrigger>
                <TooltipContent side="right">
                  <p className="text-xs font-medium">{layer.name}</p>
                  {layer.tooltip && <p className="text-[10px] text-muted-foreground">{layer.tooltip}</p>}
                </TooltipContent>
              </Tooltip>
            ))}
          </div>
        )}
      </div>

      {/* Resize handle */}
      {/* Bottom resize handle */}
      <div
        className="h-2 shrink-0 cursor-ns-resize flex items-center justify-center border-t border-sidebar-border hover:bg-primary/10 transition-colors group"
        onMouseDown={handleResizeStart}
      >
        <div className="w-8 h-0.5 rounded-full bg-muted-foreground/30 group-hover:bg-primary/50 transition-colors" />
      </div>

      {/* Right edge width resize handle */}
      {!isCollapsed && (
        <div
          className="absolute top-0 right-0 w-1.5 h-full cursor-ew-resize hover:bg-primary/20 transition-colors group/resize z-10"
          onMouseDown={handleWidthResizeStart}
        >
          <div className="absolute top-1/2 -translate-y-1/2 right-0 w-0.5 h-8 rounded-full bg-muted-foreground/30 group-hover/resize:bg-primary/50 transition-colors" />
        </div>
      )}
    </aside>
  );
}
