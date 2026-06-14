import { useState, useRef, useCallback, useEffect } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Trash2, Copy, Link, MousePointer2, Hand, Target, Layers } from 'lucide-react';
import { CanvasNode as CanvasNodeType, Connection, LayerConfig, NodeGroup } from '@/types/architecture.ts';
import { CanvasNode } from './CanvasNode.tsx';
import { GroupNode } from './GroupNode.tsx';
import { CanvasMinimap } from './CanvasMinimap.tsx';
import { SelectionBox } from './SelectionBox.tsx';
import { useMultiSelect } from '@/hooks/useMultiSelect.ts';
import { Button } from '@/components/ui/button.tsx';
import { Slider } from '@/components/ui/slider.tsx';
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuShortcut,
  ContextMenuTrigger,
} from '@/components/ui/context-menu.tsx';
import { cn } from '@/lib/utils.ts';

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 4;
const ZOOM_SENSITIVITY = 0.002;

interface ArchitectureCanvasProps {
  nodes: CanvasNodeType[];
  connections: Connection[];
  groups: NodeGroup[];
  selectedNodeId: string | null;
  onSelectNode: (id: string | null) => void;
  onUpdateNode: (id: string, updates: Partial<CanvasNodeType>) => void;
  onAddNode: (config: LayerConfig, x: number, y: number) => void;
  onDeleteNode: (id: string) => void;
  onDuplicateNode: (id: string) => void;
  onAddConnection: (from: string, to: string) => void;
  onDeleteConnection?: (id: string) => void;
  onGroupSelected?: (nodeIds: string[]) => void;
  onUngroupGroup?: (groupId: string) => void;
  onDeleteGroup?: (groupId: string) => void;
  onUpdateGroup?: (groupId: string, updates: Partial<NodeGroup>) => void;
}

export function ArchitectureCanvas({
  nodes,
  connections,
  groups,
  selectedNodeId,
  onSelectNode,
  onUpdateNode,
  onAddNode,
  onDeleteNode,
  onDuplicateNode,
  onAddConnection,
  onDeleteConnection,
  onGroupSelected,
  onUngroupGroup,
  onDeleteGroup,
  onUpdateGroup,
}: ArchitectureCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [isPanMode, setIsPanMode] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Multi-selection state
  const {
    selectedNodeIds,
    selectionBox,
    isSelecting,
    toggleNodeSelection,
    selectNode,
    clearSelection,
    selectAll,
    startBoxSelection,
    updateBoxSelection,
    endBoxSelection,
    isNodeSelected,
  } = useMultiSelect();

  // Track initial positions for group movement
  const [groupDragStart, setGroupDragStart] = useState<Map<string, { x: number; y: number }> | null>(null);

  // Connection drawing state
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionStart, setConnectionStart] = useState<{ nodeId: string; x: number; y: number } | null>(null);
  const [connectionEnd, setConnectionEnd] = useState<{ x: number; y: number } | null>(null);

  // Selected connection state
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [hoveredConnectionId, setHoveredConnectionId] = useState<string | null>(null);

  // Sync single selection with multi-selection
  useEffect(() => {
    if (selectedNodeId && !isNodeSelected(selectedNodeId)) {
      selectNode(selectedNodeId, false);
    }
  }, [selectedNodeId, isNodeSelected, selectNode]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Select all
      if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault();
        selectAll(nodes.map(n => n.id));
        return;
      }

      // Group selected (Ctrl+G)
      if ((e.ctrlKey || e.metaKey) && e.key === 'g') {
        e.preventDefault();
        if (selectedNodeIds.size >= 2 && onGroupSelected) {
          onGroupSelected(Array.from(selectedNodeIds));
          clearSelection();
        }
        return;
      }

      // Delete selected nodes or connection
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedConnectionId) {
          e.preventDefault();
          onDeleteConnection?.(selectedConnectionId);
          setSelectedConnectionId(null);
        } else if (selectedNodeIds.size > 0) {
          e.preventDefault();
          // Separate group IDs from regular node IDs
          const groupIds = groups.map(g => g.id);
          selectedNodeIds.forEach(id => {
            if (groupIds.includes(id)) {
              onDeleteGroup?.(id);
            } else {
              onDeleteNode(id);
            }
          });
          clearSelection();
          onSelectNode(null);
        } else if (selectedNodeId) {
          e.preventDefault();
          onDeleteNode(selectedNodeId);
        }
      }

      // Duplicate
      if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        if (selectedNodeIds.size > 0) {
          selectedNodeIds.forEach(id => onDuplicateNode(id));
        } else if (selectedNodeId) {
          onDuplicateNode(selectedNodeId);
        }
      }

      // Escape - clear selection
      if (e.key === 'Escape') {
        clearSelection();
        onSelectNode(null);
        setSelectedConnectionId(null);
        setIsConnecting(false);
        setConnectionStart(null);
        setConnectionEnd(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNodeId, selectedNodeIds, selectedConnectionId, nodes, onDeleteNode, onDeleteConnection, onDuplicateNode, onSelectNode, clearSelection, selectAll]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const data = e.dataTransfer.getData('layer');
    if (!data || !canvasRef.current) return;

    const config = JSON.parse(data) as LayerConfig;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left - offset.x) / zoom;
    const y = (e.clientY - rect.top - offset.y) / zoom;

    onAddNode(config, x, y);
  }, [offset, zoom, onAddNode]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    const isCanvasClick = target === canvasRef.current || target.classList.contains('canvas-dots') || target.classList.contains('canvas-transform-layer');

    if (isCanvasClick) {
      // Middle mouse button or space+click or pan mode = pan
      if (e.button === 1 || (e.button === 0 && e.shiftKey) || (e.button === 0 && isPanMode)) {
        e.preventDefault();
        setIsPanning(true);
        setPanStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
      } else if (e.button === 0 && !e.ctrlKey && !e.metaKey) {
        // Regular click = start box selection
        if (canvasRef.current) {
          const rect = canvasRef.current.getBoundingClientRect();
          const x = (e.clientX - rect.left - offset.x) / zoom;
          const y = (e.clientY - rect.top - offset.y) / zoom;
          startBoxSelection(x, y);
        }
        clearSelection();
        onSelectNode(null);
        setSelectedConnectionId(null);
      }
    }
  };

  const handleConnectionClick = (connectionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedConnectionId(connectionId);
    clearSelection();
    onSelectNode(null);
  };

  const handleDeleteConnection = (connectionId: string) => {
    onDeleteConnection?.(connectionId);
    setSelectedConnectionId(null);
  };

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      setOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    } else if (isSelecting && canvasRef.current) {
      // Update box selection
      const rect = canvasRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left - offset.x) / zoom;
      const y = (e.clientY - rect.top - offset.y) / zoom;
      updateBoxSelection(x, y);
    } else if (draggedNode && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left - offset.x) / zoom;
      const mouseY = (e.clientY - rect.top - offset.y) / zoom;

      // Check if dragging a group
      if (draggedNode.startsWith('__group__')) {
        const groupId = draggedNode.replace('__group__', '');
        const x = mouseX - dragOffset.x;
        const y = mouseY - dragOffset.y;
        onUpdateGroup?.(groupId, { x, y });
      } else if (groupDragStart && selectedNodeIds.size > 1 && isNodeSelected(draggedNode)) {
        // Move all selected nodes
        const dx = mouseX - dragOffset.x - (groupDragStart.get(draggedNode)?.x || 0);
        const dy = mouseY - dragOffset.y - (groupDragStart.get(draggedNode)?.y || 0);

        groupDragStart.forEach((startPos, nodeId) => {
          onUpdateNode(nodeId, {
            x: startPos.x + dx,
            y: startPos.y + dy
          });
        });
      } else {
        // Single node movement
        const x = mouseX - dragOffset.x;
        const y = mouseY - dragOffset.y;
        onUpdateNode(draggedNode, { x, y });
      }
    } else if (isConnecting && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      setConnectionEnd({
        x: (e.clientX - rect.left - offset.x) / zoom,
        y: (e.clientY - rect.top - offset.y) / zoom,
      });
    }
  }, [isPanning, panStart, draggedNode, dragOffset, offset, zoom, onUpdateNode, isConnecting, isSelecting, updateBoxSelection, groupDragStart, selectedNodeIds, isNodeSelected]);

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
    setDraggedNode(null);
    setGroupDragStart(null);

    if (isSelecting) {
      endBoxSelection(nodes, offset, zoom);
    }

    if (isConnecting) {
      setIsConnecting(false);
      setConnectionStart(null);
      setConnectionEnd(null);
    }
  }, [isSelecting, isConnecting, endBoxSelection, nodes, offset, zoom]);

  const handleNodeDragStart = (nodeId: string, e: React.MouseEvent) => {
    if (e.button !== 0) return;

    const node = nodes.find(n => n.id === nodeId);
    if (!node || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left - offset.x) / zoom;
    const mouseY = (e.clientY - rect.top - offset.y) / zoom;

    setDraggedNode(nodeId);
    setDragOffset({ x: mouseX - node.x, y: mouseY - node.y });

    // If this node is part of a multi-selection, set up group drag
    if (selectedNodeIds.size > 1 && isNodeSelected(nodeId)) {
      const startPositions = new Map<string, { x: number; y: number }>();
      selectedNodeIds.forEach(id => {
        const n = nodes.find(node => node.id === id);
        if (n) {
          startPositions.set(id, { x: n.x, y: n.y });
        }
      });
      setGroupDragStart(startPositions);
    } else {
      setGroupDragStart(null);
    }
  };

  const handleNodeClick = (nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    if (e.ctrlKey || e.metaKey) {
      // Ctrl/Cmd + click = toggle selection
      toggleNodeSelection(nodeId, true);
    } else if (e.shiftKey && selectedNodeIds.size > 0) {
      // Shift + click = add to selection
      selectNode(nodeId, true);
    } else {
      // Regular click = single select
      selectNode(nodeId, false);
      onSelectNode(nodeId);
    }
  };

  const handleConnectionStart = (nodeId: string, isOutput: boolean, e: React.MouseEvent) => {
    e.stopPropagation();
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    const x = isOutput ? node.x + 176 : node.x;
    const y = node.y + 40;

    setIsConnecting(true);
    setConnectionStart({ nodeId, x, y });
    setConnectionEnd({ x, y });
  };

  const handleConnectionEnd = (nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (isConnecting && connectionStart && connectionStart.nodeId !== nodeId) {
      onAddConnection(connectionStart.nodeId, nodeId);
    }
    setIsConnecting(false);
    setConnectionStart(null);
    setConnectionEnd(null);
  };

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();

    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Pinch-to-zoom (trackpad) or Ctrl+wheel
    if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) < 50) {
      const delta = -e.deltaY * ZOOM_SENSITIVITY;
      const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom * (1 + delta)));

      // Zoom toward cursor position
      const zoomRatio = newZoom / zoom;
      const newOffsetX = mouseX - (mouseX - offset.x) * zoomRatio;
      const newOffsetY = mouseY - (mouseY - offset.y) * zoomRatio;

      setZoom(newZoom);
      setOffset({ x: newOffsetX, y: newOffsetY });
    } else {
      // Regular scroll = pan
      setOffset(prev => ({
        x: prev.x - e.deltaX,
        y: prev.y - e.deltaY,
      }));
    }
  }, [zoom, offset]);

  // Handle passive wheel events separately to allow preventDefault
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  // Card widths must match CSS exactly:
  // CanvasNode  → min-w-[200px] → 200px
  // GroupNode   → min-w-[240px] → 240px
  // Port circle is `absolute -right-3 w-6` → center at card.x + cardWidth
  // Port circle is `absolute -left-3 w-6`  → center at card.x
  const NODE_WIDTH = 200;
  const GROUP_WIDTH = 240;

  /** Get position info for a node OR a group (by virtual group ID) */
  const getNodeOrGroupPos = (id: string): { x: number; y: number; width: number; height: number } | null => {
    const node = nodes.find(n => n.id === id);
    if (node) return { x: node.x, y: node.y, width: NODE_WIDTH, height: 120 };
    const group = groups.find(g => g.id === id);
    if (group) return { x: group.x, y: group.y, width: GROUP_WIDTH, height: 200 };
    return null;
  };

  /** Build a smooth cubic-bezier SVG path between two port-dot positions */
  const buildPath = (fromX: number, fromY: number, toX: number, toY: number): string => {
    const dx = Math.abs(toX - fromX);
    const dy = Math.abs(toY - fromY);
    const controlOffset = Math.max(60, Math.min(150, dx * 0.5));
    if (toX < fromX || dy > dx) {
      const curveStrength = Math.max(80, dx * 0.3);
      return `M ${fromX} ${fromY} C ${fromX + curveStrength} ${fromY}, ${toX - curveStrength} ${toY}, ${toX} ${toY}`;
    }
    return `M ${fromX} ${fromY} C ${fromX + controlOffset} ${fromY}, ${toX - controlOffset} ${toY}, ${toX} ${toY}`;
  };




  /** Connection path using virtual node-or-group positions */
  const getConnectionPathById = (fromId: string, toId: string): string => {
    const from = getNodeOrGroupPos(fromId);
    const to = getNodeOrGroupPos(toId);
    if (!from || !to) return '';
    // Output port center = from.x + from.width; input port center = to.x
    const fromX = from.x + from.width;
    const fromY = from.y + from.height / 2;
    const toX = to.x;
    const toY = to.y + to.height / 2;
    return buildPath(fromX, fromY, toX, toY);
  };

  const getTempConnectionPath = () => {
    if (!connectionStart || !connectionEnd) return '';
    const { x: startX, y: startY } = connectionStart;
    const { x: endX, y: endY } = connectionEnd;
    const dx = Math.abs(endX - startX);
    const controlOffset = Math.max(40, Math.min(120, dx * 0.4));
    return `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`;
  };

  const handleMinimapClick = (x: number, y: number) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    setOffset({ x: -x * zoom + rect.width / 2, y: -y * zoom + rect.height / 2 });
  };

  const handleDeleteSelected = () => {
    selectedNodeIds.forEach(id => onDeleteNode(id));
    clearSelection();
    onSelectNode(null);
  };

  const handleDuplicateSelected = () => {
    selectedNodeIds.forEach(id => onDuplicateNode(id));
  };

  const handleZoomToFit = useCallback(() => {
    if (!canvasRef.current || nodes.length === 0) {
      setZoom(1);
      setOffset({ x: 0, y: 0 });
      return;
    }

    const rect = canvasRef.current.getBoundingClientRect();
    const padding = 100;

    // Calculate bounds of all nodes
    const bounds = nodes.reduce(
      (acc, node) => ({
        minX: Math.min(acc.minX, node.x),
        minY: Math.min(acc.minY, node.y),
        maxX: Math.max(acc.maxX, node.x + 176),
        maxY: Math.max(acc.maxY, node.y + 80),
      }),
      { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity }
    );

    const contentWidth = bounds.maxX - bounds.minX;
    const contentHeight = bounds.maxY - bounds.minY;

    const scaleX = (rect.width - padding * 2) / contentWidth;
    const scaleY = (rect.height - padding * 2) / contentHeight;
    const newZoom = Math.min(Math.max(MIN_ZOOM, Math.min(scaleX, scaleY)), 1.5);

    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;

    setZoom(newZoom);
    setOffset({
      x: rect.width / 2 - centerX * newZoom,
      y: rect.height / 2 - centerY * newZoom,
    });
  }, [nodes]);

  // Space bar for pan mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat) {
        const t = e.target as HTMLElement | null;
        const tag = t?.tagName?.toLowerCase();
        const isTypingTarget =
          tag === 'input' ||
          tag === 'textarea' ||
          (t as any)?.isContentEditable === true;
        if (isTypingTarget) return;
        e.preventDefault();
        setIsPanMode(true);
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setIsPanMode(false);
        setIsPanning(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  return (
    <div className="h-full w-full relative bg-canvas-bg overflow-hidden">
      {/* Zoom Controls - Bottom Left */}
      <div className="absolute bottom-4 left-4 z-20 flex flex-col gap-2">
        {/* Zoom slider and controls */}
        <div className="flex items-center gap-2 bg-card/95 backdrop-blur-md rounded-xl border border-border/50 shadow-lg p-2">
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 rounded-lg hover:bg-secondary"
            onClick={() => setZoom(z => Math.max(MIN_ZOOM, z / 1.2))}
          >
            <ZoomOut className="w-4 h-4" />
          </Button>

          <div className="w-24">
            <Slider
              value={[zoom]}
              min={MIN_ZOOM}
              max={MAX_ZOOM}
              step={0.01}
              onValueChange={([v]) => setZoom(v)}
              className="cursor-pointer"
            />
          </div>

          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 rounded-lg hover:bg-secondary"
            onClick={() => setZoom(z => Math.min(MAX_ZOOM, z * 1.2))}
          >
            <ZoomIn className="w-4 h-4" />
          </Button>

          <div className="w-px h-6 bg-border" />

          <span className="text-xs font-mono w-12 text-center text-muted-foreground select-none">
            {Math.round(zoom * 100)}%
          </span>
        </div>

        {/* Quick actions */}
        <div className="flex items-center gap-1 bg-card/95 backdrop-blur-md rounded-xl border border-border/50 shadow-lg p-1">
          <Button
            variant={isPanMode ? "secondary" : "ghost"}
            size="icon"
            className="w-8 h-8 rounded-lg"
            onClick={() => setIsPanMode(!isPanMode)}
            title="Pan mode (hold Space)"
          >
            <Hand className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 rounded-lg"
            onClick={handleZoomToFit}
            title="Zoom to fit"
          >
            <Target className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 rounded-lg"
            onClick={() => { setZoom(1); setOffset({ x: 0, y: 0 }); }}
            title="Reset view"
          >
            <Maximize2 className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Selection toolbar - Top Center */}
      {selectedNodeIds.size > 1 && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 bg-card/95 backdrop-blur-md border border-border/50 shadow-xl rounded-xl px-4 py-2">
          <div className="flex items-center gap-2">
            <MousePointer2 className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">
              {selectedNodeIds.size} selected
            </span>
          </div>
          <div className="w-px h-5 bg-border" />
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-3 text-xs rounded-lg hover:bg-secondary"
            onClick={handleDuplicateSelected}
          >
            <Copy className="w-3.5 h-3.5 mr-1.5" />
            Duplicate
          </Button>
          {onGroupSelected && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-3 text-xs rounded-lg hover:bg-primary/10 text-primary"
              onClick={() => {
                onGroupSelected(Array.from(selectedNodeIds));
                clearSelection();
              }}
            >
              <Layers className="w-3.5 h-3.5 mr-1.5" />
              Group ×N
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-3 text-xs text-destructive hover:bg-destructive/10 hover:text-destructive rounded-lg"
            onClick={handleDeleteSelected}
          >
            <Trash2 className="w-3.5 h-3.5 mr-1.5" />
            Delete
          </Button>
        </div>
      )}

      {/* Keyboard hints - Bottom Right above minimap */}
      <div className="absolute bottom-28 right-4 z-10 flex flex-col items-end gap-1 text-[10px] text-muted-foreground/40">
        <span>Space: Pan</span>
        <span>Scroll: Pan • Pinch: Zoom</span>
        <span>Ctrl+A: Select all • Ctrl+G: Group</span>
      </div>

      {/* Mini-map */}
      <CanvasMinimap
        nodes={nodes}
        connections={connections}
        viewportOffset={offset}
        zoom={zoom}
        onClick={handleMinimapClick}
      />

      {/* Canvas Area - Infinite Board */}
      <div
        ref={canvasRef}
        className={cn(
          "absolute inset-0 canvas-dots transition-[cursor] duration-100",
          isPanMode && !isPanning && "cursor-grab",
          isPanning && "cursor-grabbing",
          !isPanMode && !isPanning && !isConnecting && !isSelecting && "cursor-default",
          isConnecting && "cursor-crosshair",
          isSelecting && "cursor-crosshair"
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div
          className="absolute inset-0 canvas-transform-layer"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
            transformOrigin: '0 0',
          }}
        >
          {/* Connections SVG Layer — always behind node/group cards */}
          <svg
            className="absolute inset-0 w-full h-full"
            style={{ overflow: 'visible', pointerEvents: 'none', zIndex: 0 }}
          >
            {/* Arrow marker definitions */}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="12"
                markerHeight="10"
                refX="10"
                refY="5"
                orient="auto"
                markerUnits="userSpaceOnUse"
              >
                <path
                  d="M 0 0 L 12 5 L 0 10 L 3 5 Z"
                  fill="hsl(var(--primary))"
                />
              </marker>
              <marker
                id="arrowhead-highlighted"
                markerWidth="14"
                markerHeight="12"
                refX="12"
                refY="6"
                orient="auto"
                markerUnits="userSpaceOnUse"
              >
                <path
                  d="M 0 0 L 14 6 L 0 12 L 4 6 Z"
                  fill="hsl(var(--primary))"
                />
              </marker>
              {/* Glow filter for highlighted connections */}
              <filter id="connection-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {(() => {
              // Track already-rendered group→target pairs to deduplicate
              const renderedPairs = new Set<string>();
              return connections.map((conn) => {
                // Resolve endpoints: group ID can appear directly (new port style) or via child node (old style)
                const fromIsGroupDirect = groups.some(g => g.id === conn.from);
                const toIsGroupDirect = groups.some(g => g.id === conn.to);
                const fromGroupId = fromIsGroupDirect ? conn.from : groups.find(g => g.nodeIds.includes(conn.from))?.id;
                const toGroupId = toIsGroupDirect ? conn.to : groups.find(g => g.nodeIds.includes(conn.to))?.id;
                const resolvedFrom = fromGroupId ?? conn.from;
                const resolvedTo = toGroupId ?? conn.to;

                // Hide internal group connections (both endpoints in same group)
                if (fromGroupId && fromGroupId === toGroupId) return null;

                // Skip if resolved endpoint can't be positioned (dangling reference)
                const fromPos = getNodeOrGroupPos(resolvedFrom);
                const toPos = getNodeOrGroupPos(resolvedTo);
                if (!fromPos || !toPos) return null;

                // Deduplicate: when internal child→child connections collapse to same pair
                const dedupeKey = `${resolvedFrom}::${resolvedTo}`;
                if (renderedPairs.has(dedupeKey)) return null;
                renderedPairs.add(dedupeKey);

                const isSelected = selectedConnectionId === conn.id;
                const isHovered = hoveredConnectionId === conn.id;
                const isHighlighted = isSelected || isHovered || selectedNodeId === conn.from || selectedNodeId === conn.to ||
                  isNodeSelected(conn.from) || isNodeSelected(conn.to) ||
                  selectedNodeId === resolvedFrom || selectedNodeId === resolvedTo;
                const path = getConnectionPathById(resolvedFrom, resolvedTo);

                return (
                  <g key={conn.id} className="cursor-pointer">
                    {/* Invisible wider hit area for clicking */}
                    <path
                      d={path}
                      fill="none"
                      stroke="transparent"
                      strokeWidth={20}
                      strokeLinecap="round"
                      style={{ pointerEvents: 'stroke', cursor: 'pointer' }}
                      onClick={(e) => handleConnectionClick(conn.id, e)}
                      onMouseEnter={() => setHoveredConnectionId(conn.id)}
                      onMouseLeave={() => setHoveredConnectionId(null)}
                    />
                    {/* Shadow/glow layer */}
                    <path
                      d={path}
                      fill="none"
                      stroke={isSelected ? "hsl(var(--destructive) / 0.3)" : "hsl(var(--primary) / 0.15)"}
                      strokeWidth={isHighlighted ? 10 : 5}
                      strokeLinecap="round"
                      style={{
                        filter: isHighlighted ? 'url(#connection-glow)' : undefined,
                        pointerEvents: 'none'
                      }}
                    />
                    {/* Main connection line */}
                    <path
                      d={path}
                      fill="none"
                      stroke={isSelected ? "hsl(var(--destructive))" : "hsl(var(--primary))"}
                      strokeWidth={isHighlighted ? 3 : 2}
                      strokeLinecap="round"
                      opacity={isHighlighted ? 1 : 0.7}
                      markerEnd={isHighlighted ? "url(#arrowhead-highlighted)" : "url(#arrowhead)"}
                      style={{ pointerEvents: 'none' }}
                    />
                    {/* Animated flow particles */}
                    <circle r={isHighlighted ? 5 : 4} fill={isSelected ? "hsl(var(--destructive))" : "hsl(var(--primary))"} opacity={0.9} style={{ pointerEvents: 'none' }}>
                      <animateMotion
                        dur="1.5s"
                        repeatCount="indefinite"
                        path={path}
                      />
                    </circle>
                    <circle r={isHighlighted ? 3 : 2} fill="hsl(var(--background))" opacity={0.8} style={{ pointerEvents: 'none' }}>
                      <animateMotion
                        dur="1.5s"
                        repeatCount="indefinite"
                        path={path}
                      />
                    </circle>
                    {/* Delete button when selected */}
                    {isSelected && (
                      <g
                        style={{ cursor: 'pointer' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteConnection(conn.id);
                        }}
                      >
                        <circle
                          cx={(fromPos.x + fromPos.width + toPos.x) / 2}
                          cy={(fromPos.y + fromPos.height / 2 + toPos.y + toPos.height / 2) / 2}
                          r={14}
                          fill="hsl(var(--destructive))"
                          stroke="hsl(var(--background))"
                          strokeWidth={2}
                        />
                        <text
                          x={(fromPos.x + fromPos.width + toPos.x) / 2}
                          y={(fromPos.y + fromPos.height / 2 + toPos.y + toPos.height / 2) / 2}
                          textAnchor="middle"
                          dominantBaseline="central"
                          fill="hsl(var(--destructive-foreground))"
                          fontSize={16}
                          fontWeight="bold"
                          style={{ pointerEvents: 'none' }}
                        >
                          ×
                        </text>
                      </g>
                    )}
                  </g>
                );
              });
            })()}

            {/* Temporary connection while dragging */}
            {isConnecting && connectionStart && connectionEnd && (
              <g>
                <path
                  d={getTempConnectionPath()}
                  fill="none"
                  stroke="hsl(var(--primary) / 0.3)"
                  strokeWidth={6}
                  strokeLinecap="round"
                />
                <path
                  d={getTempConnectionPath()}
                  fill="none"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  strokeDasharray="12 6"
                  strokeLinecap="round"
                  opacity={0.8}
                  markerEnd="url(#arrowhead)"
                >
                  <animate
                    attributeName="stroke-dashoffset"
                    from="18"
                    to="0"
                    dur="0.5s"
                    repeatCount="indefinite"
                  />
                </path>
              </g>
            )}
          </svg>


          {/* Node and Group Layers — explicitly on top of SVG */}
          <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 1 }}>
            {/* Selection box */}
            {isSelecting && selectionBox && (
              <div className="pointer-events-auto">
                <SelectionBox box={selectionBox} />
              </div>
            )}

            {/* Groups */}
            <div className="pointer-events-auto">
              {groups.map((group) => (
                <GroupNode
                  key={group.id}
                  group={group}
                  nodes={nodes}
                  isSelected={selectedNodeId === group.id}
                  onSelect={(e) => {
                    e.stopPropagation();
                    onSelectNode(group.id);
                  }}
                  onDragStart={(e) => {
                    if (e.button !== 0 || !canvasRef.current) return;
                    const rect = canvasRef.current.getBoundingClientRect();
                    const mouseX = (e.clientX - rect.left - offset.x) / zoom;
                    const mouseY = (e.clientY - rect.top - offset.y) / zoom;
                    // Use a pseudo-node drag for the group
                    setDraggedNode(`__group__${group.id}`);
                    setDragOffset({ x: mouseX - group.x, y: mouseY - group.y });
                  }}
                  onUpdateGroup={(id, updates) => onUpdateGroup?.(id, updates)}
                  onUngroupGroup={(id) => onUngroupGroup?.(id)}
                  onDeleteGroup={(id) => onDeleteGroup?.(id)}
                  onConnectionStart={(_, e) => {
                    e.stopPropagation();
                    if (!canvasRef.current) return;
                    // Use the actual mouse position in canvas space — this is exactly
                    // where the port circle is, regardless of the group's rendered width/height.
                    const rect = canvasRef.current.getBoundingClientRect();
                    const x = (e.clientX - rect.left - offset.x) / zoom;
                    const y = (e.clientY - rect.top - offset.y) / zoom;
                    setIsConnecting(true);
                    setConnectionStart({ nodeId: group.id, x, y });
                    setConnectionEnd({ x, y });
                  }}
                  onConnectionEnd={(e) => {
                    e.stopPropagation();
                    if (isConnecting && connectionStart && connectionStart.nodeId !== group.id) {
                      onAddConnection(connectionStart.nodeId, group.id);
                    }
                    setIsConnecting(false);
                    setConnectionStart(null);
                    setConnectionEnd(null);
                  }}
                  isConnecting={isConnecting}
                />
              ))}

              {/* Nodes (hide grouped ones) */}
              {nodes.filter(node => !groups.some(g => g.nodeIds.includes(node.id))).map((node) => (
                <ContextMenu key={node.id}>
                  <ContextMenuTrigger asChild>
                    <div>
                      <CanvasNode
                        node={node}
                        isSelected={selectedNodeId === node.id || isNodeSelected(node.id)}
                        isMultiSelected={selectedNodeIds.size > 1 && isNodeSelected(node.id)}
                        onSelect={() => handleNodeClick(node.id, { stopPropagation: () => { } } as React.MouseEvent)}
                        onClick={(e) => handleNodeClick(node.id, e)}
                        onDragStart={(e) => handleNodeDragStart(node.id, e)}
                        onConnectionStart={(isOutput, e) => handleConnectionStart(node.id, isOutput, e)}
                        onConnectionEnd={(e) => handleConnectionEnd(node.id, e)}
                        isConnecting={isConnecting}
                      />
                    </div>
                  </ContextMenuTrigger>
                  <ContextMenuContent className="bg-popover border-border w-48">
                    <ContextMenuItem
                      onClick={() => onDuplicateNode(node.id)}
                      className="cursor-pointer"
                    >
                      <Copy className="w-4 h-4 mr-2" />
                      Duplicate
                      <ContextMenuShortcut>Ctrl+D</ContextMenuShortcut>
                    </ContextMenuItem>
                    <ContextMenuItem
                      onClick={() => {
                        setIsConnecting(true);
                        setConnectionStart({ nodeId: node.id, x: node.x + 176, y: node.y + 40 });
                        setConnectionEnd({ x: node.x + 200, y: node.y + 40 });
                      }}
                      className="cursor-pointer"
                    >
                      <Link className="w-4 h-4 mr-2" />
                      Connect to...
                    </ContextMenuItem>
                    <ContextMenuSeparator className="bg-border" />
                    <ContextMenuItem
                      onClick={() => onDeleteNode(node.id)}
                      className="cursor-pointer text-destructive focus:text-destructive"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete
                      <ContextMenuShortcut>Del</ContextMenuShortcut>
                    </ContextMenuItem>
                  </ContextMenuContent>
                </ContextMenu>
              ))}
            </div>
          </div>

          {/* Empty State */}
          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center">
                <div className="text-muted-foreground/50 text-sm mb-2">
                  Drag layers from the palette to start building
                </div>
                <div className="text-muted-foreground/30 text-xs">
                  Drag to box-select • Ctrl+Click for multi-select • Shift+Drag to pan
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
