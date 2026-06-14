import { useState, useCallback, useRef } from 'react';
import { CanvasNode } from '@/types/architecture.ts';

export interface SelectionBox {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

export interface UseMultiSelectReturn {
  selectedNodeIds: Set<string>;
  selectionBox: SelectionBox | null;
  isSelecting: boolean;
  toggleNodeSelection: (nodeId: string, addToSelection: boolean) => void;
  selectNode: (nodeId: string, addToSelection: boolean) => void;
  clearSelection: () => void;
  selectAll: (nodeIds: string[]) => void;
  startBoxSelection: (x: number, y: number) => void;
  updateBoxSelection: (x: number, y: number) => void;
  endBoxSelection: (nodes: CanvasNode[], offset: { x: number; y: number }, zoom: number) => void;
  isNodeSelected: (nodeId: string) => boolean;
}

export function useMultiSelect(): UseMultiSelectReturn {
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<string>>(new Set());
  const [selectionBox, setSelectionBox] = useState<SelectionBox | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const selectionStartRef = useRef<{ x: number; y: number } | null>(null);

  const toggleNodeSelection = useCallback((nodeId: string, addToSelection: boolean) => {
    setSelectedNodeIds(prev => {
      const next = new Set(prev);
      if (addToSelection) {
        if (next.has(nodeId)) {
          next.delete(nodeId);
        } else {
          next.add(nodeId);
        }
      } else {
        next.clear();
        next.add(nodeId);
      }
      return next;
    });
  }, []);

  const selectNode = useCallback((nodeId: string, addToSelection: boolean) => {
    setSelectedNodeIds(prev => {
      if (addToSelection) {
        const next = new Set(prev);
        next.add(nodeId);
        return next;
      } else {
        return new Set([nodeId]);
      }
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedNodeIds(new Set());
  }, []);

  const selectAll = useCallback((nodeIds: string[]) => {
    setSelectedNodeIds(new Set(nodeIds));
  }, []);

  const startBoxSelection = useCallback((x: number, y: number) => {
    selectionStartRef.current = { x, y };
    setSelectionBox({ startX: x, startY: y, endX: x, endY: y });
    setIsSelecting(true);
  }, []);

  const updateBoxSelection = useCallback((x: number, y: number) => {
    if (!selectionStartRef.current) return;
    setSelectionBox({
      startX: selectionStartRef.current.x,
      startY: selectionStartRef.current.y,
      endX: x,
      endY: y,
    });
  }, []);

  const endBoxSelection = useCallback((nodes: CanvasNode[], offset: { x: number; y: number }, zoom: number) => {
    if (!selectionBox) {
      setIsSelecting(false);
      selectionStartRef.current = null;
      return;
    }

    // Convert selection box to canvas coordinates
    const minX = Math.min(selectionBox.startX, selectionBox.endX);
    const maxX = Math.max(selectionBox.startX, selectionBox.endX);
    const minY = Math.min(selectionBox.startY, selectionBox.endY);
    const maxY = Math.max(selectionBox.startY, selectionBox.endY);

    // Node dimensions (approximate)
    const nodeWidth = 176;
    const nodeHeight = 80;

    // Find nodes that intersect with the selection box
    const selectedIds = nodes.filter(node => {
      const nodeLeft = node.x;
      const nodeRight = node.x + nodeWidth;
      const nodeTop = node.y;
      const nodeBottom = node.y + nodeHeight;

      // Check for intersection
      return !(nodeRight < minX || nodeLeft > maxX || nodeBottom < minY || nodeTop > maxY);
    }).map(n => n.id);

    setSelectedNodeIds(new Set(selectedIds));
    setSelectionBox(null);
    setIsSelecting(false);
    selectionStartRef.current = null;
  }, [selectionBox]);

  const isNodeSelected = useCallback((nodeId: string) => {
    return selectedNodeIds.has(nodeId);
  }, [selectedNodeIds]);

  return {
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
  };
}
