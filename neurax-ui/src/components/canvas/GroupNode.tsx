import { useState } from 'react';
import { Layers, Repeat, Ungroup, Pencil, Trash2, Minus, Plus } from 'lucide-react';
import { NodeGroup, CanvasNode } from '@/types/architecture.ts';
import { cn } from '@/lib/utils.ts';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';

interface GroupNodeProps {
  group: NodeGroup;
  nodes: CanvasNode[];
  isSelected: boolean;
  onSelect: (e: React.MouseEvent) => void;
  onDragStart: (e: React.MouseEvent) => void;
  onUpdateGroup: (id: string, updates: Partial<NodeGroup>) => void;
  onUngroupGroup: (id: string) => void;
  onDeleteGroup: (id: string) => void;
  onConnectionStart?: (isOutput: boolean, e: React.MouseEvent) => void;
  onConnectionEnd?: (e: React.MouseEvent) => void;
  isConnecting?: boolean;
}

export function GroupNode({
  group,
  nodes,
  isSelected,
  onSelect,
  onDragStart,
  onUpdateGroup,
  onUngroupGroup,
  onDeleteGroup,
  onConnectionStart,
  onConnectionEnd,
  isConnecting,
}: GroupNodeProps) {
  const [isEditingName, setIsEditingName] = useState(false);
  const [editName, setEditName] = useState(group.name);

  const childNodes = nodes.filter(n => group.nodeIds.includes(n.id));
  const uniqueTypes = [...new Set(childNodes.map(n => n.type))];
  const blockSummary = childNodes.length <= 4
    ? childNodes.map(n => n.name).join(' → ')
    : `${childNodes.slice(0, 3).map(n => n.name).join(' → ')} … +${childNodes.length - 3}`;

  const handleNameSubmit = () => {
    onUpdateGroup(group.id, { name: editName.trim() || group.name });
    setIsEditingName(false);
  };

  return (
    <div
      className={cn(
        "absolute select-none transition-all duration-200 ease-out",
        "border-2 rounded-2xl shadow-lg w-[240px]",
        "bg-gradient-to-br from-primary/5 via-card to-accent/5",
        "border-primary/40",
        "hover:shadow-xl hover:border-primary/60",
        "cursor-move",
        isSelected && "border-primary shadow-xl ring-4 ring-primary/20"
      )}
      style={{ left: group.x, top: group.y }}
      onClick={onSelect}
      onMouseDown={onDragStart}
    >
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-primary/20 bg-primary/10 rounded-t-2xl">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary/20 border border-primary/30 text-primary">
          <Layers className="w-5 h-5" />
        </div>
        <div className="flex-1 min-w-0">
          {isEditingName ? (
            <Input
              value={editName}
              onChange={e => setEditName(e.target.value)}
              onBlur={handleNameSubmit}
              onKeyDown={e => e.key === 'Enter' && handleNameSubmit()}
              className="h-6 text-sm font-semibold px-1 py-0"
              autoFocus
              onClick={e => e.stopPropagation()}
              onMouseDown={e => e.stopPropagation()}
            />
          ) : (
            <span
              className="text-sm font-semibold truncate block text-foreground cursor-text"
              onDoubleClick={(e) => {
                e.stopPropagation();
                setIsEditingName(true);
                setEditName(group.name);
              }}
            >
              {group.name}
            </span>
          )}
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
            Group • {childNodes.length} blocks
          </span>
        </div>

        {/* Repeat badge */}
        <div className="flex items-center gap-1 bg-primary/20 rounded-lg px-1 border border-primary/30">
          <Button
            variant="ghost"
            size="icon"
            className="w-5 h-5 rounded p-0 hover:bg-primary/20"
            onClick={(e) => {
              e.stopPropagation();
              if (group.repeatCount > 1) {
                onUpdateGroup(group.id, { repeatCount: group.repeatCount - 1 });
              }
            }}
          >
            <Minus className="w-3 h-3" />
          </Button>
          <span className="text-sm font-bold text-primary min-w-[24px] text-center select-none">
            ×{group.repeatCount}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="w-5 h-5 rounded p-0 hover:bg-primary/20"
            onClick={(e) => {
              e.stopPropagation();
              onUpdateGroup(group.id, { repeatCount: group.repeatCount + 1 });
            }}
          >
            <Plus className="w-3 h-3" />
          </Button>
        </div>
      </div>

      {/* Body - block summary */}
      <div className="px-4 py-3 space-y-2">
        <div className="text-xs text-muted-foreground/80 leading-relaxed">
          {blockSummary}
        </div>
        <div className="flex flex-wrap gap-1">
          {uniqueTypes.slice(0, 5).map(t => (
            <span key={t} className="text-[10px] bg-secondary/80 text-secondary-foreground px-2 py-0.5 rounded-md">
              {t}
            </span>
          ))}
          {uniqueTypes.length > 5 && (
            <span className="text-[10px] text-muted-foreground">+{uniqueTypes.length - 5}</span>
          )}
        </div>

        {/* Effective blocks count */}
        <div className="flex items-center gap-2 pt-1 border-t border-border/50 text-[11px] text-muted-foreground">
          <Repeat className="w-3 h-3" />
          <span>{childNodes.length} × {group.repeatCount} = <strong className="text-foreground">{childNodes.length * group.repeatCount}</strong> blocks in output</span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1 px-3 py-2 border-t border-border/30">
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-[10px] rounded-md hover:bg-secondary"
          onClick={(e) => {
            e.stopPropagation();
            onUngroupGroup(group.id);
          }}
          title="Ungroup - expand back to individual blocks"
        >
          <Ungroup className="w-3 h-3 mr-1" />
          Ungroup
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-[10px] rounded-md hover:bg-secondary"
          onClick={(e) => {
            e.stopPropagation();
            setIsEditingName(true);
            setEditName(group.name);
          }}
          title="Rename group"
        >
          <Pencil className="w-3 h-3 mr-1" />
          Rename
        </Button>
        <div className="flex-1" />
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-[10px] text-destructive hover:bg-destructive/10 rounded-md"
          onClick={(e) => {
            e.stopPropagation();
            onDeleteGroup(group.id);
          }}
          title="Delete group and its blocks"
        >
          <Trash2 className="w-3 h-3" />
        </Button>
      </div>

      {/* Connection ports */}
      <div
        className={cn(
          "absolute -left-3 top-1/2 -translate-y-1/2",
          "w-6 h-6 rounded-full bg-card border-2 border-primary/40",
          "cursor-pointer transition-all duration-200 flex items-center justify-center",
          "hover:scale-125 hover:border-primary hover:bg-secondary",
          isConnecting && "animate-pulse border-primary scale-110"
        )}
        onMouseDown={(e) => { e.stopPropagation(); onConnectionStart?.(false, e); }}
        onMouseUp={(e) => { e.stopPropagation(); onConnectionEnd?.(e); }}
      >
        <div className="w-2 h-2 rounded-full bg-primary" />
      </div>
      <div
        className={cn(
          "absolute -right-3 top-1/2 -translate-y-1/2",
          "w-6 h-6 rounded-full bg-card border-2 border-primary/40",
          "cursor-pointer transition-all duration-200 flex items-center justify-center",
          "hover:scale-125 hover:border-primary hover:bg-secondary",
          isConnecting && "animate-pulse border-primary scale-110"
        )}
        onMouseDown={(e) => { e.stopPropagation(); onConnectionStart?.(true, e); }}
        onMouseUp={(e) => { e.stopPropagation(); onConnectionEnd?.(e); }}
      >
        <div className="w-2 h-2 rounded-full bg-primary" />
      </div>
    </div>
  );
}
