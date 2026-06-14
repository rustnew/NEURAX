import { useState } from 'react';
import {
  Box,
  ChevronDown,
  ChevronRight,
  Layers,
  LucideIcon,
  Network,
  GitBranch,
  Wand2,
  Workflow,
  MessageSquare,
  Repeat,
  Route,
  Shield,
  Zap,
} from 'lucide-react';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { MacroBlock } from '@/types/catalog.ts';
import { getMacroBlocksForFamily } from '@/data/architectureCatalog.ts';
import { Badge } from '@/components/ui/badge.tsx';
import { cn } from '@/lib/utils.ts';

const iconMap: Record<string, LucideIcon> = {
  Box, Layers, Network, GitBranch, Wand2, Workflow, MessageSquare, Repeat, Route, Shield, Zap,
};

interface MacroBlocksPaletteProps {
  family: ArchitectureFamily;
  onDragStart: (macro: MacroBlock) => void;
}

export function MacroBlocksPalette({ family, onDragStart }: MacroBlocksPaletteProps) {
  const macros = getMacroBlocksForFamily(family);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (macros.length === 0) return null;

  return (
    <div className="mb-4 animate-fade-in">
      <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
        Macro Blocks
      </div>
      {macros.map((macro) => {
        const Icon = iconMap[macro.icon] || Box;
        const isExpanded = expandedId === macro.id;

        return (
          <div key={macro.id} className="mx-2 mb-1">
            <div
              className="layer-item group"
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData('macro', JSON.stringify(macro));
                onDragStart(macro);
              }}
            >
              <div
                className="w-8 h-8 rounded-md flex items-center justify-center border shrink-0"
                style={{
                  backgroundColor: `${macro.color}15`,
                  borderColor: `${macro.color}30`,
                }}
              >
                <Icon className="w-4 h-4" style={{ color: macro.color }} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-sm font-medium text-foreground truncate">
                    {macro.name}
                  </span>
                  <Badge
                    variant="outline"
                    className="text-[8px] px-1 py-0 h-3.5 bg-primary/5 border-primary/20"
                  >
                    MACRO
                  </Badge>
                </div>
                <div className="text-[10px] text-muted-foreground truncate">
                  {macro.description}
                </div>
              </div>
              <button
                className="p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => {
                  e.stopPropagation();
                  setExpandedId(isExpanded ? null : macro.id);
                }}
              >
                {isExpanded ? (
                  <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
                )}
              </button>
            </div>

            {/* Expanded internal blocks */}
            {isExpanded && (
              <div className="ml-11 mb-2 space-y-1 animate-fade-in">
                <div className="text-[9px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  Internal Structure
                </div>
                {macro.internalBlocks.map((block, i) => (
                  <div
                    key={`${block}-${i}`}
                    className="flex items-center gap-2 text-[10px] text-muted-foreground"
                  >
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/50" />
                    <span className="capitalize">{block.replace(/-/g, ' ')}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
