import { ArchitectureFamily } from './plugins.ts';
import { CanvasNode, Connection } from './architecture.ts';

/**
 * A variant preset is a predefined architecture template that users can load
 * as a starting point and then customize.
 */
export interface VariantPreset {
  id: string;
  name: string;
  family: ArchitectureFamily;
  description: string;
  tags: string[];
  nodes: CanvasNode[];
  connections: Connection[];
  defaultParams: Record<string, Record<string, number | string | boolean>>;
}

/**
 * A macro block is a reusable high-level block that expands
 * into multiple atomic blocks when inspected.
 */
export interface MacroBlock {
  id: string;
  name: string;
  family: ArchitectureFamily;
  description: string;
  icon: string;
  tags: string[];
  internalBlocks: string[]; // references to block IDs
  color: string;
}

/**
 * Architecture-aware constraint rules to enforce valid designs.
 */
export interface ArchitectureConstraint {
  family: ArchitectureFamily;
  requiredBlocks: string[]; // block types that must be present
  incompatibleBlocks: string[]; // block types that should be hidden for this family
  warningRules: ConstraintWarning[];
}

export interface ConstraintWarning {
  id: string;
  condition: 'missing_block' | 'invalid_connection' | 'duplicate_block' | 'order_violation';
  blockType?: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
}
