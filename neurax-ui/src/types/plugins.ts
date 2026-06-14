import { LucideIcon } from 'lucide-react';
import { LayerConfig } from './architecture.ts';

export type ArchitectureFamily = 
  | 'transformer'
  | 'moe'
  | 'ssm'
  | 'cnn'
  | 'diffusion'
  | 'gnn'
  | 'gan'
  | 'rl'
  | 'snn'
  | 'rnn'
  | 'experimental';

export interface ArchitectureFamilyConfig {
  id: ArchitectureFamily;
  name: string;
  description: string;
  icon: string;
  color: string;
}

export interface PluginTool {
  id: string;
  name: string;
  icon: string;
  description: string;
  component: string;
}

export interface PluginMetric {
  id: string;
  name: string;
  value: string | number;
  unit?: string;
  status?: 'normal' | 'warning' | 'critical';
  tooltip?: string;
}

export interface PluginVisualization {
  id: string;
  name: string;
  type: 'chart' | 'heatmap' | 'flow' | 'graph' | 'diagram';
}

export interface ArchitecturePlugin {
  id: ArchitectureFamily;
  config: ArchitectureFamilyConfig;
  layers: LayerConfig[];
  tools: PluginTool[];
  metrics: PluginMetric[];
  visualizations: PluginVisualization[];
}

export const ARCHITECTURE_FAMILIES: ArchitectureFamilyConfig[] = [
  {
    id: 'transformer',
    name: 'Transformer / LLM',
    description: 'Attention-based sequence models',
    icon: 'Sparkles',
    color: 'hsl(199, 89%, 48%)',
  },
  {
    id: 'moe',
    name: 'Mixture of Experts',
    description: 'Sparse expert routing models',
    icon: 'Network',
    color: 'hsl(280, 70%, 55%)',
  },
  {
    id: 'ssm',
    name: 'State Space Models',
    description: 'S4, Mamba, H3 and linear recurrence models',
    icon: 'Workflow',
    color: 'hsl(160, 70%, 45%)',
  },
  {
    id: 'cnn',
    name: 'CNN / Vision',
    description: 'Convolutional neural networks',
    icon: 'Grid3X3',
    color: 'hsl(210, 70%, 50%)',
  },
  {
    id: 'diffusion',
    name: 'Diffusion Models',
    description: 'Denoising and score-based generative models',
    icon: 'Waves',
    color: 'hsl(38, 92%, 50%)',
  },
  {
    id: 'gnn',
    name: 'Graph Neural Networks',
    description: 'Node, edge, and graph-level learning',
    icon: 'Share2',
    color: 'hsl(340, 75%, 55%)',
  },
  {
    id: 'gan',
    name: 'GANs',
    description: 'Generative adversarial networks',
    icon: 'Wand2',
    color: 'hsl(45, 90%, 50%)',
  },
  {
    id: 'rl',
    name: 'Reinforcement Learning',
    description: 'Actor-critic, value, and policy networks',
    icon: 'Gamepad2',
    color: 'hsl(15, 80%, 52%)',
  },
  {
    id: 'snn',
    name: 'Spiking Neural Networks',
    description: 'Bio-inspired spike-based models',
    icon: 'Brain',
    color: 'hsl(300, 60%, 50%)',
  },
  {
    id: 'rnn',
    name: 'RNN / LSTM / GRU',
    description: 'Recurrent sequence models',
    icon: 'Repeat',
    color: 'hsl(25, 80%, 52%)',
  },
];
