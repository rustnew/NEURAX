// NEURAX SaaS Plan Types

export type PlanTier = 'free' | 'essential' | 'architect' | 'elite';

export interface PlanConfig {
  id: PlanTier;
  name: string;
  displayName: string;
  color: string;
  badge: string;
  price: {
    monthly: number;
    annual: number;
  };
  features: string[];
}

export const PLAN_CONFIGS: Record<PlanTier, PlanConfig> = {
  free: {
    id: 'free',
    name: 'Free',
    displayName: 'FREE',
    color: 'hsl(215, 16%, 47%)',
    badge: 'bg-slate-500/10 text-slate-300 border-slate-500/30',
    price: { monthly: 0, annual: 0 },
    features: [
      'Architecture designer',
      'Simulation tab (limited)',
      'Basic diagnostics (shape, VRAM, FLOPs)',
      'Up to 10 lifetime analyses after sign-in',
    ],
  },
  essential: {
    id: 'essential',
    name: 'Essential',
    displayName: 'ESSENTIAL',
    color: 'hsl(199, 89%, 48%)',
    badge: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
    price: { monthly: 29, annual: 290 },
    features: [
      'Transformer & CNN architectures',
      'Basic diagnostics (shape, VRAM, FLOPs)',
      'Consumer GPU simulation',
      'PyTorch & ONNX export',
    ],
  },
  architect: {
    id: 'architect',
    name: 'Architect',
    displayName: 'ARCHITECT',
    color: 'hsl(265, 89%, 66%)',
    badge: 'bg-primary/10 text-primary border-primary/30',
    price: { monthly: 99, annual: 990 },
    features: [
      'All Essential features',
      'MoE, SSM (incl. Mamba), Diffusion, GNN, GAN, RL',
      'Advanced diagnostics',
      'Cloud GPU simulation (A100, H100)',
      'Rust/Burn & Triton exports',
    ],
  },
  elite: {
    id: 'elite',
    name: 'Elite / Lab',
    displayName: 'ELITE',
    color: 'hsl(38, 92%, 50%)',
    badge: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    price: { monthly: 299, annual: 2990 },
    features: [
      'All Architect features',
      'Experimental architectures',
      'muP scaling & cluster analysis',
      'Multi-node simulation',
      'Megatron-LM exports',
      'Real-time collaboration',
    ],
  },
};

// Architecture access by plan
export const ARCHITECTURE_BY_PLAN: Record<PlanTier, string[]> = {
  free: ['transformer', 'cnn', 'rnn'],
  essential: ['transformer', 'cnn', 'rnn'],
  architect: ['transformer', 'cnn', 'rnn', 'moe', 'ssm', 'diffusion', 'gan', 'gnn', 'rl'],
  elite: ['transformer', 'cnn', 'rnn', 'moe', 'ssm', 'diffusion', 'gan', 'gnn', 'rl', 'snn', 'experimental'],
};

// Hardware access by plan
export interface HardwareOption {
  id: string;
  name: string;
  vram: string;
  category: 'consumer' | 'cloud' | 'cluster';
  minPlan: PlanTier;
}

export const HARDWARE_OPTIONS: HardwareOption[] = [
  { id: 'rtx3090', name: 'RTX 3090', vram: '24 GB', category: 'consumer', minPlan: 'free' },
  { id: 'rtx4090', name: 'RTX 4090', vram: '24 GB', category: 'consumer', minPlan: 'free' },
  { id: 'a100-40', name: 'A100 40GB', vram: '40 GB', category: 'cloud', minPlan: 'architect' },
  { id: 'a100-80', name: 'A100 80GB', vram: '80 GB', category: 'cloud', minPlan: 'architect' },
  { id: 'h100', name: 'H100 SXM', vram: '80 GB', category: 'cloud', minPlan: 'architect' },
  { id: 'tpu-v5p', name: 'TPU v5p', vram: '95 GB HBM', category: 'cloud', minPlan: 'architect' },
  { id: 'cluster-64', name: '64× H100 Cluster', vram: '5.12 TB', category: 'cluster', minPlan: 'elite' },
  { id: 'cluster-256', name: '256× H100 Cluster', vram: '20.48 TB', category: 'cluster', minPlan: 'elite' },
  { id: 'cluster-512', name: '512× H100 Cluster', vram: '40.96 TB', category: 'cluster', minPlan: 'elite' },
];

// Diagnostics access by plan
export interface DiagnosticFeature {
  id: string;
  name: string;
  description: string;
  minPlan: PlanTier;
}

export const DIAGNOSTIC_FEATURES: DiagnosticFeature[] = [
  { id: 'shape', name: 'Shape Validation', description: 'Verify tensor shapes flow correctly', minPlan: 'free' },
  { id: 'flops', name: 'FLOPs Estimation', description: 'Calculate computational complexity', minPlan: 'free' },
  { id: 'gradient', name: 'Gradient Curvature', description: 'Analyze gradient flow stability', minPlan: 'architect' },
  { id: 'router', name: 'Router Collapse Detection', description: 'Detect MoE routing issues', minPlan: 'architect' },
  { id: 'stability', name: 'Stability Warnings', description: 'Advanced numeric stability analysis', minPlan: 'architect' },
  { id: 'mup', name: 'μP Scaling Diagnostics', description: 'Maximal Update Parameterization analysis', minPlan: 'elite' },
  { id: 'bottleneck', name: 'Hardware Bottleneck Analysis', description: 'Identify compute/memory/bandwidth limits', minPlan: 'elite' },
  { id: 'cluster', name: 'Multi-node Feasibility', description: 'Cluster training viability assessment', minPlan: 'elite' },
];

// Export access by plan
export interface ExportOption {
  id: string;
  name: string;
  extension: string;
  icon: string;
  description: string;
  minPlan: PlanTier;
  includeAnalysis: boolean;
}

export const EXPORT_OPTIONS: ExportOption[] = [
  { id: 'pytorch', name: 'PyTorch', extension: '.py', icon: 'Code', description: 'Python model definition', minPlan: 'essential', includeAnalysis: true },
  { id: 'onnx', name: 'ONNX', extension: '.onnx', icon: 'Box', description: 'Open Neural Network Exchange', minPlan: 'essential', includeAnalysis: false },
  { id: 'json', name: 'JSON Schema', extension: '.json', icon: 'FileJson', description: 'Internal representation', minPlan: 'essential', includeAnalysis: false },
  { id: 'rust', name: 'Rust / Burn', extension: '.rs', icon: 'Cog', description: 'Rust model structure', minPlan: 'architect', includeAnalysis: true },
  { id: 'triton', name: 'Triton Kernels', extension: '.py', icon: 'Zap', description: 'Optimized GPU kernels', minPlan: 'architect', includeAnalysis: true },
  { id: 'megatron', name: 'Megatron-LM', extension: '.py', icon: 'Server', description: 'Parallelized model code', minPlan: 'elite', includeAnalysis: true },
  { id: 'cluster', name: 'Cluster Config', extension: '.yaml', icon: 'Network', description: 'Multi-node launch config', minPlan: 'elite', includeAnalysis: true },
];

// Utility functions
export function canAccessArchitecture(plan: PlanTier, architectureId: string): boolean {
  return ARCHITECTURE_BY_PLAN[plan].includes(architectureId);
}

export function canAccessHardware(plan: PlanTier, hardware: HardwareOption): boolean {
  const planOrder: PlanTier[] = ['free', 'essential', 'architect', 'elite'];
  return planOrder.indexOf(plan) >= planOrder.indexOf(hardware.minPlan);
}

export function canAccessDiagnostic(plan: PlanTier, diagnostic: DiagnosticFeature): boolean {
  const planOrder: PlanTier[] = ['free', 'essential', 'architect', 'elite'];
  return planOrder.indexOf(plan) >= planOrder.indexOf(diagnostic.minPlan);
}

export function canAccessExport(plan: PlanTier, exportOption: ExportOption): boolean {
  const planOrder: PlanTier[] = ['free', 'essential', 'architect', 'elite'];
  return planOrder.indexOf(plan) >= planOrder.indexOf(exportOption.minPlan);
}

export function getRequiredPlanName(minPlan: PlanTier): string {
  return PLAN_CONFIGS[minPlan].name;
}
