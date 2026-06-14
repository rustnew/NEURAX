// Extended analysis types for AI Architecture Designer

export interface LayerMetrics {
  id: string;
  name: string;
  type: string;
  weights: number;
  biases: number;
  totalParams: number;
  flops: number;
  memoryBytes: number;
  computeIntensity: 'low' | 'medium' | 'high' | 'extreme';
  isBottleneck: boolean;
}

export interface GradientStatus {
  connectionId: string;
  status: 'stable' | 'risk' | 'vanishing' | 'exploding';
  magnitude: number;
  explanation: string;
}

export interface MemoryBreakdown {
  fp32: string;
  fp16: string;
  int8: string;
  activations: string;
  gradients: string;
}

export interface GlobalMetrics {
  totalParams: number;
  totalWeights: number;
  totalBiases: number;
  totalFlops: number;
  memoryBreakdown: MemoryBreakdown;
  estimatedSize: string;
  depth: number;
  bottlenecks: string[];
}

export interface FrameworkComparison {
  framework: 'rust' | 'python' | 'cpp' | 'go';
  name: string;
  icon: string;
  performance: number; // 0-100
  memoryEfficiency: number; // 0-100
  parallelism: number; // 0-100
  recommendation: 'edge' | 'cloud' | 'research' | 'production';
  notes: string[];
}

export interface ArchitectureVariant {
  id: string;
  name: string;
  description: string;
  type: 'baseline' | 'optimized' | 'quantized' | 'pruned' | 'custom';
  isMain: boolean;
  paramReduction: number; // percentage
  speedup: number; // percentage
  accuracyDelta: number; // percentage change
  changes: string[];
}

export interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  icon: string;
  description: string;
  includeAnalysis: boolean;
}

export const EXPORT_FORMATS: ExportFormat[] = [
  { id: 'json', name: 'JSON Schema', extension: '.json', icon: 'FileJson', description: 'Internal representation', includeAnalysis: false },
  { id: 'yaml', name: 'YAML Config', extension: '.yaml', icon: 'FileText', description: 'Human-readable config', includeAnalysis: false },
  { id: 'onnx', name: 'ONNX', extension: '.onnx', icon: 'Box', description: 'Open Neural Network Exchange', includeAnalysis: false },
  { id: 'pytorch', name: 'PyTorch Code', extension: '.py', icon: 'Code', description: 'Python model definition', includeAnalysis: true },
  { id: 'rust', name: 'Rust Pseudo-code', extension: '.rs', icon: 'Cog', description: 'Rust model structure', includeAnalysis: true },
  { id: 'pdf', name: 'PDF Report', extension: '.pdf', icon: 'FileText', description: 'Full analysis report', includeAnalysis: true },
  { id: 'png', name: 'PNG Image', extension: '.png', icon: 'Image', description: 'Architecture diagram', includeAnalysis: false },
];
