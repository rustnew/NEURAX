import { createContext, useContext, useState, ReactNode } from 'react';
import { NeuraxEnv } from '@/utils/neuraxCompiler.ts';

export interface HardwareConfig {
  hardware: string;
  precision: NeuraxEnv['prec'];
  batchSize: number;
  seed?: number;
  device?: string;
  useCompile?: boolean;

  // ─── Training ───────────────────────────────────────────────
  learningRate: number;
  numEpochs: number;
  /** GPU count for hardware section */
  gpuCount: number;
  /** GPU memory in GB for hardware section */
  gpuMemoryGb: number;

  // ─── Data ───────────────────────────────────────────────────
  /** Total tokens / samples in dataset */
  datasetSize: number;

  // Transformers / LLM
  seqLen: number;
  vocabSize: number;
  hiddenDim: number;
  numHeads: number;
  headDim?: number;
  ffnDim: number;
  numLayers: number;
  kvHeads?: number;
  useBias: boolean;
  dropout: number;
  useFlash: boolean;
  ropeTheta?: number;
  maxSeqLen?: number;
  useAlibi?: boolean;
  useRelativeBias?: boolean;
  useCache?: boolean;
  activation?: string;

  // CNN
  imgHeight: number;
  imgWidth: number;
  inChannels: number;
  numClasses?: number;
  normType?: string;
  convActivation?: string;
  poolType?: string;

  // ViT / DiT
  patchSize?: number;
  numPatches?: number;
  numDenoisingSteps: number;
  guidanceScale: number;
  mlpRatio?: number;
  qkvBias?: boolean;
  projDrop?: number;
  attnDrop?: number;
  posEmbedType?: string;
  useFlashVit?: boolean;

  // GNN
  numNodes: number;
  numEdges: number;
  nodeFeatDim: number;
  outDim?: number;
  edgeFeatDim?: number;
  aggrType?: string;
  useNormalize?: boolean;
  addSelfLoops?: boolean;

  // RNN / SSM
  hiddenSize?: number;
  isBidirectional?: boolean;
  dState: number;
  dtRank: number;
  convKernel?: number;
  expandFactor?: number;
  useFastPath?: boolean;
  projSize?: number;
  timesteps: number;
  spikeRate: number;

  // MoE
  numExperts: number;
  topK: number;
  expertCapacity?: number;
  useSharedExpert?: boolean;

  // RL
  actionDim: number;
  stateDim: number;

  // UNet / Diffusion specific
  modelChannels?: number;
  numResBlocks?: number;
  channelMult?: string;
  attnResolutions?: string;
  useCheckpoint?: boolean;
  outChannels?: number;
}

export type ArchitectureFamily =
  | 'transformer'
  | 'cnn'
  | 'gnn'
  | 'rnn'
  | 'ssm'
  | 'moe'
  | 'diffusion'
  | 'vit'
  | 'snn'
  | 'rl'
  | 'gan'
  | 'experimental';

export const MANDATORY_FIELDS: Record<string, (keyof HardwareConfig)[]> = {
  common: ['hardware', 'precision', 'batchSize'],
  transformer: ['seqLen', 'numLayers', 'vocabSize'],
  moe: ['seqLen', 'numLayers', 'vocabSize', 'numExperts', 'topK'],
  gnn: ['numNodes', 'numEdges', 'nodeFeatDim'],
  diffusion: ['numDenoisingSteps', 'inChannels'],
  vit: ['inChannels'],
  cnn: ['inChannels'],
  ssm: ['dState'],
  rnn: [],
  snn: [],
  gan: [],
  rl: [],
  experimental: [],
};

export function validateHardwareConfig(config: HardwareConfig, family: ArchitectureFamily): { isValid: boolean; missingFields: (keyof HardwareConfig)[] } {
  const missingFields: (keyof HardwareConfig)[] = [];

  // Check common fields
  MANDATORY_FIELDS.common.forEach(field => {
    const v = config[field];
    if (typeof v === 'string') {
      if (!v.trim()) missingFields.push(field);
      return;
    }
    if (typeof v === 'number') {
      if (v <= 0) missingFields.push(field);
      return;
    }
    if (typeof v === 'boolean') {
      return;
    }
    if (v === null || v === undefined) missingFields.push(field);
  });

  // Check family specific fields
  const familyFields = MANDATORY_FIELDS[family as string] || [];
  familyFields.forEach(field => {
    const v = config[field];
    if (typeof v === 'string') {
      if (!v.trim()) missingFields.push(field);
      return;
    }
    if (typeof v === 'number') {
      if (v <= 0) missingFields.push(field);
      return;
    }
    if (v === null || v === undefined) missingFields.push(field);
  });

  return {
    isValid: missingFields.length === 0,
    missingFields,
  };
}

interface HardwareContextType {
  config: HardwareConfig;
  setConfig: (config: HardwareConfig) => void;
  updateConfig: (updates: Partial<HardwareConfig>) => void;
  lastAttemptTime: number;
  triggerAttempt: () => void;
}

export const DEFAULT_HARDWARE_CONFIG: HardwareConfig = {
  hardware: 'RTX4090',
  precision: 'fp16',
  batchSize: 64,
  seed: 0,
  device: 'cuda',
  useCompile: false,

  // Training
  learningRate: 0.0003,
  numEpochs: 100,
  gpuCount: 1,
  gpuMemoryGb: 80,

  // Data
  datasetSize: 10000000000,

  // Transformers
  seqLen: 0,
  vocabSize: 0,
  hiddenDim: 0,
  numHeads: 0,
  ffnDim: 0,
  numLayers: 0,
  useBias: false,
  dropout: 0,
  useFlash: false,

  // CNN / Spatial
  imgHeight: 0,
  imgWidth: 0,
  inChannels: 0,

  // ViT / Diffusion
  numDenoisingSteps: 0,
  guidanceScale: 0,

  // GNN
  numNodes: 0,
  numEdges: 0,
  nodeFeatDim: 0,

  // SSM / SNN
  dState: 0,
  dtRank: 0,
  timesteps: 0,
  spikeRate: 0,

  // MoE
  numExperts: 0,
  topK: 0,

  // RL
  actionDim: 0,
  stateDim: 0,
};

const HardwareContext = createContext<HardwareContextType | undefined>(undefined);

export const HardwareProvider = ({ children }: { children: ReactNode }) => {
  const [config, setConfig] = useState<HardwareConfig>(DEFAULT_HARDWARE_CONFIG);
  const [lastAttemptTime, setLastAttemptTime] = useState<number>(0);

  const updateConfig = (updates: Partial<HardwareConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
  };

  const triggerAttempt = () => {
    setLastAttemptTime(Date.now());
  };

  return (
    <HardwareContext.Provider value={{ config, setConfig, updateConfig, lastAttemptTime, triggerAttempt }}>
      {children}
    </HardwareContext.Provider>
  );
};

export function useHardware() {
  const context = useContext(HardwareContext);
  if (!context) throw new Error('useHardware must be used within HardwareProvider');
  return context;
}
