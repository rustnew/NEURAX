import { CanvasNode, Connection, LayerType } from '@/types/architecture.ts';
import { NeuraxTraining, NeuraxHardware, NeuraxData } from '@/utils/neuraxCompiler.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { HardwareConfig } from '@/contexts/HardwareContext.tsx';

interface ImportedLayer {
  id?: string;
  type: string;
  layer_type?: string;
  params?: Record<string, any>;
  input_shape?: number[];
  output_shape?: number[];
  inputs?: string[];
  outputs?: string[];
}

// Support both the internal NeuraxIR format and simpler third-party formats
interface ImportedModel {
  schema_version?: string | number;
  model_name?: string;
  model?: {
    name?: string;
    type?: string;
    kind?: string;
    layers?: ImportedLayer[];
    global_params?: Record<string, any>;
    global_config?: {
      batch_size?: number;
      seq_len?: number;
      embedding_dim?: number;
      dtype?: string;
      device_target?: string;
    };
  };
  training?: NeuraxTraining;
  hardware?: NeuraxHardware;
  data?: NeuraxData;
}

// Map imported layer types to our internal LayerType
const layerTypeMap: Record<string, LayerType> = {
  'multi_head_attention': 'attention',
  'attention': 'attention',
  'layer_norm': 'layernorm',
  'layernorm': 'layernorm',
  'feed_forward': 'dense',
  'dense': 'dense',
  'linear': 'dense',
  'conv2d': 'conv2d',
  'convolution': 'conv2d',
  'relu': 'relu',
  'gelu': 'gelu',
  'batch_norm': 'batchnorm',
  'batchnorm': 'batchnorm',
  'residual': 'residual',
  'skip_connection': 'residual',
  'transformer': 'transformer',
  'transformer_block': 'transformer',
  'input': 'input',
  'output': 'output',
  'embedding': 'input',
};

// Map layer types to display names
const layerNameMap: Record<string, string> = {
  'attention': 'MultiHead_Attn',
  'layernorm': 'LayerNorm',
  'dense': 'Dense',
  'conv2d': 'Conv2D',
  'relu': 'ReLU',
  'gelu': 'GELU',
  'batchnorm': 'BatchNorm',
  'residual': 'Residual',
  'transformer': 'Transformer',
  'input': 'Input',
  'output': 'Output',
};

function mapLayerType(importedType: string): LayerType {
  const normalized = importedType.toLowerCase().replace(/[-\s]/g, '_');
  return layerTypeMap[normalized] || 'dense';
}

function formatShape(shape: number[] | undefined): string {
  if (!shape || shape.length === 0) return 'auto';
  return `[${shape.map(d => d === 1 ? 'B' : d).join(', ')}]`;
}

export interface ImportResult {
  nodes: CanvasNode[];
  connections: Connection[];
  modelName: string;
  family?: ArchitectureFamily;
  hardwareConfig?: Partial<HardwareConfig>;
  error?: string;
}

export function parseArchitectureJSON(jsonString: string): ImportResult {
  try {
    const data: ImportedModel = JSON.parse(jsonString);

    // Validate core structure
    const layers = data.model?.layers;
    if (!layers || !Array.isArray(layers)) {
      return {
        nodes: [],
        connections: [],
        modelName: data.model_name || data.model?.name || 'Imported Model',
        error: 'Invalid JSON structure: missing model.layers array',
      };
    }

    const nodes: CanvasNode[] = [];
    const connections: Connection[] = [];

    // Extract Global Config / Training / Hardware
    const hwConfig: Partial<HardwareConfig> = {};
    const training = data.training;
    const hardware = data.hardware;
    const stats_data = data.data;
    const modelType = data.model?.type || data.model?.kind || 'transformer';

    // 1. Map Training
    if (training) {
      if (training.batch_size) hwConfig.batchSize = training.batch_size;
      if (training.learning_rate) hwConfig.learningRate = training.learning_rate;
      if (training.num_epochs) hwConfig.numEpochs = training.num_epochs;
      if (training.precision) hwConfig.precision = training.precision as any;
      if (training.sequence_length) hwConfig.seqLen = training.sequence_length;
    }

    // 2. Map Hardware
    if (hardware?.gpus?.[0]) {
      const gpu = hardware.gpus[0];
      hwConfig.hardware = gpu.name;
      hwConfig.gpuCount = gpu.count;
      if (gpu.memory_gb) hwConfig.gpuMemoryGb = gpu.memory_gb;
    }

    // 3. Map Data
    if (stats_data) {
      if (stats_data.dataset_size) hwConfig.datasetSize = stats_data.dataset_size;
      if (stats_data.vocab_size) hwConfig.vocabSize = stats_data.vocab_size;
      if (stats_data.num_classes) hwConfig.numClasses = stats_data.num_classes;
    }

    // 4. Map Model Global Params
    const globalParams = data.model?.global_params || data.model?.global_config;
    if (globalParams) {
      const gp = globalParams as any;
      if (gp.hidden_size || gp.embedding_dim)
        hwConfig.hiddenDim = gp.hidden_size || gp.embedding_dim;
      if (gp.num_layers) hwConfig.numLayers = gp.num_layers;
      if (gp.num_heads) hwConfig.numHeads = gp.num_heads;
      if (gp.vocab_size) hwConfig.vocabSize = gp.vocab_size;
    }

    // Layout configuration
    const startX = 100;
    const startY = 100;
    const nodeSpacingX = 250;
    const nodeSpacingY = 150;
    const nodesPerRow = 4;

    // Process nodes
    const nodeMap = new Map<string, string>(); // imported ID -> new internal ID

    layers.forEach((layer, index) => {
      const col = index % nodesPerRow;
      const row = Math.floor(index / nodesPerRow);

      const importedId = layer.id || `layer-${index}`;
      const internalId = `import-${index}-${Date.now()}`;
      nodeMap.set(importedId, internalId);

      const layerType = mapLayerType(layer.layer_type || layer.type);
      const baseName = layerNameMap[layerType] || layerType.charAt(0).toUpperCase() + layerType.slice(1);

      const node: CanvasNode = {
        id: internalId,
        type: layerType,
        name: layer.id || `${baseName}_${index + 1}`,
        x: startX + (col * nodeSpacingX),
        y: startY + (row * nodeSpacingY),
        params: layer.params || {},
        inputShape: layer.input_shape ? formatShape(layer.input_shape) : 'auto',
        outputShape: layer.output_shape ? formatShape(layer.output_shape) : 'auto',
      };

      nodes.push(node);
    });

    // Process connections if present (explicit or inferred)
    layers.forEach((layer, index) => {
      const toId = nodeMap.get(layer.id || `layer-${index}`);
      if (!toId) return;

      if (layer.inputs && Array.isArray(layer.inputs)) {
        // Explicit inputs
        layer.inputs.forEach(inputRef => {
          const fromId = nodeMap.get(inputRef);
          if (fromId) {
            connections.push({
              id: `import-conn-${fromId}-${toId}`,
              from: fromId,
              to: toId
            });
          }
        });
      } else if (index > 0) {
        // Sequential fallback
        const prevId = nodeMap.get(layers[index - 1].id || `layer-${index - 1}`);
        if (prevId) {
          connections.push({
            id: `import-conn-${index}`,
            from: prevId,
            to: toId
          });
        }
      }
    });

    return {
      nodes,
      connections,
      modelName: data.model_name || data.model?.name || 'Imported Model',
      family: modelType as ArchitectureFamily,
      hardwareConfig: hwConfig,
    };
  } catch (err) {
    return {
      nodes: [],
      connections: [],
      modelName: 'Error',
      error: err instanceof Error ? err.message : 'Failed to parse JSON',
    };
  }
}

// Sample JSON for reference
export const sampleTransformerJSON = `{
  "schema_version": "1.0",
  "model": {
    "name": "bert-base-sim",
    "type": "transformer",
    "global_params": {
      "hidden_size": 768,
      "num_layers": 12,
      "vocab_size": 30522,
      "sequence_length": 512,
      "num_heads": 12
    },
    "layers": [
      {
        "id": "embeddings",
        "type": "Embedding",
        "params": {
          "vocab_size": 30522,
          "d_model": 768
        },
        "outputs": ["emb_out"]
      },
      {
        "id": "encoder_block_0",
        "type": "transformer",
        "inputs": ["emb_out"],
        "params": {
          "n_heads": 12,
          "d_model": 768
        }
      }
    ]
  },
  "training": {
    "batch_size": 32,
    "precision": "fp16",
    "learning_rate": 0.0001,
    "num_epochs": 10
  },
  "hardware": {
    "gpus": [
      {
        "name": "A100",
        "count": 8,
        "memory_gb": 80
      }
    ]
  },
  "data": {
    "dataset_size": 1000000000
  }
}`;
