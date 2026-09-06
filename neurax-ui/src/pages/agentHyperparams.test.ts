/**
 * A training configuration the agent sets must reach the canvas.
 *
 * `initialize_hyperparams` and `set_hyperparams` were granted to three of the
 * agent's four modes and applied to its own snapshot copy — and nothing in the
 * studio handled either. The agent believed it had configured training, its
 * own state agreed, and the canvas never heard about it. Silently, and in both
 * directions: the agent then reasoned from values the user could not see.
 */
import { describe, expect, it } from 'vitest';

import { mapAgentHyperparams } from './Index.tsx';

describe('mapAgentHyperparams', () => {
  it('translates the agent\'s names into HardwareConfig fields', () => {
    expect(
      mapAgentHyperparams({
        learning_rate: 1e-4,
        weight_decay: 0.01,
        warmup_steps: 2000,
        max_steps: 100000,
        batch_size: 32,
        gradient_accumulation_steps: 1,
        lr_schedule: 'cosine_with_warmup',
        precision: 'bf16',
      }),
    ).toEqual({
      learningRate: 1e-4,
      weightDecay: 0.01,
      warmupSteps: 2000,
      maxSteps: 100000,
      batchSize: 32,
      gradAccumSteps: 1,
      lrScheduler: 'cosine_with_warmup',
      precision: 'bf16',
    });
  });

  it('keeps what has no field of its own instead of dropping it', () => {
    // betas, epsilon, clipping and dropout are all in the agent's own default
    // set; losing them would leave it reasoning about values nothing stores.
    const mapped = mapAgentHyperparams({
      learning_rate: 3e-4,
      beta1: 0.9,
      beta2: 0.999,
      max_grad_norm: 1.0,
      activation: 'gelu',
    });

    expect(mapped.learningRate).toBe(3e-4);
    expect(mapped.customParams).toEqual({
      beta1: 0.9,
      beta2: 0.999,
      max_grad_norm: 1.0,
      activation: 'gelu',
    });
  });

  it('merges into existing custom params rather than replacing them', () => {
    // A later set_hyperparams must not wipe what an earlier one put there.
    const mapped = mapAgentHyperparams({ dropout: 0.1 }, { beta1: 0.9, dropout: 0.5 });
    expect(mapped.customParams).toEqual({ beta1: 0.9, dropout: 0.1 });
  });

  it('ignores values no config could hold', () => {
    const mapped = mapAgentHyperparams({ schedule: { kind: 'cosine' }, notes: null });
    expect(mapped).toEqual({});
  });

  it('reports nothing to apply for an empty update', () => {
    expect(mapAgentHyperparams({})).toEqual({});
  });
});
