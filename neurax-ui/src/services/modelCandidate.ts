/**
 * One gate, for every model NEURAX is about to train — whoever wrote it.
 *
 * NEURAX's deterministic generator was never trustworthy because a machine
 * wrote it. It was trustworthy because its parameter count is confronted with
 * the analysis, and it refuses when the two disagree. That is a property of
 * the *check*, not of the author, which is what makes it safe for the
 * assistant to write model code too: code from anywhere earns the same trust
 * the same way, or it does not earn it.
 *
 * The check here is stricter than the one the export panel runs.
 * `verifyCodegenAgainstAnalysis` compares the generator's own arithmetic
 * against the compiler's — two NEURAX predictions agreeing with each other,
 * which is worth something but not much. This asks PyTorch to build the thing
 * and counts what it actually got, then confronts *that* with the analysis.
 * A generator that miscounts a layer type and a compiler that miscounts it the
 * same way would pass the first check and fail this one.
 */
import { verifyModel, type ModelVerdict } from './neuraxApi.ts';

/** A model offered for training, before anything has been checked. */
export interface ModelCandidate {
  /** Who wrote it. Changes nothing about the checking — it is for the log. */
  source: 'generator' | 'assistant';
  code: string;
  modelClassName: string;
  /** One sample's shape, without the batch dimension. */
  inputShape: number[];
  inputKind: 'tokens' | 'image' | 'features';
  vocabSize?: number;
  numClasses: number;
}

/** What happened when a candidate met the gate. */
export interface CandidateReview {
  accepted: boolean;
  /** Why, in the words the studio and the assistant both read. */
  reason: string;
  /** How far it got: `import`, `class`, `construct`, `verified`, `checker`. */
  stage: string;
  /** What PyTorch actually built, when it got that far. */
  builtParams: number | null;
  /** What the compiler said this design would be. */
  analyzedParams: number | null;
  /** Distance between the two, as a fraction. */
  deltaPct: number | null;
  outputShape: number[] | null;
  /** True when nothing could be checked, rather than something being wrong.
   *  A studio with no service running is not a studio holding a bad model,
   *  and the two must not read alike. */
  unchecked: boolean;
  verdict: ModelVerdict | null;
}

/** The same 1% the export panel uses, so the two never disagree. */
export const PARAM_TOLERANCE = 0.01;

function pct(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

function abbreviate(text: string, limit = 400): string {
  const trimmed = text.trim();
  if (trimmed.length <= limit) return trimmed;
  // The last lines of a traceback carry the exception; the first carry the
  // import machinery. Keep the end.
  return `…${trimmed.slice(trimmed.length - limit)}`;
}

/**
 * Build the candidate for real and decide whether it may be trained.
 *
 * `analyzedParams` is what the compiler said this design would be; pass 0 or
 * `null` when there is no analysis to confront it with, and the parameter
 * comparison is skipped rather than faked. Everything before that step still
 * runs: a model that will not import is refused whether or not anything was
 * analysed.
 */
export async function reviewCandidate(
  candidate: ModelCandidate,
  analyzedParams: number | null,
): Promise<CandidateReview> {
  const verdict = await verifyModel({
    modelCode: candidate.code,
    modelClass: candidate.modelClassName,
    inputShape: candidate.inputShape,
    inputKind: candidate.inputKind,
    vocabSize: candidate.vocabSize,
  });

  const base = {
    builtParams: null,
    analyzedParams: analyzedParams && analyzedParams > 0 ? analyzedParams : null,
    deltaPct: null,
    outputShape: null,
    unchecked: false,
    verdict,
  } satisfies Omit<CandidateReview, 'accepted' | 'reason' | 'stage'>;

  if (!verdict) {
    return {
      ...base,
      accepted: false,
      stage: 'checker',
      unchecked: true,
      reason:
        'The local NEURAX service did not answer, so this model could not be built and checked. ' +
        'That is not a verdict on the code — start the service and ask again.',
    };
  }

  if (!verdict.ok) {
    const detail = verdict.error ? ` ${abbreviate(verdict.error)}` : '';
    const at =
      verdict.stage === 'import'
        ? 'The file does not import.'
        : verdict.stage === 'class'
          ? 'The file imports, but the model class could not be found in it.'
          : verdict.stage === 'construct'
            ? 'The class exists but could not be instantiated.'
            : verdict.stage === 'torch'
              ? 'PyTorch is not available on this machine, so nothing could be built.'
              : 'The model could not be built.';
    return { ...base, accepted: false, stage: verdict.stage, reason: `${at}${detail}` };
  }

  const builtParams = typeof verdict.parameters === 'number' ? verdict.parameters : null;
  const withBuild = { ...base, builtParams, outputShape: verdict.outputShape ?? null };

  if (!verdict.forwardOk) {
    const detail = verdict.forwardError ? ` ${abbreviate(verdict.forwardError)}` : '';
    return {
      ...withBuild,
      accepted: false,
      stage: verdict.stage,
      reason:
        `The model builds (${builtParams?.toLocaleString('en-US') ?? '?'} parameters) but a batch of ` +
        `[${candidate.inputShape.join(', ')}] ${candidate.inputKind} does not go through it. ` +
        `A run would die on its first step.${detail}`,
    };
  }

  // Nothing to confront it with. Saying so is the honest answer; claiming a
  // match against a number that does not exist is the failure this codebase
  // keeps removing.
  if (!base.analyzedParams || builtParams === null) {
    return {
      ...withBuild,
      accepted: true,
      stage: verdict.stage,
      reason:
        `The model builds and takes its input (${builtParams?.toLocaleString('en-US') ?? '?'} parameters). ` +
        'There is no analysis to compare it against, so the parameter count is unconfirmed — ' +
        'run the analysis to close that gap.',
    };
  }

  const deltaPct = Math.abs(builtParams - base.analyzedParams) / base.analyzedParams;
  const shared = { ...withBuild, deltaPct, stage: verdict.stage };

  if (deltaPct > PARAM_TOLERANCE) {
    return {
      ...shared,
      accepted: false,
      reason:
        `PyTorch builds ${builtParams.toLocaleString('en-US')} parameters, but NEURAX analysed ` +
        `${base.analyzedParams.toLocaleString('en-US')} — ${pct(deltaPct)} apart. This is not the ` +
        'model that was costed, so every figure on screen is about something else. ' +
        'Fix the code or re-analyse the design so the two agree.',
    };
  }

  return {
    ...shared,
    accepted: true,
    reason:
      `Built by PyTorch: ${builtParams.toLocaleString('en-US')} parameters, within ${pct(deltaPct)} of ` +
      `the ${base.analyzedParams.toLocaleString('en-US')} NEURAX analysed, and one batch of ` +
      `[${candidate.inputShape.join(', ')}] goes through it` +
      `${verdict.outputShape ? ` to [${verdict.outputShape.join(', ')}]` : ''}.`,
  };
}
