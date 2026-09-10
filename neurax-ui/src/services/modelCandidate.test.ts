/**
 * The gate is the whole reason the assistant is allowed to write model code,
 * so these tests are about what it *refuses*, not what it accepts.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { reviewCandidate, type ModelCandidate } from './modelCandidate.ts';
import type { ModelVerdict } from './neuraxApi.ts';

vi.mock('./neuraxApi.ts', () => ({ verifyModel: vi.fn() }));
const { verifyModel } = await import('./neuraxApi.ts');
const mocked = verifyModel as unknown as ReturnType<typeof vi.fn>;

const candidate: ModelCandidate = {
  source: 'assistant',
  code: 'irrelevant — the checker is mocked',
  modelClassName: 'Net',
  inputShape: [3, 32, 32],
  inputKind: 'image',
  numClasses: 10,
};

const built = (parameters: number, extra: Partial<ModelVerdict> = {}): ModelVerdict => ({
  ok: true,
  stage: 'verified',
  parameters,
  className: 'Net',
  forwardOk: true,
  outputShape: [2, 10],
  ...extra,
});

beforeEach(() => mocked.mockReset());

describe('reviewCandidate', () => {
  it('accepts a model whose real parameter count matches the analysis', async () => {
    mocked.mockResolvedValue(built(1_000_000));
    const review = await reviewCandidate(candidate, 1_000_000);
    expect(review.accepted).toBe(true);
    expect(review.builtParams).toBe(1_000_000);
    expect(review.deltaPct).toBe(0);
  });

  it('accepts a small disagreement, because rounding is not a different model', async () => {
    mocked.mockResolvedValue(built(1_005_000));
    const review = await reviewCandidate(candidate, 1_000_000);
    expect(review.accepted).toBe(true);
  });

  it('refuses code that builds a different model from the one costed', async () => {
    // The failure the whole gate exists for: it trains fine, and every figure
    // the studio showed was about something else.
    mocked.mockResolvedValue(built(2_000_000));
    const review = await reviewCandidate(candidate, 1_000_000);
    expect(review.accepted).toBe(false);
    expect(review.deltaPct).toBeCloseTo(1);
    expect(review.reason).toContain('2,000,000');
    expect(review.reason).toContain('1,000,000');
  });

  it('refuses code that will not import, and names the stage', async () => {
    mocked.mockResolvedValue({
      ok: false,
      stage: 'import',
      error: 'SyntaxError: invalid syntax',
    });
    const review = await reviewCandidate(candidate, 1_000_000);
    expect(review.accepted).toBe(false);
    expect(review.stage).toBe('import');
    expect(review.reason).toContain('does not import');
    expect(review.reason).toContain('SyntaxError');
  });

  it('refuses a model whose declared input does not fit it', async () => {
    // Right weights, wrong shapes: the parameter count alone cannot see this.
    mocked.mockResolvedValue(
      built(1_000_000, {
        forwardOk: false,
        forwardError: 'RuntimeError: mat1 and mat2 shapes cannot be multiplied',
        outputShape: undefined,
      }),
    );
    const review = await reviewCandidate(candidate, 1_000_000);
    expect(review.accepted).toBe(false);
    expect(review.reason).toContain('first step');
    expect(review.builtParams).toBe(1_000_000);
  });

  it('accepts, but says the count is unconfirmed, when nothing was analysed', async () => {
    mocked.mockResolvedValue(built(1_000_000));
    const review = await reviewCandidate(candidate, null);
    expect(review.accepted).toBe(true);
    expect(review.analyzedParams).toBeNull();
    expect(review.reason).toContain('unconfirmed');
  });

  it('separates "could not check" from "checked and wrong"', async () => {
    // No service running. A studio with no checker is not a studio holding a
    // bad model, and a caller must be able to tell the two apart.
    mocked.mockResolvedValue(null);
    const review = await reviewCandidate(candidate, 1_000_000);
    expect(review.accepted).toBe(false);
    expect(review.unchecked).toBe(true);
    expect(review.stage).toBe('checker');
  });

  it('truncates a long traceback from the end, where the exception is', async () => {
    const traceback = `${'stack frame\n'.repeat(200)}RuntimeError: the real cause`;
    mocked.mockResolvedValue({ ok: false, stage: 'construct', error: traceback });
    const review = await reviewCandidate(candidate, 1_000_000);
    expect(review.reason).toContain('RuntimeError: the real cause');
    expect(review.reason.length).toBeLessThan(600);
  });
});
