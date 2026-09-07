/**
 * A client must be able to set the precision their model is stored at.
 *
 * `precision` sits in `MANDATORY_FIELDS.common` beside `hardware` and
 * `batchSize`, and it multiplies every memory figure NEURAX reports — yet no
 * surface in the studio could set it. `hyperparameterDefs.ts` has no entry for
 * it, so the hyperparameters panel never rendered one, and a canvas stayed on
 * `DEFAULT_HARDWARE_CONFIG`'s fp16 unless the design arrived through an import
 * or the agent wrote it.
 *
 * The Inference workspace's "Quantization Level" select looks like this
 * control and is not: it seeds itself from `design.precision` and writes only
 * to a local inference-simulation parameter.
 */
import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

import { SimulationTargetPanel } from './SimulationTargetPanel.tsx';
import { HardwareProvider } from '@/contexts/HardwareContext.tsx';

function renderPanel() {
  // The panel fetches the GPU catalogue; the precision control must work
  // whether or not that lands.
  vi.stubGlobal('fetch', () =>
    Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve([]), text: () => Promise.resolve('[]') } as Response),
  );
  return render(
    <HardwareProvider>
      <SimulationTargetPanel isOpen onClose={() => {}} />
    </HardwareProvider>,
  );
}

describe('the precision control', () => {
  it('offers every width the analysis understands', () => {
    renderPanel();
    for (const p of ['fp32', 'fp16', 'bf16', 'int8', 'int4']) {
      expect(screen.getByRole('button', { name: p }), p).toBeInTheDocument();
    }
  });

  it('shows the studio default as the one in effect', () => {
    renderPanel();
    expect(screen.getByRole('button', { name: 'fp16' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'int4' })).toHaveAttribute('aria-pressed', 'false');
  });

  it('changes the design when another width is chosen', () => {
    renderPanel();
    fireEvent.click(screen.getByRole('button', { name: 'int4' }));

    expect(screen.getByRole('button', { name: 'int4' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'fp16' })).toHaveAttribute('aria-pressed', 'false');
  });

  it('says what the chosen width costs per parameter', () => {
    // int4 is the one a user picks to fit smaller hardware, and the one whose
    // packing is easy to get wrong — it is stated rather than implied.
    renderPanel();
    fireEvent.click(screen.getByRole('button', { name: 'int4' }));
    expect(screen.getByText(/0\.5 byte per parameter, two packed per byte/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'fp32' }));
    expect(screen.getByText(/4 bytes per parameter/i)).toBeInTheDocument();
  });
});
