/**
 * The simulation target decides what every other number means.
 *
 * It used to be a catalogue, and these tests used to assert that five featured
 * datacenter chips were offered. They are not any more: the panel shows the
 * accelerators in this machine, and the database is the fallback for a machine
 * that has none. What is still worth guarding is that the specifications come
 * from the compiler's own database rather than a copy in the UI, that a card
 * the database does not know is *declared* rather than silently approximated,
 * and that the no-accelerator machine can still choose a target at all.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { SimulationTargetPanel } from './SimulationTargetPanel';
import { HardwareProvider } from '@/contexts/HardwareContext';
import type { HardwareProfile } from '@/types/runtime';

const CATALOGUE = [
  { name: 'H100-SXM', manufacturer: 'NVIDIA', memory_gb: 80, memory_bandwidth_gbs: 3352, tflops_fp64: 34, tflops_fp32: 67, tflops_fp16: 989, tflops_bf16: 989 },
  { name: 'RTX4090', manufacturer: 'NVIDIA', memory_gb: 24, memory_bandwidth_gbs: 1008, tflops_fp64: 0, tflops_fp32: 82, tflops_fp16: 165, tflops_bf16: 165 },
  { name: 'T4', manufacturer: 'NVIDIA', memory_gb: 16, memory_bandwidth_gbs: 300, tflops_fp64: 0, tflops_fp32: 8.1, tflops_fp16: 65, tflops_bf16: 0 },
];

vi.mock('@/services/neuraxApi.ts', () => ({
  listHardware: () => Promise.resolve(CATALOGUE),
}));

const GB = 1024 ** 3;

function profile(gpus: HardwareProfile['gpus']): HardwareProfile {
  return {
    detectedAt: '2026-09-08T09:00:00Z',
    cpu: { model: 'Ryzen 9 7950X', vendor: 'AuthenticAMD', cores: 16, threads: 32, architecture: 'x86_64' },
    ramTotalBytes: 64 * GB,
    ramAvailableBytes: 41 * GB,
    diskAvailableBytes: 892 * GB,
    gpus,
    os: 'Linux',
    notes: [],
  };
}

const open = (detected: HardwareProfile | null) =>
  render(
    <HardwareProvider>
      <SimulationTargetPanel isOpen onClose={() => {}} detected={detected} />
    </HardwareProvider>,
  );

const RTX = profile([
  {
    name: 'NVIDIA GeForce RTX 4090',
    vendor: 'NVIDIA',
    recognised: true,
    integrated: false,
    vramTotalBytes: 24 * GB,
    vramFreeBytes: 22.6 * GB,
    backend: 'cuda',
    driverVersion: '560.35.03',
  },
]);

describe('simulation target', () => {
  beforeEach(() => vi.clearAllMocks());

  it('shows the machine s own accelerator, not a catalogue', async () => {
    open(RTX);
    expect(await screen.findByText('NVIDIA GeForce RTX 4090')).toBeTruthy();
    // The featured datacenter parts are gone: nothing offers an H100 to a
    // workstation that has none.
    expect(screen.queryByText('H100-SXM')).toBeNull();
    expect(screen.queryByText('T4')).toBeNull();
  });

  it('reads its specifications from the database, matching past the driver s name', async () => {
    open(RTX);
    // "NVIDIA GeForce RTX 4090" has to resolve to the database's "RTX4090"
    // or the whole specification sheet silently disappears.
    const card = (await screen.findByText('NVIDIA GeForce RTX 4090')).closest('button')!;
    expect(within(card).getByText('165 TFLOP/s BF16')).toBeTruthy();
    expect(within(card).getByText((t) => /^1\s?008 GB\/s$/.test(t))).toBeTruthy();
  });

  it('shows free VRAM beside the total, since that is what governs whether a design fits', async () => {
    open(RTX);
    const card = (await screen.findByText('NVIDIA GeForce RTX 4090')).closest('button')!;
    expect(within(card).getByText('22.6 GB')).toBeTruthy();
    expect(within(card).getByText('24 GB')).toBeTruthy();
  });

  it('declares a card the database does not know instead of approximating quietly', async () => {
    open(
      profile([
        {
          name: 'Some Unreleased Card',
          vendor: 'Unknown',
          recognised: false,
          integrated: false,
          vramTotalBytes: 16 * GB,
          vramFreeBytes: 15 * GB,
          backend: 'vulkan',
        },
      ]),
    );
    expect(await screen.findByText(/no specification for this card/i)).toBeTruthy();
  });

  it('treats the CPU as the target when the machine has no accelerator', async () => {
    // The machine this was developed on. A CPU-only box is not a machine
    // NEURAX cannot analyse for — it is the machine most people have, and
    // offering a datacenter catalogue instead would be asking them to design
    // for hardware they do not own.
    open(profile([]));
    expect(await screen.findByText('Ryzen 9 7950X')).toBeTruthy();
    expect(await screen.findByText(/No accelerator was found, so this is what the work would run on/i)).toBeTruthy();
  });

  it('keeps the catalogue reachable, but as a deliberate act', async () => {
    open(profile([]));
    // Behind a disclosure, and named for what it is: designing for a machine
    // you do not have.
    expect(await screen.findByText(/Design for a machine you do not have/i)).toBeTruthy();
  });

  it('says plainly when the profile is an example rather than this machine', async () => {
    render(
      <HardwareProvider>
        <SimulationTargetPanel isOpen onClose={() => {}} detected={RTX} isExample />
      </HardwareProvider>,
    );
    expect(await screen.findByText(/Example machine, not this one/i)).toBeTruthy();
  });

  it('reports an integrated GPU as sharing memory, never as zero VRAM', async () => {
    // The real reading from this machine: an integrated part reports null for
    // both VRAM figures. Rendering that as "0.0 GB free" would say the card
    // is full when it has no dedicated memory at all.
    open(
      profile([
        {
          name: 'Intel WhiskeyLake-U GT2 [UHD Graphics 620]',
          vendor: 'Intel',
          recognised: false,
          integrated: true,
          vramTotalBytes: null,
          vramFreeBytes: null,
          backend: 'vulkan',
        },
      ]),
    );
    const card = (await screen.findByText(/UHD Graphics 620/)).closest('button')!;
    expect(within(card).getByText('shared')).toBeTruthy();
    expect(within(card).queryByText(/0\.0 GB/)).toBeNull();
    expect(await screen.findByText(/it has no memory of its own/i)).toBeTruthy();
  });

  it('records the chosen target under the database s name', async () => {
    open(RTX);
    fireEvent.click(await screen.findByText('NVIDIA GeForce RTX 4090'));
    await waitFor(() => expect(screen.getByText(/Analysing for RTX4090/)).toBeTruthy());
  });

  it('lets the device count be set', async () => {
    open(RTX);
    const input = (await screen.findByLabelText('Device count')) as HTMLInputElement;
    fireEvent.change(input, { target: { value: '8' } });
    await waitFor(() => expect(screen.getByText(/× 8/)).toBeTruthy());
  });
});
