/**
 * Simulation target: the chip every metric is computed against.
 *
 * Latency, throughput, VRAM headroom, cost and carbon all depend on which
 * accelerator the design is analysed for, so this is the setting that decides
 * what the rest of the numbers mean. It replaced a hyperparameter search panel
 * in this slot, which asked the user to optimise before they had said what they
 * were optimising for.
 *
 * Specifications are read from the compiler's own hardware database at open
 * time rather than restated here. A spec sheet copied into the UI is a second
 * source of truth that drifts from the one the analysis actually uses.
 */
import { useEffect, useMemo, useState } from 'react';
import { Cpu, Check, X, AlertTriangle, Loader2 } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog.tsx';
import { Input } from '@/components/ui/input.tsx';
import { cn } from '@/lib/utils.ts';
import { useHardware, type HardwareConfig } from '@/contexts/HardwareContext.tsx';
import { listHardware, type HardwareDetail } from '@/services/neuraxApi.ts';

/**
 * The five accelerators offered by default.
 *
 * Chosen for how widely they are actually deployed rather than for headline
 * throughput: the H100 and A100 carry most training, the L40S and T4 most
 * inference, and the RTX 4090 most work that happens on a desk. Every other
 * chip in the database stays reachable through the full list.
 */
const FEATURED = [
  {
    name: 'H100-SXM',
    role: 'Frontier training',
    note: 'The current datacenter standard for large-model training.',
  },
  {
    name: 'A100-SXM',
    role: 'Mainstream training',
    note: 'The most widely deployed datacenter accelerator.',
  },
  {
    name: 'L40S',
    role: 'Inference at scale',
    note: 'Serving workhorse — high throughput per watt and per rack unit.',
  },
  {
    name: 'RTX4090',
    role: 'Workstation',
    note: 'What most research and prototyping runs on locally.',
  },
  {
    name: 'T4',
    role: 'Cloud inference',
    note: 'The cheapest broadly available accelerator; the low-end floor.',
  },
] as const;

/**
 * Group thousands with a space, independent of the viewer's locale.
 *
 * `toLocaleString` renders "3,352", "3.352" or "3 352" depending on where the
 * reader is, which is the wrong kind of variation for a specification sheet.
 */
function formatNumber(value: number): string {
  return Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '\u202f');
}

function formatTflops(gpu: HardwareDetail): string {
  // bf16 is the training default; some inference parts only publish fp16.
  const value = gpu.tflops_bf16 || gpu.tflops_fp16 || gpu.tflops_fp32;
  const label = gpu.tflops_bf16 ? 'BF16' : gpu.tflops_fp16 ? 'FP16' : 'FP32';
  return `${formatNumber(value)} TFLOP/s ${label}`;
}

interface SimulationTargetPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * The precisions the analysis actually understands.
 *
 * `neuraxCompiler`'s `VALID_PRECISIONS` is the gate — anything outside this set
 * silently becomes fp32 — and `neurax-formulas::dtype_bytes` is where each one
 * turns into a byte width. Both are mirrored in the labels below so a user can
 * see why the memory figures move before they move.
 *
 * This control did not exist. `precision` sits in `MANDATORY_FIELDS.common`
 * beside `hardware` and `batchSize`, it multiplies every memory number the
 * product reports, and no surface in the studio could set it: a canvas stayed
 * on the `DEFAULT_HARDWARE_CONFIG` value of fp16 unless the design came in
 * through an import or the agent wrote it. The Inference workspace's
 * "Quantization Level" select looks like this control but is not — it seeds
 * itself from `design.precision` and writes only to a local inference-simulation
 * parameter, so it never changes the design's own weights or VRAM.
 */
const PRECISIONS: { value: HardwareConfig['precision']; label: string; bytesPerParam: string }[] = [
  { value: 'fp32', label: 'Full precision', bytesPerParam: '4 bytes per parameter' },
  { value: 'fp16', label: 'Half precision', bytesPerParam: '2 bytes per parameter' },
  { value: 'bf16', label: 'Brain float', bytesPerParam: '2 bytes per parameter' },
  { value: 'int8', label: '8-bit integer', bytesPerParam: '1 byte per parameter' },
  { value: 'int4', label: '4-bit integer', bytesPerParam: '0.5 byte per parameter, two packed per byte' },
];

export function SimulationTargetPanel({ isOpen, onClose }: SimulationTargetPanelProps) {
  const { config, updateConfig } = useHardware();
  const [catalogue, setCatalogue] = useState<HardwareDetail[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [gpuCount, setGpuCount] = useState<number>(config.gpuCount || 1);

  useEffect(() => {
    if (!isOpen || catalogue) return;
    let cancelled = false;
    listHardware()
      .then((list: HardwareDetail[]) => !cancelled && setCatalogue(list))
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [isOpen, catalogue]);

  const byName = useMemo(
    () => new Map((catalogue ?? []).map((gpu) => [gpu.name, gpu])),
    [catalogue],
  );

  const featured = FEATURED.map((entry) => ({ ...entry, gpu: byName.get(entry.name) }));
  const others = (catalogue ?? [])
    .filter((gpu) => !FEATURED.some((f) => f.name === gpu.name))
    .sort((a, b) => (b.tflops_bf16 || b.tflops_fp16) - (a.tflops_bf16 || a.tflops_fp16));

  const selected = config.hardware;

  const choose = (name: string) => {
    const gpu = byName.get(name);
    updateConfig({
      hardware: name,
      // Carry the VRAM across so memory checks reflect the chosen part rather
      // than whatever the previous target had.
      ...(gpu ? { gpuMemoryGb: gpu.memory_gb } : {}),
      gpuCount,
    });
  };

  const applyCount = (value: number) => {
    const count = Math.max(1, Math.min(4096, Math.round(value) || 1));
    setGpuCount(count);
    updateConfig({ gpuCount: count });
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-hidden flex flex-col bg-card p-0 gap-0">
        <DialogTitle className="sr-only">Simulation target</DialogTitle>
        <DialogDescription className="sr-only">
          Choose the accelerator every latency, memory and cost figure is computed for.
        </DialogDescription>

        <div className="px-4 py-3 border-b border-border flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-primary" />
            <div>
              <div className="text-[13px] font-bold tracking-[-0.02em] text-foreground">
                Simulation Target
              </div>
              <div className="text-[10px] text-muted-foreground">
                Every latency, cost and memory figure is computed for this chip
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="p-1 rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 space-y-4 scrollbar-thin">
          {error && (
            <div className="flex items-start gap-2 rounded-[8px] border border-destructive/40 bg-destructive/5 p-3">
              <AlertTriangle className="w-3.5 h-3.5 text-destructive mt-0.5 shrink-0" />
              <div className="text-[11px] text-muted-foreground">
                Could not load hardware specifications ({error}). Selecting a chip still
                works, but its specifications cannot be shown.
              </div>
            </div>
          )}

          {!catalogue && !error && (
            <div className="flex items-center gap-2 py-8 justify-center text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-[11px]">Loading hardware specifications…</span>
            </div>
          )}

            <div className="rounded-[8px] border border-border/60 bg-background/40 p-3">
              <label
                htmlFor="numeric-precision"
                className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80"
              >
                Numeric precision
              </label>
              <p className="text-[10px] text-muted-foreground mt-0.5 mb-2">
                Width every weight, activation and optimizer state is stored at. Multiplies
                the whole memory analysis — int4 packs two values per byte, an eighth of fp32.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {PRECISIONS.map((option) => {
                  const active = config.precision === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => updateConfig({ precision: option.value })}
                      aria-pressed={active}
                      title={`${option.label} — ${option.bytesPerParam}`}
                      className={
                        'px-2.5 py-1 rounded-[6px] border text-[11px] font-mono transition-colors ' +
                        (active
                          ? 'border-primary bg-primary/10 text-primary font-semibold'
                          : 'border-border/60 text-muted-foreground hover:text-foreground hover:border-border')
                      }
                    >
                      {option.value}
                    </button>
                  );
                })}
              </div>
              <p className="text-[10px] font-mono text-muted-foreground/70 mt-2">
                {PRECISIONS.find((o) => o.value === config.precision)?.bytesPerParam ??
                  'unrecognised — the analysis falls back to fp32'}
              </p>
            </div>

          {catalogue && (
            <>
              <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))' }}>
                {featured.map(({ name, role, note, gpu }) => {
                  const isSelected = selected === name;
                  return (
                    <button
                      key={name}
                      type="button"
                      onClick={() => choose(name)}
                      aria-pressed={isSelected}
                      className={cn(
                        'relative text-left rounded-[10px] border p-3 transition-all hover:border-border',
                        isSelected
                          ? 'border-primary bg-primary/[0.07]'
                          : 'border-border/60 bg-background/40',
                      )}
                    >
                      <div className="flex items-baseline justify-between gap-2 mb-1">
                        <span className="text-[13px] font-semibold text-foreground">{name}</span>
                        <span className="text-[9px] font-mono uppercase tracking-[0.12em] text-muted-foreground">
                          {role}
                        </span>
                      </div>
                      <p className="text-[10px] leading-relaxed text-muted-foreground mb-2">{note}</p>
                      {gpu ? (
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] font-mono text-muted-foreground">
                          <span>{gpu.memory_gb} GB</span>
                          <span>{formatTflops(gpu)}</span>
                          <span>{formatNumber(gpu.memory_bandwidth_gbs)} GB/s</span>
                        </div>
                      ) : (
                        <div className="text-[10px] text-muted-foreground/60">
                          Specifications unavailable
                        </div>
                      )}
                      {isSelected && (
                        <div className="absolute top-2 right-2 w-4 h-4 rounded-full bg-primary flex items-center justify-center">
                          <Check className="w-2.5 h-2.5 text-primary-foreground" />
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>

              <div className="rounded-[8px] border border-border/60 bg-background/40 p-3">
                <label
                  htmlFor="gpu-count"
                  className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80"
                >
                  Device count
                </label>
                <p className="text-[10px] text-muted-foreground mt-0.5 mb-2">
                  Devices the workload is spread over. Drives the parallelism analysis and the
                  total GPU-hour cost.
                </p>
                <Input
                  id="gpu-count"
                  type="number"
                  min={1}
                  value={gpuCount}
                  onChange={(e) => applyCount(Number(e.target.value))}
                  className="h-7 text-[11px] px-2 bg-background/50 max-w-[140px]"
                />
              </div>

              <div>
                <button
                  type="button"
                  onClick={() => setShowAll((v) => !v)}
                  className="text-[10px] font-bold uppercase tracking-[0.15em] text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showAll ? 'Hide' : 'Show'} the other {others.length} chips
                </button>
                {showAll && (
                  <div className="mt-2 grid gap-1.5" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
                    {others.map((gpu) => {
                      const isSelected = selected === gpu.name;
                      return (
                        <button
                          key={gpu.name}
                          type="button"
                          onClick={() => choose(gpu.name)}
                          aria-pressed={isSelected}
                          className={cn(
                            'text-left rounded-[8px] border px-2.5 py-2 transition-all',
                            isSelected
                              ? 'border-primary bg-primary/[0.07]'
                              : 'border-border/50 bg-background/30 hover:border-border',
                          )}
                        >
                          <div className="text-[11px] font-medium text-foreground">{gpu.name}</div>
                          <div className="text-[9px] font-mono text-muted-foreground">
                            {gpu.memory_gb} GB · {formatTflops(gpu)}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        <div className="border-t border-border px-4 py-3 flex items-center justify-between shrink-0">
          <div className="text-[10px] font-mono text-muted-foreground">
            {selected ? `Analysing for ${selected} × ${gpuCount}` : 'No target selected'}
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-[8px] text-[11px] font-semibold bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
          >
            Done
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
