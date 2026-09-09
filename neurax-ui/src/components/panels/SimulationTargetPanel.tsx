/**
 * Simulation target: the chip every metric is computed against.
 *
 * Latency, throughput, VRAM headroom, cost and carbon all depend on which
 * accelerator the design is analysed for, so this is the setting that decides
 * what the rest of the numbers mean.
 *
 * It used to open on a catalogue: five featured datacenter parts, and the rest
 * of the database behind a toggle. That made sense while NEURAX knew nothing
 * about the machine it was running on — you had to say what you were designing
 * for, because nothing could tell. Now something can, and the catalogue was
 * answering a question the user no longer has. Worse, it put an H100 one click
 * from the default on a workstation that has none, so the easiest path led to
 * a verdict about a machine the user cannot run.
 *
 * So this panel shows the accelerators in *this* machine and their
 * specifications, and nothing else. The full database stays reachable, but only
 * where it is genuinely needed: on a machine with no accelerator detected,
 * where there is nothing else to analyse against.
 *
 * Specifications are read from the compiler's own hardware database rather
 * than restated here. A spec sheet copied into the UI is a second source of
 * truth that drifts from the one the analysis actually uses — and where the
 * driver's name does not match anything in that database, the panel says the
 * figures are approximate instead of showing a confident wrong number.
 */
import { useEffect, useMemo, useState } from 'react';
import { Cpu, Check, X, AlertTriangle, Loader2, MonitorCog } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog.tsx';
import { Input } from '@/components/ui/input.tsx';
import { cn } from '@/lib/utils.ts';
import { useHardware, type HardwareConfig } from '@/contexts/HardwareContext.tsx';
import type { GpuInfo, HardwareProfile } from '@/types/runtime.ts';
import { listHardware, type HardwareDetail } from '@/services/neuraxApi.ts';

const GB = 1024 ** 3;

/**
 * Group thousands with a space, independent of the viewer's locale.
 *
 * `toLocaleString` renders "3,352", "3.352" or "3 352" depending on where the
 * reader is, which is the wrong kind of variation for a specification sheet.
 */
function formatNumber(value: number): string {
  return Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

function formatTflops(gpu: HardwareDetail): string {
  // bf16 is the training default; some inference parts only publish fp16.
  const value = gpu.tflops_bf16 || gpu.tflops_fp16 || gpu.tflops_fp32;
  const label = gpu.tflops_bf16 ? 'BF16' : gpu.tflops_fp16 ? 'FP16' : 'FP32';
  return `${formatNumber(value)} TFLOP/s ${label}`;
}

/**
 * Match a driver's name to a database entry.
 *
 * Drivers report "NVIDIA GeForce RTX 4090" where the database says "RTX4090",
 * so an exact comparison would declare every real card unrecognised and quietly
 * drop the whole specification sheet. Spaces, dashes and the vendor prefix
 * carry no information here; what is left is compared case-insensitively.
 */
function normaliseName(name: string): string {
  return name
    .toLowerCase()
    .replace(/\b(nvidia|amd|intel|geforce|radeon|arc)\b/g, '')
    .replace(/[^a-z0-9]/g, '');
}

function findSpec(name: string, catalogue: HardwareDetail[] | null): HardwareDetail | undefined {
  if (!catalogue) return undefined;
  const wanted = normaliseName(name);
  return catalogue.find((gpu) => normaliseName(gpu.name) === wanted);
}

interface SimulationTargetPanelProps {
  /** What was detected on this machine. Absent when detection has not run or
   *  found nothing — a normal state, and then the database is offered instead,
   *  because a target still has to be chosen for anything to be analysed. */
  detected?: HardwareProfile | null;
  /** True when `detected` is the example profile rather than a reading from
   *  this machine — the local service is not running. Said plainly, because a
   *  fallback mistaken for a measurement is worse than no measurement. */
  isExample?: boolean;
  isMeasuring?: boolean;
  /** Runs the throughput measurement. Absent when the caller cannot. */
  onMeasure?: () => void;
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
 */
const PRECISIONS: { value: HardwareConfig['precision']; label: string; bytesPerParam: string }[] = [
  { value: 'fp32', label: 'Full precision', bytesPerParam: '4 bytes per parameter' },
  { value: 'fp16', label: 'Half precision', bytesPerParam: '2 bytes per parameter' },
  { value: 'bf16', label: 'Brain float', bytesPerParam: '2 bytes per parameter' },
  { value: 'int8', label: '8-bit integer', bytesPerParam: '1 byte per parameter' },
  { value: 'int4', label: '4-bit integer', bytesPerParam: '0.5 byte per parameter, two packed per byte' },
];

/** One line of the specification sheet. */
function Spec({ label, value, tone }: { label: string; value: string; tone?: 'primary' }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] font-mono uppercase tracking-[0.12em] text-muted-foreground/70">{label}</span>
      <span
        className={cn(
          'text-[12px] font-mono tabular-nums',
          tone === 'primary' ? 'text-primary font-semibold' : 'text-foreground',
        )}
      >
        {value}
      </span>
    </div>
  );
}

/** One detected accelerator, with everything NEURAX knows about it. */
function DetectedCard({
  gpu,
  spec,
  isSelected,
  onSelect,
}: {
  gpu: GpuInfo;
  spec: HardwareDetail | undefined;
  isSelected: boolean;
  onSelect: () => void;
}) {
  // An integrated part reports no memory of its own, and a driver that will
  // not answer reports none either. Neither is "zero bytes free", so neither
  // is rendered as a number.
  const hasVram = gpu.vramTotalBytes != null && gpu.vramFreeBytes != null;
  const freePct = hasVram ? (gpu.vramFreeBytes! / gpu.vramTotalBytes!) * 100 : 0;

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={isSelected}
      className={cn(
        'relative w-full text-left rounded-[10px] border p-4 transition-all hover:border-border',
        isSelected ? 'border-primary bg-primary/[0.07]' : 'border-border/60 bg-background/40',
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="flex items-center gap-2 text-[14px] font-semibold text-foreground">
          <MonitorCog className="w-4 h-4 text-primary shrink-0" />
          {gpu.name}
        </span>
        <span className="text-[9px] font-mono uppercase tracking-[0.12em] text-primary shrink-0">detected</span>
      </div>

      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-3">
        {hasVram ? (
          <>
            <Spec label="VRAM free" value={`${(gpu.vramFreeBytes! / GB).toFixed(1)} GB`} tone="primary" />
            <Spec label="VRAM total" value={`${(gpu.vramTotalBytes! / GB).toFixed(0)} GB`} />
            <Spec label="In use" value={`${(100 - freePct).toFixed(0)} %`} />
          </>
        ) : (
          <>
            <Spec
              label="Memory"
              value={gpu.integrated ? 'shared' : 'not reported'}
              tone={gpu.integrated ? 'primary' : undefined}
            />
            <Spec label="Type" value={gpu.integrated ? 'integrated' : 'discrete'} />
          </>
        )}
        <Spec label="Backend" value={gpu.backend.toUpperCase()} />
        {spec ? (
          <>
            <Spec label="Compute" value={formatTflops(spec)} />
            <Spec label="Bandwidth" value={`${formatNumber(spec.memory_bandwidth_gbs)} GB/s`} />
            {spec.tflops_fp32 ? <Spec label="FP32" value={`${formatNumber(spec.tflops_fp32)} TFLOP/s`} /> : null}
            {spec.manufacturer ? <Spec label="Vendor" value={spec.manufacturer} /> : null}
          </>
        ) : null}
        {gpu.driverVersion ? <Spec label="Driver" value={gpu.driverVersion} /> : null}
        {gpu.temperatureC != null ? <Spec label="Temperature" value={`${Math.round(gpu.temperatureC)} °C`} /> : null}
        {gpu.utilisationPct != null ? <Spec label="Utilisation" value={`${Math.round(gpu.utilisationPct)} %`} /> : null}
        {gpu.powerWatts != null ? <Spec label="Power" value={`${Math.round(gpu.powerWatts)} W`} /> : null}
      </div>

      {gpu.integrated ? (
        <p className="mt-3 flex items-start gap-1.5 text-[10px] text-amber-600 dark:text-amber-400 leading-relaxed">
          <AlertTriangle className="w-3 h-3 mt-px shrink-0" />
          This processor is integrated: it has no memory of its own and borrows the system RAM already counted above.
          A design sized against a dedicated VRAM budget does not describe what would happen here.
        </p>
      ) : !spec ? (
        <p className="mt-3 flex items-start gap-1.5 text-[10px] text-amber-600 dark:text-amber-400 leading-relaxed">
          <AlertTriangle className="w-3 h-3 mt-px shrink-0" />
          NEURAX has no specification for this card, so throughput and latency computed against it are approximations
          rather than the database's measured figures. Memory checks are unaffected — those come from the driver.
        </p>
      ) : null}

      {isSelected && (
        <div className="absolute top-3 right-3 w-4 h-4 rounded-full bg-primary flex items-center justify-center">
          <Check className="w-2.5 h-2.5 text-primary-foreground" />
        </div>
      )}
    </button>
  );
}

export function SimulationTargetPanel({
  isOpen,
  onClose,
  detected = null,
  isExample = false,
  isMeasuring = false,
  onMeasure,
}: SimulationTargetPanelProps) {
  const { config, updateConfig } = useHardware();
  const [catalogue, setCatalogue] = useState<HardwareDetail[] | null>(null);
  const [error, setError] = useState<string | null>(null);
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

  const detectedGpus = detected?.gpus ?? [];
  const selected = config.hardware;

  /** The database, ordered by capability, for the no-accelerator case. */
  const fallbackList = useMemo(
    () =>
      (catalogue ?? [])
        .slice()
        .sort((a, b) => (b.tflops_bf16 || b.tflops_fp16) - (a.tflops_bf16 || a.tflops_fp16)),
    [catalogue],
  );

  const choose = (name: string, memoryGb?: number) => {
    updateConfig({
      hardware: name,
      // Carry the VRAM across so memory checks reflect the chosen part rather
      // than whatever the previous target had.
      ...(memoryGb !== undefined ? { gpuMemoryGb: memoryGb } : {}),
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
          The accelerator every latency, memory and cost figure is computed for.
        </DialogDescription>

        <div className="px-4 py-3 border-b border-border flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-primary" />
            <div>
              <div className="text-[13px] font-bold tracking-[-0.02em] text-foreground">This machine</div>
              <div className="text-[10px] text-muted-foreground">
                Every latency, cost and memory figure is computed for the accelerator below
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
                Could not load hardware specifications ({error}). The detected card is still the target, and its memory
                figures still come from the driver — only the database's throughput numbers are missing.
              </div>
            </div>
          )}

          {detectedGpus.length > 0 ? (
            <div className="space-y-2">
              {detectedGpus.map((gpu) => {
                const spec = findSpec(gpu.name, catalogue);
                const targetName = spec?.name ?? gpu.name;
                return (
                  <DetectedCard
                    key={gpu.name}
                    gpu={gpu}
                    spec={spec}
                    isSelected={selected === targetName}
                    onSelect={() =>
                      choose(targetName, spec?.memory_gb ?? (gpu.vramTotalBytes != null ? gpu.vramTotalBytes / GB : undefined))
                    }
                  />
                );
              })}
            </div>
          ) : (
            /*
              A machine with no accelerator is a machine, not a missing one.

              This used to say "nothing here can be analysed against" and fall
              back to the catalogue, which was the old habit in a new place:
              it asked the user to design for hardware they do not own. A CPU
              is a real compute target — it is the only one a great many
              people have — and NEURAX can now measure exactly what it
              sustains. So the CPU *is* the target, and the catalogue moves to
              where it belongs: an explicit "design for something else".
            */
            <div className="space-y-4">
              <button
                type="button"
                onClick={() => choose(detected?.cpu.model ?? 'CPU', undefined)}
                aria-pressed={selected === (detected?.cpu.model ?? 'CPU')}
                className={cn(
                  'relative w-full text-left rounded-[10px] border p-4 transition-all hover:border-border',
                  selected === (detected?.cpu.model ?? 'CPU')
                    ? 'border-primary bg-primary/[0.07]'
                    : 'border-border/60 bg-background/40',
                )}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="flex items-center gap-2 text-[14px] font-semibold text-foreground">
                    <Cpu className="w-4 h-4 text-primary shrink-0" />
                    {detected?.cpu.model ?? 'This CPU'}
                  </span>
                  <span className="text-[9px] font-mono uppercase tracking-[0.12em] text-primary shrink-0">
                    detected
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-3">
                  <Spec label="Cores" value={`${detected?.cpu.cores ?? '—'}`} />
                  <Spec label="Threads" value={`${detected?.cpu.threads ?? '—'}`} />
                  <Spec
                    label="RAM free"
                    value={detected ? `${(detected.ramAvailableBytes / GB).toFixed(1)} GB` : '—'}
                    tone="primary"
                  />
                  <Spec
                    label="RAM total"
                    value={detected ? `${(detected.ramTotalBytes / GB).toFixed(0)} GB` : '—'}
                  />
                </div>
                <p className="mt-3 text-[10px] text-muted-foreground leading-relaxed">
                  No accelerator was found, so this is what the work would run on. Training here is slower than on a
                  GPU by a wide margin — but it is real, it is what you have, and every figure computed against it
                  describes your machine rather than someone else's.
                </p>
                {selected === (detected?.cpu.model ?? 'CPU') && (
                  <div className="absolute top-3 right-3 w-4 h-4 rounded-full bg-primary flex items-center justify-center">
                    <Check className="w-2.5 h-2.5 text-primary-foreground" />
                  </div>
                )}
              </button>
            </div>
          )}

          {/* The catalogue, as a deliberate act rather than the default. */}
          {detectedGpus.length === 0 ? (
            <details className="rounded-[8px] border border-border/60 bg-background/40 p-3">
              <summary className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80 cursor-pointer">
                Design for a machine you do not have
              </summary>
              <p className="mt-1.5 text-[10px] text-muted-foreground leading-relaxed">
                Analyse against a chip from the specification database instead — for planning a rented instance, or a
                machine you intend to buy. The figures are then predictions about that hardware, not about this one.
              </p>
              {!catalogue && !error ? (
                <div className="flex items-center gap-2 py-6 justify-center text-muted-foreground">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-[11px]">Loading hardware specifications…</span>
                </div>
              ) : (
                <div
                  className="mt-3 grid gap-1.5"
                  style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}
                >
                  {fallbackList.map((gpu) => (
                    <button
                      key={gpu.name}
                      type="button"
                      onClick={() => choose(gpu.name, gpu.memory_gb)}
                      aria-pressed={selected === gpu.name}
                      className={cn(
                        'text-left rounded-[8px] border px-2.5 py-2 transition-all',
                        selected === gpu.name
                          ? 'border-primary bg-primary/[0.07]'
                          : 'border-border/50 bg-background/30 hover:border-border',
                      )}
                    >
                      <div className="text-[11px] font-medium text-foreground">{gpu.name}</div>
                      <div className="text-[9px] font-mono text-muted-foreground">
                        {gpu.memory_gb} GB · {formatTflops(gpu)}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </details>
          ) : null}

          {/*
            What this machine sustains, measured.

            The database covers datacenter accelerators. It has no entry for
            an integrated graphics processor, a laptop CPU, or a part newer
            than the database itself — and for those, latency and throughput
            would simply be unavailable, which makes the whole Optimization
            workspace blank. Measuring is the only way to have a real number,
            and it is offered here because this is the panel that owns the
            question "what am I computing against".
          */}
          {detected && !isExample ? (
            <div className="rounded-[8px] border border-border/60 bg-background/40 p-3">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80">
                    Measured throughput
                  </span>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    {detected.compute
                      ? 'Measured on this machine, not read from a specification sheet.'
                      : 'NEURAX can measure what this machine actually sustains — about half a second, and remembered afterwards.'}
                  </p>
                </div>
                {onMeasure ? (
                  <button
                    type="button"
                    onClick={onMeasure}
                    disabled={isMeasuring}
                    className="shrink-0 px-3 py-1.5 rounded-[6px] border border-border text-[11px] font-medium hover:bg-secondary transition-colors disabled:opacity-50"
                  >
                    {isMeasuring ? (
                      <span className="flex items-center gap-1.5">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        Measuring…
                      </span>
                    ) : detected.compute ? (
                      'Measure again'
                    ) : (
                      'Measure this machine'
                    )}
                  </button>
                ) : null}
              </div>

              {detected.compute ? (
                <>
                  <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-3">
                    <Spec
                      label="Sustained"
                      value={`${detected.compute.gflopsF32.toFixed(1)} GFLOP/s`}
                      tone="primary"
                    />
                    <Spec label="Bandwidth" value={`${detected.compute.memoryBandwidthGbs.toFixed(1)} GB/s`} />
                    <Spec label="Threads" value={String(detected.compute.threads)} />
                    <Spec label="Took" value={`${detected.compute.durationMs} ms`} />
                  </div>
                  <p className="mt-2 text-[10px] font-mono text-muted-foreground/70 leading-relaxed">
                    {detected.compute.method}
                  </p>
                  <p className="mt-1.5 text-[10px] text-muted-foreground leading-relaxed">
                    This is what a good ordinary kernel sustains here, not the chip's theoretical peak and not a tuned
                    BLAS. A real run using a vendor library will exceed it, so treat latency derived from this figure as
                    a conservative bound rather than a prediction to the percent.
                  </p>
                </>
              ) : null}
            </div>
          ) : null}

          {isExample ? (
            <div className="rounded-[8px] border border-amber-500/40 bg-amber-500/5 p-3">
              <div className="flex items-center gap-1.5 text-[11px] font-semibold text-amber-700 dark:text-amber-400">
                <AlertTriangle className="w-3.5 h-3.5" />
                Example machine, not this one
              </div>
              <p className="mt-1 text-[10px] text-muted-foreground leading-relaxed">
                The local NEURAX service is not answering, so nothing could be detected. What is shown is an example
                profile — design against it if you like, but the memory verdicts do not describe your machine.
              </p>
            </div>
          ) : null}

          <div className="rounded-[8px] border border-border/60 bg-background/40 p-3">
            <label
              htmlFor="numeric-precision"
              className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80"
            >
              Numeric precision
            </label>
            <p className="text-[10px] text-muted-foreground mt-0.5 mb-2">
              Width every weight, activation and optimizer state is stored at. Multiplies the whole memory analysis —
              int4 packs two values per byte, an eighth of fp32.
            </p>
            <div className="flex flex-wrap gap-1.5" id="numeric-precision">
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

          <div className="rounded-[8px] border border-border/60 bg-background/40 p-3">
            <label htmlFor="gpu-count" className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80">
              Device count
            </label>
            <p className="text-[10px] text-muted-foreground mt-0.5 mb-2">
              Devices the workload is spread over. Drives the parallelism analysis and the total GPU-hour cost.
              {detectedGpus.length > 1 ? ` This machine reports ${detectedGpus.length}.` : ''}
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
