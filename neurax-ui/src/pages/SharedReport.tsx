import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { NeuraxLogo } from '@/components/brand/NeuraxLogo.tsx';
import { Button } from '@/components/ui/button.tsx';
import { getShare, shareDownloadUrl, type Share } from '@/services/neuraxApi.ts';
import { formatMetric, type Unit } from '@/utils/designComparison.ts';
import { AnalysisResult } from '@/types/architecture.ts';
import { Download, ExternalLink, Eye, AlertTriangle, FileQuestion } from 'lucide-react';

interface FieldSpec {
  key: keyof AnalysisResult;
  label: string;
  unit: Unit;
}

interface Section {
  title: string;
  fields: FieldSpec[];
}

// The full report, not the short curated list `designComparison.ts` shows in
// a side-by-side compare — a public "here's what this costs" page is the one
// place showing everything the compiler produced is the point, not a defect.
const SECTIONS: Section[] = [
  {
    title: 'Architecture',
    fields: [
      { key: 'numLayers', label: 'Layers', unit: 'count' },
      { key: 'totalParams', label: 'Total parameters', unit: 'count' },
      { key: 'activeParams', label: 'Active parameters', unit: 'count' },
      { key: 'hiddenSize', label: 'Hidden size', unit: 'count' },
      { key: 'vocabSize', label: 'Vocabulary size', unit: 'count' },
      { key: 'sequenceLength', label: 'Sequence length', unit: 'count' },
      { key: 'numAttentionHeads', label: 'Attention heads', unit: 'count' },
      { key: 'numKeyValueHeads', label: 'KV heads', unit: 'count' },
      { key: 'intermediateSize', label: 'Intermediate size', unit: 'count' },
      { key: 'graphDepth', label: 'Graph depth', unit: 'count' },
      { key: 'totalOperations', label: 'Total operations', unit: 'count' },
      { key: 'totalTensorCount', label: 'Tensors', unit: 'count' },
    ],
  },
  {
    title: 'Compute',
    fields: [
      { key: 'totalFlops', label: 'Total FLOPs', unit: 'flops' },
      { key: 'forwardFlops', label: 'Forward FLOPs', unit: 'flops' },
      { key: 'backwardFlops', label: 'Backward FLOPs', unit: 'flops' },
      { key: 'flopsPerToken', label: 'FLOPs per token', unit: 'flops' },
      { key: 'arithmeticIntensity', label: 'Arithmetic intensity', unit: 'ratio' },
      { key: 'rooflinePosition', label: 'Roofline position', unit: 'ratio' },
    ],
  },
  {
    title: 'Memory',
    fields: [
      { key: 'peakVramBytes', label: 'Peak VRAM', unit: 'bytes' },
      { key: 'parameterMemoryBytes', label: 'Weights', unit: 'bytes' },
      { key: 'activationMemoryBytes', label: 'Activations', unit: 'bytes' },
      { key: 'gradientMemoryBytes', label: 'Gradients', unit: 'bytes' },
      { key: 'optimizerStateBytes', label: 'Optimizer state', unit: 'bytes' },
      { key: 'maxBatchSizeFit', label: 'Max batch that fits', unit: 'count' },
    ],
  },
  {
    title: 'Hardware & parallelism',
    fields: [
      { key: 'gpuCount', label: 'GPU count', unit: 'count' },
      { key: 'gpuMemoryGb', label: 'GPU memory (GB)', unit: 'count' },
      { key: 'gpuTflops', label: 'GPU TFLOPS', unit: 'count' },
      { key: 'gpuBandwidthGbs', label: 'GPU bandwidth (GB/s)', unit: 'count' },
      { key: 'dataParallel', label: 'Data parallel degree', unit: 'count' },
      { key: 'tensorParallel', label: 'Tensor parallel degree', unit: 'count' },
      { key: 'pipelineParallel', label: 'Pipeline parallel degree', unit: 'count' },
      { key: 'optimalGpuCount', label: 'Optimal GPU count', unit: 'count' },
    ],
  },
  {
    title: 'Performance',
    fields: [
      { key: 'latencyMs', label: 'Latency', unit: 'ms' },
      { key: 'throughputTokensPerS', label: 'Throughput (tokens/s)', unit: 'count' },
      { key: 'gpuUtilization', label: 'GPU utilisation', unit: 'percent' },
      { key: 'effectiveTflops', label: 'Effective TFLOPS', unit: 'count' },
    ],
  },
  {
    title: 'Cost & impact',
    fields: [
      { key: 'trainingCostUsd', label: 'Training cost', unit: 'usd' },
      { key: 'trainingTimeHours', label: 'Training time', unit: 'hours' },
      { key: 'costPerMillionTokensUsd', label: 'Cost per million tokens', unit: 'usd' },
      { key: 'energyKwh', label: 'Energy', unit: 'kwh' },
      { key: 'co2Kg', label: 'CO₂', unit: 'kg' },
    ],
  },
];

// Fields where 0 never describes a real model — every real architecture has
// a non-zero hidden size, and effective throughput of exactly 0 TFLOPS means
// "not computed for this run", not "computed as zero". Showing 0 for these
// reads as broken data rather than an honest absence, so they're treated the
// same as a missing value. Fields where 0 is a real, meaningful answer (a
// model with zero CO₂, zero pipeline parallelism) are left alone.
const NEVER_LEGITIMATELY_ZERO = new Set(['hiddenSize', 'effectiveTflops']);

function readNumeric(report: Record<string, unknown>, key: string): number | null {
  const v = report[key];
  if (typeof v !== 'number' || !Number.isFinite(v)) return null;
  if (v === 0 && NEVER_LEGITIMATELY_ZERO.has(key)) return null;
  return v;
}

function readString(report: Record<string, unknown>, key: string): string | null {
  const v = report[key];
  return typeof v === 'string' && v.trim() ? v : null;
}

type LoadState =
  | { status: 'loading' }
  | { status: 'not_found' }
  | { status: 'error'; message: string }
  | { status: 'ready'; share: Share };

export default function SharedReport() {
  const { id } = useParams<{ id: string }>();
  const [state, setState] = useState<LoadState>({ status: 'loading' });

  useEffect(() => {
    if (!id) {
      setState({ status: 'not_found' });
      return;
    }
    let cancelled = false;
    getShare(id)
      .then((share) => {
        if (!cancelled) setState({ status: 'ready', share });
      })
      .catch((err) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setState(
          message.includes('404') || message.includes('not_found')
            ? { status: 'not_found' }
            : { status: 'error', message },
        );
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <NeuraxLogo variant="mark" size={30} />
            <span className="font-semibold">NEURAX</span>
          </Link>
          <a
            href="https://raw.githubusercontent.com/rustnew/NEURAX/main/install.sh"
            className="text-sm text-muted-foreground hover:text-foreground inline-flex items-center gap-1.5"
          >
            Get NEURAX <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10">
        {state.status === 'loading' && (
          <div className="py-24 text-center text-muted-foreground" aria-busy="true">
            Loading report…
          </div>
        )}

        {state.status === 'not_found' && (
          <div className="py-24 text-center">
            <FileQuestion className="w-10 h-10 mx-auto text-muted-foreground mb-4" />
            <h1 className="text-xl font-semibold">This link doesn't exist</h1>
            <p className="text-muted-foreground mt-2">
              It may have been deleted, or the link was mistyped.
            </p>
          </div>
        )}

        {state.status === 'error' && (
          <div className="py-24 text-center">
            <AlertTriangle className="w-10 h-10 mx-auto text-destructive mb-4" />
            <h1 className="text-xl font-semibold">Couldn't load this report</h1>
            <p className="text-muted-foreground mt-2">{state.message}</p>
          </div>
        )}

        {state.status === 'ready' && (
          <ReportView share={state.share} />
        )}
      </main>
    </div>
  );
}

function ReportView({ share }: { share: Share }) {
  const report = share.report as Record<string, unknown>;
  const gpuName = readString(report, 'gpuName');
  const bottleneck = readString(report, 'bottleneck');
  const modelType = readString(report, 'modelType');
  const warnings = Array.isArray(report.reportWarnings)
    ? (report.reportWarnings as string[])
    : [];

  return (
    <div className="space-y-8">
      <div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
          <Eye className="w-3.5 h-3.5" />
          {share.view_count} {share.view_count === 1 ? 'view' : 'views'}
          {share.family && <span>· {share.family}</span>}
          {modelType && modelType !== share.family && <span>· {modelType}</span>}
        </div>
        <h1 className="text-2xl font-bold">{share.display_name}</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Analysed with NEURAX — deterministic, no GPU touched to produce these numbers.
        </p>
      </div>

      {gpuName && (
        <div className="rounded-lg border border-border bg-secondary/30 px-4 py-3 text-sm">
          Sized for <span className="font-medium">{gpuName}</span>
          {readNumeric(report, 'gpuCount') ? ` × ${readNumeric(report, 'gpuCount')}` : ''}
          {bottleneck && (
            <span className="text-muted-foreground">
              {' '}
              — {bottleneck.endsWith('-bound') ? bottleneck : `${bottleneck}-bound`}
            </span>
          )}
        </div>
      )}

      {SECTIONS.map((section) => {
        const rows = section.fields
          .map((f) => ({ ...f, value: readNumeric(report, f.key as string) }))
          .filter((f) => f.value !== null);
        if (rows.length === 0) return null;
        return (
          <section key={section.title}>
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">
              {section.title}
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-px rounded-lg border border-border overflow-hidden bg-border">
              {rows.map((row) => (
                <div key={row.key as string} className="bg-background px-4 py-3">
                  <div className="text-xs text-muted-foreground">{row.label}</div>
                  <div className="text-lg font-semibold tabular-nums">
                    {formatMetric(row.value, row.unit)}
                  </div>
                </div>
              ))}
            </div>
          </section>
        );
      })}

      {warnings.length > 0 && (
        <section>
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Diagnostics
          </h2>
          <ul className="space-y-1.5">
            {warnings.map((w, i) => (
              <li key={i} className="text-sm text-muted-foreground flex gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-500" />
                {w}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="flex flex-wrap items-center gap-3 pt-4 border-t border-border">
        <a href={shareDownloadUrl(share.id)}>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-1.5" />
            Download report
          </Button>
        </a>
        {share.mode === 'full' && share.design && (
          // Honest about what this link does: it installs NEURAX, it does not
          // open this specific design — there is no handoff from a public
          // share link into a local install. The button used to read "Open
          // this design in NEURAX", which promised something this doesn't do.
          <a href="https://raw.githubusercontent.com/rustnew/NEURAX/main/install.sh">
            <Button size="sm" variant="outline">
              Get NEURAX to open designs like this
            </Button>
          </a>
        )}
      </div>
    </div>
  );
}
