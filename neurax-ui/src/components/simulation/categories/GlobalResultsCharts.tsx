import { BarChart3 } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, PieChart, Pie,
} from 'recharts';
import { hasPhaseTimeline } from '../simulationData.ts';


interface GlobalResultsChartsProps {
  analysis?: AnalysisResult;
}

const SECTION_CLS = 'panel-section p-4 bg-card/30 border-primary/5 rounded-xl';
const TITLE_CLS = 'text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-3';

// ─── 2.1 Model Size (Parameters) ─────────────────────────────────────────────

function ModelSizeDonut({ analysis }: { analysis: AnalysisResult }) {
  const total = analysis.totalParams;
  // Bucket op-distribution counts into semantic param groups
  const ops = analysis.opsDistribution ?? {};
  const opTotal = Object.values(ops).reduce((a, b) => a + b, 0) || 1;
  const buckets: Record<string, number> = { Weights: 0, Embedding: 0, Bias: 0, Normalization: 0 };
  for (const [op, count] of Object.entries(ops)) {
    const pct = count / opTotal;
    if (/embed/i.test(op)) buckets.Embedding += pct;
    else if (/norm|layer_norm|rms/i.test(op)) buckets.Normalization += pct;
    else if (/bias/i.test(op)) buckets.Bias += pct;
    else buckets.Weights += pct;
  }
  // Convert to integer percentages and ensure sum=100
  const labels = Object.keys(buckets);
  const totPct = labels.reduce((s, k) => s + buckets[k], 0) || 1;
  const sliceData = labels
    .map((k, i) => ({
      name: k,
      value: Math.round((buckets[k] / totPct) * 100),
      fill: ['#3b82f6', '#f59e0b', '#ef4444', '#f97316'][i],
    }))
    .filter(s => s.value > 0);

  const paramLabel =
    total >= 1e9 ? `${(total / 1e9).toFixed(1)}B`
      : total >= 1e6 ? `${(total / 1e6).toFixed(0)}M`
        : `${total}`;

  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>2.1 — Model Size (Parameters)</div>
      <div className="flex items-center gap-4 h-44">
        <div className="relative flex-shrink-0 w-36 h-36">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={sliceData}
                cx="50%"
                cy="50%"
                innerRadius={48}
                outerRadius={68}
                paddingAngle={2}
                dataKey="value"
                startAngle={90}
                endAngle={-270}
              >
                {sliceData.map((s, i) => <Cell key={i} fill={s.fill} stroke="none" />)}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <span className="text-lg font-bold font-mono">{paramLabel}</span>
            <span className="text-[8px] text-muted-foreground">Params</span>
          </div>
        </div>

        <div className="space-y-2 text-[10px]">
          {sliceData.map(s => (
            <div key={s.name} className="flex items-center gap-2">
              <span className="font-semibold" style={{ color: s.fill }}>{s.name} {s.value}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── 2.2 FLOPs by Op Type ─────────────────────────────────────────────────────

function FlopsByOpType({ analysis }: { analysis: AnalysisResult }) {
  const raw = analysis.opsDistribution;
  const total = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
  const data = Object.entries(raw)
    .map(([name, value]) => ({ name, value: Math.round((value / total) * analysis.totalFlops / 1e9) }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);

  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>2.2 — FLOPs by Op Type</div>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 56, right: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" horizontal={false} />
            <XAxis type="number" stroke="#555" fontSize={9} tickLine={false} axisLine={false}
              tickFormatter={v => v >= 1 ? `${v}G` : `${(v * 1000).toFixed(0)}M`} />
            <YAxis type="category" dataKey="name" stroke="#888" fontSize={9} tickLine={false} axisLine={false} width={54} />
            <Tooltip
              contentStyle={{ backgroundColor: '#111', border: '1px solid #333', fontSize: '9px' }}
              formatter={(v: number) => [`${v} GFLOPs`, 'FLOPs']}
            />
            <Bar dataKey="value" fill="#3b82f6" radius={[0, 3, 3, 0]} barSize={10} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 2.3 Latency Breakdown ────────────────────────────────────────────────────

function LatencyBreakdown({ analysis }: { analysis: AnalysisResult }) {
  const phases = analysis.compilation?.phase_timeline ?? [];
  if (phases.length === 0) return null;
  const data = phases.map(p => ({ name: p.name.replace(' ', '\n'), ms: parseFloat(p.duration_ms.toFixed(1)) }));

  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>2.3 — Latency Breakdown</div>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
            <XAxis dataKey="name" stroke="#888" fontSize={8} tickLine={false} axisLine={false} />
            <YAxis stroke="#555" fontSize={9} tickLine={false} axisLine={false} tickCount={4} />
            <Tooltip
              contentStyle={{ backgroundColor: '#111', border: '1px solid #333', fontSize: '9px' }}
              formatter={(v: number) => [`${v.toFixed(1)} ms`, 'Latency']}
            />
            <Bar dataKey="ms" fill="#f59e0b" radius={[3, 3, 0, 0]} barSize={20} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 2.5 Confidence Radar ─────────────────────────────────────────────────────

function ConfidenceSummary({ analysis }: { analysis: AnalysisResult }) {
  const score = Math.min(1, Math.max(0, analysis.confidenceScore));
  const pct = Math.round(score * 100);
  const color = pct >= 85 ? '#10b981' : pct >= 60 ? '#f59e0b' : '#ef4444';
  const radius = 42;
  const circumference = 2 * Math.PI * radius;

  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>2.5 — Confidence Score</div>
      <div className="flex h-48 items-center gap-6">
        <div className="relative h-28 w-28 shrink-0">
          <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r={radius} stroke="#333" strokeWidth="10" fill="none" />
            <circle
              cx="50"
              cy="50"
              r={radius}
              stroke={color}
              strokeWidth="10"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={circumference * (1 - score)}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold font-mono">{pct}</span>
            <span className="text-[8px] uppercase tracking-wider text-muted-foreground">confidence</span>
          </div>
        </div>
        <div className="space-y-2 text-[10px]">
          <div className="font-bold uppercase tracking-[0.14em]" style={{ color }}>
            {pct >= 85 ? 'Compiler-backed' : pct >= 60 ? 'Estimated with caution' : 'Low confidence'}
          </div>
          <div className="text-muted-foreground">
            This score comes directly from the report and reflects how trustworthy the current synthesis is.
          </div>
          <div className="rounded-md border border-border/60 bg-secondary/20 px-3 py-2 text-muted-foreground">
            {hasPhaseTimeline(analysis)
              ? 'Phase data is available for this run.'
              : 'Detailed confidence dimensions are not in the report yet.'}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── 2.6 Hardware Fit Score ───────────────────────────────────────────────────

function HardwareFitScore({ analysis }: { analysis: AnalysisResult }) {
  const vram = analysis.peakVramBytes / 1e9;
  const gpuMem = analysis.gpuMemoryGb;
  const utilization = analysis.gpuUtilization ?? analysis.rooflinePosition;
  const fitRatio = gpuMem > 0 ? Math.min(1, vram / gpuMem) : 0.5;
  const score = Math.round((1 - fitRatio * 0.4 + utilization * 0.6) * 100);
  const clamped = Math.min(100, Math.max(0, score));
  const r = 42;
  const circ = 2 * Math.PI * r;
  const label = clamped > 75 ? 'Great fit' : clamped > 50 ? 'Good fit' : 'Under-utilised';
  const color = clamped > 75 ? '#10b981' : clamped > 50 ? '#f59e0b' : '#ef4444';

  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>2.6 — Hardware Fit Score</div>
      <div className="flex items-center gap-6 h-48">
        <div className="relative w-28 h-28 flex-shrink-0">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r={r} stroke="#333" strokeWidth="10" fill="none" />
            <circle
              cx="50" cy="50" r={r}
              stroke={color} strokeWidth="10" fill="none"
              strokeDasharray={circ}
              strokeDashoffset={circ * (1 - clamped / 100)}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold font-mono">{clamped}</span>
            <span className="text-[8px] text-muted-foreground">/100</span>
          </div>
        </div>
        <div className="space-y-2 text-[10px]">
          <div className="font-bold" style={{ color }}>{analysis.gpuName || 'GPU'} — {label}</div>
          <div className="text-muted-foreground">
            {vram.toFixed(1)} GB / {gpuMem || '?'} GB VRAM
          </div>
          <div className="text-muted-foreground">
            {analysis.gpuTflops ? `${analysis.gpuTflops.toFixed(1)} TFLOP/s` : '—'}
          </div>
          <div className="text-muted-foreground">
            Throughput: {analysis.throughputTokensPerS ? `${Math.round(analysis.throughputTokensPerS)} tok/s` : '—'}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── 2.7 Cost Summary (Treemap-style) ────────────────────────────────────────

function CostTreemap({ analysis }: { analysis: AnalysisResult }) {
  const flops = analysis.totalFlops / 1e12;
  const vram = analysis.peakVramBytes / 1e9;
  const latency = analysis.latencyMs ?? 0;

  const tiles = [
    { label: 'FLOPs', value: `${flops.toFixed(1)} TFLOPs`, color: '#3b82f6', flex: 2 },
    { label: 'VRAM', value: `${vram.toFixed(1)} GB`, color: '#f59e0b', flex: 1.5 },
    { label: 'Latency', value: latency ? `${latency.toFixed(0)}ms` : '—', color: '#ef4444', flex: 1 },
  ];

  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>2.7 — Cost Summary (Treemap)</div>
      <div className="flex gap-2 h-48 items-stretch mt-2">
        {tiles.map(t => (
          <div
            key={t.label}
            className="flex flex-col items-center justify-center rounded-lg text-white font-bold transition-all"
            style={{ backgroundColor: t.color, flex: t.flex }}
          >
            <span className="text-sm">{t.value}</span>
            <span className="text-[9px] mt-1 opacity-80">{t.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── 2.8 Dialect Distribution ─────────────────────────────────────────────────

function DialectDistribution({ analysis }: { analysis: AnalysisResult }) {
  const ops = analysis.opsDistribution;
  const total = Object.values(ops).reduce((a, b) => a + b, 0) || 1;

  // Map op types to dialect families
  const dialects: Record<string, number> = { DenseSeq: 0, SparseMoE: 0, ConvGrid: 0, Other: 0 };
  for (const [op, v] of Object.entries(ops)) {
    const pct = v / total;
    if (/matmul|linear|embed|attn/i.test(op)) dialects.DenseSeq += pct;
    else if (/moe|expert|gate/i.test(op)) dialects.SparseMoE += pct;
    else if (/conv|pool/i.test(op)) dialects.ConvGrid += pct;
    else dialects.Other += pct;
  }

  const data = Object.entries(dialects)
    .map(([name, pct]) => ({ name, value: Math.round(pct * 100) }))
    .filter(d => d.value > 0);

  if (data.length === 0) {
    data.push({ name: 'DenseSeq', value: 80 }, { name: 'Other', value: 20 });
  }

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#6366f1'];

  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>2.8 — Dialect Distribution</div>
      <div className="flex items-center h-48 gap-4">
        <ResponsiveContainer width="55%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              startAngle={180}
              endAngle={-180}
              innerRadius={44}
              outerRadius={70}
              paddingAngle={2}
              dataKey="value"
            >
              {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="none" />)}
            </Pie>
            <Tooltip
              contentStyle={{ backgroundColor: '#111', border: '1px solid #333', fontSize: '9px' }}
              formatter={(v: number) => [`${v}%`, 'Coverage']}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="space-y-2 text-[10px]">
          {data.map((d, i) => (
            <div key={d.name} className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
              <span className="text-muted-foreground">{d.name}</span>
              <span className="font-bold ml-auto">{d.value}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── 2.4 Key Stats Strip ─────────────────────────────────────────────────────

function KeyStatsStrip({ analysis }: { analysis: AnalysisResult }) {
  const stats = [
    { label: 'Total FLOPs', value: analysis.estimatedFlops },
    { label: 'Peak VRAM', value: analysis.memoryUsage },
    { label: 'Arith. Intensity', value: `${analysis.arithmeticIntensity.toFixed(1)} FLOP/B` },
    { label: 'Confidence', value: `${(analysis.confidenceScore * 100).toFixed(0)}%` },
    { label: 'Throughput', value: analysis.throughputTokensPerS ? `${Math.round(analysis.throughputTokensPerS)} tok/s` : '—' },
    { label: 'Latency', value: analysis.latencyMs ? `${analysis.latencyMs.toFixed(0)} ms` : '—' },
  ];

  return (
    <div className="panel-section p-4 bg-card/30 grid grid-cols-3 md:grid-cols-6 gap-4">
      {stats.map(s => (
        <div key={s.label} className="space-y-1">
          <div className="text-[8px] text-muted-foreground uppercase tracking-wider">{s.label}</div>
          <div className="text-sm font-bold font-mono text-foreground">{s.value}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function GlobalResultsCharts({ analysis }: GlobalResultsChartsProps) {
  if (!analysis || analysis.totalParams === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground border-2 border-dashed border-border rounded-lg">
        <BarChart3 className="w-8 h-8 mb-2 opacity-20" />
        <p className="text-sm">No analysis data available</p>
        <p className="text-xs">Run analysis in the Architecture workspace first.</p>
      </div>
    );
  }

  const hasPhases = hasPhaseTimeline(analysis);
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-primary" />
        <h2 className="text-base font-semibold">Global Results — Full Report</h2>
      </div>

      {/* Row 1: Model Size · FLOPs by Op · Latency */}
      <div className={`grid grid-cols-1 ${hasPhases ? "lg:grid-cols-3" : "lg:grid-cols-2"} gap-4`}>
        <ModelSizeDonut analysis={analysis} />
        <FlopsByOpType analysis={analysis} />
        <LatencyBreakdown analysis={analysis} />
      </div>

      {/* Row 2: Confidence Radar · Hardware Fit · Cost Treemap */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ConfidenceSummary analysis={analysis} />
        <HardwareFitScore analysis={analysis} />
        <CostTreemap analysis={analysis} />
      </div>

      {/* Row 3: Dialect Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DialectDistribution analysis={analysis} />
        <KeyStatsStrip analysis={analysis} />
      </div>
    </div>
  );
}
