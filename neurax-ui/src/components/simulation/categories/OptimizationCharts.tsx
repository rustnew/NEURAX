import { Target, Info } from 'lucide-react';
import { AnalysisResult, CanvasNode, Connection, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  SIMULATION_COLORS,
  buildDerivedLayerMetrics,
  deriveFusionCandidates,
  deriveOptimizationOpportunities,
  formatPercent,
} from '../simulationData.ts';

interface OptimizationChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  nodes?: CanvasNode[];
  connections?: Connection[];
}

const CARD_CLS = 'panel-section p-4 bg-card/30 border-primary/5 rounded-xl';
const TITLE_CLS = 'text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-4';
const TOOLTIP_STYLE = { backgroundColor: '#111', border: '1px solid #333', fontSize: '10px' };

function buildRooflineRows(analysis: AnalysisResult) {
  const intensity = Math.max(analysis.arithmeticIntensity, 0.1);
  const crossover = analysis.gpuBandwidthGbs > 0 ? (analysis.gpuTflops / analysis.gpuBandwidthGbs) * 1000 : intensity;
  const maxIntensity = Math.max(intensity * 3, crossover * 1.6, 8);
  const intensities = [0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128].filter((value) => value <= maxIntensity);
  if (!intensities.includes(maxIntensity)) intensities.push(maxIntensity);

  return intensities.map((x) => ({
    intensity: x,
    memoryRoof: (x * analysis.gpuBandwidthGbs) / 1000,
    computeRoof: analysis.gpuTflops,
  }));
}

export function OptimizationCharts({
  analysis,
  perLayer = [],
  nodes = [],
  connections = [],
}: OptimizationChartsProps) {
  if (!analysis || analysis.totalFlops === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground border-2 border-dashed border-border rounded-lg">
        <Target className="w-8 h-8 mb-2 opacity-20" />
        <p className="text-sm">No optimization data available</p>
        <p className="text-xs">Run analysis to see roofline and bottlenecks.</p>
      </div>
    );
  }

  const achievedTflops = analysis.latencyMs
    ? (analysis.totalFlops / 1e12) / (analysis.latencyMs / 1000)
    : Math.max(0.1, (analysis.gpuTflops || 1) * (analysis.gpuUtilization ?? 0.45));
  const rooflineRows = buildRooflineRows(analysis);
  const layerRows = buildDerivedLayerMetrics(analysis, perLayer)
    .sort((a, b) => b.flops - a.flops)
    .map((row, index, list) => {
      const cumulative = list
        .slice(0, index + 1)
        .reduce((sum, current) => sum + current.flops, 0);
      const total = list.reduce((sum, current) => sum + current.flops, 0) || 1;
      return {
        name: row.name,
        flops: row.flops,
        cumulativePct: (cumulative / total) * 100,
      };
    });
  const rooflineMix = (() => {
    const compute = Math.max(0, Math.min(analysis.rooflinePosition, 1));
    const mixed = Math.min(18, (1 - analysis.confidenceScore) * 30 + Math.min(compute, 1 - compute) * 20);
    const memory = Math.max(0, 100 - mixed - compute * 100);
    return [
      { name: 'Compute-bound', value: compute * 100, fill: SIMULATION_COLORS.green },
      { name: 'Memory-bound', value: memory, fill: SIMULATION_COLORS.red },
      { name: 'Mixed', value: mixed, fill: SIMULATION_COLORS.amber },
    ].filter((entry) => entry.value > 0);
  })();
  const opportunities = deriveOptimizationOpportunities(analysis);
  const fusionCandidates = deriveFusionCandidates(nodes, connections, perLayer);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Target className="w-5 h-5 text-primary" />
          Optimization — Roofline & Bottlenecks
        </h2>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground bg-secondary/50 px-2 py-1 rounded-md">
          <Info className="w-3 h-3" />
          Compiler-backed hardware ceilings, bottlenecks, and fusion opportunities
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className={`${CARD_CLS} xl:col-span-2`}>
          <div className={TITLE_CLS}>6.1 — Roofline Model</div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rooflineRows} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
                <XAxis dataKey="intensity" stroke="#888" fontSize={9} tickFormatter={(value) => `${value.toFixed(1)}`} label={{ value: 'Arithmetic Intensity (FLOP/B)', position: 'insideBottom', offset: -4, fill: '#666', fontSize: 10 }} />
                <YAxis stroke="#888" fontSize={9} tickFormatter={(value) => `${value.toFixed(1)}`} label={{ value: 'TFLOP/s', angle: -90, position: 'insideLeft', fill: '#666', fontSize: 10 }} />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  formatter={(value: number, name: string) => [`${value.toFixed(2)} TFLOP/s`, name === 'memoryRoof' ? 'Memory roof' : 'Compute roof']}
                />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Line type="monotone" dataKey="memoryRoof" name="Memory roof" stroke={SIMULATION_COLORS.red} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="computeRoof" name="Compute roof" stroke={SIMULATION_COLORS.green} strokeWidth={2} dot={false} />
                <ReferenceDot
                  x={Math.max(analysis.arithmeticIntensity, 0.1)}
                  y={achievedTflops}
                  r={6}
                  fill={SIMULATION_COLORS.blue}
                  stroke="white"
                  label={{ value: 'Current model', position: 'top', fill: SIMULATION_COLORS.blue, fontSize: 10 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className={CARD_CLS}>
          <div className={TITLE_CLS}>6.2 — Bottleneck Pareto (80/20)</div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={layerRows} margin={{ top: 8, right: 16, bottom: 36, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
                <XAxis dataKey="name" stroke="#888" fontSize={9} angle={-30} textAnchor="end" interval={0} height={60} />
                <YAxis yAxisId="left" stroke="#888" fontSize={9} tickFormatter={(value) => `${value.toFixed(0)}G`} />
                <YAxis yAxisId="right" orientation="right" stroke="#888" fontSize={9} domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  formatter={(value: number, name: string) => [
                    name === 'cumulativePct' ? `${value.toFixed(0)}%` : `${value.toFixed(2)} GFLOPs`,
                    name === 'cumulativePct' ? 'Cumulative' : 'Layer FLOPs',
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Bar yAxisId="left" dataKey="flops" name="Layer FLOPs" fill={SIMULATION_COLORS.blue} radius={[3, 3, 0, 0]} />
                <Line yAxisId="right" type="monotone" dataKey="cumulativePct" name="Cumulative %" stroke={SIMULATION_COLORS.amber} strokeWidth={2} dot={{ r: 3, fill: SIMULATION_COLORS.amber }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className={CARD_CLS}>
          <div className={TITLE_CLS}>6.3 — Compute vs Memory Bound</div>
          <div className="h-72 flex items-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={rooflineMix}
                  dataKey="value"
                  innerRadius={62}
                  outerRadius={92}
                  paddingAngle={3}
                >
                  {rooflineMix.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value: number) => [`${value.toFixed(1)}%`, 'Share']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-2 text-[10px]">
            {rooflineMix.map((entry) => (
              <div key={entry.name} className="text-center">
                <div className="font-semibold" style={{ color: entry.fill }}>{entry.name}</div>
                <div>{entry.value.toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>

        <div className={CARD_CLS}>
          <div className={TITLE_CLS}>6.4 — Optimization Opportunities</div>
          <div className="space-y-3">
            {opportunities.map((entry) => (
              <div key={entry.title}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span>{entry.title}</span>
                  <span className="font-mono">{entry.score.toFixed(0)}%</span>
                </div>
                <div className="h-2 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${entry.score}%`,
                      backgroundColor:
                        entry.priority === 'high'
                          ? SIMULATION_COLORS.green
                          : entry.priority === 'medium'
                            ? SIMULATION_COLORS.amber
                            : SIMULATION_COLORS.blue,
                    }}
                  />
                </div>
                <div className="text-[10px] text-muted-foreground mt-1">{entry.description}</div>
              </div>
            ))}
          </div>
        </div>

        <div className={CARD_CLS}>
          <div className={TITLE_CLS}>6.5 — Layer Fusion Candidates</div>
          <div className="space-y-3">
            {fusionCandidates.map((candidate) => (
              <div key={candidate.label} className="rounded-lg bg-secondary/20 px-3 py-2">
                <div className="flex items-center justify-between gap-3 text-xs">
                  <span className="font-medium">{candidate.label}</span>
                  <span className="font-mono" style={{ color: SIMULATION_COLORS.green }}>
                    {candidate.gainPct.toFixed(0)}%
                  </span>
                </div>
                <div className="mt-2 h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${candidate.gainPct}%`, backgroundColor: SIMULATION_COLORS.green }}
                  />
                </div>
                <div className="mt-1 text-[10px] text-muted-foreground">
                  {candidate.difficulty} integration • estimated uplift from reduced kernel launch / memory traffic
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 rounded-md border border-border/50 bg-secondary/10 px-3 py-2 text-[10px] text-muted-foreground">
            Roofline position: {formatPercent(analysis.rooflinePosition, 0)} toward compute-bound. Peak utilization is approximately{' '}
            {formatPercent((achievedTflops || 0) / Math.max(analysis.gpuTflops || 1, 1), 0)} of hardware peak.
          </div>
        </div>
      </div>
    </div>
  );
}
