/**
 * Diagnostics — what the compiler could not resolve, and where.
 *
 * Six charts stood here, and four of them were counts rendered as geometry: a
 * severity pie, a confidence bar, an unsupported-ops bar, a resolution
 * donut. Four numbers do not need four charts, and a pie of three integers is
 * strictly harder to read than the three integers. Those are strip cells now.
 *
 * What genuinely wanted drawing is *where* along the graph the trouble sits.
 * A design with nine warnings spread evenly is a different problem from one
 * with nine warnings on a single block, and no aggregate distinguishes them —
 * so the two charts kept are both profiles across depth: how many findings
 * each layer attracted, and how much of each layer's shape the compiler
 * actually resolved.
 *
 * Unresolved dimensions matter more than their count suggests. Every downstream
 * number — memory, FLOPs, latency — is computed from shapes, so a layer the
 * compiler could not fully resolve is a layer whose costs are estimates
 * standing on an estimate. That is worth seeing per layer rather than as one
 * ratio for the whole model.
 */
import { Bug, CircleCheck, Lightbulb, OctagonAlert, ShieldQuestion, TriangleAlert } from 'lucide-react';
import { Area, AreaChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts';

import { AnalysisResult, CanvasNode, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';
import {
  CHART_GRID, CHART_MARGINS, ChartErrorBoundary, ChartGrid, ChartSlot, StatCard, StatStrip,
  ViewNote, XA, YA, chartTooltipStyle,
} from '../shared';
import {
  SIMULATION_COLORS, buildDerivedLayerMetrics, deriveDiagnosticsByLayer, deriveIssueSummary,
  deriveUnsupportedOps, hasAnalysisReportData, normalizeSeverity,
} from '../simulationData.ts';
import { cn } from '@/lib/utils.ts';

interface DebuggingChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  warnings?: Warning[];
  nodes?: CanvasNode[];
}

const SEVERITY_TONE: Record<string, string> = {
  critical: 'text-red-600 dark:text-red-400',
  warning: 'text-amber-600 dark:text-amber-400',
  info: 'text-blue-600 dark:text-blue-400',
  hint: 'text-muted-foreground',
};

export function DebuggingCharts({ analysis, perLayer = [], warnings = [], nodes = [] }: DebuggingChartsProps) {
  if (!hasAnalysisReportData(analysis)) {
    return (
      <ChartErrorBoundary>
        <div className="flex flex-col items-center justify-center gap-2 py-24 text-center">
          <Bug className="w-8 h-8 text-muted-foreground/30" />
          <div className="text-[13px] font-semibold text-foreground">No diagnostics yet</div>
          <p className="text-[11px] text-muted-foreground max-w-[52ch] leading-relaxed">
            Run the analysis. Findings come from the compiler's own passes — shape resolution, memory, unsupported
            operators — not from a separate checker.
          </p>
        </div>
      </ChartErrorBoundary>
    );
  }

  const severityRows = deriveIssueSummary(analysis, warnings);
  const totalIssues = severityRows.reduce((sum, row) => sum + row.count, 0);
  const bySeverity = Object.fromEntries(severityRows.map((r) => [normalizeSeverity(r.severity), r.count]));
  const unsupported = deriveUnsupportedOps(analysis);
  const byLayer = deriveDiagnosticsByLayer(analysis, warnings, perLayer, nodes);
  const layers = buildDerivedLayerMetrics(analysis, perLayer);

  /**
   * Findings per layer, in graph order.
   *
   * `deriveDiagnosticsByLayer` groups by *category* — shape, memory,
   * parallel, op, config, general — not by severity, so the curve is split
   * the way the data actually is. Severity stays in the strip above, where
   * it is a count rather than a position.
   */
  const CATEGORIES = [
    { key: 'shape', label: 'shape', color: SIMULATION_COLORS.red },
    { key: 'memory', label: 'memory', color: SIMULATION_COLORS.amber },
    { key: 'op', label: 'operator', color: SIMULATION_COLORS.violet },
    { key: 'parallel', label: 'parallel', color: SIMULATION_COLORS.blue },
    { key: 'config', label: 'config', color: SIMULATION_COLORS.teal },
    { key: 'general', label: 'general', color: SIMULATION_COLORS.cyan },
  ] as const;

  const counts = new Map(byLayer.map((row) => [String(row.layer), row]));
  const num = (row: Record<string, string | number> | undefined, key: string): number => {
    const value = row?.[key];
    return typeof value === 'number' ? value : 0;
  };
  const depth = layers.map((layer, i) => {
    const row = counts.get(layer.name);
    const entry: Record<string, string | number> = { i: i + 1, name: layer.name };
    for (const c of CATEGORIES) entry[c.key] = num(row, c.key);
    return entry;
  });
  const hasDepth = depth.length > 1;
  const anyFinding = depth.some((d) => CATEGORIES.reduce((sum, c) => sum + Number(d[c.key] ?? 0), 0) > 0);

  const resolutionPct = (analysis.tensorResolutionRatio ?? 0) * 100;
  const confidence = analysis.confidenceScore ?? 0;

  // Resolution per layer: a layer whose output shape the compiler fully
  // resolved contributes full confidence; the model-wide ratio is the floor
  // everything else is judged against.
  const resolution = layers.map((layer, i) => ({
    i: i + 1,
    name: layer.name,
    // A layer with no parameters and no FLOPs is one the compiler could not
    // cost — the strongest per-layer signal of an unresolved shape available
    // without a second report field.
    resolved: layer.params > 0 || layer.flops > 0 ? 100 : 0,
  }));
  const hasResolution = resolution.length > 1;

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <StatStrip>
          <StatCard
            icon={<OctagonAlert className="w-3.5 h-3.5" />}
            label="Critical"
            value={String(bySeverity.critical ?? 0)}
            variant={(bySeverity.critical ?? 0) > 0 ? 'danger' : 'success'}
          />
          <StatCard
            icon={<TriangleAlert className="w-3.5 h-3.5" />}
            label="Warnings"
            value={String(bySeverity.warning ?? 0)}
            variant={(bySeverity.warning ?? 0) > 0 ? 'warning' : 'default'}
          />
          <StatCard label="Informational" value={String((bySeverity.info ?? 0) + (bySeverity.hint ?? 0))} />
          <StatCard
            icon={<ShieldQuestion className="w-3.5 h-3.5" />}
            label="Shapes resolved"
            value={resolutionPct > 0 ? `${resolutionPct.toFixed(0)} %` : '—'}
            sublabel={
              analysis.unresolvedDimCount > 0 ? `${analysis.unresolvedDimCount} dims unresolved` : 'all dimensions known'
            }
            variant={resolutionPct > 0 && resolutionPct < 90 ? 'warning' : 'default'}
          />
          <StatCard
            label="Confidence"
            value={confidence > 0 ? `${confidence.toFixed(0)} / 100` : '—'}
            sublabel="in the derived figures"
            variant={confidence > 0 && confidence < 70 ? 'warning' : 'default'}
          />
          <StatCard
            label="Unsupported ops"
            value={String(unsupported.length)}
            sublabel={analysis.customLayerCount ? `${analysis.customLayerCount} custom blocks` : undefined}
            variant={unsupported.length > 0 ? 'warning' : 'default'}
          />
        </StatStrip>

        <ChartGrid>
          <ChartSlot
            title="Findings across depth"
            has={hasDepth && anyFinding}
            emptyIcon={CircleCheck}
            emptyTitle={totalIssues === 0 ? 'Nothing to report' : 'Findings are not layer-attributed'}
            emptyHint={
              totalIssues === 0
                ? 'The compiler resolved this design without raising anything. Findings appear here the moment it stops.'
                : 'The findings on this design are model-wide rather than attached to individual layers, so there is no profile to draw. They are listed below.'
            }
            action={
              <div className="flex items-center gap-3 flex-wrap">
                {CATEGORIES.map((e) => (
                  <span key={e.key} className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    <span className="w-2 h-2 rounded-[2px]" style={{ background: e.color }} />
                    {e.label}
                  </span>
                ))}
              </div>
            }
            reading="Where the findings land along the graph. Nine warnings spread evenly is a different problem from nine on one block, and only their position tells you which you have."
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={depth} margin={CHART_MARGINS.area}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA allowDecimals={false} width={40} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => depth[Number(i) - 1]?.name ?? `layer ${i}`}
                />
                {CATEGORIES.map((c) => (
                  <Area
                    key={c.key}
                    type="monotone"
                    dataKey={c.key}
                    name={c.label}
                    stackId="findings"
                    stroke={c.color}
                    fill={c.color}
                    fillOpacity={0.3}
                    strokeWidth={1.25}
                    isAnimationActive={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="What the compiler could cost, layer by layer"
            has={hasResolution}
            emptyIcon={ShieldQuestion}
            emptyTitle="No per-layer resolution"
            emptyHint="This needs per-layer parameters or FLOPs, which this analysis did not produce."
            reading={
              <>
                A layer at zero is one the compiler could not attach a cost to — usually an unresolved shape or an
                operator it does not model. Every memory and latency figure downstream of such a layer is an estimate
                resting on an estimate.
                {resolutionPct > 0 ? ` Across the whole graph it resolved ${resolutionPct.toFixed(0)} % of dimensions.` : ''}
              </>
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={resolution} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" %" domain={[0, 100]} width={44} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => resolution[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => (v > 0 ? 'costed' : 'not costed')}
                />
                {resolutionPct > 0 ? (
                  <ReferenceLine
                    y={resolutionPct}
                    stroke="hsl(var(--muted-foreground))"
                    strokeDasharray="2 3"
                    label={{
                      value: 'graph-wide resolution',
                      position: 'insideTopRight',
                      fontSize: 9,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                  />
                ) : null}
                <Line
                  type="stepAfter"
                  dataKey="resolved"
                  name="costed"
                  stroke={SIMULATION_COLORS.green}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>
        </ChartGrid>

        {totalIssues > 0 ? (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              What the compiler raised
            </h3>
            <div className="border border-border rounded-lg divide-y divide-border/60">
              {(analysis.diagnostics ?? []).slice(0, 12).map((d, i) => {
                const severity = normalizeSeverity(d.severity);
                return (
                  <div key={`${d.code ?? d.category}-${i}`} className="p-3">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={cn('text-[10px] font-semibold uppercase tracking-wider', SEVERITY_TONE[severity])}>
                        {severity}
                      </span>
                      {d.code ? <span className="font-mono text-[11px] text-muted-foreground">{d.code}</span> : null}
                      {d.layer_id ? (
                        <span className="ml-auto font-mono text-[10px] text-muted-foreground/70">{d.layer_id}</span>
                      ) : null}
                    </div>
                    <p className="mt-1 text-[12px] text-foreground/90 leading-relaxed max-w-[96ch]">{d.message}</p>
                    {d.suggestion ? (
                      <p className="mt-1 text-[11px] text-muted-foreground leading-relaxed max-w-[96ch]">
                        {d.suggestion}
                      </p>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}

        {(analysis.recommendations?.length ?? 0) > 0 ? (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              What the compiler suggests
            </h3>
            {/* The report has carried `recommendations` all along and no view
                ever showed them. They are the half of a diagnostic that says
                what to do, which makes them the more useful half. */}
            <div className="border border-border rounded-lg divide-y divide-border/60">
              {(analysis.recommendations ?? []).slice(0, 8).map((rec, i) => {
                const r = rec as unknown as Record<string, unknown>;
                const title = String(r.title ?? r.category ?? 'Recommendation');
                const detail = String(r.message ?? r.description ?? r.suggestion ?? '');
                const impact = typeof r.impact === 'string' ? r.impact : undefined;
                return (
                  <div key={`${title}-${i}`} className="p-3">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Lightbulb className="w-3.5 h-3.5 text-primary shrink-0" />
                      <span className="text-[12px] font-semibold text-foreground">{title}</span>
                      {impact ? (
                        <span className="ml-auto font-mono text-[10px] text-muted-foreground/70">{impact}</span>
                      ) : null}
                    </div>
                    {detail ? (
                      <p className="mt-1 text-[12px] text-foreground/90 leading-relaxed max-w-[96ch]">{detail}</p>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}

        {unsupported.length > 0 ? (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Operators the compiler does not model
            </h3>
            <div className="flex flex-wrap gap-1.5">
              {unsupported.map((op) => (
                <span
                  key={op.name}
                  className="rounded-[6px] border border-amber-500/40 bg-amber-500/5 px-2 py-1 font-mono text-[11px] text-amber-700 dark:text-amber-400"
                >
                  {op.name}
                  {op.count > 1 ? <span className="text-muted-foreground"> ×{op.count}</span> : null}
                </span>
              ))}
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground leading-relaxed max-w-[96ch]">
              These contribute no FLOPs and no memory to the analysis. Their cost is not zero — it is unknown, and
              every total on this workspace is a lower bound while they are present.
            </p>
          </div>
        ) : null}

        <ViewNote icon={Bug}>
          Findings here are static: they come from reading the graph, not from running it. The ones only a live process
          can raise — a dataloader starving the GPU, a loss going non-finite — appear in{' '}
          <span className="font-medium text-foreground">Training</span>.
        </ViewNote>
      </div>
    </ChartErrorBoundary>
  );
}
