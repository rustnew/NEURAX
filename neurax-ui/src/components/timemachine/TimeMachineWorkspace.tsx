import { useState, useRef, useEffect } from 'react';
import {
  Clock, TrendingUp, DollarSign, Leaf, Shield, SlidersHorizontal,
  AlertTriangle, Zap, Award, Info, Loader2
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Slider } from '@/components/ui/slider.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { TooltipProvider } from '@/components/ui/tooltip.tsx';
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, ResponsiveContainer,
  Legend, ReferenceLine,
  Tooltip as RechartsTooltip
} from 'recharts';
import { CanvasNode, Connection } from '@/types/architecture.ts';
import {
  runTimeMachine,
  TimeMachineProjection,
  TmScenarioPoint,
  TmCarbonPoint,
  TmCostBreakdownPoint,
  TmRecommendation,
  getComplianceConfig,
  type ComplianceConfig,
  type ComplianceRegulation,
  type ComplianceThresholds,
} from '@/services/neuraxApi.ts';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';

// ── Type aliases so sub-view components keep their signatures ──
type ScenarioPoint = TmScenarioPoint;
type CarbonPoint = TmCarbonPoint;
type CostBreakdown = TmCostBreakdownPoint;
type Recommendation = TmRecommendation;


// ── Custom chart tooltip ──────────────────────────────────────
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload as ScenarioPoint;
  return (
    <div className="bg-popover border border-border rounded-lg p-3 shadow-lg text-xs space-y-1">
      <p className="font-semibold text-foreground">Year {label}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: ${p.value?.toLocaleString()}/mo
        </p>
      ))}
      {point?.breakingPoint && (
        <p className="text-destructive font-medium flex items-center gap-1">
          <AlertTriangle className="w-3 h-3" /> Budget exceeded
        </p>
      )}
      {point?.hardwareEvent && (
        <p className="text-primary flex items-center gap-1">
          <Zap className="w-3 h-3" /> {point.hardwareEvent}
        </p>
      )}
      {point?.migration && (
        <p className="text-accent-foreground italic">{point.migration}</p>
      )}
    </div>
  );
}

// ── Breaking point dot renderer ───────────────────────────────
function BreakingDot(props: any) {
  const { cx, cy, payload } = props;
  if (!payload?.breakingPoint) return null;
  return (
    <svg x={cx - 8} y={cy - 8} width={16} height={16}>
      <circle cx="8" cy="8" r="6" fill="hsl(var(--destructive))" opacity={0.8} />
      <circle cx="8" cy="8" r="3" fill="hsl(var(--destructive-foreground))" />
    </svg>
  );
}

interface TimeMachineWorkspaceProps {
  nodes: CanvasNode[];
  connections: Connection[];
  analysis?: unknown;
}

export function TimeMachineWorkspace({ nodes, connections }: TimeMachineWorkspaceProps) {
  const [growthRate, setGrowthRate] = useState(100);
  const [budgetMax, setBudgetMax] = useState(500000);
  const [horizon, setHorizon] = useState(5);
  const [hardware, setHardware] = useState<'a100' | 'h200' | 'b100'>('a100');
  const [activeView, setActiveView] = useState('timeline');
  const [projection, setProjection] = useState<TimeMachineProjection | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [projError, setProjError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (nodes.length === 0) { setProjection(null); return; }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setIsLoading(true);
      setProjError(null);
      try {
        const ir = compileToNeuraxIR(nodes, connections, { modelName: 'NeuraxModel' });
        const { projection: proj } = await runTimeMachine({
          topology: ir as unknown as Record<string, unknown>,
          params: {
            growth_rate_pct: growthRate,
            horizon_years: horizon,
            annual_budget_usd: budgetMax,
            hardware_track: hardware,
          },
        });
        setProjection(proj);
      } catch (e) {
        setProjError(e instanceof Error ? e.message : 'Projection failed');
      } finally {
        setIsLoading(false);
      }
    }, 600);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [nodes, connections, growthRate, horizon, budgetMax, hardware]);

  const timeline = projection?.timeline ?? [];
  const recommendations = projection?.recommendations ?? [];
  const carbon = projection?.carbon ?? [];
  const costBreakdown = projection?.costBreakdown ?? [];
  const totalCostNominal = projection?.summary.totalCostNominalUsd ?? 0;
  const firstBreakYear = projection?.summary.firstBreakYear;

  const priorityColor = (p: string) =>
    p === 'high' ? 'text-destructive' : p === 'medium' ? 'text-yellow-500' : 'text-muted-foreground';

  return (
    <TooltipProvider>
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Header */}
        <div className="border-b border-border bg-card px-3 sm:px-6 py-3 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            <div className="p-1.5 sm:p-2 rounded-lg bg-primary/10 shrink-0">
              <Clock className="w-4 h-4 sm:w-5 sm:h-5 text-primary" />
            </div>
            <div className="min-w-0">
              <h2 className="text-xs sm:text-sm font-semibold text-foreground flex items-center gap-2">
                <span className="truncate">Time Machine</span>
                {isLoading && <Loader2 className="w-3 h-3 animate-spin text-primary shrink-0" />}
              </h2>
              <p className="text-[10px] sm:text-xs text-muted-foreground hidden sm:block">
                Compiler-backed multi-scenario cost &amp; scaling projection
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-4 text-xs shrink-0">
            {projection && (
              <div className="text-right hidden sm:block">
                <span className="text-muted-foreground">Total ({horizon}yr nominal)</span>
                <p className="font-mono font-semibold text-foreground">${(totalCostNominal / 1000).toFixed(0)}k</p>
              </div>
            )}
            {firstBreakYear && (
              <Badge variant="destructive" className="text-[10px] gap-1">
                <AlertTriangle className="w-3 h-3" />
                <span className="hidden sm:inline">Budget break:</span> {firstBreakYear}
              </Badge>
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
          {/* What-If Panel (left sidebar on desktop, collapsible on mobile) */}
          <div className="w-full md:w-72 border-b md:border-b-0 md:border-r border-border bg-card p-3 sm:p-4 space-y-4 sm:space-y-5 overflow-y-auto scrollbar-thin max-h-48 md:max-h-none">
            <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
              <SlidersHorizontal className="w-4 h-4 text-primary" />
              What-If Parameters
            </div>

            {/* Growth Rate */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-muted-foreground">User Growth</label>
                <span className="text-xs font-mono font-medium text-foreground">{growthRate}%/yr</span>
              </div>
              <Slider
                value={[growthRate]}
                onValueChange={([v]) => setGrowthRate(v)}
                min={0} max={500} step={10}
              />
            </div>

            {/* Budget Max */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-muted-foreground">Annual Budget</label>
                <span className="text-xs font-mono font-medium text-foreground">${(budgetMax / 1000).toFixed(0)}k</span>
              </div>
              <Slider
                value={[budgetMax]}
                onValueChange={([v]) => setBudgetMax(v)}
                min={10000} max={5000000} step={10000}
              />
            </div>

            {/* Horizon */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-muted-foreground">Horizon</label>
                <span className="text-xs font-mono font-medium text-foreground">{horizon} yrs</span>
              </div>
              <Slider
                value={[horizon]}
                onValueChange={([v]) => setHorizon(v)}
                min={1} max={10} step={1}
              />
            </div>

            {/* Hardware */}
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground">Target Hardware</label>
              <Select value={hardware} onValueChange={(v) => setHardware(v as 'a100' | 'h200' | 'b100')}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="a100">NVIDIA A100</SelectItem>
                  <SelectItem value="h200">NVIDIA H200</SelectItem>
                  <SelectItem value="b100">NVIDIA B100</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Recommendations */}
            <div className="pt-3 border-t border-border space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
                <Award className="w-4 h-4 text-primary" />
                Recommendations
              </div>
              {isLoading && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="w-3 h-3 animate-spin" /> Computing…
                </div>
              )}
              {!isLoading && projError && (
                <p className="text-xs text-destructive">{projError}</p>
              )}
              {!isLoading && !projError && recommendations.map((rec: Recommendation, i: number) => (
                <div key={i} className="p-2.5 rounded-md bg-secondary/50 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] font-bold uppercase ${priorityColor(rec.priority)}`}>
                      {rec.priority}
                    </span>
                    <span className="text-xs font-medium text-foreground">{rec.title}</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground leading-relaxed">{rec.description}</p>
                  <div className="flex items-center gap-3 text-[10px]">
                    <span className="text-primary font-medium">Savings: {rec.savings}</span>
                    <span className="text-muted-foreground">{rec.timing}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Main Charts Area */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Sub-tabs */}
            <div className="border-b border-border bg-card px-2 sm:px-4 py-1.5">
              <div className="flex items-center gap-1 overflow-x-auto scrollbar-thin">
                {[
                  { id: 'timeline', label: 'Cost Timeline', icon: TrendingUp },
                  { id: 'breakdown', label: 'Cost Breakdown', icon: DollarSign },
                  { id: 'carbon', label: 'Carbon Impact', icon: Leaf },
                  { id: 'compliance', label: 'Compliance', icon: Shield },
                ].map(tab => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveView(tab.id)}
                      className={`flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap ${activeView === tab.id
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                        }`}
                    >
                      <Icon className="w-3.5 h-3.5 shrink-0" />
                      <span className="hidden sm:inline">{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Chart content */}
            <div className="flex-1 overflow-auto p-4 scrollbar-thin">
              {nodes.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground gap-3">
                  <Clock className="w-10 h-10 opacity-20" />
                  <p className="text-sm font-medium">No model on the canvas</p>
                  <p className="text-xs">Add layers to generate a Time Machine projection.</p>
                </div>
              ) : isLoading && !projection ? (
                <div className="h-full flex flex-col items-center justify-center gap-3 text-muted-foreground">
                  <Loader2 className="w-8 h-8 animate-spin text-primary" />
                  <p className="text-sm">Computing projection…</p>
                </div>
              ) : projError && !projection ? (
                <div className="h-full flex flex-col items-center justify-center gap-3">
                  <AlertTriangle className="w-8 h-8 text-destructive" />
                  <p className="text-sm font-medium text-destructive">Projection failed</p>
                  <p className="text-xs text-muted-foreground">{projError}</p>
                </div>
              ) : (
                <>
                  {activeView === 'timeline' && <TimelineView timeline={timeline} budgetMax={budgetMax} />}
                  {activeView === 'breakdown' && <BreakdownView data={costBreakdown} />}
                  {activeView === 'carbon' && <CarbonView data={carbon} />}
                  {activeView === 'compliance' && <ComplianceView horizon={horizon} />}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}

// ── Sub-views ─────────────────────────────────────────────────

function TimelineView({ timeline, budgetMax }: { timeline: ScenarioPoint[]; budgetMax: number }) {
  const monthlyBudget = Math.round(budgetMax / 12);
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Multi-Scenario Cost Projection</h3>
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-primary" /> Nominal (75%)</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-400" /> Optimistic (15%)</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-orange-400" /> Pessimistic (10%)</span>
        </div>
      </div>
      <div className="h-80 bg-card rounded-lg border border-border p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timeline} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="year" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
            <ReferenceLine y={monthlyBudget} stroke="hsl(var(--destructive))" strokeDasharray="6 3" label={{ value: 'Budget limit', fill: 'hsl(var(--destructive))', fontSize: 10 }} />
            <Line type="monotone" dataKey="pessimistic" stroke="hsl(25, 95%, 53%)" strokeWidth={1.5} strokeDasharray="4 2" name="Pessimistic" dot={false} />
            <Line type="monotone" dataKey="nominal" stroke="hsl(var(--primary))" strokeWidth={2.5} name="Nominal" dot={<BreakingDot />} />
            <Line type="monotone" dataKey="optimistic" stroke="hsl(210, 80%, 60%)" strokeWidth={1.5} strokeDasharray="4 2" name="Optimistic" dot={false} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <RechartsTooltip content={<CustomTooltip />} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Year 1 Cost', value: `$${((timeline[1]?.nominal || 0) * 12 / 1000).toFixed(0)}k`, sub: 'annual' },
          { label: `Year ${timeline.length - 1} Cost`, value: `$${((timeline[timeline.length - 1]?.nominal || 0) * 12 / 1000).toFixed(0)}k`, sub: 'annual' },
          { label: 'Cost Growth', value: `${timeline.length > 1 ? ((timeline[timeline.length - 1]?.nominal / timeline[0]?.nominal - 1) * 100).toFixed(0) : 0}%`, sub: 'total' },
          { label: 'Break Point', value: timeline.find(t => t.breakingPoint)?.year?.toString() || 'None', sub: 'year' },
        ].map((card, i) => (
          <Card key={i} className="bg-card">
            <CardContent className="p-3">
              <p className="text-[10px] text-muted-foreground">{card.label}</p>
              <p className="text-lg font-mono font-bold text-foreground">{card.value}</p>
              <p className="text-[10px] text-muted-foreground">{card.sub}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function BreakdownView({ data }: { data: CostBreakdown[] }) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-foreground">Cost Breakdown by Component</h3>
      <div className="h-80 bg-card rounded-lg border border-border p-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="year" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
            <Area type="monotone" dataKey="compute" stackId="1" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.6} name="Compute" />
            <Area type="monotone" dataKey="storage" stackId="1" stroke="hsl(210, 80%, 60%)" fill="hsl(210, 80%, 60%)" fillOpacity={0.5} name="Storage" />
            <Area type="monotone" dataKey="network" stackId="1" stroke="hsl(150, 60%, 45%)" fill="hsl(150, 60%, 45%)" fillOpacity={0.4} name="Network" />
            <Area type="monotone" dataKey="egress" stackId="1" stroke="hsl(40, 90%, 55%)" fill="hsl(40, 90%, 55%)" fillOpacity={0.3} name="Egress" />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CarbonView({ data }: { data: CarbonPoint[] }) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-foreground">Carbon Footprint Projection (tonnes CO₂/yr)</h3>
      <div className="h-80 bg-card rounded-lg border border-border p-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="year" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <Area type="monotone" dataKey="baseline" stroke="hsl(0, 70%, 55%)" fill="hsl(0, 70%, 55%)" fillOpacity={0.3} name="Baseline" />
            <Area type="monotone" dataKey="optimized" stroke="hsl(40, 90%, 55%)" fill="hsl(40, 90%, 55%)" fillOpacity={0.3} name="Optimized" />
            <Area type="monotone" dataKey="withGreenRegions" stroke="hsl(140, 70%, 45%)" fill="hsl(140, 70%, 45%)" fillOpacity={0.4} name="Green Regions" />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {data.length > 0 && [
          { label: 'Baseline Reduction', value: `${((1 - data[data.length - 1].optimized / data[data.length - 1].baseline) * 100).toFixed(0)}%`, desc: 'With optimizations' },
          { label: 'Green Regions', value: `${((1 - data[data.length - 1].withGreenRegions / data[data.length - 1].baseline) * 100).toFixed(0)}%`, desc: 'Max reduction possible' },
          { label: `CO₂ Saved (${data[data.length - 1]?.year})`, value: `${(data[data.length - 1].baseline - data[data.length - 1].withGreenRegions).toFixed(1)}t`, desc: 'tonnes/year' },
        ].map((c, i) => (
          <Card key={i} className="bg-card">
            <CardContent className="p-3">
              <p className="text-[10px] text-muted-foreground">{c.label}</p>
              <p className="text-lg font-mono font-bold text-primary">{c.value}</p>
              <p className="text-[10px] text-muted-foreground">{c.desc}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function ComplianceView({ horizon }: { horizon: number }) {
  const [config, setConfig] = useState<ComplianceConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getComplianceConfig()
      .then((data) => setConfig(data))
      .catch(() => setConfig(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">Loading compliance data…</span>
      </div>
    );
  }

  const regulations = config?.regulations ?? [
    { name: 'EU AI Act Phase 1', year: 2027, limit: 300, unit: 'GFLOPs/req', status: 'upcoming', description: 'General-purpose AI models trained with >10²⁵ FLOPs must comply with transparency and safety obligations.', region: 'EU' },
    { name: 'EU AI Act Phase 2', year: 2028, limit: 150, unit: 'GFLOPs/req', status: 'upcoming', description: 'Stricter limits for high-risk AI applications in critical infrastructure, law enforcement, and biometrics.', region: 'EU' },
    { name: 'Carbon Reporting (CSRD)', year: 2026, limit: null, unit: null, status: 'active', description: 'Corporate Sustainability Reporting Directive requires disclosure of energy consumption and CO₂ emissions for large companies.', region: 'EU' },
    { name: 'Digital Services Act', year: 2026, limit: null, unit: null, status: 'active', description: 'Requires transparency reporting for very large online platforms using AI, including compute disclosure.', region: 'EU' },
  ];

  const thresholds = config?.thresholds ?? {
    high_risk_gflops: 300,
    carbon_report_tonnes: 50,
    dsa_disclosure_flops: 1e25,
    cost_review_usd: 100000,
  };

  const recommendations = config?.recommendations ?? [
    'Monitor EU AI Act Phase 1 compliance for models exceeding 300 GFLOPs/request',
    'Prepare CSRD carbon reporting for training runs exceeding 50 tonnes CO₂e/year',
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-foreground">Regulatory Compliance Timeline</h3>
      <div className="space-y-3">
        {regulations.map((reg: ComplianceRegulation, i: number) => {
          const inScope = reg.year <= 2026 + horizon;
          return (
            <Card key={i} className={`bg-card ${!inScope ? 'opacity-50' : ''}`}>
              <CardContent className="p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Shield className={`w-5 h-5 ${reg.status === 'active' ? 'text-primary' : 'text-yellow-500'}`} />
                  <div>
                    <p className="text-sm font-medium text-foreground">{reg.name}</p>
                    <p className="text-xs text-muted-foreground">
                      Effective: {reg.year} {reg.limit ? `· Limit: ${reg.limit} ${reg.unit}` : '· Disclosure required'}
                      {reg.region && ` · ${reg.region}`}
                    </p>
                    {reg.description && (
                      <p className="text-[10px] text-muted-foreground mt-0.5 line-clamp-2">{reg.description}</p>
                    )}
                  </div>
                </div>
                <Badge variant={reg.status === 'active' ? 'default' : 'secondary'} className="text-[10px]">
                  {reg.status === 'active' ? 'Active' : reg.status === 'proposed' ? 'Proposed' : `${reg.year - 2026} yr away`}
                </Badge>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Thresholds */}
      <Card className="bg-secondary/30">
        <CardContent className="p-4">
          <h4 className="text-xs font-semibold text-foreground mb-2">Compliance Thresholds</h4>
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div>
              <span className="text-muted-foreground">High-risk threshold:</span>{' '}
              <span className="font-mono font-bold text-primary">{thresholds.high_risk_gflops} GFLOPs</span>
            </div>
            <div>
              <span className="text-muted-foreground">Carbon report:</span>{' '}
              <span className="font-mono font-bold text-primary">{thresholds.carbon_report_tonnes}t CO₂e/yr</span>
            </div>
            <div>
              <span className="text-muted-foreground">DSA disclosure:</span>{' '}
              <span className="font-mono font-bold text-primary">{thresholds.dsa_disclosure_flops.toExponential(0)} FLOPs</span>
            </div>
            <div>
              <span className="text-muted-foreground">Cost review:</span>{' '}
              <span className="font-mono font-bold text-primary">${(thresholds.cost_review_usd / 1000).toFixed(0)}k</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <Card className="bg-secondary/30 border-primary/20">
          <CardContent className="p-4">
            <h4 className="text-xs font-semibold text-foreground mb-2">Recommendations</h4>
            <ul className="space-y-1">
              {recommendations.map((rec: string, i: number) => (
                <li key={i} className="text-[11px] text-muted-foreground flex items-start gap-1.5">
                  <span className="text-primary mt-0.5">•</span>
                  {rec}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <Card className="bg-secondary/30 border-primary/20">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <Info className="w-4 h-4 text-primary mt-0.5" />
            <div className="text-xs text-muted-foreground space-y-1">
              <p className="font-medium text-foreground">Compliance analysis is based on current regulatory proposals.</p>
              <p>Actual limits and timelines may change. NEURAX will update projections as regulations are finalized.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
