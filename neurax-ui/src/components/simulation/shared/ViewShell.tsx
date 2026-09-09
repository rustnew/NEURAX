/**
 * One rhythm for every analysis view.
 *
 * Simulation had drifted into six differently-shaped screens: some opened on a
 * giant square tile holding a single number, some on a donut with an empty
 * state behind it, some on a grid of figures that was a table wearing a
 * chart's clothes. The numbers were right — the compiler produces them — but
 * the reading was work, and a screen you have to decode is a screen you stop
 * opening.
 *
 * The Training workspace's Live view had already settled the shape that works,
 * and it is not a NEURAX invention: the figures you check at a glance go in one
 * compact strip along the top, and everything below is charts of equal size in
 * one grid. This module is that shape, extracted, so the six analysis views
 * share it rather than each re-deciding.
 *
 * Two rules it enforces by construction:
 *
 *  - **A number is a strip cell, never a chart.** A tile whose whole content is
 *    "0.0" wastes a chart-sized hole in the page. If it is one figure, it
 *    belongs in the strip.
 *  - **A chart with no data does not get drawn.** `ChartSlot` renders the empty
 *    state in place, at the card's own size, so the grid keeps its rhythm
 *    instead of collapsing around a missing tile — and so a missing series is
 *    visibly missing rather than silently rendered as zero.
 */
import { type ComponentType, type ReactNode } from 'react';
import { CartesianGrid, XAxis, YAxis } from 'recharts';

import { ChartCard, ChartContainer, EmptyChartState } from './index';
import { cn } from '@/lib/utils';

/** Axis styling every chart in Simulation shares, so two charts side by side
 *  are read at the same weight rather than one shouting over the other. */
export const AXIS = { stroke: 'hsl(var(--muted-foreground))', fontSize: 10 } as const;

/** The grid behind every chart. Faint on purpose: it is a reading aid, not a
 *  layer of the drawing. */
export const CHART_GRID = (
  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
);

export function XA(props: React.ComponentProps<typeof XAxis>) {
  return <XAxis {...AXIS} {...props} />;
}

export function YA(props: React.ComponentProps<typeof YAxis>) {
  return <YAxis {...AXIS} {...props} />;
}

/** The row of headline figures. Six across on a wide screen — the number the
 *  Live view arrived at, because a seventh stops being scannable. */
export function StatStrip({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">{children}</div>;
}

/** The chart grid: two equal columns, one column on a narrow window. */
export function ChartGrid({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('grid grid-cols-1 lg:grid-cols-2 gap-4', className)}>{children}</div>;
}

interface ChartSlotProps {
  title: string;
  /** What the figures are: `derived` for anything the compiler computed,
   *  `measured` for the rare value read from hardware. Stated on every card
   *  because Simulation and Training now sit beside each other, and a reader
   *  moving between them has to know which kind of number they are looking at. */
  badge?: { text: string; variant?: 'live' | 'derived' | 'warning' | 'info' };
  /** False when the series behind this chart does not exist in the analysis.
   *  The card is still drawn, at its normal size, saying so. */
  has: boolean;
  emptyIcon: ComponentType<{ className?: string }>;
  emptyTitle: string;
  emptyHint: string;
  /** A sentence under the chart naming what the reader should take from it.
   *  Optional, and worth writing: a chart nobody can interpret is decoration. */
  reading?: ReactNode;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function ChartSlot({
  title,
  badge = { text: 'derived', variant: 'derived' },
  has,
  emptyIcon,
  emptyTitle,
  emptyHint,
  reading,
  action,
  className,
  children,
}: ChartSlotProps) {
  return (
    <ChartCard title={title} badge={badge} size="wide" action={action} className={className}>
      {has ? (
        <div className="flex flex-col h-full min-h-0">
          <ChartContainer className="flex-1 min-h-0">{children}</ChartContainer>
          {reading ? (
            <p className="pt-2 text-[10px] text-muted-foreground leading-relaxed shrink-0">{reading}</p>
          ) : null}
        </div>
      ) : (
        <EmptyChartState icon={emptyIcon} title={emptyTitle} description={emptyHint} />
      )}
    </ChartCard>
  );
}

/** The closing line of a view: what the whole screen is, and is not. */
export function ViewNote({ icon: Icon, children }: { icon: ComponentType<{ className?: string }>; children: ReactNode }) {
  return (
    <p className="flex items-start gap-1.5 text-[11px] text-muted-foreground leading-relaxed max-w-[100ch]">
      <Icon className="w-3.5 h-3.5 mt-px shrink-0 text-muted-foreground/60" />
      <span>{children}</span>
    </p>
  );
}
