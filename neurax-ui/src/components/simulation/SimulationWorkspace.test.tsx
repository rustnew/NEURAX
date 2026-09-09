import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SIMULATION_CATEGORIES } from './SimulationWorkspace';

const here = dirname(fileURLToPath(import.meta.url));

/** Module rendering each category, in the same order as the tab list. */
const MODULE_FOR: Record<string, string> = {
  overview: 'GlobalResultsCharts',
  perlayer: 'PerLayerCharts',
  memory: 'MemoryCharts',
  optimization: 'OptimizationCharts',
  diagnostics: 'DebuggingCharts',
};

/** Views are being rewritten onto `ChartSlot`, which is a `ChartCard` that
 *  also carries its own empty state; both spellings count as one chart. */
function countChartCards(module: string): number {
  const source = readFileSync(join(here, 'categories', `${module}.tsx`), 'utf8');
  return (source.match(/<ChartCard|<ChartSlot/g) ?? []).length;
}

describe('simulation category tabs', () => {
  it('advertises the number of charts each view actually renders', () => {
    for (const category of SIMULATION_CATEGORIES) {
      const actual = countChartCards(MODULE_FOR[category.id]);
      expect(
        category.chartCount,
        `${category.label} advertises ${category.chartCount} charts but renders ${actual}`,
      ).toBe(actual);
    }
  });

  it('has a module wired for every tab', () => {
    for (const category of SIMULATION_CATEGORIES) {
      expect(MODULE_FOR[category.id], `${category.id} has no module`).toBeTruthy();
    }
    expect(Object.keys(MODULE_FOR).length).toBe(SIMULATION_CATEGORIES.length);
  });

  it('gives every tab a distinct id and a hint', () => {
    const ids = SIMULATION_CATEGORIES.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const category of SIMULATION_CATEGORIES) {
      expect(category.hint.length, `${category.label} needs a hint`).toBeGreaterThan(10);
    }
  });

  it('opens on the overview', () => {
    expect(SIMULATION_CATEGORIES[0].id).toBe('overview');
  });
});

describe('chart titles', () => {
  it('carry no internal spec numbering', () => {
    for (const module of Object.values(MODULE_FOR)) {
      const source = readFileSync(join(here, 'categories', `${module}.tsx`), 'utf8');
      const numbered = source.match(/title="\d+\.\d+/g) ?? [];
      expect(numbered, `${module} still has spec-numbered titles`).toEqual([]);
    }
  });
});
