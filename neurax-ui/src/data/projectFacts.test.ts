/**
 * The landing page's numbers must be checkable against the repository.
 *
 * A wrong figure here is a claim, not a bug: the previous copy advertised 680+
 * blocks against a catalogue of 208, with a per-family breakdown in the FAQ that
 * matched nothing.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  CATALOGUE,
  BLOCK_COUNT,
  ENVIRONMENT,
  FAMILY_COUNT,
  HERO_STATS,
  IR_PASS_COUNT,
  ON_YOUR_MACHINE,
  WORKSPACES,
} from './projectFacts';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
/** A file from the studio's own source, so a claim can be checked against
 *  the code that implements it rather than against a copy. */
const source = (relative: string) =>
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', relative), 'utf8');

const catalogue = JSON.parse(
  readFileSync(join(repoRoot, 'neurax-agent', 'catalogue.json'), 'utf8'),
);

describe('project facts', () => {
  it('lists exactly the families the catalogue defines', () => {
    expect(Object.keys(CATALOGUE).sort()).toEqual(Object.keys(catalogue).sort());
    expect(FAMILY_COUNT).toBe(Object.keys(catalogue).length);
  });

  it('counts each family s blocks as the catalogue does', () => {
    for (const [family, counts] of Object.entries(CATALOGUE)) {
      expect(counts.blocks, `${family} blocks`).toBe(catalogue[family].blocks?.length ?? 0);
      expect(counts.macroBlocks, `${family} macro-blocks`).toBe(
        catalogue[family].macroBlocks?.length ?? 0,
      );
    }
  });

  it('totals the blocks it advertises', () => {
    const expected = Object.values(catalogue).reduce(
      (total: number, family: any) =>
        total + (family.blocks?.length ?? 0) + (family.macroBlocks?.length ?? 0),
      0,
    );
    expect(BLOCK_COUNT).toBe(expected);
  });

  it('counts the IR passes the pipeline actually runs', () => {
    const source = readFileSync(
      join(repoRoot, 'neurax-core', 'src', 'lib.rs'),
      'utf8',
    );
    // Each pass appears as a `<Name>Pass` type in the analysis pipeline.
    const passes = new Set(source.match(/\b(\w+)Pass\b/g) ?? []);
    expect(passes.size).toBeGreaterThanOrEqual(IR_PASS_COUNT);
  });

  it('gives every hero stat a value, a label and an explanation', () => {
    expect(HERO_STATS).toHaveLength(4);
    for (const stat of HERO_STATS) {
      expect(stat.value).toBeTruthy();
      expect(stat.label).toBeTruthy();
      expect(stat.detail.length).toBeGreaterThan(20);
    }
  });

  it('names exactly the workspaces the studio has, in the same order', () => {
    // The claim that broke this page was "from transformers to spiking
    // networks", advertising a family that had been deleted. The same drift
    // is possible for workspaces — Inference Intelligence was removed and the
    // page would have gone on listing it — so the list is checked against the
    // component that draws the tabs rather than maintained by hand.
    const tabs = source('components/layout/WorkspaceTabs.tsx');
    const labels = [...tabs.matchAll(/label: '([^']+)'/g)].map((m) => m[1]);
    expect(WORKSPACES.map((w) => w.name)).toEqual(labels);
  });

  it('describes every workspace and every part of the environment', () => {
    for (const workspace of WORKSPACES) {
      expect(workspace.does.length, `${workspace.name} has no description`).toBeGreaterThan(20);
    }
    for (const part of ENVIRONMENT) {
      expect(part.detail.length, `${part.name} has no description`).toBeGreaterThan(20);
    }
    for (const item of ON_YOUR_MACHINE) {
      expect(item.detail.length, `${item.title} has no description`).toBeGreaterThan(20);
    }
  });

  it('claims no architecture family the catalogue does not define', () => {
    // Any family named in prose has to exist. `spiking` is checked by name
    // because it is the one that was advertised for real.
    const landing = source('pages/Landing.tsx');
    const prose = landing.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
    expect(prose.toLowerCase()).not.toContain('spiking');
  });
});
