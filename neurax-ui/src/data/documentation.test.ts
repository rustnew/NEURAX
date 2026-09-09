/**
 * Documentation that is wrong is worse than none.
 *
 * A guide describing a shortcut that does not exist, or a button that was
 * renamed two versions ago, teaches the reader to stop trusting the rest of it
 * — and they are right to. So the checks here are less about prose and more
 * about the guide agreeing with the application: the shortcuts it lists are the
 * ones the page binds, the workspaces it names are the ones that exist, and the
 * export formats it promises are the ones the export panel offers.
 *
 * The structural checks matter too. A section with no id cannot be linked to; a
 * duplicate id makes navigation jump to the wrong place; a table row with the
 * wrong number of cells renders as a broken grid.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DOCUMENTATION, allSections, searchDocs } from './documentation';

const source = (relative: string) =>
  readFileSync(join(__dirname, '..', relative), 'utf8');

describe('the user guide', () => {
  describe('is structurally sound', () => {
    it('gives every section a unique id', () => {
      const ids = allSections().map((s) => s.id);
      expect(new Set(ids).size, `duplicate section ids: ${ids.join(', ')}`).toBe(ids.length);
    });

    it('gives every chapter a unique id and at least one section', () => {
      const ids = DOCUMENTATION.map((c) => c.id);
      expect(new Set(ids).size).toBe(ids.length);
      for (const chapter of DOCUMENTATION) {
        expect(chapter.sections.length, `${chapter.id} is empty`).toBeGreaterThan(0);
      }
    });

    it('gives every section a title and a summary', () => {
      for (const section of allSections()) {
        expect(section.title.length, `${section.id} has no title`).toBeGreaterThan(0);
        expect(section.summary.length, `${section.id} has no summary`).toBeGreaterThan(0);
        expect(section.blocks.length, `${section.id} has no content`).toBeGreaterThan(0);
      }
    });

    it('gives every table row as many cells as there are columns', () => {
      for (const section of allSections()) {
        for (const block of section.blocks) {
          if (block.kind !== 'table') continue;
          for (const [i, row] of block.rows.entries()) {
            expect(
              row.length,
              `${section.id}: table row ${i} has ${row.length} cells for ${block.columns.length} columns`,
            ).toBe(block.columns.length);
          }
        }
      }
    });

    it('closes every emphasis and code span it opens', () => {
      // An unclosed `**` or backtick renders as literal punctuation mid-sentence.
      for (const section of allSections()) {
        for (const block of section.blocks) {
          const texts: string[] = [];
          if (block.kind === 'text' || block.kind === 'heading') texts.push(block.text);
          if (block.kind === 'list' || block.kind === 'steps') texts.push(...block.items);
          if (block.kind === 'note') texts.push(block.text);
          if (block.kind === 'table') texts.push(...block.rows.flat());

          for (const text of texts) {
            const bold = (text.match(/\*\*/g) ?? []).length;
            expect(bold % 2, `${section.id}: unclosed ** in "${text.slice(0, 60)}…"`).toBe(0);
            const ticks = (text.match(/`/g) ?? []).length;
            expect(ticks % 2, `${section.id}: unclosed backtick in "${text.slice(0, 60)}…"`).toBe(0);

            // The colour marks. An unclosed one renders as literal `{+` in the
            // middle of a sentence, which looks like a bug in the guide.
            for (const [open, close, name] of [
              ['{+', '+}', 'green'],
              ['{-', '-}', 'red'],
            ]) {
              const opens = text.split(open).length - 1;
              const closes = text.split(close).length - 1;
              expect(
                opens,
                `${section.id}: ${opens} ${name} opens but ${closes} closes in "${text.slice(0, 60)}…"`,
              ).toBe(closes);
            }
          }
        }
      }
    });
  });

  describe('uses colour to mean something', () => {
    /** Every text the panel renders through `RichText`. */
    const prose = allSections().flatMap((section) =>
      section.blocks.flatMap((block) => {
        if (block.kind === 'text' || block.kind === 'heading') return [block.text];
        if (block.kind === 'list' || block.kind === 'steps') return block.items;
        if (block.kind === 'note') return [block.text];
        if (block.kind === 'table') return block.rows.flat();
        return [];
      }),
    );

    const greens = prose.flatMap((t) => [...t.matchAll(/\{\+([^}]+)\+\}/g)].map((m) => m[1]));
    const reds = prose.flatMap((t) => [...t.matchAll(/\{-([^}]+)-\}/g)].map((m) => m[1]));

    it('marks what can be relied on, and what will cost you', () => {
      expect(greens.length, 'nothing is marked as reliable').toBeGreaterThan(5);
      expect(reds.length, 'nothing is marked as a limit').toBeGreaterThan(5);
    });

    it('keeps colour rare enough to mean something', () => {
      // A page where everything is coloured says nothing. This is a ceiling on
      // enthusiasm, not a target.
      const marked = greens.length + reds.length;
      expect(
        marked / prose.length,
        `${marked} marks across ${prose.length} passages — too much of the guide is coloured`,
      ).toBeLessThan(0.5);
    });

    it('never marks an empty or whitespace-only span', () => {
      for (const span of [...greens, ...reds]) {
        expect(span.trim().length, `empty colour mark: "${span}"`).toBeGreaterThan(0);
      }
    });

    it('marks the mixture-of-experts shortfall in red, since assuming it costs you', () => {
      const accuracy = allSections().find((s) => s.id === 'accuracy')!;
      const text = JSON.stringify(accuracy);
      expect(text).toMatch(/\{-[^}]*22 %[^}]*-\}/);
    });

    it('does not colour a limit green or a strength red', () => {
      // A narrow check on phrases that can only mean one thing. Deliberately
      // not a keyword sweep: "cannot" is a limitation in "NEURAX cannot export
      // PyTorch" and a guarantee in "switching tabs cannot change your
      // architecture", and a test that cannot tell them apart would push the
      // guide towards worse sentences.
      for (const span of greens) {
        expect(span, `"${span}" is marked as a strength`).not.toMatch(
          /\bnot verified\b|\bunder-count\b|\blower bound\b/i,
        );
      }
      for (const span of reds) {
        expect(span, `"${span}" is marked as a limit`).not.toMatch(/\bfully verified\b/i);
      }
    });
  });

  describe('describes the application that exists', () => {
    /** Every shortcut the guide lists, normalised. */
    const documentedKeys = allSections()
      .flatMap((s) => s.blocks)
      .flatMap((b) => (b.kind === 'keys' ? b.items.map((i) => i.keys) : []))
      .flatMap((k) => k.split('/').map((part) => part.trim()))
      .filter(Boolean);

    it('lists only shortcuts the application actually binds', () => {
      const page = source('pages/Index.tsx');
      const canvas = source('components/canvas/ArchitectureCanvas.tsx');
      const bound = `${page}\n${canvas}`;

      // Map a documented chord onto the literal the handler compares against.
      const keyLiteral: Record<string, string> = {
        'Ctrl+S': "'s'",
        'Ctrl+Shift+S': "'s'",
        'Ctrl+O': "'o'",
        'Ctrl+Z': "'z'",
        'Ctrl+Shift+Z': "'z'",
        'Ctrl+Y': "'y'",
        'Ctrl+A': "'a'",
        'Ctrl+D': "'d'",
        'Ctrl+G': "'g'",
        'Delete': "'Delete'",
        'Backspace': "'Backspace'",
        'Escape': "'Escape'",
        'F1': "'F1'",
        // Canvas modifiers, bound by state rather than by a key comparison.
        'Space': "'Space'",
        'Shift': 'shiftKey',
      };

      for (const chord of documentedKeys) {
        const literal = keyLiteral[chord];
        expect(literal, `the guide documents "${chord}", which this test does not know about`).toBeDefined();
        expect(
          bound.includes(literal!),
          `the guide documents ${chord}, but no handler compares against ${literal}`,
        ).toBe(true);
      }
    });

    it('documents every shortcut the document handler binds', () => {
      // The other direction: a binding nobody wrote down is a feature nobody
      // finds.
      for (const chord of ['Ctrl+S', 'Ctrl+O', 'Ctrl+Z', 'Ctrl+Y', 'F1']) {
        expect(
          documentedKeys.includes(chord),
          `${chord} is bound but not in the guide`,
        ).toBe(true);
      }
    });

    it('names the five workspaces the application has', () => {
      const tabs = source('components/layout/WorkspaceTabs.tsx');
      const labels = [...tabs.matchAll(/label: '([^']+)'/g)].map((m) => m[1]);
      expect(labels.length).toBe(5);

      const guide = JSON.stringify(DOCUMENTATION);
      for (const label of labels) {
        expect(guide.includes(label), `the guide never mentions the "${label}" workspace`).toBe(true);
      }
    });

    it('promises only the export formats the export panel offers', () => {
      const panel = source('components/panels/ExportPanel.tsx');
      const offered = [...panel.matchAll(/\bname: '([^']+)'/g)].map((m) => m[1]);

      const exporting = allSections().find((s) => s.id === 'exporting');
      expect(exporting).toBeDefined();

      const table = exporting!.blocks.find((b) => b.kind === 'table');
      expect(table?.kind).toBe('table');
      const documented = (table as { rows: string[][] }).rows.map((r) =>
        r[0].replace(/\*/g, ''),
      );

      for (const format of documented) {
        expect(offered.includes(format), `the guide promises a "${format}" export that does not exist`).toBe(true);
      }
    });

    it('states the profile path the desktop build actually uses', () => {
      const persistence = readFileSync(
        join(__dirname, '..', '..', '..', 'neurax-service', 'src', 'persistence.rs'),
        'utf8',
      );
      const appId = persistence.match(/APP_ID: &str = "([^"]+)"/)?.[1];
      const file = persistence.match(/PROJECTS_FILE: &str = "([^"]+)"/)?.[1];

      // Without these, a renamed constant makes the regex return undefined and
      // the assertions below search the guide for the literal "undefined" —
      // failing with a message that points nowhere near the real cause.
      expect(appId, 'APP_ID not found in persistence.rs — has it been renamed?').toBeDefined();
      expect(file, 'PROJECTS_FILE not found in persistence.rs — has it been renamed?').toBeDefined();

      const guide = JSON.stringify(DOCUMENTATION);
      expect(guide.includes(appId!), `the guide should name the profile directory ${appId}`).toBe(true);
      expect(guide.includes(file!), `the guide should name the projects file ${file}`).toBe(true);
    });
  });

  describe('is honest about what NEURAX cannot do', () => {
    it('has a section on accuracy and limits', () => {
      const accuracy = allSections().find((s) => s.id === 'accuracy');
      expect(accuracy, 'the guide must state its own limits').toBeDefined();
    });

    it('records the mixture-of-experts under-count rather than hiding it', () => {
      // Verified against the running compiler in
      // `huggingfaceImporter.integration.test.ts`. If that gap is ever closed,
      // both this test and the guide should change together.
      const guide = JSON.stringify(DOCUMENTATION).toLowerCase();
      expect(guide).toMatch(/under-count|lower bound/);
      expect(guide).toMatch(/mixtral/);
    });

    it('is accurate about the PyTorch export that exists — real, but checked, not guaranteed', () => {
      // A real, verified `projectExport.ts` codegen path shipped after this
      // test was first written to guard against a PyTorch export the guide
      // shouldn't promise. The guide should now describe it accurately —
      // not deny it exists, and not oversell it as guaranteed-correct either.
      const guide = JSON.stringify(DOCUMENTATION);
      const exporting = allSections().find((s) => s.id === 'exporting')!;
      const text = JSON.stringify(exporting);
      expect(text).toMatch(/Full Project/);
      expect(text).toMatch(/verified|needs review/i);
      expect(guide).not.toMatch(/Export to PyTorch/i);
    });
  });

  describe('can be searched', () => {
    it('finds a section by a word in its title', () => {
      const hits = searchDocs('undo');
      expect(hits.length).toBeGreaterThan(0);
      expect(hits[0].id).toBe('undo');
    });

    it('finds a section by a word in its body', () => {
      expect(searchDocs('config.json').some((s) => s.id === 'import-hf')).toBe(true);
      expect(searchDocs('vram').length).toBeGreaterThan(0);
    });

    it('requires every term, rather than any', () => {
      const both = searchDocs('mixture experts');
      const nonsense = searchDocs('mixture zzzznotaword');
      expect(both.length).toBeGreaterThan(0);
      expect(nonsense).toEqual([]);
    });

    it('returns nothing for an empty query rather than everything', () => {
      expect(searchDocs('')).toEqual([]);
      expect(searchDocs('   ')).toEqual([]);
    });

    it('is case-insensitive', () => {
      expect(searchDocs('LLAMA').length).toBe(searchDocs('llama').length);
    });
  });

    it('names no workspace the application does not have', () => {
      // The drift this catches: Inference Intelligence was removed and the
      // guide went on describing its widgets in the accuracy section, where
      // no test was looking. Any workspace named in prose has to be a tab
      // that exists.
      const tabs = source('components/layout/WorkspaceTabs.tsx');
      const labels = new Set([...tabs.matchAll(/label: '([^']+)'/g)].map((m) => m[1]));
      const guide = JSON.stringify(DOCUMENTATION);
      for (const removed of ['Inference Intelligence', 'Comparison workspace']) {
        if (!labels.has(removed)) {
          expect(
            guide.includes(removed),
            `the guide names "${removed}", which the application no longer has`,
          ).toBe(false);
        }
      }
    });

    it('lists only the numeric precisions the target panel offers', () => {
      // `fp8` was documented and never offered — a width a reader would go
      // looking for and not find.
      const panel = source('components/panels/SimulationTargetPanel.tsx');
      const offered = new Set(
        [...panel.matchAll(/value: '(fp\d+|bf\d+|int\d+)'/g)].map((m) => m[1]),
      );
      expect(offered.size, 'no precisions were found in the target panel').toBeGreaterThan(0);

      const guide = JSON.stringify(DOCUMENTATION);
      for (const width of ['fp8', 'fp64', 'int2']) {
        if (!offered.has(width)) {
          expect(
            guide.includes(width),
            `the guide names "${width}", which the target panel does not offer`,
          ).toBe(false);
        }
      }
    });

    it('does not claim the analysis is never checked against a run', () => {
      // True while NEURAX only predicted. The Training workspace exists to do
      // exactly this, so the claim now reads as a missing feature.
      const guide = JSON.stringify(DOCUMENTATION);
      expect(guide).not.toContain('No measured run has been compared');
      expect(guide).not.toContain('It never runs your model');
    });
});
