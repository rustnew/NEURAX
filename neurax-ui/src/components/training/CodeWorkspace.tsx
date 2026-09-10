/**
 * The project, as files — read, edited, and held to the same check.
 *
 * Until now the code NEURAX generates was visible only inside the Export
 * panel, on the way out of the product. That is the wrong place for it: the
 * code is the thing being trained, and it belongs beside the run, where you
 * can read what is about to happen and change it.
 *
 * Only `model.py` is editable, and that is a statement about what is true
 * rather than a limitation. A run takes the model source and a class name;
 * everything else in the folder — the harness, the requirements, the README —
 * is regenerated from the canvas for export and is *not* what the runtime
 * uses. Letting someone edit `train.py` here and quietly ignoring it would be
 * a lie told by a text box.
 *
 * An edit is not trusted for having been typed by a human. It goes through
 * exactly the gate the assistant's code goes through: PyTorch builds it, the
 * parameter count it really has is confronted with the analysis, and one batch
 * of the declared input is pushed through it. Three authors now — the
 * translator, the assistant, and you — and one check.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Check, CircleAlert, FileCode, FileText, Loader2, Package, RotateCcw, ShieldCheck,
} from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';
import type { CandidateReview } from '@/services/modelCandidate.ts';
import { cn } from '@/lib/utils.ts';

import { CodeEditor } from './CodeEditor.tsx';

export interface ProjectFile {
  path: string;
  content: string;
}

interface CodeWorkspaceProps {
  files: ProjectFile[];
  /** The one file the run actually trains. */
  modelPath: string;
  modelClassName: string;
  /**
   * Build this code and say what it is. Absent when the studio has no service
   * to ask — then the editor still reads, and says why it cannot verify.
   */
  onVerify?: (code: string) => Promise<CandidateReview>;
  /** Who wrote the model a run would use right now. */
  activeSource: 'generator' | 'assistant' | 'you';
  /** The verdict on that model, when it has one. */
  activeReview?: CandidateReview | null;
}

const SOURCE_LABEL: Record<CodeWorkspaceProps['activeSource'], string> = {
  generator: 'Translated from your canvas',
  assistant: 'Written by the assistant',
  you: 'Edited by you',
};

function iconFor(path: string) {
  if (path.endsWith('.py')) return FileCode;
  if (path.endsWith('.json') || path.endsWith('.txt')) return Package;
  return FileText;
}

/** `src/model.py` → `src`. Top-level files group under the project root. */
function directoryOf(path: string): string {
  const cut = path.lastIndexOf('/');
  return cut === -1 ? '' : path.slice(0, cut);
}

export function CodeWorkspace({
  files,
  modelPath,
  modelClassName,
  onVerify,
  activeSource,
  activeReview = null,
}: CodeWorkspaceProps) {
  const [selectedPath, setSelectedPath] = useState(modelPath);
  const [draft, setDraft] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  /**
   * The verdict, and the exact text it is about.
   *
   * Kept together on purpose. A verdict held on its own goes on reading
   * "accepted" over text that has since been typed over — which is the most
   * misleading thing this view could do, and it is one keystroke away.
   */
  const [draftReview, setDraftReview] = useState<{ code: string; review: CandidateReview } | null>(null);

  const generated = useMemo(
    () => files.find((f) => f.path === modelPath)?.content ?? '',
    [files, modelPath],
  );

  // An editor showing yesterday's model beside today's analysis is worse than
  // no editor, so an untouched draft follows the canvas. `draft === null` does
  // that on its own; this covers the case where someone typed and then undid
  // their change back to the generated text — the draft is no longer null but
  // no longer differs either, and it must not pin the editor to a stale file.
  useEffect(() => {
    setDraft((current) => (current === generated ? null : current));
  }, [generated]);

  const modelText = draft ?? generated;
  const dirty = draft !== null && draft !== generated;

  const selected = useMemo(
    () => files.find((f) => f.path === selectedPath) ?? files[0] ?? null,
    [files, selectedPath],
  );
  const isModelFile = selected?.path === modelPath;

  const grouped = useMemo(() => {
    const byDirectory = new Map<string, ProjectFile[]>();
    for (const file of files) {
      const directory = directoryOf(file.path);
      const bucket = byDirectory.get(directory);
      if (bucket) bucket.push(file);
      else byDirectory.set(directory, [file]);
    }
    // Root first, then directories alphabetically — the order someone reading
    // an unfamiliar repository expects.
    return [...byDirectory.entries()].sort(([a], [b]) => (a === '' ? -1 : b === '' ? 1 : a.localeCompare(b)));
  }, [files]);

  const verify = useCallback(async () => {
    if (!onVerify) return;
    setVerifying(true);
    const review = await onVerify(modelText).catch(() => null);
    setDraftReview(review ? { code: modelText, review } : null);
    setVerifying(false);
  }, [modelText, onVerify]);

  const revert = useCallback(() => {
    setDraft(null);
    setDraftReview(null);
  }, []);

  // The verdict on screen is about the text on screen, or there is none.
  const reviewShown =
    draftReview?.code === modelText ? draftReview.review : dirty ? null : activeReview;

  return (
    <div className="flex flex-col gap-3">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-foreground">The project, as it will be trained</h3>
          <p className="text-xs text-muted-foreground">
            {SOURCE_LABEL[activeSource]} · <code className="font-mono">{modelClassName}</code> in{' '}
            <code className="font-mono">{modelPath}</code> is what a run instantiates.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {dirty ? (
            <Button variant="ghost" size="sm" onClick={revert} className="h-8 text-xs">
              <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
              Discard edit
            </Button>
          ) : null}
          <Button
            size="sm"
            className="h-8 text-xs"
            disabled={!onVerify || verifying || !dirty}
            onClick={() => void verify()}
            title={
              !onVerify
                ? 'The local NEURAX service is not answering, so nothing can be built here.'
                : !dirty
                  ? 'Nothing has been changed yet.'
                  : 'Build this file and compare it with the analysis'
            }
          >
            {verifying ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <ShieldCheck className="mr-1.5 h-3.5 w-3.5" />
            )}
            {verifying ? 'Building…' : 'Verify and use'}
          </Button>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[210px_minmax(0,1fr)]">
        <nav aria-label="Project files" className="rounded-lg border border-border bg-card p-1.5">
          {grouped.map(([directory, entries]) => (
            <div key={directory || 'root'} className="mb-1 last:mb-0">
              {directory ? (
                <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground/70">
                  {directory}
                </div>
              ) : null}
              {entries.map((file) => {
                const Icon = iconFor(file.path);
                const isActive = selected?.path === file.path;
                const isModel = file.path === modelPath;
                return (
                  <button
                    key={file.path}
                    type="button"
                    onClick={() => setSelectedPath(file.path)}
                    aria-current={isActive}
                    className={cn(
                      'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left font-mono text-[11px] transition-colors',
                      isActive
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-secondary hover:text-foreground',
                    )}
                  >
                    <Icon className="h-3.5 w-3.5 shrink-0" />
                    <span className="truncate">{file.path.split('/').pop()}</span>
                    {isModel && dirty ? (
                      <span
                        title="Edited, not yet verified"
                        className={cn(
                          'ml-auto h-1.5 w-1.5 shrink-0 rounded-full',
                          isActive ? 'bg-primary-foreground' : 'bg-amber-500',
                        )}
                      />
                    ) : null}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-border">
          <div className="flex items-center justify-between gap-3 border-b border-border bg-muted/40 px-3 py-1.5">
            <span className="truncate font-mono text-[11px] text-muted-foreground">{selected?.path ?? ''}</span>
            <span className="shrink-0 text-[10px] text-muted-foreground/70">
              {isModelFile
                ? dirty
                  ? 'edited'
                  : 'editable'
                : 'read-only — regenerated from the canvas'}
            </span>
          </div>

          <CodeEditor
            key={selected?.path}
            value={isModelFile ? modelText : selected?.content ?? ''}
            onChange={isModelFile ? setDraft : undefined}
            readOnly={!isModelFile}
            language={selected?.path.endsWith('.py') ? 'python' : 'plain'}
            ariaLabel={selected?.path ?? 'file'}
            className="h-[min(60vh,560px)]"
          />

          {isModelFile ? (
            <div
              className={cn(
                'flex items-start gap-2 border-t border-border px-3 py-2 text-[11px] leading-relaxed',
                reviewShown === null
                  ? 'bg-muted/30 text-muted-foreground'
                  : reviewShown.accepted
                    ? 'bg-emerald-500/5 text-emerald-700 dark:text-emerald-400'
                    : 'bg-red-500/5 text-red-700 dark:text-red-400',
              )}
            >
              {reviewShown === null ? (
                <>
                  <FileCode className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>
                    {dirty
                      ? 'Changed and not yet built. Verify before a run uses it — a model that trains with a different parameter count is a different model.'
                      : 'As generated. Edit it and verify to train something else.'}
                  </span>
                </>
              ) : (
                <>
                  {reviewShown.accepted ? (
                    <Check className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  ) : (
                    <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  )}
                  <span className="min-w-0 whitespace-pre-wrap break-words">{reviewShown.reason}</span>
                </>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
