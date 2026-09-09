/**
 * The dataset the model is being built for.
 *
 * NEURAX has never looked at data. `data.dataset_size` is a single scalar the
 * Cost phase reads to estimate a duration, and every shape field — number of
 * classes, channels, image size — is typed in by hand. This panel is where
 * that changes: point at a file or a folder, and the fields fill themselves.
 *
 * What is shown, and what is deliberately not:
 *
 *  - **Structure and statistics only.** There is no preview, no first rows, no
 *    thumbnails. Not an oversight — a design rule. NEURAX reads how a dataset
 *    is shaped, never what it contains, and nothing about it leaves the
 *    machine. The fingerprint is a digest.
 *  - **Integrity is stated, never fatal.** A few unreadable files or images at
 *    the wrong size do not block a profile; they are counted and named, because
 *    finding out at step 340 of a training run is the expensive way to learn it.
 *  - **The derived fields are the point.** `Suggested configuration` is what
 *    empties `missing_mandatory_fields` — the panel exists to fill the form the
 *    user would otherwise fill.
 */
import { Database, FileWarning, FolderOpen, Hash, Layers3 } from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';
import { formatBytes, formatCount } from '@/services/format.ts';
import type { ClassCount, DatasetProfile } from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

interface DatasetPanelProps {
  profile: DatasetProfile | null;
  onChoose: () => void;
  isExample?: boolean;
}

const KIND_LABEL: Record<DatasetProfile['kind'], string> = {
  csv: 'Tabular · CSV',
  'image-folder': 'Images · one folder per class',
  jsonl: 'Records · JSONL',
  unknown: 'Unrecognised',
};

/** Class balance as bars. The imbalance is the reading that changes a design —
 *  a 4:1 majority class is a training decision, not a footnote — so it is
 *  drawn rather than tabulated. */
function ClassBalance({ classes }: { classes: ClassCount[] }) {
  const total = classes.reduce((sum, c) => sum + c.count, 0);
  const largest = Math.max(...classes.map((c) => c.count));
  const ratio = largest / Math.min(...classes.map((c) => c.count));

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground">
          Class balance
        </span>
        <span
          className={cn(
            'text-[11px] font-mono tabular-nums',
            ratio >= 3 ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground',
          )}
        >
          {ratio.toFixed(1)}:1 majority to minority
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {classes.map((c) => (
          <div key={c.label} className="grid grid-cols-[minmax(64px,110px)_1fr_auto] items-center gap-2">
            <span className="text-[12px] text-foreground truncate">{c.label}</span>
            <span className="h-2.5 rounded-[3px] bg-muted overflow-hidden">
              <span
                className="block h-full bg-primary/70"
                style={{ width: `${(c.count / largest) * 100}%` }}
              />
            </span>
            <span className="text-[11px] font-mono tabular-nums text-muted-foreground">
              {formatCount(c.count)}
              <span className="text-muted-foreground/50"> · {((c.count / total) * 100).toFixed(0)}%</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground/70">{label}</span>
      <span className="text-[16px] font-semibold tabular-nums text-foreground">{value}</span>
      {detail ? <span className="text-[11px] font-mono text-muted-foreground/60">{detail}</span> : null}
    </div>
  );
}

export function DatasetPanel({ profile, onChoose, isExample }: DatasetPanelProps) {
  if (!profile) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 p-8 text-center">
        <Database className="w-7 h-7 text-muted-foreground/40" />
        <div>
          <div className="text-[15px] font-semibold text-foreground">No dataset selected</div>
          <p className="mt-1 text-[12px] text-muted-foreground max-w-[46ch] leading-relaxed">
            Point NEURAX at a file or a folder and it will read its shape — how many samples, what
            they look like, how the classes are distributed — so the model can be built for it.
          </p>
        </div>
        <Button size="sm" onClick={onChoose} className="mt-1">
          <FolderOpen className="w-3.5 h-3.5 mr-1.5" />
          Choose a file or folder
        </Button>
        <p className="text-[11px] text-muted-foreground/60 max-w-[44ch]">
          Structure and statistics only. Your data is never read for its content and never leaves
          this machine.
        </p>
      </div>
    );
  }

  const { integrity, suggested } = profile;
  const hasIssues =
    integrity.unreadableFiles + integrity.corruptHeaders + integrity.inconsistentShapes + integrity.missingValues > 0;

  return (
    <div className="flex flex-col gap-5 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[15px] font-semibold text-foreground truncate">{profile.displayPath}</span>
            {isExample ? (
              <span className="rounded-[4px] border border-border bg-background/60 px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-[0.1em] text-muted-foreground shrink-0">
                example
              </span>
            ) : null}
          </div>
          <div className="mt-0.5 text-[11px] font-mono text-muted-foreground">{KIND_LABEL[profile.kind]}</div>
        </div>
        <Button size="sm" variant="outline" onClick={onChoose} className="shrink-0">
          <FolderOpen className="w-3.5 h-3.5 mr-1.5" />
          Change
        </Button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat label="Samples" value={formatCount(profile.samples)} />
        <Stat label="On disk" value={formatBytes(profile.storageBytes)} />
        {profile.sampleShape ? (
          <Stat
            label="Sample shape"
            value={profile.sampleShape.join(' × ')}
            detail={profile.kind === 'image-folder' ? 'C × H × W' : 'features'}
          />
        ) : null}
        {profile.splits ? (
          <Stat
            label="Split"
            value={`${formatCount(profile.splits.train)} / ${formatCount(profile.splits.validation)}`}
            detail="train / validation"
          />
        ) : null}
      </div>

      {profile.classes ? <ClassBalance classes={profile.classes} /> : null}

      {profile.columns ? (
        <div>
          <div className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground mb-2">Columns</div>
          <div className="overflow-x-auto border border-border rounded-[6px]">
            <table className="w-full text-[12px] font-mono">
              <thead>
                <tr className="bg-muted/40 text-muted-foreground">
                  <th className="text-left px-3 py-2 font-medium">name</th>
                  <th className="text-left px-3 py-2 font-medium">type</th>
                  <th className="text-right px-3 py-2 font-medium">missing</th>
                </tr>
              </thead>
              <tbody>
                {profile.columns.map((c) => (
                  <tr key={c.name} className="border-t border-border/60">
                    <td className="px-3 py-2 text-foreground">{c.name}</td>
                    <td className="px-3 py-2 text-muted-foreground">{c.type}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                      {c.missingCount || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {hasIssues ? (
        <div className="rounded-[6px] border border-amber-500/40 bg-amber-500/5 p-3">
          <div className="flex items-center gap-1.5 text-[12px] font-semibold text-amber-700 dark:text-amber-400">
            <FileWarning className="w-3.5 h-3.5" />
            Integrity
          </div>
          <ul className="mt-1.5 flex flex-col gap-1 text-[12px] text-muted-foreground">
            {integrity.notes.map((note) => (
              <li key={note} className="leading-relaxed">
                {note}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="rounded-[6px] border border-primary/30 bg-primary/5 p-3">
        <div className="flex items-center gap-1.5 text-[12px] font-semibold text-primary">
          <Layers3 className="w-3.5 h-3.5" />
          Filled in from this dataset
        </div>
        <p className="mt-1 text-[11px] text-muted-foreground leading-relaxed">
          These would otherwise be typed by hand, and they are what the agent designs against.
        </p>
        <div className="mt-2.5 flex flex-wrap gap-x-5 gap-y-2">
          {suggested.numClasses !== undefined ? <Stat label="Classes" value={String(suggested.numClasses)} /> : null}
          {suggested.inChannels !== undefined ? <Stat label="Channels" value={String(suggested.inChannels)} /> : null}
          {suggested.imgHeight !== undefined && suggested.imgWidth !== undefined ? (
            <Stat label="Image size" value={`${suggested.imgHeight} × ${suggested.imgWidth}`} />
          ) : null}
          {suggested.seqLen !== undefined ? <Stat label="Sequence" value={String(suggested.seqLen)} /> : null}
          {suggested.familyHint ? <Stat label="Family hint" value={suggested.familyHint} /> : null}
        </div>
      </div>

      <div className="flex items-center gap-1.5 text-[11px] font-mono text-muted-foreground/60">
        <Hash className="w-3 h-3" />
        {profile.fingerprint}
        <span className="text-muted-foreground/40">· a digest, never a sample</span>
      </div>
    </div>
  );
}
