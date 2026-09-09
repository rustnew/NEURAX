/**
 * Choosing the data, from the toolbar.
 *
 * It belongs beside Templates rather than inside a tab, and the reason is what
 * the two controls have in common: a template decides what the design starts
 * as, and a dataset decides what it has to be shaped for. Both are chosen once,
 * early, and both change every number downstream. Burying the dataset inside
 * the Training tab would have implied it only matters when you train — but the
 * number of classes, the input channels and the image size are what the
 * *design* is built around, long before a run exists.
 *
 * The picker opens the real file dialog. NEURAX cannot profile the file yet, so
 * the statistics that come back are generated — but the choice is genuine: the
 * name is the file the user actually picked, and the kind is read from its real
 * extension. A fake picker that ignored the selection would teach the wrong
 * thing about how this works.
 *
 * Both a file and a folder are offered, because the two shapes of dataset are
 * genuinely different objects: a table is one file, and an image dataset is a
 * directory whose subfolder names *are* the class labels. Asking for a folder
 * through a file dialog is the standard way that goes wrong.
 */
import { useRef, useState } from 'react';
import { ChevronDown, Database, FileSpreadsheet, FolderOpen, Images, Braces } from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover.tsx';
import { formatCount } from '@/services/format.ts';
import type { DatasetKind, DatasetProfile } from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

/**
 * What NEURAX will read, by extension.
 *
 * Listed rather than left to a filter string, because "supported formats" is a
 * question users ask before they go looking for a file, and an `accept`
 * attribute answers it only once the dialog is already open.
 */
export const DATASET_FORMATS: {
  kind: DatasetKind;
  label: string;
  extensions: string[];
  note: string;
  icon: typeof Database;
}[] = [
  {
    kind: 'csv',
    label: 'Tabular',
    extensions: ['.csv', '.tsv', '.parquet'],
    note: 'One row per sample. Column types and missing values are read from the header and a scan.',
    icon: FileSpreadsheet,
  },
  {
    kind: 'jsonl',
    label: 'Records',
    extensions: ['.jsonl', '.ndjson', '.json'],
    note: 'One record per line. Keys become fields; nested objects are flattened.',
    icon: Braces,
  },
  {
    kind: 'image-folder',
    label: 'Images',
    extensions: ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif'],
    note: 'A folder per class. The subfolder names become the labels.',
    icon: Images,
  },
];

/** Every extension the file dialog should accept, as one `accept` string. */
const ACCEPT = DATASET_FORMATS.flatMap((f) => f.extensions).join(',');

/** The kind a filename implies. `unknown` when the extension is not one of
 *  ours — said plainly rather than guessed at. */
export function kindForFilename(name: string): DatasetKind {
  const dot = name.lastIndexOf('.');
  if (dot < 0) return 'unknown';
  const ext = name.slice(dot).toLowerCase();
  const match = DATASET_FORMATS.find((f) => f.extensions.includes(ext));
  return match ? match.kind : 'unknown';
}

export interface DatasetSelection {
  displayPath: string;
  kind: DatasetKind;
  fileCount: number;
}

/**
 * The file and folder dialogs, shared.
 *
 * Two places offer this choice — the toolbar, and the Training panel's own
 * "Change" — and they must be the same choice. Duplicating the inputs would
 * have been the easy version and the one where the two drift apart.
 */
export function useDatasetSelection(onPick: (selection: DatasetSelection) => void) {
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null, asFolder: boolean) => {
    if (!files || files.length === 0) return;
    if (asFolder) {
      // `webkitRelativePath` is "<root>/<class>/<file>" — the root is the
      // dataset, and the level below it is what the labels come from.
      const root = files[0].webkitRelativePath.split('/')[0] || 'dataset';
      onPick({ displayPath: root, kind: 'image-folder', fileCount: files.length });
    } else {
      onPick({ displayPath: files[0].name, kind: kindForFilename(files[0].name), fileCount: 1 });
    }
  };

  const inputs = (
    <>
      <input
        ref={fileInput}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files, false)}
      />
      <input
        ref={folderInput}
        type="file"
        // Non-standard but universally supported, and the only way a browser
        // will hand over a directory.
        {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
        multiple
        className="hidden"
        onChange={(e) => handleFiles(e.target.files, true)}
      />
    </>
  );

  return {
    inputs,
    chooseFile: () => fileInput.current?.click(),
    chooseFolder: () => folderInput.current?.click(),
  };
}

interface DatasetPickerProps {
  profile: DatasetProfile | null;
  /** Given what the user actually picked: the display path and the kind read
   *  from its extension (or `image-folder` when a directory was chosen). */
  onPick: (selection: DatasetSelection) => void;
}

export function DatasetPicker({ profile, onPick }: DatasetPickerProps) {
  const [open, setOpen] = useState(false);
  const { inputs, chooseFile, chooseFolder } = useDatasetSelection((selection) => {
    onPick(selection);
    setOpen(false);
  });

  return (
    <>
      {inputs}

      <Popover open={open} onOpenChange={setOpen} modal={false}>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className={cn('text-muted-foreground hover:text-foreground', profile && 'text-foreground')}
            title={profile ? `${profile.displayPath} — ${formatCount(profile.samples)} samples` : 'Choose the data this model is built for'}
          >
            <Database className={cn('w-4 h-4 mr-1.5', profile && 'text-primary')} />
            <span className="max-w-[112px] truncate">{profile ? profile.displayPath.split("/").pop() : "Dataset"}</span>
            <ChevronDown className="w-3.5 h-3.5 ml-1 shrink-0" />
          </Button>
        </PopoverTrigger>

        <PopoverContent className="w-[340px] p-0" align="start" onOpenAutoFocus={(e) => e.preventDefault()}>
          <div className="px-4 py-3 border-b border-border">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Dataset</p>
            <p className="mt-1 text-[11px] text-muted-foreground leading-relaxed">
              The shape of your data fills in the fields the design needs — classes, channels, image size — instead of
              you typing them.
            </p>
          </div>

          {profile ? (
            <div className="px-4 py-2.5 border-b border-border bg-primary/[0.04]">
              <div className="text-[12px] font-semibold text-foreground truncate">{profile.displayPath}</div>
              <div className="mt-0.5 flex flex-wrap gap-x-3 text-[10px] font-mono text-muted-foreground">
                <span>{formatCount(profile.samples)} samples</span>
                {profile.suggested.numClasses ? <span>{profile.suggested.numClasses} classes</span> : null}
                {profile.sampleShape ? <span>{profile.sampleShape.join('×')}</span> : null}
              </div>
            </div>
          ) : null}

          <div className="p-3 flex gap-2">
            <Button size="sm" variant="outline" className="flex-1" onClick={chooseFile}>
              <FileSpreadsheet className="w-3.5 h-3.5 mr-1.5" />
              Choose a file
            </Button>
            <Button size="sm" variant="outline" className="flex-1" onClick={chooseFolder}>
              <FolderOpen className="w-3.5 h-3.5 mr-1.5" />
              Choose a folder
            </Button>
          </div>

          <div className="px-4 pb-3">
            <p className="text-[9px] font-mono uppercase tracking-[0.12em] text-muted-foreground/70 mb-2">
              What NEURAX reads
            </p>
            <div className="flex flex-col gap-2.5">
              {DATASET_FORMATS.map((format) => {
                const Icon = format.icon;
                return (
                  <div key={format.kind} className="flex gap-2.5">
                    <Icon className="w-3.5 h-3.5 mt-0.5 shrink-0 text-muted-foreground/60" />
                    <div className="min-w-0">
                      <div className="flex items-baseline gap-2 flex-wrap">
                        <span className="text-[11px] font-medium text-foreground">{format.label}</span>
                        <span className="font-mono text-[10px] text-muted-foreground/70">
                          {format.extensions.join(' ')}
                        </span>
                      </div>
                      <p className="text-[10px] text-muted-foreground leading-snug mt-0.5">{format.note}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="px-4 py-2 border-t border-border">
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              Structure and statistics only. Your data is never read for its content and never leaves this machine.
            </p>
          </div>
        </PopoverContent>
      </Popover>
    </>
  );
}
