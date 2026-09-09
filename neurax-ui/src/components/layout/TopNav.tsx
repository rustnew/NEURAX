import { useState } from 'react';
import {
  ChevronDown,
  Play,
  Plus,
  Save,
  Trash2,
  Upload,
  BookOpen,
  Menu,
  MessageSquareText,
  FolderOpen,
  CloudUpload,
  SlidersHorizontal,
  Cpu,
  Undo2,
  Redo2,
  GitCompare,
  HelpCircle,
  Pencil,
  Share2,
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { ArchitectureSelector } from '@/components/architecture/ArchitectureSelector.tsx';
import { AuthControl } from '@/components/auth/AuthControl.tsx';
import { NeuraxLogo } from '@/components/brand/NeuraxLogo.tsx';
import { ThemeToggle } from '@/components/layout/ThemeToggle.tsx';
import { VariantPresetsPanel } from '@/components/catalog/VariantPresetsPanel.tsx';
import { DatasetPicker } from '@/components/layout/DatasetPicker.tsx';
import { cn } from '@/lib/utils.ts';
import { isCpuOnly, primaryGpu } from '@/types/runtime.ts';
import type { DatasetKind, DatasetProfile, HardwareProfile } from '@/types/runtime.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { VariantPreset } from '@/types/catalog.ts';
import { CanvasNode, Connection } from '@/types/architecture.ts';
import type { Project } from '@/services/neuraxApi.ts';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover.tsx';
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from '@/components/ui/sheet.tsx';

interface TopNavProps {
  /** The dataset the design is being built for, and how to change it. Sits
   *  beside Templates because the two answer the same kind of question: what
   *  the design starts from, and what it has to fit. */
  datasetProfile?: DatasetProfile | null;
  onPickDataset?: (selection: { displayPath: string; kind: DatasetKind; fileCount: number }) => void;
  /** What was detected on this machine. When present, the Target control is
   *  labelled with the real card instead of the word "Target" — the analysis
   *  runs against the machine you have unless you say otherwise, and the
   *  control should read that way. */
  detectedHardware?: HardwareProfile | null;
  onRunAnalysis: () => void;
  isAnalyzing: boolean;
  onNewCanvas?: () => void;
  onSaveCanvas?: () => void;
  onExport?: () => void;
  onImport?: () => void;
  onShare?: () => void;

  // ─── Document ───────────────────────────────────────────────
  /** Open a `.neurax` design from disk. */
  onOpenDesign?: () => void;
  /** Save to the design's own file, asking for a name only if it has none. */
  onSaveDesign?: () => void;
  /** Name of the open document, or undefined when it has never been saved. */
  documentName?: string;
  /** The same name, stripped of its `.neurax` extension — what the rename field shows and edits. */
  documentBaseName?: string;
  /**
   * Rename the open document. Renaming a design that came from a saved file
   * does not itself rewrite that file — same as any other unsaved change,
   * it takes effect in the document and reaches disk on the next Save.
   */
  onRenameDocument?: (name: string) => void;
  /** Whether there are changes not yet written to the file. */
  isDirty?: boolean;

  // ─── History ────────────────────────────────────────────────
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;

  /** Open the A/B comparison panel. */
  onCompare?: () => void;
  /** Whether a baseline has been captured, for the button's state. */
  hasBaseline?: boolean;

  /** Open the in-app guide. */
  onOpenDocumentation?: () => void;
  isChatOpen?: boolean;
  onToggleChat?: () => void;
  selectedArchitecture: ArchitectureFamily;
  onArchitectureChange: (family: ArchitectureFamily) => void;
  onLoadPreset?: (preset: VariantPreset) => void;
  onClearCanvas?: () => void;
  currentPresetId?: string | null;
  nodes?: CanvasNode[];
  connections?: Connection[];
  // Hyperparameter optimization
  onSelectTarget?: () => void;
  onHyperparameters?: () => void;
  // Project management
  projects?: Project[];
  currentProjectId?: string | null;
  onSaveProject?: () => void;
  onLoadProject?: (project: Project) => void;
  onDeleteProject?: (projectId: string) => void;
  isProjectsLoading?: boolean;
}

/**
 * The open document's name, editable in place.
 *
 * Click (or Enter/Space on it) to start typing; Enter or losing focus
 * commits, Escape reverts to the name before editing started. An empty
 * name is rejected rather than committed — a blank title is worse than
 * "Untitled design", which at least says what it is.
 */
export function DocumentNameField({
  name,
  onRename,
}: {
  name: string;
  onRename?: (name: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== name) onRename?.(trimmed);
    setEditing(false);
  };

  if (editing) {
    return (
      <Input
        autoFocus
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commit();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            setDraft(name);
            setEditing(false);
          }
        }}
        onFocus={(e) => e.currentTarget.select()}
        className="h-6 w-40 px-1.5 text-xs font-medium"
        aria-label="Design name"
      />
    );
  }

  return (
    <button
      type="button"
      onClick={() => {
        setDraft(name);
        setEditing(true);
      }}
      className="group flex items-center gap-1 max-w-[10rem] rounded px-1.5 py-0.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
      title="Rename this design"
      aria-label={`Rename this design (currently "${name}")`}
    >
      <span className="truncate">{name}</span>
      <Pencil className="w-3 h-3 shrink-0 opacity-0 group-hover:opacity-60" />
    </button>
  );
}

export function TopNav({
  datasetProfile,
  onPickDataset,
  detectedHardware,
  onRunAnalysis,
  isAnalyzing,
  onNewCanvas,
  onSaveCanvas,
  onExport,
  onImport,
  onShare,
  onOpenDesign,
  onSaveDesign,
  documentName,
  documentBaseName,
  onRenameDocument,
  isDirty,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  onCompare,
  hasBaseline,
  onOpenDocumentation,
  isChatOpen,
  onToggleChat,
  selectedArchitecture,
  onArchitectureChange,
  onLoadPreset,
  onClearCanvas,
  currentPresetId,
  nodes = [],
  connections = [],
  onSelectTarget,
  onHyperparameters,
  projects = [],
  currentProjectId,
  onSaveProject,
  onLoadProject,
  onDeleteProject,
  isProjectsLoading,
}: TopNavProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const canClearCanvas = nodes.length > 0 || connections.length > 0;
  const detectedGpu = primaryGpu(detectedHardware ?? null);
  const cpuOnly = isCpuOnly(detectedHardware ?? null);
  const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(0)} GB`;

  // Deliberately not a ternary chain inline in the markup: which of these
  // three a machine is decides what the whole studio analyses against, and it
  // should be readable in one place.
  const { targetLabel, targetDetail, targetTitle } = (() => {
    if (!detectedHardware) {
      return {
        targetLabel: 'Target',
        targetDetail: null as string | null,
        targetTitle: 'Detecting this machine…',
      };
    }
    if (cpuOnly) {
      return {
        targetLabel: detectedHardware.cpu.model.replace(/\(R\)|\(TM\)|CPU|@.*$/g, '').trim(),
        targetDetail: `${detectedHardware.cpu.cores}c · ${gb(detectedHardware.ramAvailableBytes)} free`,
        targetTitle: `No accelerator detected — analysing for this CPU (${detectedHardware.cpu.model}, ${detectedHardware.cpu.cores} cores, ${detectedHardware.cpu.threads} threads).`,
      };
    }
    if (detectedGpu?.integrated) {
      return {
        targetLabel: detectedGpu.name,
        targetDetail: `shares ${gb(detectedHardware.ramAvailableBytes)}`,
        targetTitle: `${detectedGpu.name} is integrated: it has no dedicated VRAM and borrows system memory.`,
      };
    }
    const free = detectedGpu?.vramFreeBytes;
    return {
      targetLabel: detectedGpu?.name ?? 'Target',
      targetDetail: free != null ? `${gb(free)} free` : null,
      targetTitle:
        free != null
          ? `Analysing for ${detectedGpu?.name} — ${(free / 1024 ** 3).toFixed(1)} GB free.`
          : `Analysing for ${detectedGpu?.name}. The driver did not report its memory.`,
    };
  })();

  return (
    <header className="h-12 bg-card border-b border-border flex items-center justify-between gap-2 px-2 sm:px-4">
      {/* Left - Logo & Name */}
      <div className="flex items-center gap-2 sm:gap-3 shrink-0">
        <NeuraxLogo size={30} variant="mark" showText={false} />
        <h1 className="text-xs sm:text-sm font-bold tracking-tight" style={{ color: 'hsl(var(--foreground))' }}>
          NEURAX
        </h1>

        <div className="hidden sm:flex items-center gap-1.5">
          <span className="text-border">/</span>
          <DocumentNameField
            name={documentBaseName ?? 'Untitled design'}
            onRename={onRenameDocument}
          />
        </div>

        <AuthControl />
      </div>

      {/* Center - Architecture Selector & Actions (desktop).
          `min-w-0` lets this group shrink below its content width, and the
          overflow scrolls, so a narrow window loses scrolling convenience
          rather than losing access to the buttons. */}
      <div className="hidden md:flex items-center gap-4 min-w-0 flex-1 overflow-x-auto scrollbar-thin">
        <ArchitectureSelector
          value={selectedArchitecture}
          onChange={onArchitectureChange}
        />

        <Popover modal={false}>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
              <BookOpen className="w-4 h-4 mr-1.5" />
              Templates
              <ChevronDown className="w-3.5 h-3.5 ml-1" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-80 p-0 max-h-[500px] overflow-hidden flex flex-col" align="start" onOpenAutoFocus={(e) => e.preventDefault()}>
            <VariantPresetsPanel
              key={selectedArchitecture}
              family={selectedArchitecture}
              onLoadPreset={(preset) => onLoadPreset?.(preset)}
              onClonePreset={(preset) => {
                const cloned = { ...preset, id: `${preset.id}-clone-${Date.now()}`, name: `${preset.name} (Custom)` };
                onLoadPreset?.(cloned);
              }}
              currentPresetId={currentPresetId}
              currentNodes={nodes}
              currentConnections={connections}
            />
          </PopoverContent>
        </Popover>

        {onPickDataset ? (
          <DatasetPicker profile={datasetProfile ?? null} onPick={onPickDataset} />
        ) : null}

        {/*
          What the analysis is computed for, named on the toolbar.

          Three states, because a machine really has three. A discrete card
          shows its free VRAM, which is the figure that decides whether a
          design starts. An integrated part has no VRAM of its own — it
          borrows system memory — so showing "0 GB free" would be a lie in the
          most alarming possible direction; it shows the shared-memory pool
          instead. A machine with no accelerator names its CPU, because that
          is genuinely what the work would run on, and pretending otherwise is
          how a user ends up designing for hardware they do not have.
        */}
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            'text-muted-foreground hover:text-foreground',
            detectedHardware && 'text-foreground',
          )}
          onClick={onSelectTarget}
          title={targetTitle}
        >
          <Cpu className={cn('w-4 h-4 mr-1.5', detectedHardware && 'text-primary')} />
          <span className="max-w-[132px] truncate">{targetLabel}</span>
          {targetDetail ? (
            <span className="ml-1.5 font-mono text-[10px] text-muted-foreground tabular-nums shrink-0">
              {targetDetail}
            </span>
          ) : null}
        </Button>

        <div className="h-6 w-px bg-border" />

        <nav className="flex items-center gap-1">
          {/* Undo and redo. Icon-only: these two arrows are the most universally
              recognised controls in any editor, and the toolbar has no room to
              spell out what everyone already knows. */}
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 text-muted-foreground hover:text-foreground"
            onClick={onUndo}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            aria-label="Undo"
          >
            <Undo2 className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 text-muted-foreground hover:text-foreground"
            onClick={onRedo}
            disabled={!canRedo}
            title="Redo (Ctrl+Shift+Z)"
            aria-label="Redo"
          >
            <Redo2 className="w-4 h-4" />
          </Button>

          <div className="h-6 w-px bg-border mx-1" />

          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
            onClick={onNewCanvas}
            disabled={!canClearCanvas}
            title="Start a fresh architecture from scratch"
          >
            <Plus className="w-4 h-4 mr-1.5" />
            New
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
            onClick={onOpenDesign}
            title="Open a .neurax design (Ctrl+O)"
          >
            <FolderOpen className="w-4 h-4 mr-1.5" />
            Open
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
            onClick={onImport}
            title="Import a HuggingFace config.json or a NEURAX design"
          >
            <Upload className="w-4 h-4 mr-1.5" />
            Import
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
            onClick={onSaveDesign ?? onSaveCanvas}
            disabled={!canClearCanvas}
            title={
              documentName
                ? `Save ${documentName} (Ctrl+S)`
                : 'Save this design to a .neurax file (Ctrl+S)'
            }
          >
            <Save className="w-4 h-4 mr-1.5" />
            Save
            {/* A dot, not a word: the toolbar has no room and every editor
                marks unsaved work this way. */}
            {/* A bare span is `role="generic"` and carries no accessible name,
                so the dot alone conveys the state by colour and shape only.
                The text is for assistive technology; the dot is for everyone
                else. */}
            {isDirty && (
              <>
                <span className="ml-1.5 w-1.5 h-1.5 rounded-full bg-primary" aria-hidden="true" />
                <span className="sr-only">unsaved changes</span>
              </>
            )}
          </Button>
          <Button
            variant={hasBaseline ? 'secondary' : 'ghost'}
            size="sm"
            className={hasBaseline ? '' : 'text-muted-foreground hover:text-foreground'}
            onClick={onCompare}
            title="Compare this design against a captured baseline"
          >
            <GitCompare className="w-4 h-4 mr-1.5" />
            Compare
          </Button>

          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground hover:text-foreground"
                title="Cloud Projects"
              >
                <FolderOpen className="w-4 h-4 mr-1.5" />
                Projects
                <ChevronDown className="w-3.5 h-3.5 ml-1" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80 p-3" align="start">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Cloud Projects</p>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={onSaveProject}
                    disabled={!onSaveProject}
                  >
                    <CloudUpload className="w-3.5 h-3.5 mr-1" />
                    Save Current
                  </Button>
                </div>
                {isProjectsLoading && (
                  <p className="text-xs text-muted-foreground py-2">Loading projects...</p>
                )}
                {!isProjectsLoading && projects.length === 0 && (
                  <p className="text-xs text-muted-foreground py-2">No saved projects yet. Click "Save Current" to save your work.</p>
                )}
                <div className="max-h-[300px] overflow-y-auto space-y-1.5">
                  {projects.map((project) => (
                    <div
                      key={project.id}
                      className={`flex items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-accent/50 cursor-pointer ${project.id === currentProjectId ? 'bg-accent' : ''}`}
                      onClick={() => onLoadProject?.(project)}
                    >
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-xs truncate">{project.name}</p>
                        <p className="text-[10px] text-muted-foreground">
                          {project.architecture || 'No architecture'} · {new Date(project.updated_at).toLocaleDateString()}
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 shrink-0 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteProject?.(project.id);
                        }}
                      >
                        <Trash2 className="w-3 h-3" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            </PopoverContent>
          </Popover>
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:text-destructive hover:bg-destructive/10"
            onClick={onClearCanvas}
            disabled={!canClearCanvas}
          >
            <Trash2 className="w-4 h-4 mr-1.5" />
            Clear
          </Button>

          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
            onClick={onHyperparameters}
          >
            <SlidersHorizontal className="w-4 h-4 mr-1.5" />
            Hyperparameters
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
            onClick={onShare}
            title="Publish a public link to this analysis"
          >
            <Share2 className="w-4 h-4 mr-1.5" />
            Share
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
            onClick={onExport}
          >
            Export
            <ChevronDown className="w-3.5 h-3.5 ml-1" />
          </Button>
        </nav>


      </div>

      {/* Right - Run Analysis & Actions. Never shrinks: this is what the page
          exists to let you do. */}
      <div className="flex items-center gap-1 sm:gap-2 shrink-0">
        {/* The guide. Icon-only and always present: help that is hard to find
            is help nobody uses. */}
        <Button
          variant="ghost"
          size="icon"
          className="w-8 h-8 text-muted-foreground hover:text-foreground"
          onClick={onOpenDocumentation}
          title="Open the NEURAX guide (F1)"
          aria-label="Open the NEURAX guide"
        >
          <HelpCircle className="w-4 h-4" />
        </Button>

        <ThemeToggle />

        <Button
          variant={isChatOpen ? 'secondary' : 'ghost'}
          size="sm"
          className="text-muted-foreground hover:text-foreground hidden sm:flex"
          onClick={onToggleChat}
        >
          <MessageSquareText className="w-4 h-4 mr-1.5" />
          Neurax AI
        </Button>

        <Button
          size="sm"
          className="bg-primary text-primary-foreground hover:bg-primary/90 shadow-glow-sm text-xs sm:text-sm"
          onClick={onRunAnalysis}
          disabled={isAnalyzing}
        >
          <Play className="w-4 h-4 sm:mr-1.5" />
          <span className="hidden sm:inline">{isAnalyzing ? 'Analyzing...' : 'Run Analysis'}</span>
        </Button>

        {/* Mobile hamburger menu */}
        <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="md:hidden w-8 h-8">
              <Menu className="w-5 h-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="w-72 p-4 space-y-4">
            <div className="space-y-2">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Architecture</p>
              <ArchitectureSelector
                value={selectedArchitecture}
                onChange={(f) => { onArchitectureChange(f); }}
              />
            </div>

            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Actions</p>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => {
                  onToggleChat?.();
                  setMobileMenuOpen(false);
                }}
              >
                <MessageSquareText className="w-4 h-4 mr-2" /> Chat with Neurax AI
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => {
                  onNewCanvas?.();
                  setMobileMenuOpen(false);
                }}
                disabled={!canClearCanvas}
              >
                <Plus className="w-4 h-4 mr-2" /> New
              </Button>
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onOpenDesign?.(); setMobileMenuOpen(false); }}>
                <FolderOpen className="w-4 h-4 mr-2" /> Open
              </Button>
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onImport?.(); setMobileMenuOpen(false); }}>
                <Upload className="w-4 h-4 mr-2" /> Import
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => {
                  (onSaveDesign ?? onSaveCanvas)?.();
                  setMobileMenuOpen(false);
                }}
                disabled={!canClearCanvas}
              >
                <Save className="w-4 h-4 mr-2" /> Save
                {isDirty && (
                  <>
                    <span className="ml-1.5 w-1.5 h-1.5 rounded-full bg-primary" aria-hidden="true" />
                    <span className="sr-only">unsaved changes</span>
                  </>
                )}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => {
                  onCompare?.();
                  setMobileMenuOpen(false);
                }}
              >
                <GitCompare className="w-4 h-4 mr-2" /> Compare
              </Button>
              <div className="flex gap-1 py-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex-1 justify-center"
                  onClick={() => onUndo?.()}
                  disabled={!canUndo}
                  aria-label="Undo"
                >
                  <Undo2 className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex-1 justify-center"
                  onClick={() => onRedo?.()}
                  disabled={!canRedo}
                  aria-label="Redo"
                >
                  <Redo2 className="w-4 h-4" />
                </Button>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => {
                  onSaveProject?.();
                  setMobileMenuOpen(false);
                }}
              >
                <CloudUpload className="w-4 h-4 mr-2" /> Save to Cloud
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-destructive hover:text-destructive hover:bg-destructive/10"
                onClick={() => {
                  onClearCanvas?.();
                  setMobileMenuOpen(false);
                }}
                disabled={!canClearCanvas}
              >
                <Trash2 className="w-4 h-4 mr-2" /> Clear Canvas
              </Button>
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onHyperparameters?.(); setMobileMenuOpen(false); }}>
                <SlidersHorizontal className="w-4 h-4 mr-2" /> Set Hyperparameters
              </Button>
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onSelectTarget?.(); setMobileMenuOpen(false); }}>
                <Cpu className="w-4 h-4 mr-2" /> Simulation Target
              </Button>
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onShare?.(); setMobileMenuOpen(false); }}>
                <Share2 className="w-4 h-4 mr-2" /> Share
              </Button>
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onExport?.(); setMobileMenuOpen(false); }}>
                <ChevronDown className="w-4 h-4 mr-2" /> Export
              </Button>

            </div>

            <div className="space-y-2">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Cloud Projects</p>
              <div className="max-h-[200px] overflow-y-auto space-y-1">
                {isProjectsLoading && <p className="text-xs text-muted-foreground py-1">Loading...</p>}
                {!isProjectsLoading && projects.length === 0 && <p className="text-xs text-muted-foreground py-1">No saved projects</p>}
                {projects.map((project) => (
                  <div
                    key={project.id}
                    className={`flex items-center justify-between rounded-md px-2 py-1.5 text-sm hover:bg-accent/50 cursor-pointer ${project.id === currentProjectId ? 'bg-accent' : ''}`}
                    onClick={() => { onLoadProject?.(project); setMobileMenuOpen(false); }}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-xs truncate">{project.name}</p>
                      <p className="text-[10px] text-muted-foreground">{project.architecture || 'N/A'}</p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-5 w-5 shrink-0 text-destructive hover:text-destructive"
                      onClick={(e) => { e.stopPropagation(); onDeleteProject?.(project.id); }}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Templates</p>
              <div className="max-h-[300px] overflow-y-auto">
                <VariantPresetsPanel
                  family={selectedArchitecture}
                  onLoadPreset={(preset) => { onLoadPreset?.(preset); setMobileMenuOpen(false); }}
                  onClonePreset={(preset) => {
                    const cloned = { ...preset, id: `${preset.id}-clone-${Date.now()}`, name: `${preset.name} (Custom)` };
                    onLoadPreset?.(cloned);
                    setMobileMenuOpen(false);
                  }}
                  currentPresetId={currentPresetId}
                  currentNodes={nodes}
                  currentConnections={connections}
                />
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </header>
  );
}
