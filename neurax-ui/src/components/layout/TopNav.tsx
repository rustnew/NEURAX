import { useState } from 'react';
import {
  ChevronDown,
  Play,
  Plus,
  Save,
  Trash2,
  CreditCard,
  Upload,
  BookOpen,
  Menu,
  MessageSquareText,
  FolderOpen,
  CloudUpload,
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { ArchitectureSelector } from '@/components/architecture/ArchitectureSelector.tsx';
import { AuthControl } from '@/components/auth/AuthControl.tsx';
import { ThemeToggle } from '@/components/layout/ThemeToggle.tsx';
import { VariantPresetsPanel } from '@/components/catalog/VariantPresetsPanel.tsx';
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
  onRunAnalysis: () => void;
  isAnalyzing: boolean;
  onNewCanvas?: () => void;
  onSaveCanvas?: () => void;
  onExport?: () => void;
  onImport?: () => void;
  isChatOpen?: boolean;
  onToggleChat?: () => void;
  selectedArchitecture: ArchitectureFamily;
  onArchitectureChange: (family: ArchitectureFamily) => void;
  onOpenPricing?: () => void;
  onLoadPreset?: (preset: VariantPreset) => void;
  onClearCanvas?: () => void;
  currentPresetId?: string | null;
  nodes?: CanvasNode[];
  connections?: Connection[];
  // Project management
  projects?: Project[];
  currentProjectId?: string | null;
  onSaveProject?: () => void;
  onLoadProject?: (project: Project) => void;
  onDeleteProject?: (projectId: string) => void;
  isProjectsLoading?: boolean;
}

export function TopNav({
  onRunAnalysis,
  isAnalyzing,
  onNewCanvas,
  onSaveCanvas,
  onExport,
  onImport,
  isChatOpen,
  onToggleChat,
  selectedArchitecture,
  onArchitectureChange,
  onOpenPricing,
  onLoadPreset,
  onClearCanvas,
  currentPresetId,
  nodes = [],
  connections = [],
  projects = [],
  currentProjectId,
  onSaveProject,
  onLoadProject,
  onDeleteProject,
  isProjectsLoading,
}: TopNavProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const canClearCanvas = nodes.length > 0 || connections.length > 0;

  return (
    <header className="h-12 bg-card border-b border-border flex items-center justify-between px-2 sm:px-4">
      {/* Left - Logo & Name */}
      <div className="flex items-center gap-2 sm:gap-3 shrink-0">
        <h1 className="text-xs sm:text-sm font-semibold tracking-tight">
          <span className="text-gradient-primary">NEURAX</span>
        </h1>

        <AuthControl />
      </div>

      {/* Center - Architecture Selector & Actions (desktop) */}
      <div className="hidden md:flex items-center gap-4">
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
          <PopoverContent className="w-80 p-0 max-h-[500px]" align="start" onOpenAutoFocus={(e) => e.preventDefault()}>
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

        <div className="h-6 w-px bg-border" />

        <nav className="flex items-center gap-1">
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
            onClick={onImport}
            title="Import an existing architecture from JSON"
          >
            <Upload className="w-4 h-4 mr-1.5" />
            Import
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
            onClick={onSaveCanvas}
            disabled={!canClearCanvas}
            title="Download the current architecture as a .neurax.json file"
          >
            <Save className="w-4 h-4 mr-1.5" />
            Save
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
            onClick={onExport}
          >
            Export
            <ChevronDown className="w-3.5 h-3.5 ml-1" />
          </Button>
        </nav>


      </div>

      {/* Right - Run Analysis & Actions */}
      <div className="flex items-center gap-1 sm:gap-2">
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
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground"
          onClick={onOpenPricing}
        >
          <CreditCard className="w-4 h-4 mr-1.5" />
          <span className="hidden sm:inline">Pricing</span>
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
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onImport?.(); setMobileMenuOpen(false); }}>
                <Upload className="w-4 h-4 mr-2" /> Import
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => {
                  onSaveCanvas?.();
                  setMobileMenuOpen(false);
                }}
                disabled={!canClearCanvas}
              >
                <Save className="w-4 h-4 mr-2" /> Save
              </Button>
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
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onExport?.(); setMobileMenuOpen(false); }}>
                <ChevronDown className="w-4 h-4 mr-2" /> Export
              </Button>
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => { onOpenPricing?.(); setMobileMenuOpen(false); }}>
                <CreditCard className="w-4 h-4 mr-2" /> Pricing
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
