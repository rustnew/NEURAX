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
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { ArchitectureSelector } from '@/components/architecture/ArchitectureSelector.tsx';
import { AuthControl } from '@/components/auth/AuthControl.tsx';
import { ThemeToggle } from '@/components/layout/ThemeToggle.tsx';
import { VariantPresetsPanel } from '@/components/catalog/VariantPresetsPanel.tsx';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { VariantPreset } from '@/types/catalog.ts';
import { CanvasNode, Connection } from '@/types/architecture.ts';
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

        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
              <BookOpen className="w-4 h-4 mr-1.5" />
              Templates
              <ChevronDown className="w-3.5 h-3.5 ml-1" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-80 p-0 max-h-[500px]" align="start">
            <VariantPresetsPanel
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
