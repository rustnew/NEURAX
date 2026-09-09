import { ArchitectureFamily, ARCHITECTURE_FAMILIES } from '@/types/plugins.ts';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '@/components/ui/select.tsx';
import { FamilyGlyph } from '@/components/architecture/FamilyGlyph.tsx';
import { cn } from '@/lib/utils.ts';

interface ArchitectureSelectorProps {
  value: ArchitectureFamily;
  onChange: (value: ArchitectureFamily) => void;
  className?: string;
}

export function ArchitectureSelector({ value, onChange, className }: ArchitectureSelectorProps) {
  const currentFamily = ARCHITECTURE_FAMILIES.find(f => f.id === value);

  return (
    <Select value={value} onValueChange={(v) => onChange(v as ArchitectureFamily)}>
      <SelectTrigger
        title={currentFamily ? `${currentFamily.name} — ${currentFamily.description}` : undefined}
        className={cn(
          "w-[220px] h-9 shrink-0",
          "bg-secondary/50 border-border/50 hover:bg-secondary transition-colors",
          "focus:ring-1 focus:ring-primary/50",
          className
        )}
      >
        {/*
          The trigger draws the family name itself rather than delegating to
          `SelectValue`. `SelectValue` mirrors the whole selected row — icon,
          name, description and badge — which is right in a dropdown and far
          too much for a trigger that already carries an icon: inside the
          toolbar it overflowed and left a fragment of the name on screen,
          "Transformer / LLM" showing as "LLM". The full name and its
          description are on the trigger's tooltip.
        */}
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="w-6 h-6 rounded-[5px] flex items-center justify-center shrink-0"
            style={{ backgroundColor: `${currentFamily?.color}1f` }}
          >
            {currentFamily ? (
              <FamilyGlyph
                family={currentFamily.id}
                color={currentFamily.color}
                className="w-[18px] h-[18px] text-foreground/70"
              />
            ) : null}
          </div>
          <span className="truncate text-sm font-medium">
            {currentFamily?.name ?? 'Select architecture'}
          </span>
        </div>
      </SelectTrigger>
      <SelectContent className="bg-popover border-border z-50 max-h-[70vh]">
        {ARCHITECTURE_FAMILIES.map((family) => (
          <SelectItem
            key={family.id}
            value={family.id}
            className="cursor-pointer focus:bg-secondary"
          >
            <div className="flex items-center gap-3 py-1">
              {/* The glyph gets real room here — it is a diagram, and at the
                  trigger's size only its silhouette survives. */}
              <div
                className="w-9 h-9 rounded-[6px] flex items-center justify-center shrink-0"
                style={{ backgroundColor: `${family.color}1f` }}
              >
                <FamilyGlyph
                  family={family.id}
                  color={family.color}
                  className="w-[26px] h-[26px] text-foreground/70"
                />
              </div>
              <div className="flex flex-col min-w-0">
                <span className="text-sm font-medium leading-tight">{family.name}</span>
                <span className="text-[10px] text-muted-foreground leading-tight mt-0.5">{family.description}</span>
              </div>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
