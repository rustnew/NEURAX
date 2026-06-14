import { SelectionBox as SelectionBoxType } from '@/hooks/useMultiSelect.ts';

interface SelectionBoxProps {
  box: SelectionBoxType;
}

export function SelectionBox({ box }: SelectionBoxProps) {
  const left = Math.min(box.startX, box.endX);
  const top = Math.min(box.startY, box.endY);
  const width = Math.abs(box.endX - box.startX);
  const height = Math.abs(box.endY - box.startY);

  if (width < 5 && height < 5) return null;

  return (
    <div
      className="absolute pointer-events-none border-2 border-primary/60 bg-primary/10 rounded-sm"
      style={{
        left,
        top,
        width,
        height,
      }}
    />
  );
}
