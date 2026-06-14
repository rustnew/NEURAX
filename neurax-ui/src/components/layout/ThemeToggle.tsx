import { Palette } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Theme, useTheme } from '@/contexts/ThemeContext.tsx';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip.tsx';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu.tsx';

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  const label = (t: Theme) => {
    switch (t) {
      case 'light':
        return 'Light';
      case 'dark':
        return 'Dark';
      case 'gruvbox':
        return 'Gruvbox';
      case 'nord':
        return 'Nord';
      case 'onedark':
        return 'One Dark';
    }
  };

  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
              <Palette className="w-4 h-4" />
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent>
          <p>Theme: {label(theme)}</p>
        </TooltipContent>
      </Tooltip>

      <DropdownMenuContent align="end">
        {(['light', 'dark', 'gruvbox', 'nord', 'onedark'] as Theme[]).map((t) => (
          <DropdownMenuItem key={t} onClick={() => setTheme(t)}>
            {label(t)}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
