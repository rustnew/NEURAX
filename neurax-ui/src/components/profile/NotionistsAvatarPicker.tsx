/**
 * Avatar picker.
 *
 * The set used to be twelve emoji on coloured discs. It is now generated
 * identicons, in the style GitHub gives accounts without a picture: a
 * deterministic symmetric grid drawn from a seed. They read as an identity
 * rather than a decoration, they cannot be confused with the emoji the rest of
 * the product uses for status, and nothing is fetched to render one.
 */
import { useState } from 'react';
import { Check } from 'lucide-react';
import { cn } from '@/lib/utils.ts';
import { Identicon } from '@/components/profile/Identicon.tsx';

export interface AvatarOption {
  id: string;
  name: string;
  /** Seed the identicon is drawn from; stable for the lifetime of the option. */
  seed: string;
}

/**
 * The offered set.
 *
 * Seeds are fixed words rather than random values so the same avatar appears
 * for the same person on every device, and so this list can be extended without
 * changing anyone's existing picture.
 */
export const AVATAR_OPTIONS: AvatarOption[] = [
  { id: 'ax-01', name: 'Vector', seed: 'neurax-vector' },
  { id: 'ax-02', name: 'Tensor', seed: 'neurax-tensor' },
  { id: 'ax-03', name: 'Kernel', seed: 'neurax-kernel' },
  { id: 'ax-04', name: 'Gradient', seed: 'neurax-gradient' },
  { id: 'ax-05', name: 'Lattice', seed: 'neurax-lattice' },
  { id: 'ax-06', name: 'Cipher', seed: 'neurax-cipher' },
  { id: 'ax-07', name: 'Quanta', seed: 'neurax-quanta' },
  { id: 'ax-08', name: 'Circuit', seed: 'neurax-circuit' },
  { id: 'ax-09', name: 'Entropy', seed: 'neurax-entropy' },
  { id: 'ax-10', name: 'Manifold', seed: 'neurax-manifold' },
  { id: 'ax-11', name: 'Spectra', seed: 'neurax-spectra' },
  { id: 'ax-12', name: 'Nucleus', seed: 'neurax-nucleus' },
  { id: 'ax-13', name: 'Photon', seed: 'neurax-photon' },
  { id: 'ax-14', name: 'Fractal', seed: 'neurax-fractal' },
  { id: 'ax-15', name: 'Vertex', seed: 'neurax-vertex' },
  { id: 'ax-16', name: 'Synapse', seed: 'neurax-synapse' },
  { id: 'ax-17', name: 'Cortex', seed: 'neurax-cortex' },
  { id: 'ax-18', name: 'Neutrino', seed: 'neurax-neutrino' },
  { id: 'ax-19', name: 'Plasma', seed: 'neurax-plasma' },
  { id: 'ax-20', name: 'Helix', seed: 'neurax-helix' },
  { id: 'ax-21', name: 'Prism', seed: 'neurax-prism' },
  { id: 'ax-22', name: 'Quark', seed: 'neurax-quark' },
  { id: 'ax-23', name: 'Nebula', seed: 'neurax-nebula' },
  { id: 'ax-24', name: 'Axiom', seed: 'neurax-axiom' },
  { id: 'ax-25', name: 'Codex', seed: 'neurax-codex' },
  { id: 'ax-26', name: 'Matrix', seed: 'neurax-matrix' },
  { id: 'ax-27', name: 'Vortex', seed: 'neurax-vortex' },
  { id: 'ax-28', name: 'Zenith', seed: 'neurax-zenith' },
  { id: 'ax-29', name: 'Pulsar', seed: 'neurax-pulsar' },
  { id: 'ax-30', name: 'Cascade', seed: 'neurax-cascade' },
  { id: 'ax-31', name: 'Nexus', seed: 'neurax-nexus' },
  { id: 'ax-32', name: 'Radian', seed: 'neurax-radian' },
  { id: 'ax-33', name: 'Simplex', seed: 'neurax-simplex' },
  { id: 'ax-34', name: 'Boson', seed: 'neurax-boson' },
  { id: 'ax-35', name: 'Cypher', seed: 'neurax-cypher' },
  { id: 'ax-36', name: 'Halcyon', seed: 'neurax-halcyon' },
];

/**
 * Kept under its previous name so existing imports keep working.
 * @deprecated Prefer {@link AVATAR_OPTIONS}.
 */
export const NOTIONISTS_AVATARS = AVATAR_OPTIONS;

/**
 * Resolve a stored identifier to the option it names — an avatar id (how
 * the picker itself refers to one) or an avatar seed (how a profile's
 * `avatarSeed` field refers to the same one; the two are different strings
 * for the same option, `{ id: 'ax-07', seed: 'neurax-quanta' }`, and a
 * profile only ever stores the seed), tolerating values written by the
 * emoji set that predated both.
 */
export function resolveAvatar(stored: string | null | undefined): AvatarOption {
  if (!stored) return AVATAR_OPTIONS[0];
  const byId = AVATAR_OPTIONS.find((option) => option.id === stored);
  if (byId) return byId;
  const bySeed = AVATAR_OPTIONS.find((option) => option.seed === stored);
  if (bySeed) return bySeed;
  // Anything else — including an emoji saved by the previous picker — is
  // hashed to a stable option, so an existing profile keeps a consistent
  // avatar instead of silently resetting to the first one.
  let sum = 0;
  for (let i = 0; i < stored.length; i++) sum = (sum + stored.charCodeAt(i)) % 9973;
  return AVATAR_OPTIONS[sum % AVATAR_OPTIONS.length];
}

interface AvatarPickerProps {
  selectedId?: string;
  onSelect: (avatarId: string) => void;
  /**
   * `strip` shows one scrollable row with no heading; `grid` shows all
   * thirty-six with their names.
   *
   * The grid was the only mode, and in the sign-in dialog it was the largest
   * thing on screen — thirty-six named tiles above the name and email fields
   * that are the actual point of the form. Picking a pattern is a pleasant
   * detail, not the task; it gets a row, and the full set stays one click
   * away for anyone who wants to browse it.
   */
  variant?: 'grid' | 'strip';
}

export const NotionistsAvatarPicker = ({
  selectedId,
  onSelect,
  variant = 'grid',
}: AvatarPickerProps) => {
  const selected = resolveAvatar(selectedId);

  if (variant === 'strip') {
    // The right edge fades instead of slicing a tile in half. A hard cut
    // through the middle of an avatar looks like a layout fault; a fade says
    // there is more and it scrolls.
    return (
      <div
        className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin w-full min-w-0"
        role="radiogroup"
        aria-label="Avatar"
        style={{
          maskImage: 'linear-gradient(to right, black calc(100% - 32px), transparent)',
          WebkitMaskImage: 'linear-gradient(to right, black calc(100% - 32px), transparent)',
        }}
      >
        {AVATAR_OPTIONS.map((avatar) => {
          const isSelected = selected.id === avatar.id;
          return (
            <button
              key={avatar.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              aria-label={avatar.name}
              title={avatar.name}
              onClick={() => onSelect(avatar.id)}
              className={cn(
                'shrink-0 p-1 rounded-[7px] border transition-all duration-150',
                isSelected
                  ? 'border-primary bg-primary/10'
                  : 'border-border/60 bg-card hover:border-border',
              )}
            >
              <div className="text-foreground">
                <Identicon seed={avatar.seed} size={28} />
              </div>
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div
        className="grid grid-cols-4 sm:grid-cols-6 gap-2.5 max-h-[280px] overflow-y-auto pr-1 scrollbar-thin"
        role="radiogroup"
        aria-label="Avatar"
      >
        {AVATAR_OPTIONS.map((avatar) => {
          const isSelected = selected.id === avatar.id;
          return (
            <button
              key={avatar.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              aria-label={avatar.name}
              onClick={() => onSelect(avatar.id)}
              className={cn(
                'relative flex flex-col items-center gap-1.5 p-2 rounded-[8px] border transition-all duration-150 hover:scale-105',
                isSelected
                  ? 'border-primary bg-primary/10'
                  : 'border-border/60 bg-card hover:border-border',
              )}
            >
              <div className="text-foreground">
                <Identicon seed={avatar.seed} size={38} />
              </div>
              <span
                className={cn(
                  'text-[10px] font-medium text-center leading-tight',
                  isSelected ? 'text-primary' : 'text-muted-foreground',
                )}
              >
                {avatar.name}
              </span>

              {isSelected && (
                <div className="absolute top-1 right-1 w-5 h-5 rounded-full flex items-center justify-center bg-primary">
                  <Check className="w-3 h-3 text-primary-foreground" />
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

/** Track the selected avatar. */
export const useNotionistAvatar = (initialId?: string) => {
  const [selectedId, setSelectedId] = useState<string>(
    () => resolveAvatar(initialId).id,
  );
  return {
    selectedId,
    selectedAvatar: resolveAvatar(selectedId),
    setSelectedId,
  };
};

interface AvatarDisplayProps {
  avatarId: string;
  size?: 'sm' | 'md' | 'lg';
  showName?: boolean;
}

const SIZES = { sm: 32, md: 40, lg: 56 } as const;

export const NotionistAvatarDisplay = ({
  avatarId,
  size = 'md',
  showName = false,
}: AvatarDisplayProps) => {
  const avatar = resolveAvatar(avatarId);
  return (
    <div className="flex items-center gap-2">
      <div className="text-foreground">
        <Identicon seed={avatar.seed} size={SIZES[size]} />
      </div>
      {showName && (
        <span className="font-medium text-[12px] text-foreground">{avatar.name}</span>
      )}
    </div>
  );
};
