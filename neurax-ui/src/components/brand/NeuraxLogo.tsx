/**
 * The NEURAX mark.
 *
 * Replaces a hand-drawn SVG spider in an amber gradient — a placeholder that
 * had outlived its purpose and no longer matched anything: not the product's
 * name, not its palette, and not the identity that now exists.
 *
 * ## Why the mark is an image and the word is text
 *
 * The pangolin is a rendered illustration — layered scales, gradients, a glow
 * — and nothing is gained by tracing it into SVG paths that would be larger
 * than the PNG and still not editable. The word beside it is different: as
 * live text it inherits `currentColor`, so it is the right colour in every
 * theme, at every size, with no second asset to keep in step. A wordmark
 * baked into an image is a wordmark that is white on a white page.
 *
 * ## Why there are two files
 *
 * The artwork is silver and acid green on black. Measured against the themes
 * it has to survive: on the dark ones it is exactly as designed; on white, a
 * quarter of it is above 200 luminance and simply vanishes, and the rest reads
 * as pale grey on pale grey.
 *
 * A dark tile behind it was the first fix and it was a compromise — a black
 * square on a white page is a lockup, not a logo. `neurax-mark-light` is the
 * same illustration with its *lightness* inverted and nothing else: hue and
 * saturation are untouched, so the silver becomes charcoal and the acid green
 * stays acid green. A plain colour inversion was the obvious alternative and
 * is wrong — it inverts hue too, and turns the green magenta.
 *
 * The right file is chosen by the theme, except where a surface paints its own
 * ground regardless of it. The landing page is always dark whatever the studio
 * is set to, so it asks for the variant it needs rather than inheriting one
 * that would be invisible on it.
 *
 * The source `logo.png` could not be used directly: it is a near-black square
 * at roughly two-thirds opacity across the whole canvas — 0.5% of its pixels
 * are fully transparent and none are fully opaque — so dropping it anywhere
 * would paint a dark translucent box. Both files here are the mark cut out of
 * it with a real alpha channel and the halo removed.
 */
import { cn } from '@/lib/utils.ts';

export interface NeuraxLogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
  variant?: 'mark' | 'full';
  /**
   * Which ground the logo is being placed on.
   *
   * `auto` follows the studio's theme, which is right almost everywhere. A
   * surface that paints its own background regardless of the theme has to say
   * so, or it gets the variant for the theme rather than the one for the
   * surface — dark ink on a dark page.
   */
  tone?: 'auto' | 'onDark' | 'onLight';
}

/** Light artwork, for dark surfaces. */
const MARK_ON_DARK = '/neurax-mark.png';
/** Dark artwork, for light surfaces. */
const MARK_ON_LIGHT = '/neurax-mark-light.png';

function Mark({ size, tone = 'auto' }: { size: number; tone?: 'auto' | 'onDark' | 'onLight' }) {
  const common = {
    alt: '',
    width: size,
    height: size,
    draggable: false,
    style: { width: size, height: size },
  } as const;

  if (tone === 'onDark') return <img src={MARK_ON_DARK} {...common} className="block select-none" />;
  if (tone === 'onLight') return <img src={MARK_ON_LIGHT} {...common} className="block select-none" />;

  // Both are in the markup and one is hidden, rather than reading the theme
  // from a context: the correct one is painted on the first frame, with no
  // flash of the wrong artwork while a provider resolves.
  return (
    <span className="inline-flex shrink-0" style={{ width: size, height: size }}>
      <img src={MARK_ON_LIGHT} {...common} className="block dark:hidden select-none" />
      <img src={MARK_ON_DARK} {...common} className="hidden dark:block select-none" />
    </span>
  );
}

export const NeuraxLogo = ({
  size = 28,
  className = '',
  showText = true,
  variant = 'full',
  tone = 'auto',
}: NeuraxLogoProps) => {
  const mark = <Mark size={size} tone={tone} />;

  if (variant === 'mark' || !showText) {
    return <span className={cn('inline-flex', className)}>{mark}</span>;
  }

  return (
    <span className={cn('inline-flex items-center gap-2.5', className)}>
      {mark}
      {/* `currentColor`, not a gradient. The word is the one part that has to
          be legible on every surface the logo lands on, and the surface's own
          text colour is always the right answer. */}
      <span
        className="font-semibold tracking-[0.04em] leading-none"
        style={{ fontSize: `${Math.round(size * 0.5)}px` }}
      >
        NEURAX
      </span>
    </span>
  );
};
