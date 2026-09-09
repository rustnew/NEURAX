/**
 * Each architecture family, drawn as the figure its own literature draws.
 *
 * The selector used emoji — a brick for Transformers, an ice cube for CNNs, a
 * spiral for Diffusion. They were distinct from one another, which is all an
 * emoji can be, and they said nothing: nobody looks at 🧊 and thinks about a
 * shrinking feature map. Worse, emoji render as the host system's font, so the
 * same picker looked different on every machine and carried none of the
 * product's own drawing language.
 *
 * These are the canonical figures instead, reduced until they survive 20 px:
 *
 *  - **Transformer** — two stacked sub-blocks with the residual rail beside
 *    them, which is what Vaswani et al.'s Figure 1 is once you remove the
 *    labels: attention, add-and-norm, feed-forward, add-and-norm, ×N.
 *  - **CNN** — the pyramid of feature maps, tall and shallow at the input,
 *    short and deep at the classifier. Every convnet diagram since LeNet.
 *  - **Mixture of Experts** — one token, a router, and a bank of experts with
 *    only the routed ones lit. The sparsity *is* the architecture.
 *  - **Diffusion** — the U-Net: contracting path, bottleneck, expanding path,
 *    and the skip connections across the top that give it its name.
 *  - **GNN** — a graph with one node receiving from its neighbours, because
 *    message passing is the operation, not the picture of a network.
 *  - **RNN** — the unrolled recurrence: cells in a chain, state passed along,
 *    an input under each step and an output over it.
 *  - **SSM** — a long decaying kernel over the sequence axis. What separates
 *    S4 and Mamba from attention is that the past reaches the present through
 *    one continuous convolution rather than an all-pairs comparison.
 *  - **GAN** — the two facing cones: a generator widening noise into a sample,
 *    a discriminator narrowing a sample to one bit.
 *
 * Structure is drawn in `currentColor` so the glyph inherits the text colour
 * of wherever it sits, and the one accent stroke takes the family's own colour
 * — the same two-tone treatment across all eight, so the set reads as a set.
 */
import type { ArchitectureFamily } from '@/types/plugins.ts';
import { cn } from '@/lib/utils.ts';

interface FamilyGlyphProps {
  family: ArchitectureFamily;
  /** The family's colour, for the single accented element. */
  color?: string;
  className?: string;
}

/** Paths per family. Every glyph is authored on the same 24×24 grid with the
 *  same stroke width, so no one family reads heavier than its neighbours in
 *  the list. */
function paths(family: ArchitectureFamily, accent: string) {
  switch (family) {
    case 'transformer':
      return (
        <>
          {/* Residual rail, and the two taps that make it an "add" */}
          <path d="M4.5 4.5v15" />
          <path d="M4.5 7h4M4.5 17h4" />
          {/* Attention, then feed-forward */}
          <rect x="8.5" y="3.5" width="11" height="7" rx="1.5" />
          <rect x="8.5" y="13.5" width="11" height="7" rx="1.5" stroke={accent} />
          <path d="M11 7h6" stroke={accent} />
        </>
      );

    case 'cnn':
      return (
        <>
          {/* Feature maps: tall and shallow to short and deep */}
          <rect x="2" y="4" width="4.5" height="16" rx="1" />
          <rect x="8.5" y="6.5" width="4.5" height="11" rx="1" />
          <rect x="15" y="9" width="4" height="6" rx="1" stroke={accent} />
          <path d="M20.5 11.5v1" stroke={accent} />
        </>
      );

    case 'moe':
      return (
        <>
          {/* Experts, only the routed ones lit */}
          <rect x="2" y="3" width="6" height="5" rx="1.25" />
          <rect x="9" y="3" width="6" height="5" rx="1.25" stroke={accent} />
          <rect x="16" y="3" width="6" height="5" rx="1.25" stroke={accent} />
          {/* Router, and the token entering it */}
          <rect x="8" y="12" width="8" height="4" rx="1.25" />
          <path d="M12 20v-4" />
          <path d="M10.5 12 6 8.5" />
          <path d="M12 12V8" stroke={accent} />
          <path d="M13.5 12 18 8.5" stroke={accent} />
        </>
      );

    case 'diffusion':
      return (
        <>
          {/* Contracting path, bottleneck, expanding path */}
          <path d="M3 4v4l3.5 3.5v4L10 19h4l3.5-3.5v-4L21 8V4" />
          {/* Skip connections */}
          <path d="M3 5.5h18" strokeDasharray="2 2.5" stroke={accent} />
          <path d="M6.5 11h11" strokeDasharray="2 2.5" stroke={accent} />
        </>
      );

    case 'gnn':
      return (
        <>
          {/* Neighbours, and the node aggregating from them */}
          <path d="M5 6.5 12 12M19.5 7 12 12M6.5 18 12 12M18 18.5 12 12" />
          <circle cx="5" cy="6" r="2" />
          <circle cx="19.5" cy="6.5" r="2" />
          <circle cx="6" cy="18.5" r="2" />
          <circle cx="18.5" cy="19" r="2" />
          <circle cx="12" cy="12" r="2.75" stroke={accent} />
        </>
      );

    case 'rnn':
      return (
        <>
          {/* Unrolled cells, state passed along the chain */}
          <rect x="2" y="9" width="5.5" height="6" rx="1.5" />
          <rect x="9.25" y="9" width="5.5" height="6" rx="1.5" />
          <rect x="16.5" y="9" width="5.5" height="6" rx="1.5" />
          <path d="M7.5 12h1.75M14.75 12h1.75" stroke={accent} />
          <path d="M4.75 9V5.5M12 9V5.5M19.25 9V5.5" />
          <path d="M4.75 15v3.5M12 15v3.5M19.25 15v3.5" />
        </>
      );

    case 'ssm':
      return (
        <>
          {/* The sequence, and one long decaying kernel over it */}
          <path d="M2.5 20h19" />
          <path d="M2.5 20V4" />
          <path d="M4 5.5c6.5 0 3.5 13 17 13" stroke={accent} />
          <path d="M8 20v1.5M13 20v1.5M18 20v1.5" />
        </>
      );

    case 'gan':
      return (
        <>
          {/* Generator widens noise into a sample */}
          <path d="M2.5 11.25v1.5L9.5 19V5Z" />
          {/* Discriminator narrows a sample to one bit */}
          <path d="M21.5 11.25v1.5L14.5 19V5Z" stroke={accent} />
          <path d="M9.5 12h5" strokeDasharray="2 2" />
        </>
      );

    default:
      return <rect x="4" y="4" width="16" height="16" rx="2" />;
  }
}

export function FamilyGlyph({ family, color = 'currentColor', className }: FamilyGlyphProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={cn('shrink-0', className)}
    >
      {paths(family, color)}
    </svg>
  );
}
