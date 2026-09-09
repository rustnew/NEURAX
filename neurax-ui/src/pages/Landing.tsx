/**
 * Landing page.
 *
 * NEURAX is a commercial engineering product, and the page it had did not say
 * so. It led with an open-source badge, made GitHub a primary action, closed
 * on an MIT licence, and explained itself in paragraphs — the visual language
 * of a repository, not of a platform a team adopts. Worse, in removing the
 * clutter an earlier pass removed the only way to sign in, so a first-time
 * visitor pressing the main call to action was bounced straight back here by
 * the route guard with nothing said.
 *
 * Three rules shape what replaced it:
 *
 *  - **The product is the subject.** Every section shows a surface the studio
 *    actually draws, in the studio's own typography and figures, rather than
 *    describing it. A reader should understand what NEURAX does before they
 *    have read a full sentence.
 *
 *  - **The figures are real.** The worked example is BERT-base analysed
 *    against the machine this was built on — an Intel laptop with no discrete
 *    GPU, 668 hours, 360.9 kWh. An unflattering number, kept because knowing
 *    it before starting is the entire argument.
 *
 *  - **One accent, used sparingly.** Acid green on near-black, on the things
 *    that matter: the primary action, a verdict, a live value. Not on blocks.
 */
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Check, Menu, X } from 'lucide-react';

import { AuthControl } from '@/components/auth/AuthControl.tsx';
import { NeuraxLogo } from '@/components/brand/NeuraxLogo.tsx';
import {
  BLOCK_COUNT,
  ENVIRONMENTS,
  EXAMPLE_ACCURACY,
  EXAMPLE_COMPUTE,
  EXAMPLE_MODEL,
  EXAMPLE_PREDICTION,
  FAMILY_COUNT,
  METRIC_COUNT,
  TEAMS,
  WORKFLOW,
} from '@/data/projectFacts.ts';

// ─── Palette ────────────────────────────────────────────────────────
//
// Written as custom properties rather than fixed hexes, so the page follows
// the theme the reader chose instead of forcing one. The enterprise identity
// survives the switch: the same near-neutral greys inverted, and the same acid
// green.
//
// The accent needs two values, and that is the one non-obvious part. `#B8FF5A`
// is a fill colour — it carries dark text at better than 12:1 — but as *text*
// on white it is about 1.4:1, which is invisible. So the light theme keeps the
// green for fills and uses a deep green for anything set in it. Using one
// value for both is how a bright accent becomes unreadable on half the pages
// it appears on.
const PALETTE_CSS = `
  .nx {
    --nx-bg: #FFFFFF;
    --nx-surface: #F7F8FA;
    --nx-surface-2: #FFFFFF;
    --nx-border: #E4E6EB;
    --nx-text: #0C0D10;
    --nx-muted: #5D636E;
    --nx-accent: #B8FF5A;
    --nx-accent-ink: #3E6B00;
    --nx-on-accent: #0C0D10;
  }
  .dark .nx {
    --nx-bg: #08090B;
    --nx-surface: #101216;
    --nx-surface-2: #15171C;
    --nx-border: #24272D;
    --nx-text: #F5F5F2;
    --nx-muted: #8B9099;
    --nx-accent: #B8FF5A;
    --nx-accent-ink: #B8FF5A;
    --nx-on-accent: #08090B;
  }
`;

const BG = 'var(--nx-bg)';
const SURFACE = 'var(--nx-surface)';
const SURFACE_2 = 'var(--nx-surface-2)';
const BORDER = 'var(--nx-border)';
const TEXT = 'var(--nx-text)';
const MUTED = 'var(--nx-muted)';
/** Fills only — buttons, dots, rails. Carries `ON_ACCENT`, never body text. */
const ACCENT = 'var(--nx-accent)';
/** The accent as text on the page's own ground, legible in both themes. */
const ACCENT_INK = 'var(--nx-accent-ink)';
const ON_ACCENT = 'var(--nx-on-accent)';

/**
 * Where a demo request goes.
 *
 * Empty until there is an address to send it to. A "Request a demo" button
 * that opens nothing is worse than no button at all — it is the first promise
 * the product makes, broken — so the control is not rendered while this is
 * unset rather than rendered inert.
 */
const DEMO_CONTACT = '';

// ─── Building blocks ────────────────────────────────────────────────

function Section({
  children,
  id,
  surface,
}: {
  children: React.ReactNode;
  id?: string;
  surface?: boolean;
}) {
  return (
    <section
      id={id}
      style={{ background: surface ? SURFACE : BG, borderTop: `1px solid ${BORDER}` }}
    >
      <div className="mx-auto max-w-6xl px-6 py-20 sm:py-24">{children}</div>
    </section>
  );
}

function Kicker({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] font-mono uppercase tracking-[0.24em] mb-5" style={{ color: ACCENT_INK }}>
      {children}
    </p>
  );
}

function Title({ children }: { children: React.ReactNode }) {
  return (
    <h2
      className="text-[clamp(1.75rem,3.6vw,2.5rem)] font-semibold leading-[1.12] tracking-[-0.025em] max-w-3xl text-balance"
      style={{ color: TEXT }}
    >
      {children}
    </h2>
  );
}

function Lede({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-5 text-[16px] leading-[1.65] max-w-2xl" style={{ color: MUTED }}>
      {children}
    </p>
  );
}

/** A product surface. The frame the studio uses, so the page shows the tool
 *  rather than a picture of one. */
function Panel({
  label,
  children,
  accent,
}: {
  label: string;
  children: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ background: SURFACE_2, border: `1px solid ${accent ? '#3A4A28' : BORDER}` }}
    >
      <div
        className="px-4 py-2.5 flex items-center gap-2"
        style={{ borderBottom: `1px solid ${BORDER}`, background: SURFACE }}
      >
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: accent ? ACCENT : MUTED }} />
        <span
          className="text-[10px] font-mono uppercase tracking-[0.16em]"
          style={{ color: MUTED }}
        >
          {label}
        </span>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

/** One label/value line, in the studio's monospace. */
function Field({
  label,
  value,
  note,
  accent,
}: {
  label: string;
  value: string;
  note?: string;
  accent?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <span className="text-[12px]" style={{ color: MUTED }}>
        {label}
      </span>
      <span className="flex items-baseline gap-2 text-right">
        <span
          className="text-[13px] font-mono tabular-nums font-semibold"
          style={{ color: accent ? ACCENT_INK : TEXT }}
        >
          {value}
        </span>
        {note ? (
          <span className="text-[11px] font-mono" style={{ color: MUTED }}>
            {note}
          </span>
        ) : null}
      </span>
    </div>
  );
}

export default function Landing() {
  const [menuOpen, setMenuOpen] = useState(false);

  const navLinks = [
    { href: '#platform', label: 'Platform' },
    { href: '#workflow', label: 'Workflow' },
    { href: '#accuracy', label: 'Accuracy' },
    { href: '#teams', label: 'Teams' },
  ];

  return (
    <div className="nx min-h-screen antialiased" style={{ background: BG, color: TEXT }}>
      {/* The palette lives in CSS so it can switch on the theme's `dark`
          class. Injected here rather than in the global stylesheet because it
          belongs to this page and nothing else reads it. */}
      <style>{PALETTE_CSS}</style>

      {/* ── Navigation ── */}
      <header
        className="sticky top-0 z-40 backdrop-blur-xl"
        style={{
          // `color-mix` rather than a fixed rgba: the header has to be the
          // page's own ground at 85%, and the page's ground changes with the
          // theme. A literal `rgba(8,9,11,.82)` put a black bar across the
          // top of the light themes.
          background: 'color-mix(in srgb, var(--nx-bg) 85%, transparent)',
          borderBottom: `1px solid ${BORDER}`,
        }}
      >
        <nav className="mx-auto max-w-6xl px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2.5" aria-label="NEURAX home">
            <NeuraxLogo variant="mark" size={36} />
            <span className="text-[15px] font-semibold tracking-[0.02em]">NEURAX</span>
          </Link>

          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="text-[13.5px] transition-colors hover:text-white"
                style={{ color: MUTED }}
              >
                {link.label}
              </a>
            ))}
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            {/*
              Sign in, restored.

              An earlier pass took the auth control off this page entirely.
              `/app` is behind a route guard that redirects an unauthenticated
              visitor back to `/`, so the primary action sent a first-time
              reader in a circle with nothing said. This is the only surface
              that offers an account, and it has to be here.
            */}
            {/* Quiet on purpose.
                `AuthControl`'s default variant is the app's gold, which put
                two competing bright colours side by side in the header — the
                amber of the studio's theme next to the green of this page.
                Only one action in a header should read as primary, and here
                it is Open NEURAX. */}
            <AuthControl
              triggerLabel="Sign in"
              triggerSize="sm"
              triggerVariant="ghost"
              triggerClassName="h-9 px-3.5 rounded-lg text-[13.5px] font-medium border border-[#24272D] bg-transparent text-[#F5F5F2] hover:bg-white/[0.06] hover:text-white"
            />
            <Link
              to="/app"
              className="hidden sm:inline-flex items-center gap-1.5 h-9 px-4 rounded-lg text-[13.5px] font-semibold transition-transform hover:-translate-y-px"
              style={{ background: ACCENT, color: ON_ACCENT }}
            >
              Open NEURAX
            </Link>
            <button
              type="button"
              className="md:hidden p-2 -mr-2"
              onClick={() => setMenuOpen((v) => !v)}
              aria-label={menuOpen ? 'Close menu' : 'Open menu'}
              aria-expanded={menuOpen}
              style={{ color: MUTED }}
            >
              {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </nav>

        {menuOpen ? (
          <div className="md:hidden px-6 pb-4 flex flex-col gap-3" style={{ borderTop: `1px solid ${BORDER}` }}>
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setMenuOpen(false)}
                className="text-[14px] pt-3"
                style={{ color: MUTED }}
              >
                {link.label}
              </a>
            ))}
          </div>
        ) : null}
      </header>

      {/* ── Hero ── */}
      <div className="mx-auto max-w-6xl px-6 pt-20 pb-16 sm:pt-28">
        <div className="max-w-3xl">
          <Kicker>AI workload intelligence</Kicker>
          {/*
            "before you run it" alone stopped being true.

            It was the right headline while NEURAX only predicted: a promise
            nobody else makes, in one line. But the product now runs the
            workload, and the phrase reads as "instead of running it" — which
            describes half of what is on offer and quietly contradicts the
            three sections below it.

            Extended rather than replaced. The prediction is still the promise;
            running is not a second promise, it is what makes the first one
            credible, and no competitor can say the second half at all.
          */}
          <h1
            className="text-[clamp(2.25rem,5vw,3.5rem)] font-semibold leading-[1.06] tracking-[-0.035em] text-balance"
            style={{ color: TEXT }}
          >
            Know what your AI workload costs —
            <br />
            <span style={{ color: MUTED }}>before it runs, and after.</span>
          </h1>
          <p className="mt-7 text-[18px] leading-[1.6] max-w-2xl" style={{ color: MUTED }}>
            NEURAX reads your model, your data and the compute you actually have, predicts memory,
            performance and cost, then runs the workload and measures every figure it got wrong.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-3">
            <Link
              to="/app"
              className="inline-flex items-center gap-2 h-12 px-7 rounded-xl text-[15.5px] font-semibold transition-transform hover:-translate-y-px"
              style={{ background: ACCENT, color: ON_ACCENT }}
            >
              Open NEURAX
              <ArrowRight className="w-4 h-4" />
            </Link>
            {DEMO_CONTACT ? (
              <a
                href={DEMO_CONTACT}
                className="inline-flex items-center h-12 px-7 rounded-xl text-[15.5px] font-medium transition-colors hover:bg-white/5"
                style={{ border: `1px solid ${BORDER}`, color: TEXT }}
              >
                Request a demo
              </a>
            ) : (
              <a
                href="#platform"
                className="inline-flex items-center h-12 px-7 rounded-xl text-[15.5px] font-medium transition-colors hover:bg-white/5"
                style={{ border: `1px solid ${BORDER}`, color: TEXT }}
              >
                See the platform
              </a>
            )}
          </div>
        </div>

        {/* ── The product, as the studio draws it ── */}
        <div className="mt-16 grid gap-4 lg:grid-cols-3">
          <Panel label="Model">
            <div className="text-[15px] font-semibold mb-1">{EXAMPLE_MODEL.name}</div>
            <div className="text-[11px] font-mono mb-3" style={{ color: MUTED }}>
              {EXAMPLE_MODEL.family}
            </div>
            <div style={{ borderTop: `1px solid ${BORDER}` }} className="pt-1">
              <Field label="Parameters" value={EXAMPLE_MODEL.parameters} />
              <Field label="Layers" value={EXAMPLE_MODEL.layers} />
              <Field label="Hidden size" value={EXAMPLE_MODEL.hidden} />
              <Field label="Attention heads" value={EXAMPLE_MODEL.heads} />
            </div>
          </Panel>

          <Panel label="Compute target">
            <div className="text-[15px] font-semibold mb-1">{EXAMPLE_COMPUTE.device}</div>
            <div className="text-[11px] font-mono mb-3" style={{ color: MUTED }}>
              {EXAMPLE_COMPUTE.kind}
            </div>
            <div style={{ borderTop: `1px solid ${BORDER}` }} className="pt-1">
              <Field label="Processor" value={EXAMPLE_COMPUTE.cores} />
              <Field label="Memory" value={EXAMPLE_COMPUTE.memory} />
              <Field label="Measured" value={EXAMPLE_COMPUTE.measured} accent />
            </div>
          </Panel>

          <Panel label="Prediction" accent>
            <div className="text-[15px] font-semibold mb-1">Before anything runs</div>
            <div className="text-[11px] font-mono mb-3" style={{ color: MUTED }}>
              Computed from the design
            </div>
            <div style={{ borderTop: `1px solid ${BORDER}` }} className="pt-1">
              {EXAMPLE_PREDICTION.map((row) => (
                <Field
                  key={row.label}
                  label={row.label}
                  value={row.value}
                  note={row.note}
                  accent={row.label === 'Verdict'}
                />
              ))}
            </div>
          </Panel>
        </div>

        <p className="mt-4 text-[11.5px] font-mono" style={{ color: MUTED }}>
          A real reading: BERT-base analysed against a laptop with no discrete GPU.
        </p>
      </div>

      {/* ── The problem ── */}
      <Section surface>
        <Kicker>The problem</Kicker>
        <Title>Compute is expensive. Guessing is more expensive.</Title>
        <Lede>
          The choices that decide what a model costs — how wide, how deep, which precision, what
          batch — are made before anything can measure them, and paid for weeks later.
        </Lede>

        <div className="mt-12 grid gap-px md:grid-cols-3" style={{ background: BORDER }}>
          {[
            {
              t: 'You find out by running it',
              d: 'A workload that will not fit announces itself hours in, on hardware that was already billing.',
            },
            {
              t: 'The estimate describes another machine',
              d: 'Spreadsheets, blog posts, and the last model that happened to work. None of them is your infrastructure.',
            },
            {
              t: 'Nothing checks the answer',
              d: 'Tools that predict do not run. Tools that run do not predict. Nobody produces the error between them.',
            },
          ].map((item) => (
            <div key={item.t} className="p-7" style={{ background: BG }}>
              <h3 className="text-[17px] font-semibold tracking-[-0.01em]">{item.t}</h3>
              <p className="mt-3 text-[14px] leading-[1.6]" style={{ color: MUTED }}>
                {item.d}
              </p>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Workflow ── */}
      <Section id="workflow">
        <Kicker>Workflow</Kicker>
        <Title>One model. One environment. One complete workflow.</Title>
        <Lede>
          Every stage feeds the next, and the last one feeds the first. The four in green happen on
          real hardware — that half is what no other tool closes.
        </Lede>

        <ol className="mt-12 grid gap-px sm:grid-cols-2 lg:grid-cols-4" style={{ background: BORDER }}>
          {WORKFLOW.map((step, i) => (
            <li key={step.stage} className="p-6" style={{ background: BG }}>
              <div className="flex items-baseline gap-3">
                {/* The second half is accented because it is the half that
                    did not exist before: everything from Train onward happens
                    on real hardware. The lede below the grid says so, rather
                    than leaving the colour to be guessed at. */}
                <span
                  className="text-[11px] font-mono tabular-nums"
                  style={{ color: i >= 4 ? ACCENT_INK : MUTED }}
                >
                  {String(i + 1).padStart(2, '0')}
                </span>
                <h3 className="text-[15.5px] font-semibold tracking-[-0.01em]">{step.stage}</h3>
              </div>
              <p className="mt-2 text-[13px] leading-relaxed pl-7" style={{ color: MUTED }}>
                {step.gives}
              </p>
            </li>
          ))}
        </ol>
      </Section>

      {/* ── Platform ── */}
      <Section id="platform" surface>
        <Kicker>Platform</Kicker>
        <Title>Your infrastructure becomes part of the model.</Title>
        <Lede>
          NEURAX does not size a workload against a machine from a catalogue. It reads the hardware
          that will run it, measures what that hardware sustains, and designs against the answer.
        </Lede>

        <div className="mt-12 grid gap-8 lg:grid-cols-[1.1fr_1fr] items-start">
          <div className="grid gap-px sm:grid-cols-2" style={{ background: BORDER }}>
            {[
              { t: 'Detects the machine', d: 'Processor, memory, disk and any accelerator — including an integrated one, reported as sharing system memory rather than as a card with nothing left.' },
              { t: 'Measures what it sustains', d: 'Published specifications cover datacenter cards. Everything else gets measured once and remembered.' },
              { t: 'Reads the dataset', d: 'Shape, size, class balance and integrity. Structure and statistics only — never content, and nothing leaves the machine.' },
              { t: 'Sizes everything to both', d: 'Batch, precision and memory budget follow from the hardware and the data, not from a default.' },
            ].map((item) => (
              <div key={item.t} className="p-6" style={{ background: SURFACE }}>
                <h3 className="text-[15px] font-semibold tracking-[-0.01em]">{item.t}</h3>
                <p className="mt-2.5 text-[13.5px] leading-[1.6]" style={{ color: MUTED }}>
                  {item.d}
                </p>
              </div>
            ))}
          </div>

          <Panel label="Detected environment" accent>
            <Field label="Processor" value="Intel Core i5-8365U" />
            <Field label="Cores" value="4 / 8 threads" />
            <Field label="Memory" value="23 GB" note="5.46 free" />
            <Field label="Accelerator" value="none detected" />
            <Field label="Sustained" value="48.2 GFLOP/s" accent />
            <Field label="Bandwidth" value="14.3 GB/s" accent />
            <div className="mt-4 pt-4 text-[12px] leading-relaxed" style={{ borderTop: `1px solid ${BORDER}`, color: MUTED }}>
              Measured on this machine, not read from a specification sheet.
            </div>
          </Panel>
        </div>
      </Section>

      {/* ── Accuracy ── */}
      <Section id="accuracy">
        <Kicker>What nothing else does</Kicker>
        <Title>Measure where the prediction was wrong.</Title>
        <Lede>
          The run reports its real parameter count before the first step, then its memory, speed and
          duration as it goes — each beside the figure it was meant to be.
        </Lede>

        <div className="mt-12 rounded-xl overflow-hidden" style={{ border: `1px solid ${BORDER}` }}>
          <div className="overflow-x-auto">
            <table className="w-full text-[13.5px]">
              <thead>
                <tr style={{ background: SURFACE }}>
                  {['Metric', 'Predicted', 'Observed', 'Error'].map((h, i) => (
                    <th
                      key={h}
                      className={`px-5 py-3 text-[10.5px] font-mono uppercase tracking-[0.16em] font-medium ${
                        i === 0 ? 'text-left' : 'text-right'
                      }`}
                      style={{ color: MUTED }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {EXAMPLE_ACCURACY.map((row) => (
                  <tr key={row.metric} style={{ borderTop: `1px solid ${BORDER}`, background: SURFACE_2 }}>
                    <td className="px-5 py-3.5">{row.metric}</td>
                    <td className="px-5 py-3.5 text-right font-mono tabular-nums" style={{ color: MUTED }}>
                      {row.predicted}
                    </td>
                    <td className="px-5 py-3.5 text-right font-mono tabular-nums font-semibold">
                      {row.observed}
                    </td>
                    {/*
                      The error column is data, not a verdict.

                      Every value here was rendered in the accent, which reads
                      as "good" — and made the page claim, in colour, that
                      NEURAX is always right to within three percent. An error
                      is a measurement; it is neither good nor bad by itself.
                      The accent is spent on the one entry that is genuinely
                      remarkable: a parameter count that matches exactly,
                      reported before the first step.
                    */}
                    <td
                      className="px-5 py-3.5 text-right font-mono tabular-nums font-semibold"
                      style={{ color: row.error === 'exact' ? ACCENT_INK : TEXT }}
                    >
                      {row.error}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-10 grid gap-8 lg:grid-cols-2 items-start">
          <p className="text-[19px] font-medium leading-snug" style={{ color: TEXT }}>
            Every run makes the next prediction better.
          </p>
          <p className="text-[14.5px] leading-[1.65]" style={{ color: MUTED }}>
            The parameter count is arithmetic: it matches or a formula is wrong, and it costs a
            second to find out. Memory and timing carry real error bands — a few percent is a good
            analytical prediction, and knowing which few percent is what makes the next estimate
            better than the last.
          </p>
        </div>
      </Section>

      {/* ── Environments ── */}
      <Section surface>
        <Kicker>Compute environments</Kicker>
        <Title>One intelligence layer across the machines you run on.</Title>
        <Lede>
          The same analysis, the same plan, the same comparison — whether the workload runs on the
          laptop in front of you or on hardware you rent.
        </Lede>

        <div className="mt-12 grid gap-px sm:grid-cols-2 lg:grid-cols-4" style={{ background: BORDER }}>
          {ENVIRONMENTS.map((env) => (
            <div key={env.name} className="p-7" style={{ background: SURFACE }}>
              <div className="flex items-center gap-2">
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: env.status === 'available' ? ACCENT : BORDER }}
                />
                <h3 className="text-[15.5px] font-semibold tracking-[-0.01em]">{env.name}</h3>
              </div>
              {/* Planned is said, not implied. A roadmap item drawn like a
                  shipped one is the fastest way to lose an engineering
                  audience. */}
              <p className="mt-2 text-[11px] font-mono uppercase tracking-[0.14em]" style={{ color: MUTED }}>
                {env.status}
              </p>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Teams ── */}
      <Section id="teams">
        <Kicker>Who it is for</Kicker>
        <Title>Built for teams operating AI.</Title>

        <div className="mt-12 grid gap-px sm:grid-cols-2 lg:grid-cols-3" style={{ background: BORDER }}>
          {TEAMS.map((team) => (
            <div key={team.name} className="p-7" style={{ background: BG }}>
              <h3
                className="text-[11px] font-mono uppercase tracking-[0.18em]"
                style={{ color: ACCENT_INK }}
              >
                {team.name}
              </h3>
              <p className="mt-3 text-[14.5px] leading-[1.6]" style={{ color: TEXT }}>
                {team.does}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-wrap gap-x-10 gap-y-4">
          {[
            `${FAMILY_COUNT} architecture families`,
            `${BLOCK_COUNT} catalogue blocks`,
            `${METRIC_COUNT} metrics per analysis`,
          ].map((line) => (
            <span key={line} className="flex items-center gap-2 text-[13.5px]" style={{ color: MUTED }}>
              <Check className="w-3.5 h-3.5" style={{ color: ACCENT_INK }} />
              {line}
            </span>
          ))}
        </div>
      </Section>

      {/* ── Close ── */}
      <Section surface>
        <div className="text-center">
          <h2
            className="text-[clamp(1.875rem,4vw,2.75rem)] font-semibold leading-[1.1] tracking-[-0.03em] text-balance mx-auto max-w-3xl"
            style={{ color: TEXT }}
          >
            Understand your AI workload before it runs.
          </h2>
          <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
            <Link
              to="/app"
              className="inline-flex items-center gap-2 h-12 px-8 rounded-xl text-[15.5px] font-semibold transition-transform hover:-translate-y-px"
              style={{ background: ACCENT, color: ON_ACCENT }}
            >
              Open NEURAX
              <ArrowRight className="w-4 h-4" />
            </Link>
            <AuthControl
              triggerLabel="Create an account"
              triggerVariant="ghost"
              triggerClassName="h-12 px-8 rounded-xl text-[15.5px] font-medium border border-[#24272D] bg-transparent text-[#F5F5F2] hover:bg-white/[0.06] hover:text-white"
            />
          </div>
        </div>
      </Section>

      {/* ── Footer ── */}
      <footer style={{ background: BG, borderTop: `1px solid ${BORDER}` }}>
        <div className="mx-auto max-w-6xl px-6 py-10 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <NeuraxLogo variant="mark" size={30} />
            <span className="text-[13.5px] font-medium">NEURAX</span>
          </div>
          <p className="text-[12.5px]" style={{ color: MUTED }}>
            AI workload intelligence
          </p>
        </div>
      </footer>
    </div>
  );
}
