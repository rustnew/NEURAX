/**
 * Figures the landing page states about NEURAX.
 *
 * Every number here is checked against the repository by
 * `projectFacts.test.ts`, because a landing page is the one surface where a
 * wrong number is a claim rather than a bug. The previous copy advertised "680+
 * configurable block types" and broke that down per family in the FAQ; the
 * catalogue holds 193. It also carried market statistics — a $2M average cost
 * for a failed training run, 73% of models needing redesign, a 5x production
 * overrun — with no source behind any of them.
 *
 * If a number cannot be derived from the repository or attributed to a source,
 * it does not belong on this page.
 */

/** Blocks and macro-blocks in the architecture catalogue, per family. */
export const CATALOGUE = {
  cnn: { blocks: 20, macroBlocks: 3 },
  diffusion: { blocks: 16, macroBlocks: 3 },
  gan: { blocks: 13, macroBlocks: 2 },
  gnn: { blocks: 17, macroBlocks: 3 },
  moe: { blocks: 19, macroBlocks: 2 },
  rnn: { blocks: 14, macroBlocks: 3 },
  ssm: { blocks: 13, macroBlocks: 2 },
  transformer: { blocks: 17, macroBlocks: 3 },
} as const;

export const FAMILY_COUNT = Object.keys(CATALOGUE).length;

export const BLOCK_COUNT = Object.values(CATALOGUE).reduce(
  (total, family) => total + family.blocks + family.macroBlocks,
  0,
);

/** Metrics returned by a single `/analyze` call. */
export const METRIC_COUNT = 66;

/** Passes in the IR pipeline, from architecture through to the report. */
export const IR_PASS_COUNT = 10;

/** Reference architectures shipped as ready-to-load presets. */
export const PRESET_COUNT = 30;

/**
 * Headline latency.
 *
 * The published benchmark is an 8B-parameter architecture analysed end to end;
 * the CLI reports well under this on the shipped examples.
 */
export const ANALYSIS_BUDGET_MS = 50;

/**
 * Export targets.
 *
 * A design leaving NEURAX is either archived for re-import, handed to the
 * compiler, or committed somewhere — nothing else is offered, because nothing
 * else could be offered truthfully.
 *
 * The framework emitters that used to sit here produced a class whose
 * `__init__` was empty and whose `forward` was a chain of `x2 = x1`: for
 * LLaMA 2 7B, an identity function carrying the model's name. The GitHub
 * target used to push exactly that into users' repositories, and now pushes
 * the same two files this list names.
 */
export const EXPORT_FORMATS = [
  'JSON — architecture and analysis, re-importable',
  'NEURAX IR — the exact topology the compiler analyses',
  'PyTorch — the design as a runnable nn.Module, checked against the analysis',
  'GitHub — the same files, committed to a repository',
] as const;

/**
 * What NEURAX does on the machine it is installed on.
 *
 * New, and the reason this page needed rewriting: until now every figure the
 * product produced was derived from a graph, and the page said so proudly.
 * That half is unchanged — the compiler still never executes a design — but
 * it is no longer the whole product. NEURAX now reads the machine in front of
 * it, measures what that machine actually sustains, and can run the design to
 * see whether the prediction held.
 *
 * Each line is something that works today, on a machine with no accelerator
 * included. Nothing here describes an intention.
 */
export const WORKSPACES = [
  {
    name: 'Architecture',
    does: 'The canvas. Drag blocks, connect them, set their parameters — and see what each one contributes as you go.',
  },
  {
    name: 'Simulation',
    does: 'What the compiler derives: cost and memory across depth, the roofline, where the budget runs out, what the graph is made of.',
  },
  {
    name: 'Production',
    does: 'Weight initialisation, computed per layer from its resolved shape — the recipe, not a few million random floats.',
  },
  {
    name: 'Training',
    does: 'A real run on your machine, beside the prediction it is testing. Start it, pause it, stop it, and pick it up after closing the window.',
  },
  {
    name: 'Time Machine',
    does: 'The same design projected forward: cost and carbon over a horizon, hardware migration, regulatory tracking.',
  },
] as const;

/**
 * The parts of NEURAX that are not the studio.
 *
 * Stated because the studio is the visible half and not the whole thing: the
 * compiler is a Rust workspace with its own CLI and terminal interface, the
 * agent is a separate process, and the desktop build embeds the service so it
 * runs with no network at all.
 */
export const ENVIRONMENT = [
  {
    name: 'The compiler',
    detail:
      'A Rust workspace: the IR pipeline, the formula library, the hardware database and the operator specifications. Used by every other part, and by nothing else.',
  },
  {
    name: 'The agent',
    detail:
      'Designs, optimises and explains architectures using your own API key, with the compiler as its tools. It reads the machine and the dataset, so what it proposes fits what you have.',
  },
  {
    name: 'The desktop build',
    detail:
      'The studio and the service in one application, with the compiler embedded. No network, no account, nothing leaves the machine.',
  },
  {
    name: 'The terminal',
    detail:
      'The same analysis without a browser, for a machine you reach over SSH.',
  },
] as const;

export const ON_YOUR_MACHINE = [
  {
    title: 'It reads your machine',
    detail:
      'CPU, cores, memory, disk and any accelerator — including an integrated one, which is reported as sharing system memory rather than as a card with no VRAM left.',
  },
  {
    title: 'It measures what that machine sustains',
    detail:
      'A blocked matrix multiply and a memory sweep, once, cached afterwards. Published specifications cover datacenter cards; a laptop needs measuring, and a measured number beats a missing one.',
  },
  {
    title: 'It trains, and the run outlives the window',
    detail:
      'Close NEURAX, shut the laptop, come back tomorrow — the run kept going, and the session is where you left it. One that was cut off mid-way picks up from its last save rather than starting again.',
  },
  {
    title: 'It checks the prediction against what happened',
    detail:
      'Within a second of starting, before any real work, it confirms the model it built is the model it costed. Everything after that — memory, speed, spend — appears beside the figure it was meant to be.',
  },
] as const;

/** The four figures shown under the hero. */
export const HERO_STATS = [
  {
    value: String(FAMILY_COUNT),
    label: 'Architecture families',
    detail: 'Transformer, CNN, MoE, SSM, diffusion, GAN, GNN, RNN',
  },
  {
    value: String(BLOCK_COUNT),
    label: 'Catalogue blocks',
    detail: 'Individual blocks and macro-blocks across every family',
  },
  {
    value: String(METRIC_COUNT),
    label: 'Metrics per analysis',
    detail: 'Parameters, FLOPs, memory, latency, cost, energy and carbon',
  },
  {
    value: `<${ANALYSIS_BUDGET_MS}ms`,
    label: 'End-to-end analysis',
    detail: `${IR_PASS_COUNT}-pass IR pipeline, no GPU required`,
  },
] as const;

// ─── The product, shown rather than described ───────────────────────
//
// A landing page for an engineering tool has to show the tool. These are the
// surfaces the studio actually draws, with the figures it actually produces,
// so the page can render the product in the product's own language instead of
// describing it in paragraphs.
//
// The training-cost example is a real reading, not an invented one: BERT-base,
// analysed by NEURAX against an Intel i5-8365U with no discrete GPU. It is an
// unflattering example on purpose — 668 hours is what that combination really
// costs, and knowing it before starting is the entire product.

export const EXAMPLE_MODEL = {
  name: 'BERT-base',
  family: 'Transformer',
  parameters: '110 M',
  layers: '12',
  hidden: '768',
  heads: '12',
} as const;

export const EXAMPLE_COMPUTE = {
  device: 'Intel Core i5-8365U',
  kind: 'CPU · no discrete GPU',
  cores: '4 cores / 8 threads',
  memory: '5.46 GB free of 23 GB',
  measured: '48.2 GFLOP/s · 14.3 GB/s',
} as const;

export const EXAMPLE_PREDICTION = [
  { label: 'Peak memory', value: '3.29 GB', note: 'of 5.46 GB free' },
  { label: 'Duration', value: '668 h 15', note: '2,960 steps' },
  { label: 'Energy', value: '360.9 kWh', note: 'about $2,004' },
  { label: 'Verdict', value: 'Fits', note: '2.17 GB of headroom' },
] as const;

/**
 * Predicted against observed.
 *
 * The differentiating screen, and the reason the loop exists. These are the
 * metrics the Accuracy view compares; the figures are representative of a run
 * that behaved normally, which is what a reader needs to understand what the
 * column means.
 */
export const EXAMPLE_ACCURACY = [
  { metric: 'Parameters', predicted: '109,482,240', observed: '109,482,240', error: 'exact' },
  { metric: 'Peak memory', predicted: '3.29 GB', observed: '3.38 GB', error: '+2.7%' },
  { metric: 'Time per step', predicted: '812 ms', observed: '836 ms', error: '+3.0%' },
  { metric: 'Throughput', predicted: '19.7/s', observed: '19.1/s', error: '−3.0%' },
] as const;

/** The stages a workload passes through, named as the user experiences them. */
export const WORKFLOW = [
  { stage: 'Design', gives: 'The model, block by block' },
  { stage: 'Analyze', gives: 'Parameters, FLOPs, tensor shapes' },
  { stage: 'Simulate', gives: 'Memory, latency, cost, carbon' },
  { stage: 'Plan', gives: 'Batch, precision, checkpoint schedule' },
  { stage: 'Train', gives: 'A real run on real hardware' },
  { stage: 'Observe', gives: 'Loss, throughput, memory, live' },
  { stage: 'Compare', gives: 'Predicted against observed' },
  { stage: 'Optimize', gives: 'The next design, better informed' },
] as const;

/** Where a workload can run. Local is implemented; the rest is the shape the
 *  contract was designed to accept, and the page says which is which. */
export const ENVIRONMENTS = [
  { name: 'Workstation', status: 'available' },
  { name: 'Local server', status: 'available' },
  { name: 'Cloud GPU', status: 'planned' },
  { name: 'AI cluster', status: 'planned' },
] as const;

export const TEAMS = [
  { name: 'AI Research', does: 'Evaluate architectures before committing compute.' },
  { name: 'ML Engineering', does: 'Understand training requirements before execution.' },
  { name: 'MLOps', does: 'Run workloads against real infrastructure, not a spec sheet.' },
  { name: 'Platform', does: 'Standardise how AI compute is measured across environments.' },
  { name: 'FinOps', does: 'Put a defensible number on every workload before it runs.' },
  { name: 'Engineering leadership', does: 'Make infrastructure decisions from measured data.' },
] as const;
