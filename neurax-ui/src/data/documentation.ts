/**
 * The user guide, as data.
 *
 * Kept as structured content rather than a folder of Markdown files, for two
 * reasons. The desktop application ships as a single bundle with no network
 * and no filesystem reads at runtime, so documentation that lives in files
 * would have to be fetched or inlined at build time anyway. And keeping it
 * typed means the search index, the table of contents and the panel all read
 * the same source — documentation cannot fall out of step with its own
 * navigation.
 *
 * What matters more than the format: **everything stated here is true of this
 * build**. A guide that describes a button that is not there costs more than no
 * guide at all, because it teaches the reader to distrust the rest. Where
 * NEURAX is approximate or incomplete, the guide says so in the same voice it
 * uses for everything else — see the "Accuracy and limits" section, which is
 * not an appendix but part of knowing how to use the tool.
 */

/** A block of content inside a section. */
export type DocBlock =
  | { kind: 'text'; text: string }
  | { kind: 'heading'; text: string }
  | { kind: 'list'; items: string[] }
  | { kind: 'steps'; items: string[] }
  | { kind: 'code'; text: string; caption?: string }
  | { kind: 'keys'; items: Array<{ keys: string; action: string }> }
  | { kind: 'table'; columns: string[]; rows: string[][] }
  | { kind: 'note'; tone: 'info' | 'warning'; title: string; text: string };

export interface DocSection {
  id: string;
  title: string;
  /** One line shown under the title in the contents. */
  summary: string;
  blocks: DocBlock[];
}

export interface DocChapter {
  id: string;
  title: string;
  sections: DocSection[];
}

export const DOCUMENTATION: DocChapter[] = [
  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'start',
    title: 'Getting started',
    sections: [
      {
        id: 'what-is-neurax',
        title: 'What NEURAX does for you',
        summary: 'The questions it answers, and the ones it does not.',
        blocks: [
          {
            kind: 'text',
            text:
              'NEURAX answers questions about a neural network **before you train it**. You describe an architecture — by drawing it, loading a template, or importing a config — and it tells you what building that model would cost: how much VRAM it needs, how many FLOPs it takes, how long training would run, what that costs in dollars and in carbon.',
          },
          {
            kind: 'text',
            text:
              'Answers arrive in {+well under a second+}, {+with no GPU+}, and {+the same design always gives the same numbers+}.',
          },
          { kind: 'heading', text: 'What it will not do' },
          {
            kind: 'list',
            items: [
              '{-It never runs your model.-} Training is PyTorch and JAX; NEURAX works before there is anything to run.',
              '{-It does not measure.-} A profiler reports a model that ran. NEURAX predicts one that has not.',
              '{-It does not export runnable code.-} See Exporting for what does leave the tool.',
            ],
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'The question it is built for',
            text:
              '"I am considering 32 layers at width 4096 with grouped-query attention. Will it fit on eight H100s, and what will a full training run cost?" That has a real answer before any GPU is booked, and finding it should take seconds.',
          },
        ],
      },
      {
        id: 'first-analysis',
        title: 'Your first analysis',
        summary: 'From an empty window to a full report in about a minute.',
        blocks: [
          {
            kind: 'steps',
            items: [
              'Open **Templates** in the toolbar and pick a model — LLaMA 2 7B is a good first choice.',
              'The canvas fills with the architecture. Each box is a block; the lines are the tensors flowing between them.',
              'Press **Run Analysis** at the top right.',
              'The panel on the right fills with results: parameters, peak VRAM, FLOPs, latency, training cost.',
              'Click any block. The inspector along the bottom shows what that block contributes on its own.',
              'Open **Hyperparameters**, raise the layer count, and run the analysis again. Every figure moves.',
            ],
          },
          {
            kind: 'text',
            text:
              'That loop — change, analyse, read — is the whole tool. Everything else in this guide makes it faster or tells you what the numbers mean.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Nothing leaves your machine',
            text:
              'On the desktop, {+no architecture you design is sent anywhere+} and {+NEURAX works with no internet connection+}. The one exception is the AI Copilot, which contacts the provider whose key you supply.',
          },
        ],
      },
      {
        id: 'workspace-tour',
        title: 'The window',
        summary: 'What each area is for.',
        blocks: [
          {
            kind: 'table',
            columns: ['Area', 'What it is for'],
            rows: [
              ['**Toolbar** (top)', 'Family selector, templates, file actions, undo, compare, hyperparameters, target, export, Run Analysis.'],
              ['**Block palette** (left)', 'Every block available for the chosen family. Drag one onto the canvas.'],
              ['**Canvas** (centre)', 'The architecture. Move blocks, connect them, zoom and pan.'],
              ['**Inspector** (bottom)', "The selected block's parameters, and what it contributes."],
              ['**Analysis panel** (right)', 'Results, in three tabs: Architecture, Performance, Hardware.'],
              ['**Workspace tabs**', 'Five views of the same design, and of the run it produces.'],
            ],
          },
          {
            kind: 'text',
            text:
              'The **?** button at the top right opens this guide, and the theme toggle beside it switches between light and dark.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'designing',
    title: 'Building a model',
    sections: [
      {
        id: 'canvas-basics',
        title: 'Working on the canvas',
        summary: 'Adding, connecting, selecting, zooming.',
        blocks: [
          { kind: 'heading', text: 'Blocks' },
          {
            kind: 'list',
            items: [
              'Drag a block from the palette onto the canvas to add it.',
              'Click a block to select it; its parameters appear in the inspector below.',
              'Drag a block to move it. Position is cosmetic — it changes nothing in the analysis.',
              'Edit a parameter in the inspector and it takes effect on the next analysis.',
              '**Ctrl+D** duplicates the selection, **Delete** removes it.',
            ],
          },
          { kind: 'heading', text: 'Connecting blocks' },
          {
            kind: 'text',
            text:
              'Drag from the output port on the right of one block to the input port on the left of another. Invalid connections are refused as you make them. Click a line to select it, then press **Delete** to remove it.',
          },
          { kind: 'heading', text: 'Selecting several at once' },
          {
            kind: 'list',
            items: [
              'Drag on empty canvas to draw a selection box.',
              'Hold **Shift** and click to add or remove one block.',
              '**Ctrl+A** selects everything.',
              '**Escape** clears the selection and cancels a connection in progress.',
            ],
          },
          { kind: 'heading', text: 'Moving around' },
          {
            kind: 'table',
            columns: ['To do this', 'Do this'],
            rows: [
              ['Zoom', 'The **+** and **−** buttons, or the zoom slider, on the canvas toolbar'],
              ['Fit the whole design on screen', 'The **fit** button on the canvas toolbar'],
              ['Pan', 'Hold **Space** and drag, hold **Shift** and drag, or drag with the middle mouse button'],
              ['Pan continuously', 'Switch to the **hand** tool on the canvas toolbar'],
            ],
          },
        ],
      },
      {
        id: 'stacks-and-groups',
        title: 'Repeated layers',
        summary: 'How to describe 32 identical blocks without drawing 32 of them.',
        blocks: [
          {
            kind: 'text',
            text:
              'Real models repeat. A transformer is one decoder block, thirty-two times. You describe that once.',
          },
          { kind: 'heading', text: 'The layer stack' },
          {
            kind: 'steps',
            items: [
              'Place a `layer_stack` block on the trunk of the design.',
              'Set its `num_layers` to how many times the body repeats.',
              'Build the body of one layer — the norms, attention, feed-forward — off to the side.',
              'Connect the positional or embedding block **into** the stack.',
              'Connect the end of each branch of the body **back to** the stack.',
            ],
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'Connect the body back, or every number is wrong',
            text:
              'If a layer body is not connected back to its stack, {-it is counted once instead of once per layer-} — and the parameter count, memory, cost and FLOPs are all wrong by that factor. When a design looks right but comes out far too small, this is almost always why. Load a template and compare the wiring.',
          },
          { kind: 'heading', text: 'Groups' },
          {
            kind: 'text',
            text:
              'Select two or more blocks and press **Ctrl+G** to group them. A group can be collapsed to tidy the canvas and carries its own repeat count. Groups are for organising your view; the layer stack is what the analysis counts.',
          },
        ],
      },
      {
        id: 'families',
        title: 'Architecture families',
        summary: 'Choosing the right one, and why it matters.',
        blocks: [
          {
            kind: 'text',
            text:
              'The selector at the top left sets the family. It decides which blocks the palette offers and how the analysis treats them. {-A mixture-of-experts model left on "transformer" has its experts counted once instead of per expert.-}',
          },
          {
            kind: 'list',
            items: [
              '**Transformer** — encoder, decoder and encoder-decoder models.',
              '**CNN** — convolutional vision models.',
              '**MoE** — mixture of experts, with routers and expert blocks.',
              '**SSM** — state-space models such as Mamba and RWKV.',
              '**Diffusion** — denoisers, U-Nets and DiTs.',
              '**GNN** — graph networks.',
              '**GAN** — generator and discriminator pairs.',
              '**RNN** — LSTM and GRU models.',
            ],
          },
          {
            kind: 'text',
            text:
              'There are **150 blocks** across these 8 families and **30 reference templates** built from them.',
          },
        ],
      },
      {
        id: 'templates',
        title: 'Templates',
        summary: 'Starting from a model that already exists.',
        blocks: [
          {
            kind: 'steps',
            items: [
              'Press **Templates** in the toolbar.',
              'The list shows the templates for the family you have selected. Switch family to see others.',
              'Press a template to load it. It replaces whatever is on the canvas.',
              'Press **clone** instead to load a copy you can edit freely under a new name.',
            ],
          },
          {
            kind: 'text',
            text:
              'Every block in a loaded template is editable — a template is a starting point, not a fixed object.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Loading a template starts a new document',
            text:
              'Undo will not reach back into whatever you had open before. Save first if you want to keep it.',
          },
        ],
      },
      {
        id: 'hyperparameters',
        title: 'Hyperparameters and target hardware',
        summary: 'The settings every number is computed against.',
        blocks: [
          {
            kind: 'text',
            text:
              'Two toolbar dialogs set the terms of the analysis. They matter as much as the architecture: the same model on different hardware, or at a different precision, gives entirely different numbers.',
          },
          { kind: 'heading', text: 'Hyperparameters' },
          {
            kind: 'table',
            columns: ['Group', 'What you set'],
            rows: [
              ['Model', 'Width, depth, attention and key-value heads, head dimension, feed-forward width, vocabulary, sequence length'],
              ['Training', 'Batch size, learning rate, epochs, optimizer, weight decay, warmup, schedule, gradient checkpointing, early stopping'],
              ['Parallelism', 'Tensor, pipeline and expert parallel degrees, micro-batch size, gradient accumulation, ZeRO stage'],
              ['Data', 'Dataset size, vocabulary, number of classes'],
              ['Family-specific', 'Expert count and top-k for MoE, state dimension for SSM, image size for CNN, and so on'],
            ],
          },
          {
            kind: 'text',
            text:
              'You can also add your own named parameters, for a block that takes something the schema does not list.',
          },
          { kind: 'heading', text: 'Target' },
          {
            kind: 'text',
            text:
              'The chip every metric is computed for: which GPU, how many, and how they are connected. NEURAX ships specifications for {+more than twenty GPUs+} — A100, H100, V100, the RTX 40 series and others — with their memory, bandwidth and per-precision throughput.',
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'Precision moves every memory figure at once',
            text:
              'Switching between fp32, bf16 and fp8 changes weights, activations, gradients and optimizer state together. When comparing two designs, {-keep the precision the same-} or you are measuring the precision, not the architecture.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'importing',
    title: 'Bringing models in',
    sections: [
      {
        id: 'import-hf',
        title: 'Importing a HuggingFace config.json',
        summary: 'Load any open-weights model from the file that defines its shape.',
        blocks: [
          {
            kind: 'text',
            text:
              'Every model on the HuggingFace Hub ships a `config.json` beside its weights, and that file fully determines the architecture. NEURAX reads it directly, so {+working on a model that already exists does not mean rebuilding it block by block+}.',
          },
          {
            kind: 'steps',
            items: [
              'Press **Import** in the toolbar.',
              'Press **Open File** and pick a `config.json`, or paste its contents into the box.',
              'The badge above the box names the format it recognised.',
              'Press **Read File**. A summary appears: depth, width, heads, vocabulary, and the structural choices it read.',
              'Read the amber box, if there is one — it lists anything the file did not state.',
              'Press **Import**. The architecture appears on the canvas and the analysis runs.',
            ],
          },
          { kind: 'heading', text: 'What it reads' },
          {
            kind: 'list',
            items: [
              '**Dense decoders** — LLaMA, Mistral, Qwen, Gemma, Phi, OLMo, StableLM, Falcon, GPT-2, GPT-NeoX, OPT, Bloom.',
              '**Encoders** — BERT, RoBERTa, DistilBERT, ELECTRA and relatives, which get a classification head instead of an LM head.',
              '**Mixture of experts** — Mixtral, Qwen-MoE and DeepSeek-style configs, including shared experts.',
              '**Multimodal** — the language tower, from `text_config`. {-Vision and audio towers are not imported.-}',
            ],
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'It tells you what it assumed',
            text:
              'Configs vary in what they state. Where a field is missing, NEURAX takes the conventional default and lists every such choice before you import, so {+a guess is never silent+}.',
          },
          { kind: 'heading', text: 'How accurate is an import?' },
          {
            kind: 'table',
            columns: ['Model', 'Published', 'NEURAX', 'Error'],
            rows: [
              ['LLaMA-2 7B', '6.74 B', '6.63 B', '−1.7 %'],
              ['Mistral 7B', '7.24 B', '7.13 B', '−1.5 %'],
              ['Qwen2 7B', '7.62 B', '7.62 B', '−0.04 %'],
              ['GPT-2', '124 M', '124 M', '< 1 %'],
            ],
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'Routed models come out too small',
            text:
              'A mixture-of-experts model imported from its config analyses {-about 22 % below its published size-} — Mixtral 8x7B lands near 36 B against a published 46.7 B. {-Treat MoE parameter counts, and the memory and cost figures that follow from them, as a lower bound.-}',
          },
        ],
      },
      {
        id: 'import-neurax',
        title: 'Importing a NEURAX design',
        summary: 'Bringing back an architecture exported as JSON.',
        blocks: [
          {
            kind: 'text',
            text:
              'The same Import dialog accepts a NEURAX design — the JSON that **Export** produces. It is recognised automatically; you never pick a format.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Import or Open?',
            text:
              '**Open** is for a `.neurax` document you saved, and restores everything including hardware settings. **Import** brings in a model from elsewhere. If you want a design back exactly as it was, save it as a `.neurax` file and use Open.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'documents',
    title: 'Saving and sharing',
    sections: [
      {
        id: 'neurax-files',
        title: 'The .neurax file',
        summary: 'Your design as a document on disk.',
        blocks: [
          {
            kind: 'text',
            text:
              'A design saves to a `.neurax` file holding everything: every block and connection, the groups, the family, the full hardware and training configuration, and the last analysis for reference.',
          },
          {
            kind: 'keys',
            items: [
              { keys: 'Ctrl+S', action: 'Save to the file this design came from' },
              { keys: 'Ctrl+Shift+S', action: 'Save As — choose a new file' },
              { keys: 'Ctrl+O', action: 'Open a .neurax file' },
            ],
          },
          {
            kind: 'text',
            text:
              'A dot beside **Save** in the toolbar means there are changes not yet written to the file. Closing with unsaved work asks first.',
          },
          {
            kind: 'text',
            text:
              'A file can go in a repository next to the training code it describes, be attached to a review, and be sent to a colleague who then opens it. It is written so that {+changing one hyperparameter changes exactly one line in `git diff`+}.',
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'In a browser, Save always asks',
            text:
              '{-A web page cannot write to a path-}, so every save in the browser downloads a fresh copy. {+Only the desktop application re-saves a file in place+}, and only for files you opened or saved in that session.',
          },
        ],
      },
      {
        id: 'projects',
        title: 'Projects',
        summary: 'Designs the application keeps for you.',
        blocks: [
          {
            kind: 'text',
            text:
              'Alongside files, NEURAX keeps a list of projects, reachable from the toolbar. Press **Save Current** to store the design on screen, pick one from the list to load it, or delete one you no longer want. {+Projects survive restarts+} — on the desktop they are written to your profile and saved again when you quit.',
          },
          {
            kind: 'table',
            columns: ['', 'Project', '.neurax file'],
            rows: [
              ['Where it lives', 'Application profile', 'Wherever you put it'],
              ['Survives a restart', 'Yes', 'Yes'],
              ['Can go in git', '{-No-}', '{+Yes+}'],
              ['Can be sent to someone', '{-No-}', '{+Yes+}'],
              ['Best for', 'Work in progress', 'Anything you want to keep or share'],
            ],
          },
        ],
      },
      {
        id: 'exporting',
        title: 'Exporting',
        summary: 'Getting the architecture out in other forms.',
        blocks: [
          {
            kind: 'text',
            text: 'Press **Export** in the toolbar and choose a format.',
          },
          {
            kind: 'table',
            columns: ['Format', 'What you get'],
            rows: [
              ['**JSON**', 'The architecture and its analysis, re-importable into NEURAX.'],
              ['**NEURAX IR**', 'The exact topology that was analysed. Attach this to a bug report.'],
              ['**TOML**', 'The same topology, in a format meant to be hand-edited and diffed.'],
              ['**GitHub**', 'Push the architecture straight to a repository.'],
            ],
          },
          {
            kind: 'text',
            text: 'The **Full Project** tab is different: it generates a runnable PyTorch project — model.py, train.py, and the rest of the package — from the compiled design, not just a description of it.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Full Project is checked, not guessed',
            text:
              'Generated code is cross-checked against the parameter count this design actually compiled to. A **verified** badge means they match; a **needs review** badge means they don\'t, and says so rather than staying silent about the gap.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'iterating',
    title: 'Working day to day',
    sections: [
      {
        id: 'undo',
        title: 'Undo and redo',
        summary: 'Nothing on the canvas is permanent.',
        blocks: [
          {
            kind: 'keys',
            items: [
              { keys: 'Ctrl+Z', action: 'Undo' },
              { keys: 'Ctrl+Shift+Z', action: 'Redo' },
              { keys: 'Ctrl+Y', action: 'Redo (Windows convention)' },
            ],
          },
          {
            kind: 'text',
            text:
              'History covers {+every change to the design+}: blocks added, moved, deleted or duplicated, connections made and broken, parameters edited, groups formed. A drag is one step, so a single Ctrl+Z puts the block back where it started. {+Two hundred steps are kept.+}',
          },
          {
            kind: 'text',
            text:
              'Opening a file, loading a template or starting a blank page begins a new history. {-Undo will not reach back into the document you left.-}',
          },
        ],
      },
      {
        id: 'compare',
        title: 'Comparing two designs',
        summary: 'Hold a baseline and see exactly what your change moved.',
        blocks: [
          {
            kind: 'steps',
            items: [
              'Analyse the design you want to measure against.',
              'Press **Compare**, then **Capture current design**.',
              'Close the panel, change the architecture, and run the analysis again.',
              'Reopen **Compare**. Both are shown side by side, with the change in each metric.',
            ],
          },
          {
            kind: 'text',
            text:
              'Changes are coloured by whether they are an improvement, not by their sign: {+less VRAM is green+}, {-less throughput is red-}. Parameter count is left uncoloured, because more parameters is neither good nor bad. A change under half a percent shows as `=` rather than being dressed up as a result.',
          },
          {
            kind: 'text',
            text:
              'Press **Make current the baseline** to move the comparison forward, or **Clear baseline** to start again.',
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'It tells you when a comparison is not fair',
            text:
              'If the two were analysed on different GPUs, at a different precision, batch size, sequence length or family, an amber box lists which terms differ. {-Part of the difference is then the settings, not the architecture.-}',
          },
        ],
      },
      {
        id: 'copilot',
        title: 'The AI Copilot',
        summary: 'Describing an architecture in words.',
        blocks: [
          {
            kind: 'steps',
            items: [
              'Press **Neurax AI** in the toolbar to open the chat panel.',
              'The first time, choose a provider — OpenAI, Anthropic, Google or Mistral — and paste your own API key.',
              'Ask for what you want: "create a transformer for image classification", "add grouped-query attention", "why is this running out of memory".',
              'Review the blocks it produced on the canvas, then run the analysis.',
            ],
          },
          {
            kind: 'text',
            text:
              'The key is stored in your profile and {+sent only to the provider you chose — never to NEURAX+}. You can change or remove it from your account page.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Check what it builds',
            text:
              'The Copilot is a fast way to get a first draft onto the canvas. {-It is not a source of truth about your model.-} Read the blocks and run the analysis before trusting the shape.',
          },
        ],
      },
      {
        id: 'shortcuts',
        title: 'Keyboard shortcuts',
        summary: 'Everything bound to a key.',
        blocks: [
          { kind: 'heading', text: 'Document' },
          {
            kind: 'keys',
            items: [
              { keys: 'Ctrl+S', action: 'Save' },
              { keys: 'Ctrl+Shift+S', action: 'Save As' },
              { keys: 'Ctrl+O', action: 'Open a .neurax file' },
            ],
          },
          { kind: 'heading', text: 'Editing' },
          {
            kind: 'keys',
            items: [
              { keys: 'Ctrl+Z', action: 'Undo' },
              { keys: 'Ctrl+Shift+Z  /  Ctrl+Y', action: 'Redo' },
              { keys: 'Ctrl+A', action: 'Select every block' },
              { keys: 'Ctrl+D', action: 'Duplicate the selection' },
              { keys: 'Ctrl+G', action: 'Group the selected blocks' },
              { keys: 'Delete  /  Backspace', action: 'Delete the selection' },
              { keys: 'Escape', action: 'Clear the selection, cancel a connection' },
            ],
          },
          { kind: 'heading', text: 'Canvas' },
          {
            kind: 'keys',
            items: [
              { keys: 'Space', action: 'Hold and drag to pan' },
              { keys: 'Shift', action: 'Hold and drag to pan, or hold and click to extend a selection' },
            ],
          },
          { kind: 'heading', text: 'Help' },
          {
            kind: 'keys',
            items: [
              { keys: 'F1', action: 'Open this guide — works even while typing' },
            ],
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Typing always wins',
            text:
              'While the cursor is in a text field these shortcuts stand down, so Ctrl+Z there undoes your typing.',
          },
          { kind: 'text', text: 'On macOS use **Cmd** wherever **Ctrl** is written.' },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'workspaces',
    title: 'The five workspaces',
    sections: [
      {
        id: 'workspace-list',
        title: 'What each one shows',
        summary: 'Five views of the same design.',
        blocks: [
          {
            kind: 'text',
            text:
              'The tabs switch what you are looking at, never what you have built. {+Switching tabs cannot change your architecture.+}',
          },
          { kind: 'heading', text: 'Architecture' },
          {
            kind: 'text',
            text:
              'Where you design. The canvas, the palette, the inspector and the analysis panel. This is where you spend most of your time.',
          },
          { kind: 'heading', text: 'Simulation' },
          {
            kind: 'text',
            text:
              'Predicted training-time behaviour: memory over the course of a step, throughput, gradient and optimizer breakdowns, per-layer latency and VRAM, and KV-cache growth against sequence length. Every figure here is derived, never measured — what a run actually does is in Training.',
          },
          { kind: 'heading', text: 'Production' },
          {
            kind: 'text',
            text:
              'What deploying this model looks like — serving shape and the practical consequences of the design.',
          },
          { kind: 'heading', text: 'Training' },
          {
            kind: 'text',
            text:
              'The one workspace where something is running. Start a run on your own machine and watch it: loss, throughput, memory and GPU state as the process reports them, {+beside what NEURAX predicted+} — so you can see where the analysis held and where it did not.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'A run is a directory, not a window',
            text:
              'Closing NEURAX {+does not stop a run+}, and reopening it picks the session back up. If the studio is closed abruptly, the run is listed as {-interrupted-} and resumes from its last checkpoint — nothing is lost.',
          },
          { kind: 'heading', text: 'Time Machine' },
          {
            kind: 'text',
            text:
              'Three- to five-year projections of cost, carbon and scaling, hardware migration planning, and regulatory tracking for the EU AI Act, CSRD and DSA.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'results',
    title: 'Reading the results',
    sections: [
      {
        id: 'metrics',
        title: 'What the numbers mean',
        summary: 'The figures that decide whether a design is viable.',
        blocks: [
          {
            kind: 'text',
            text:
              'The analysis panel on the right has three tabs — Architecture, Performance and Hardware. Between them they show the figures below; the full set of 66 metrics is in the exported report.',
          },
          { kind: 'heading', text: 'Model' },
          {
            kind: 'table',
            columns: ['Metric', 'Meaning'],
            rows: [
              ['Parameters', 'Total learnable weights — the headline size.'],
              ['Layers', 'Depth, taken from the layer stack, not from how many blocks you drew.'],
              ['Graph depth', 'Longest path through the design.'],
            ],
          },
          { kind: 'heading', text: 'Memory' },
          {
            kind: 'table',
            columns: ['Metric', 'Meaning'],
            rows: [
              ['Peak VRAM', 'The high-water mark during a training step. This decides whether the model fits.'],
              ['Weights', 'Parameters at the chosen precision.'],
              ['Activations', 'Intermediates held for the backward pass. Gradient checkpointing trades these for compute.'],
              ['Gradients', 'One value per parameter.'],
              ['Optimizer state', 'AdamW adds two more values per parameter on top of the gradients.'],
              ['Max batch that fits', 'The largest batch that stays inside the target GPU memory.'],
            ],
          },
          { kind: 'heading', text: 'Compute' },
          {
            kind: 'table',
            columns: ['Metric', 'Meaning'],
            rows: [
              ['Total FLOPs', 'Operations for a full step, forward and backward.'],
              ['FLOPs per token', 'The usual scale for comparing language models.'],
              ['Arithmetic intensity', 'FLOPs per byte moved. Low means memory-bound, high means compute-bound.'],
              ['Roofline position', "Where the model sits against the target chip's ridge point."],
            ],
          },
          { kind: 'heading', text: 'Performance and cost' },
          {
            kind: 'table',
            columns: ['Metric', 'Meaning'],
            rows: [
              ['Latency', 'Estimated time for one forward pass.'],
              ['Throughput', 'Tokens per second at the configured batch size.'],
              ['Training cost', 'Dollars for a full run at the configured epochs and dataset size.'],
              ['Training time', 'Wall-clock hours on the configured GPU count.'],
              ['Energy / CO₂', 'Kilowatt-hours and kilograms for the run.'],
            ],
          },
        ],
      },
      {
        id: 'diagnostics',
        title: 'Diagnostics — where the optimisation is',
        summary: 'The most useful thing NEURAX produces, and how to read it.',
        blocks: [
          {
            kind: 'text',
            text:
              'The metrics tell you what a design costs. The diagnostics tell you {+what to do about it+}. They appear under the analysis in the Architecture tab.',
          },
          { kind: 'heading', text: 'Reading a row by its colour' },
          {
            kind: 'table',
            columns: ['Colour', 'Label', 'What it means'],
            rows: [
              ['{-Red-}', 'Blocking', 'The model will not run as configured. Fix this first.'],
              ['Amber', 'Warning', 'It may run, but something is wrong or wasteful.'],
              ['{+Green+}', 'Opportunity', 'A specific, quantified way to make the model better.'],
              ['Grey', 'Note', 'Context. Nothing to act on.'],
            ],
          },
          {
            kind: 'text',
            text:
              'Rows are ordered by that list, so what blocks you is always at the top. A summary line above them says what was found. Each row carries the category it belongs to — Memory, Bottleneck, Parallelism, Architecture, Cost, Custom block, Shapes, Configuration — and, after the arrow, {+what to do about it+}.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'The green rows are the point',
            text:
              'A green row is not a problem to silence. It is NEURAX telling you, for example, that optimizer state is 43 % of your memory and that ZeRO stage 1 would shard it across ranks without changing the maths. {+That is the fastest route to a model that fits.+}',
          },
          { kind: 'heading', text: 'Recommendations' },
          {
            kind: 'text',
            text:
              'Below the diagnostics, recommendations are ordered by priority and carry a quantified impact — "Save ~12.0 GB VRAM". {+That number is the reason to act on one rather than another.+}',
          },
          { kind: 'heading', text: 'Issues' },
          {
            kind: 'text',
            text:
              'The Issues block lists problems found in the design itself: unresolved shapes, blocks with no path from the input, a missing input or output. The inspector shows a count and jumps you there.',
          },
        ],
      },
      {
        id: 'accuracy',
        title: 'Accuracy and limits',
        summary: 'What to trust, what to treat as an estimate, and what is missing.',
        blocks: [
          {
            kind: 'text',
            text:
              'NEURAX predicts. Some of what it predicts is checked against published figures; some is not. Knowing which is which is part of using it well.',
          },
          { kind: 'heading', text: 'Checked against published figures' },
          {
            kind: 'table',
            columns: ['Model', 'Published', 'NEURAX', 'Error'],
            rows: [
              ['VGG-16', '138.0 M', '138.4 M', '+0.3 %'],
              ['Mixtral 8x7B', '46.7 B', '47.4 B', '+1.5 %'],
              ['LLaMA-2 70B', '70.0 B', '68.7 B', '−1.8 %'],
              ['ResNet-50', '25.6 M', '26.5 M', '+3.5 %'],
              ['RWKV 7B', '7.5 B', '7.2 B', '−4.2 %'],
              ['DeepSeek-V3', '671 B', '701 B', '+4.5 %'],
              ['Mamba 2.8B', '2.80 B', '2.66 B', '−4.9 %'],
            ],
          },
          {
            kind: 'text',
            text:
              '{+Parameter counts for these models are verified on every build+}, together with four imported configs checked {+end to end through the real compiler+}. Seven models is what is measured; it is not a claim about every architecture that exists.',
          },
          { kind: 'heading', text: 'Not checked against measurement' },
          {
            kind: 'list',
            items: [
              '**Latency and throughput** are roofline estimates from the chip\'s specifications. {-No measured run has been compared against them.-}',
              '**Training cost and time** follow from those estimates and inherit their uncertainty, plus whatever your cloud actually charges.',
              '**Energy and CO₂** rest on published chip power figures and an assumed grid intensity.',
              '**Inference Intelligence** widgets are analytical indicators, {-not measurements of a served model-}.',
            ],
          },
          { kind: 'heading', text: 'Known gaps' },
          {
            kind: 'list',
            items: [
              '**Mixture of experts.** Routed models drawn or imported in the studio {-under-count by roughly 22 %-} against their published size. {-Treat MoE numbers from the canvas as a lower bound.-}',
              '**Multimodal models.** Only the language tower is imported, so {-totals cover the text side alone-}.',
              '**No calibration.** {-NEURAX does not learn from your measured runs.-} Every prediction is the formula\'s, not your cluster\'s.',
            ],
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'The short version',
            text:
              '{+Trust parameters, memory and FLOPs for dense models, within a few percent.+} {-Treat latency, cost and carbon as estimates for comparing designs against each other, not as forecasts of your invoice.-}',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'troubleshooting',
    title: 'Troubleshooting',
    sections: [
      {
        id: 'common-problems',
        title: 'When something looks wrong',
        summary: 'The usual causes, in the order to check them.',
        blocks: [
          { kind: 'heading', text: 'The parameter count is far too low' },
          {
            kind: 'text',
            text:
              'Almost always the layer stack. Check the layer body is connected back to the `layer_stack` block and that `num_layers` is right. See "Repeated layers".',
          },
          { kind: 'heading', text: 'A block seems to be ignored' },
          {
            kind: 'text',
            text:
              'Check it is connected at both ends. A block with no path from the input is not part of the model, and the Issues block will say so.',
          },
          { kind: 'heading', text: 'The analysis will not run' },
          {
            kind: 'text',
            text:
              'Read the message — it names the cause. The usual ones are no target GPU selected (open **Target**), an empty canvas, or a missing Input or Output block. If a shape cannot be resolved, the diagnostic names the block.',
          },
          { kind: 'heading', text: 'Two analyses of the same design differ' },
          {
            kind: 'text',
            text:
              '{+They should not.+} Something changed in the hyperparameters or the target between the runs. Capture a baseline and use **Compare** — it lists exactly which setting moved.',
          },
          { kind: 'heading', text: 'A file will not open' },
          {
            kind: 'text',
            text:
              '**Open** takes `.neurax` documents only. Exported IR and HuggingFace configs go through **Import**. A file written by a newer NEURAX says so rather than opening partially.',
          },
          { kind: 'heading', text: 'Save says it cannot write in place' },
          {
            kind: 'text',
            text:
              'The file may have moved or become read-only since you opened it. NEURAX falls back to Save As so you can choose a new location.',
          },
          { kind: 'heading', text: 'An error mentions a defect in NEURAX' },
          {
            kind: 'text',
            text:
              'Then it is not your design. Export the **NEURAX IR** and attach it to a bug report — it is the exact input that failed.',
          },
        ],
      },
      {
        id: 'where-things-live',
        title: 'Where NEURAX keeps things',
        summary: 'Files on disk, on each platform.',
        blocks: [
          { kind: 'text', text: 'Saved projects live in your profile, under `dev.neurax.desktop`:' },
          {
            kind: 'table',
            columns: ['Platform', 'Location'],
            rows: [
              ['Linux', '`~/.local/share/dev.neurax.desktop/projects.json`'],
              ['macOS', '`~/Library/Application Support/dev.neurax.desktop/projects.json`'],
              ['Windows', '`%APPDATA%\\dev.neurax.desktop\\projects.json`'],
            ],
          },
          {
            kind: 'text',
            text:
              'Your `.neurax` files live wherever you saved them. The Copilot API key is kept in your profile and {+never travels with a design+}.',
          },
        ],
      },
    ],
  },
];

/** Every section, flattened, for search and for direct linking. */
export function allSections(): Array<DocSection & { chapter: string; chapterId: string }> {
  return DOCUMENTATION.flatMap((chapter) =>
    chapter.sections.map((section) => ({
      ...section,
      chapter: chapter.title,
      chapterId: chapter.id,
    })),
  );
}

/** The searchable text of a section, title and body together. */
function searchableText(section: DocSection): string {
  const parts: string[] = [section.title, section.summary];

  for (const block of section.blocks) {
    switch (block.kind) {
      case 'text':
      case 'heading':
        parts.push(block.text);
        break;
      case 'code':
        parts.push(block.text, block.caption ?? '');
        break;
      case 'list':
      case 'steps':
        parts.push(...block.items);
        break;
      case 'keys':
        parts.push(...block.items.map((k) => `${k.keys} ${k.action}`));
        break;
      case 'table':
        parts.push(...block.columns, ...block.rows.flat());
        break;
      case 'note':
        parts.push(block.title, block.text);
        break;
    }
  }

  return parts.join(' ').toLowerCase();
}

/**
 * Sections matching a query, best first.
 *
 * Deliberately simple: every term must appear somewhere in the section, and
 * sections whose title matches sort first. A guide this size does not need an
 * index, and a fuzzy matcher would return the wrong section confidently rather
 * than nothing honestly.
 */
export function searchDocs(query: string): Array<DocSection & { chapter: string; chapterId: string }> {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [];

  return allSections()
    .map((section) => {
      const haystack = searchableText(section);
      if (!terms.every((term) => haystack.includes(term))) return null;

      const title = section.title.toLowerCase();
      const score = terms.filter((term) => title.includes(term)).length;
      return { section, score };
    })
    .filter((hit): hit is { section: ReturnType<typeof allSections>[number]; score: number } => hit !== null)
    .sort((a, b) => b.score - a.score)
    .map((hit) => hit.section);
}
