/**
 * The Code view is the first place in NEURAX where someone can change what
 * gets trained by typing. Everything here is about the two ways that goes
 * wrong: an edit that silently does nothing, and an edit that silently does
 * something without being checked.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

import { CodeWorkspace } from './CodeWorkspace.tsx';
import type { CandidateReview } from '@/services/modelCandidate.ts';

const FILES = [
  { path: 'README.md', content: '# My Model\n' },
  { path: 'requirements.txt', content: 'torch>=2.0\n' },
  { path: 'train.py', content: 'from src.model import NeuraxModel\n' },
  { path: 'src/model.py', content: 'import torch.nn as nn\n\n\nclass NeuraxModel(nn.Module):\n    pass\n' },
];

const accepted: CandidateReview = {
  accepted: true, reason: 'Built by PyTorch: 212 parameters.', stage: 'verified',
  builtParams: 212, analyzedParams: 212, deltaPct: 0, outputShape: [2, 4],
  unchecked: false, verdict: null,
};
const refused: CandidateReview = {
  accepted: false, reason: 'The file does not import. SyntaxError: invalid syntax', stage: 'import',
  builtParams: null, analyzedParams: 212, deltaPct: null, outputShape: null,
  unchecked: false, verdict: null,
};

function setup(props: Partial<React.ComponentProps<typeof CodeWorkspace>> = {}) {
  const onVerify = vi.fn().mockResolvedValue(accepted);
  render(
    <CodeWorkspace
      files={FILES}
      modelPath="src/model.py"
      modelClassName="NeuraxModel"
      onVerify={onVerify}
      activeSource="generator"
      {...props}
    />,
  );
  return { onVerify };
}

const editor = () => screen.getByLabelText('src/model.py') as HTMLTextAreaElement;

beforeEach(() => vi.clearAllMocks());

describe('CodeWorkspace — the files', () => {
  it('lists every file in the project, grouped by directory', () => {
    setup();
    for (const file of FILES) {
      expect(screen.getByText(file.path.split('/').pop()!)).toBeInTheDocument();
    }
    expect(screen.getByText('src')).toBeInTheDocument();
  });

  it('opens on the model, which is the file that decides what trains', () => {
    setup();
    expect(editor()).toHaveValue(FILES[3].content);
  });

  it('shows another file when it is picked', () => {
    setup();
    fireEvent.click(screen.getByText('train.py'));
    expect(screen.getByLabelText('train.py')).toHaveValue(FILES[2].content);
  });

  it('makes every file but the model read-only, and says why', () => {
    // A text box that accepts edits the runtime then ignores is a lie told by
    // a UI. `train.py` is regenerated for export; the run uses its own harness.
    setup();
    fireEvent.click(screen.getByText('train.py'));
    expect(screen.getByLabelText('train.py')).toHaveAttribute('readonly');
    expect(screen.getByText(/regenerated from the canvas/i)).toBeInTheDocument();
  });

  it('leaves the model editable', () => {
    setup();
    expect(editor()).not.toHaveAttribute('readonly');
  });
});

describe('CodeWorkspace — an edit is not trusted for being typed', () => {
  it('cannot be verified until something has actually changed', () => {
    setup();
    expect(screen.getByRole('button', { name: /verify and use/i })).toBeDisabled();
  });

  it('offers verification once the model is edited', () => {
    setup();
    fireEvent.change(editor(), { target: { value: 'changed' } });
    expect(screen.getByRole('button', { name: /verify and use/i })).toBeEnabled();
  });

  it('sends exactly what is on screen to be built', async () => {
    const { onVerify } = setup();
    fireEvent.change(editor(), { target: { value: 'class Other:\n    pass\n' } });
    fireEvent.click(screen.getByRole('button', { name: /verify and use/i }));
    await waitFor(() => expect(onVerify).toHaveBeenCalledWith('class Other:\n    pass\n'));
  });

  it('reports a refusal in the same words the assistant would get', async () => {
    setup({ onVerify: vi.fn().mockResolvedValue(refused) });
    fireEvent.change(editor(), { target: { value: 'broken(' } });
    fireEvent.click(screen.getByRole('button', { name: /verify and use/i }));
    await screen.findByText(/SyntaxError: invalid syntax/);
  });

  it('reports an acceptance', async () => {
    setup();
    fireEvent.change(editor(), { target: { value: 'fine' } });
    fireEvent.click(screen.getByRole('button', { name: /verify and use/i }));
    await screen.findByText(/Built by PyTorch: 212 parameters\./);
  });

  it('drops a verdict as soon as the text moves past what was checked', async () => {
    // The most misleading thing this view could do: show "accepted" over text
    // that is not the text that was accepted.
    setup();
    fireEvent.change(editor(), { target: { value: 'fine' } });
    fireEvent.click(screen.getByRole('button', { name: /verify and use/i }));
    await screen.findByText(/Built by PyTorch/);

    fireEvent.change(editor(), { target: { value: 'fine, but different' } });
    expect(screen.queryByText(/Built by PyTorch/)).not.toBeInTheDocument();
    expect(screen.getByText(/not yet built/i)).toBeInTheDocument();
  });

  it('cannot verify when there is no service to ask, and says so', () => {
    setup({ onVerify: undefined });
    fireEvent.change(editor(), { target: { value: 'changed' } });
    const button = screen.getByRole('button', { name: /verify and use/i });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('title', expect.stringContaining('not answering'));
  });
});

describe('CodeWorkspace — discarding', () => {
  it('offers to discard only once there is something to discard', () => {
    setup();
    expect(screen.queryByRole('button', { name: /discard edit/i })).not.toBeInTheDocument();
    fireEvent.change(editor(), { target: { value: 'changed' } });
    expect(screen.getByRole('button', { name: /discard edit/i })).toBeInTheDocument();
  });

  it('puts the generated file back, and clears the verdict with it', async () => {
    setup({ onVerify: vi.fn().mockResolvedValue(refused) });
    fireEvent.change(editor(), { target: { value: 'broken(' } });
    fireEvent.click(screen.getByRole('button', { name: /verify and use/i }));
    await screen.findByText(/SyntaxError/);

    fireEvent.click(screen.getByRole('button', { name: /discard edit/i }));
    expect(editor()).toHaveValue(FILES[3].content);
    expect(screen.queryByText(/SyntaxError/)).not.toBeInTheDocument();
  });
});

describe('CodeWorkspace — who wrote what is running', () => {
  it('names the translator when the code came from the canvas', () => {
    setup();
    expect(screen.getByText(/Translated from your canvas/i)).toBeInTheDocument();
  });

  it('names the assistant when it wrote the model', () => {
    setup({ activeSource: 'assistant' });
    expect(screen.getByText(/Written by the assistant/i)).toBeInTheDocument();
  });

  it('names you when your edit is the one in force', () => {
    setup({ activeSource: 'you' });
    expect(screen.getByText(/Edited by you/i)).toBeInTheDocument();
  });

  it('carries the verdict on the model in force before anything is edited', () => {
    setup({ activeSource: 'assistant', activeReview: accepted });
    expect(screen.getByText(/Built by PyTorch: 212 parameters\./)).toBeInTheDocument();
  });
});
