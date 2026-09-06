/**
 * A run must be stoppable, and the composer must not offer what it cannot do.
 *
 * There was no way to end a run early: once started, the only exits were the
 * agent's own step/time ceilings, and every step until then is a real LLM call
 * billed to the user's own key. The microphone button beside Send was worse
 * than useless — it had no handler at all, so it promised voice input the
 * studio has never implemented.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

import AIChatDrawer from './AIChatDrawer.tsx';
import { ApiKeyProvider } from '@/contexts/ApiKeyContext.tsx';

/** jsdom has no EventSource; the drawer opens one for every run. */
class FakeEventSource {
  static last: FakeEventSource | null = null;
  closed = false;
  onerror: ((e: Event) => void) | null = null;
  constructor(public url: string) {
    FakeEventSource.last = this;
  }
  addEventListener() {}
  close() {
    this.closed = true;
  }
}

const snapshot = { family: 'cnn', nodes: [], connections: [] } as never;

function renderDrawer() {
  return render(
    <ApiKeyProvider>
      <AIChatDrawer open onOpenChange={() => {}} getSnapshot={() => snapshot} />
    </ApiKeyProvider>,
  );
}

/** Stub `fetch`, recording every call, and hand back the record. */
function stubFetch(): { url: string; method: string }[] {
  const calls: { url: string; method: string }[] = [];
  vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), method: String(init?.method ?? 'GET') });
    return Promise.resolve({
      ok: true,
      status: 200,
      text: () => Promise.resolve('{}'),
      json: () => Promise.resolve({ run_id: 'run-123' }),
    } as Response);
  });
  return calls;
}

async function startARun() {
  fireEvent.change(screen.getByPlaceholderText(/how can neurax help/i), {
    target: { value: 'build me a small cnn' },
  });
  fireEvent.click(screen.getByRole('button', { name: /send message/i }));
  await waitFor(() => expect(screen.getByRole('button', { name: /stop the agent/i })).toBeInTheDocument());
}

afterEach(() => {
  vi.unstubAllGlobals();
  FakeEventSource.last = null;
});

describe('the composer', () => {
  it('no longer offers controls it never implemented', () => {
    renderDrawer();
    // None of the three had a handler at all: the microphone promised voice
    // input the studio has never had, "Plan" a mode that does not exist (the
    // agent's roadmap is produced by the backend, unprompted), and "+" an
    // attachment flow with nothing behind it.
    expect(screen.queryByRole('button', { name: /voice input/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^plan$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /add context/i })).not.toBeInTheDocument();
  });

  it('turns Send into Stop while the agent is working', async () => {
    stubFetch();
    vi.stubGlobal('EventSource', FakeEventSource);
    renderDrawer();

    // Idle: Send, and no way to stop something that is not running.
    expect(screen.getByRole('button', { name: /send message/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /stop the agent/i })).not.toBeInTheDocument();

    await startARun();
    expect(screen.queryByRole('button', { name: /send message/i })).not.toBeInTheDocument();
  });

  it('cancels the run on the server, not just locally', async () => {
    const calls = stubFetch();
    vi.stubGlobal('EventSource', FakeEventSource);
    renderDrawer();
    await startARun();

    fireEvent.click(screen.getByRole('button', { name: /stop the agent/i }));

    await waitFor(() => {
      // Closing the stream alone would leave the backend spending the user's
      // key until it happened to notice the socket was gone.
      expect(calls.some((c) => c.method === 'DELETE' && c.url.endsWith('/runs/run-123'))).toBe(true);
    });
    expect(FakeEventSource.last?.closed).toBe(true);
  });

  it('says it stopped, and hands the composer back', async () => {
    stubFetch();
    vi.stubGlobal('EventSource', FakeEventSource);
    renderDrawer();
    await startARun();

    fireEvent.click(screen.getByRole('button', { name: /stop the agent/i }));

    await waitFor(() => {
      // "Stopped", never "Done" — what is on the canvas is only whatever the
      // run had already applied.
      expect(screen.getByText(/stopped/i)).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: /send message/i })).toBeInTheDocument();
  });
});
