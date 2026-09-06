/**
 * The Model field must actually offer the provider's real models on screen.
 *
 * The unit tests next to `providerModels.ts` prove the fetch is shaped right;
 * this proves a user can reach it — that the control is rendered, enabled by
 * a key, and that a successful load really does replace the free-text field
 * (which is what let a stale default like Fireworks'
 * `llama-v3p1-70b-instruct` reach the agent and 404).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import Account from './Account.tsx';
import { AuthProvider } from '@/contexts/AuthContext.tsx';
import { ApiKeyProvider } from '@/contexts/ApiKeyContext.tsx';

function renderAccount() {
  return render(
    <MemoryRouter>
      <AuthProvider>
        <ApiKeyProvider>
          <Account />
        </ApiKeyProvider>
      </AuthProvider>
    </MemoryRouter>,
  );
}

/** Open the "API & Agent" tab, where the API key form lives. */
function openAgentTab() {
  // Radix activates a tab on mousedown, which fireEvent.click does not send.
  const tab = screen.getByRole('tab', { name: /api & agent/i });
  fireEvent.mouseDown(tab);
  fireEvent.click(tab);
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Stub `fetch` and hand back the list of calls it received. */
function stubFetch(body: unknown, ok = true, status = 200): string[] {
  const calls: string[] = [];
  vi.stubGlobal('fetch', (url: string) => {
    calls.push(String(url));
    return Promise.resolve({
      ok,
      status,
      text: () => Promise.resolve(JSON.stringify(body)),
    } as Response);
  });
  return calls;
}

describe('the Model field on the Account page', () => {
  it('asks for nothing while there is no key to ask with', () => {
    const calls = stubFetch({ data: [{ id: 'gpt-5-mini' }] });
    renderAccount();
    openAgentTab();

    // Nothing to authenticate with yet — the provider would just reject it.
    expect(screen.getByRole('button', { name: /load my models/i })).toBeDisabled();
    // And a half-typed key must not fire a request per keystroke.
    fireEvent.change(screen.getByPlaceholderText(/sk-\.\.\./i), { target: { value: 'sk-' } });
    expect(calls).toHaveLength(0);
  });

  it('lists the key\'s own models on its own, with nothing to click', async () => {
    stubFetch({ data: [{ id: 'gpt-5-mini' }, { id: 'text-embedding-3-large' }] });
    renderAccount();
    openAgentTab();

    fireEvent.change(screen.getByPlaceholderText(/sk-\.\.\./i), {
      target: { value: 'sk-a-real-key' },
    });

    // No click anywhere: entering the key is the whole interaction.
    await waitFor(
      () => expect(screen.getByText(/1 model this key can reach/i)).toBeInTheDocument(),
      { timeout: 3000 },
    );
    expect(screen.getByText('Choose a model')).toBeInTheDocument();
  });

  it('replaces the free-text field with the models the key can reach', async () => {
    stubFetch({ data: [{ id: 'gpt-5-mini' }, { id: 'text-embedding-3-large' }] });

    renderAccount();
    openAgentTab();
    fireEvent.change(screen.getByPlaceholderText(/sk-\.\.\./i), {
      target: { value: 'sk-a-real-key' },
    });

    // Before loading: a free-text box carrying the hand-maintained default,
    // and only the provider picker is a dropdown on this form.
    expect(screen.getByPlaceholderText('gpt-4o')).toBeInTheDocument();
    expect(screen.getAllByRole('combobox')).toHaveLength(1);

    fireEvent.click(screen.getByRole('button', { name: /load my models/i }));

    await waitFor(() => {
      // One model survived the non-chat filter, and it is now a real choice.
      expect(screen.getByText(/1 model this key can reach/i)).toBeInTheDocument();
    });
    // The text box is gone, replaced by a second dropdown awaiting a choice.
    expect(screen.queryByPlaceholderText('gpt-4o')).not.toBeInTheDocument();
    expect(screen.getAllByRole('combobox')).toHaveLength(2);
    expect(screen.getByText('Choose a model')).toBeInTheDocument();
    // And a way back, so a model the listing omits is still reachable.
    expect(screen.getByRole('button', { name: /enter manually/i })).toBeInTheDocument();
  });

  it("shows the provider's own reason and keeps the field usable on failure", async () => {
    stubFetch({ error: { message: 'Incorrect API key' } }, false, 401);

    renderAccount();
    openAgentTab();
    fireEvent.change(screen.getByPlaceholderText(/sk-\.\.\./i), {
      target: { value: 'wrong-key' },
    });
    fireEvent.click(screen.getByRole('button', { name: /load my models/i }));

    await waitFor(() => {
      expect(screen.getByText(/401.*Incorrect API key/i)).toBeInTheDocument();
    });
    // A failed listing must never take the manual field away.
    expect(screen.getByPlaceholderText('gpt-4o')).toBeInTheDocument();
  });
});
