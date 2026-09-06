/**
 * Holds one provider model list for a key-entry screen.
 *
 * A hook rather than a shared component: the two screens that let a user set
 * a key (`AuthControl`'s setup dialog and the Account page) are skinned
 * completely differently — dark glass versus the light shadcn surface — so
 * they share the behaviour and each keeps its own markup, instead of one
 * component growing a theme prop to satisfy both.
 */
import { useCallback, useEffect, useState } from 'react';

import type { ApiProvider } from '@/contexts/ApiKeyContext.tsx';
import { listProviderModels } from '@/services/providerModels.ts';

export type ModelListStatus = 'idle' | 'loading' | 'loaded' | 'error';

/** Long enough that typing a key does not fire a request per keystroke. */
const AUTOLOAD_DEBOUNCE_MS = 700;

/** Below this a key is being typed, not finished — every provider's are longer. */
const MIN_PLAUSIBLE_KEY_LENGTH = 8;

export function useProviderModels(
  provider: ApiProvider,
  apiKey: string,
  customEndpoint?: string,
) {
  const [models, setModels] = useState<string[]>([]);
  const [status, setStatus] = useState<ModelListStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setModels([]);
    setStatus('idle');
    setError(null);
  }, []);

  const load = useCallback(async () => {
    setStatus('loading');
    setError(null);
    try {
      const found = await listProviderModels(provider, apiKey, customEndpoint);
      setModels(found);
      if (found.length) {
        setStatus('loaded');
      } else {
        // Reached the provider, got nothing usable back. Distinct from a
        // failed call, and worth saying so: the key works, this account
        // just has no chat model the agent could drive.
        setStatus('error');
        setError('The provider answered, but lists no chat model for this key.');
      }
    } catch (e) {
      setModels([]);
      setStatus('error');
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [provider, apiKey, customEndpoint]);

  // A list belongs to the exact provider/key/endpoint it was fetched with, so
  // any change to that triple drops it — keeping it would go on offering
  // models the new combination may have no access to, the same staleness this
  // whole feature exists to remove, just with a shorter shelf life.
  //
  // And then it reloads itself. Entering a key is already the moment a user
  // states which account to use; making them press a second button before the
  // studio will say what that account can actually run is a step with no
  // decision in it. The debounce is what makes that safe: a key arrives one
  // character at a time, and only the pause after the last one is a real
  // intent to use it.
  useEffect(() => {
    reset();

    const key = apiKey.trim();
    if (key.length < MIN_PLAUSIBLE_KEY_LENGTH) return;
    // A custom provider has nowhere to ask until its endpoint is set.
    if (provider === 'custom' && !(customEndpoint ?? '').trim()) return;

    const timer = window.setTimeout(() => {
      void load();
    }, AUTOLOAD_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [provider, apiKey, customEndpoint, load, reset]);

  return { models, status, error, load, reset };
}
