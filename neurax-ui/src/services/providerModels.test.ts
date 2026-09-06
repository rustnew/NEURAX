/**
 * The Model field must offer what the key can actually reach.
 *
 * This exists because a hand-maintained default could not: the studio
 * proposed `accounts/fireworks/models/llama-v3p1-70b-instruct` long after
 * Fireworks stopped serving it, and every run died on
 * `404 Model not found, inaccessible, and/or not deployed` with a perfectly
 * valid key, endpoint and provider.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

import { listProviderModels } from './providerModels.ts';

type FetchCall = { url: string; headers: Record<string, string> };

function mockFetch(body: unknown, ok = true, status = 200): FetchCall[] {
  const calls: FetchCall[] = [];
  vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), headers: (init?.headers ?? {}) as Record<string, string> });
    return Promise.resolve({
      ok,
      status,
      text: () => Promise.resolve(JSON.stringify(body)),
    } as Response);
  });
  return calls;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('OpenAI-shaped providers', () => {
  it('reaches each provider at its own endpoint with a Bearer key', async () => {
    for (const [provider, host] of [
      ['openai', 'https://api.openai.com/v1/models'],
      ['mistral', 'https://api.mistral.ai/v1/models'],
      ['fireworks', 'https://api.fireworks.ai/inference/v1/models'],
      ['deepseek', 'https://api.deepseek.com/v1/models'],
      ['glm', 'https://open.bigmodel.cn/api/paas/v4/models'],
    ] as const) {
      const calls = mockFetch({ data: [{ id: 'a-chat-model' }] });
      const models = await listProviderModels(provider, 'the-key');
      expect(calls[0].url).toBe(host);
      expect(calls[0].headers.Authorization).toBe('Bearer the-key');
      expect(models).toEqual(['a-chat-model']);
    }
  });

  it('leaves out models the agent could never drive', async () => {
    // Every step is a with_structured_output call, so offering an embedding
    // or image model would swap a clear 404 for a confusing one.
    mockFetch({
      data: [
        { id: 'llama-v4-instruct' },
        { id: 'text-embedding-3-large' },
        { id: 'whisper-v3' },
        { id: 'stable-diffusion-xl' },
        { id: 'llama-guard-3' },
      ],
    });
    expect(await listProviderModels('fireworks', 'k')).toEqual(['llama-v4-instruct']);
  });

  it('de-duplicates and sorts what it offers', async () => {
    mockFetch({ data: [{ id: 'zeta' }, { id: 'alpha' }, { id: 'zeta' }] });
    expect(await listProviderModels('openai', 'k')).toEqual(['alpha', 'zeta']);
  });
});

describe('Anthropic', () => {
  it('sends the headers its API requires from a browser', async () => {
    // Without the direct-browser-access opt-in, api.anthropic.com returns no
    // CORS headers at all and the call fails before it is ever authenticated.
    const calls = mockFetch({ data: [{ id: 'claude-sonnet-5' }] });
    const models = await listProviderModels('anthropic', 'sk-ant-key');

    expect(calls[0].url).toContain('https://api.anthropic.com/v1/models');
    expect(calls[0].headers['x-api-key']).toBe('sk-ant-key');
    expect(calls[0].headers['anthropic-version']).toBe('2023-06-01');
    expect(calls[0].headers['anthropic-dangerous-direct-browser-access']).toBe('true');
    expect(models).toEqual(['claude-sonnet-5']);
  });
});

describe('Google', () => {
  it('keeps only what the API itself says can generate content', async () => {
    const calls = mockFetch({
      models: [
        { name: 'models/gemini-2.5-pro', supportedGenerationMethods: ['generateContent'] },
        { name: 'models/text-embedding-004', supportedGenerationMethods: ['embedContent'] },
      ],
    });
    const models = await listProviderModels('google', 'AIza-key');

    // Gemini authenticates by query parameter, not by header.
    expect(calls[0].url).toContain('key=AIza-key');
    // The `models/` prefix is the API's own namespacing, not part of the id
    // callers send back as `model`.
    expect(models).toEqual(['gemini-2.5-pro']);
  });
});

describe('custom endpoints', () => {
  it('asks the endpoint the user configured', async () => {
    const calls = mockFetch({ data: [{ id: 'qwen3-72b' }] });
    await listProviderModels('custom', 'k', 'http://192.168.1.50:8000/v1/');
    expect(calls[0].url).toBe('http://192.168.1.50:8000/v1/models');
  });

  it('says what is missing rather than calling nowhere', async () => {
    mockFetch({});
    await expect(listProviderModels('custom', 'k')).rejects.toThrow(/endpoint/i);
  });
});

describe('failures stay readable', () => {
  it('asks for a key before calling anything', async () => {
    const calls = mockFetch({});
    await expect(listProviderModels('openai', '   ')).rejects.toThrow(/API key/i);
    expect(calls).toHaveLength(0);
  });

  it("surfaces the provider's own reason", async () => {
    mockFetch({ error: { message: 'Incorrect API key provided' } }, false, 401);
    await expect(listProviderModels('openai', 'wrong')).rejects.toThrow(/401.*Incorrect API key/);
  });
});
