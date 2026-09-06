/**
 * The real model list for a provider, fetched with the user's own key.
 *
 * The Model field used to be free text prefilled from `PROVIDER_DEFAULTS`, a
 * hand-maintained constant. Providers retire model ids, so that default goes
 * stale silently and the studio then proposes a name the provider no longer
 * serves: `accounts/fireworks/models/llama-v3p1-70b-instruct` returned
 * `404 Model not found, inaccessible, and/or not deployed` while the key,
 * endpoint and routing were all correct. A list the provider itself gives us
 * cannot go stale that way, and it is scoped to what *this* key can actually
 * reach — a hardcoded default is not (an account without access to a model
 * fails the same way as one asking for a retired one).
 *
 * Called from the browser, not through `neurax-agent`. Verified rather than
 * assumed: all seven provider endpoints answer a cross-origin request
 * (`access-control-allow-origin` present for `tauri://localhost` on every
 * one; Anthropic only once `anthropic-dangerous-direct-browser-access` is
 * sent, which is exactly what that header is for). That keeps key discovery
 * working when the agent is not running — which is the state a user is
 * most likely in while first configuring a key.
 */
import type { ApiProvider } from '@/contexts/ApiKeyContext.tsx';

/** Providers whose `/models` is OpenAI-shaped: `Bearer` auth, `{data:[{id}]}`. */
const OPENAI_SHAPED_BASE: Partial<Record<ApiProvider, string>> = {
  openai: 'https://api.openai.com/v1',
  mistral: 'https://api.mistral.ai/v1',
  fireworks: 'https://api.fireworks.ai/inference/v1',
  deepseek: 'https://api.deepseek.com/v1',
  glm: 'https://open.bigmodel.cn/api/paas/v4',
};

/**
 * Substrings that mark a model as something the agent cannot drive.
 *
 * The agent's every step is a `with_structured_output` call, so an embedding,
 * transcription, moderation or image model in the list is not a choice a user
 * could make successfully — it would swap a clear 404 for a confusing
 * structured-output failure. Deliberately a deny-list, not an allow-list:
 * missing a genuinely new chat model is a worse failure than leaving one
 * unusable entry in, and provider naming changes far too often to enumerate
 * what is good.
 */
const NON_CHAT_MARKERS = [
  'embed', 'rerank', 'whisper', 'tts', 'dall-e', 'moderation', 'guard',
  'stable-diffusion', 'sdxl', 'flux', 'playground-v', 'image', 'audio',
  'transcribe', 'bge-', 'clip-', 'text-similarity', 'text-search',
  'davinci', 'babbage', 'ada-', 'curie',
];

const isChatModel = (id: string): boolean => {
  const lower = id.toLowerCase();
  return !NON_CHAT_MARKERS.some((marker) => lower.includes(marker));
};

const TIMEOUT_MS = 15_000;

async function getJson(url: string, headers: Record<string, string>): Promise<any> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const resp = await fetch(url, { headers, signal: controller.signal });
    const text = await resp.text();
    let body: any = null;
    try {
      body = JSON.parse(text);
    } catch {
      // Left null — a non-JSON body is reported through the status below.
    }
    if (!resp.ok) {
      const detail = String(body?.error?.message ?? body?.message ?? text ?? '').slice(0, 200);
      throw new Error(detail ? `${resp.status} — ${detail}` : `HTTP ${resp.status}`);
    }
    return body;
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new Error(`No answer within ${TIMEOUT_MS / 1000}s.`);
    }
    throw e;
  } finally {
    window.clearTimeout(timer);
  }
}

/**
 * Every chat model `apiKey` can actually reach on `provider`, sorted.
 *
 * Throws with a readable reason — a wrong key, a provider that is down, an
 * unreachable custom endpoint. Callers keep the free-text field as the
 * fallback rather than treating this as required.
 */
export async function listProviderModels(
  provider: ApiProvider,
  apiKey: string,
  customEndpoint?: string,
): Promise<string[]> {
  const key = apiKey.trim();
  if (!key) throw new Error('Enter your API key first.');

  if (provider === 'anthropic') {
    const body = await getJson('https://api.anthropic.com/v1/models?limit=100', {
      'x-api-key': key,
      'anthropic-version': '2023-06-01',
      // Anthropic returns no CORS headers without this; it is the documented
      // opt-in for calling the API straight from a browser, which is what a
      // bring-your-own-key studio does by design.
      'anthropic-dangerous-direct-browser-access': 'true',
    });
    return dedupeSorted(
      (body?.data ?? []).map((m: any) => String(m?.id ?? '')).filter(Boolean).filter(isChatModel),
    );
  }

  if (provider === 'google') {
    const body = await getJson(
      `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(key)}&pageSize=200`,
      {},
    );
    const models = (body?.models ?? [])
      // Gemini's list mixes in embedding and image models; the API states per
      // model which methods it supports, so this filter is the provider's own
      // answer rather than a guess from the name.
      .filter((m: any) => (m?.supportedGenerationMethods ?? []).includes('generateContent'))
      .map((m: any) => String(m?.name ?? '').replace(/^models\//, ''))
      .filter(Boolean);
    return dedupeSorted(models.filter(isChatModel));
  }

  const base =
    provider === 'custom'
      ? (customEndpoint ?? '').trim().replace(/\/$/, '')
      : OPENAI_SHAPED_BASE[provider];
  if (!base) throw new Error('Set the API endpoint for this provider first.');

  const body = await getJson(`${base}/models`, { Authorization: `Bearer ${key}` });
  const raw = Array.isArray(body?.data) ? body.data : Array.isArray(body) ? body : [];
  return dedupeSorted(
    raw.map((m: any) => String(m?.id ?? m?.name ?? '')).filter(Boolean).filter(isChatModel),
  );
}

function dedupeSorted(ids: string[]): string[] {
  return Array.from(new Set(ids)).sort((a, b) => a.localeCompare(b));
}
