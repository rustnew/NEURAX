import { useState, useCallback, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Check, Key, LogOut, User,
  Settings, Bot, Save, X, Eye, EyeOff,
  Palette, AtSign, Mail, Shield,
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { useApiKey, PROVIDER_DEFAULTS, type ApiProvider, type ApiKeyConfig } from '@/contexts/ApiKeyContext.tsx';
import {
  OpenAIIcon, AnthropicIcon, GeminiIcon, MistralIcon,
  FireworksIcon, DeepSeekIcon, GlmIcon, CustomProviderIcon,
} from '@/components/icons/ProviderIcons.tsx';

import { useProviderModels } from '@/hooks/useProviderModels.ts';
import { useToast } from '@/hooks/use-toast.ts';

// ─── Notionists Avatar Family ────────────────────────────────────
import {
  AVATAR_OPTIONS,
  resolveAvatar,
} from '@/components/profile/NotionistsAvatarPicker.tsx';
import { Identicon } from '@/components/profile/Identicon.tsx';

const PROVIDERS: { value: ApiProvider; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { value: 'openai', label: 'OpenAI', icon: OpenAIIcon },
  { value: 'anthropic', label: 'Anthropic', icon: AnthropicIcon },
  { value: 'google', label: 'Google AI (Gemini)', icon: GeminiIcon },
  { value: 'mistral', label: 'Mistral', icon: MistralIcon },
  { value: 'fireworks', label: 'Fireworks AI', icon: FireworksIcon },
  { value: 'deepseek', label: 'DeepSeek', icon: DeepSeekIcon },
  { value: 'glm', label: 'GLM (Zhipu)', icon: GlmIcon },
  { value: 'custom', label: 'Custom (OpenAI-compatible)', icon: CustomProviderIcon },
];

export default function Account() {
  const { isAuthenticated, user, signOut, updateProfile } = useAuth();
  const { config: apiKeyConfig, isConfigured: hasApiKey, setApiKey, clearApiKey } = useApiKey();

  const navigate = useNavigate();
  const { toast } = useToast();

  // ── Navigation tabs ──
  const [activeTab, setActiveTab] = useState('profile');

  // Every action on this page is a local write — nothing async to be
  // "busy" doing, so the buttons are never disabled for it. Kept as a named
  // constant (not deleted from every call site) so a future action that
  // genuinely does await something has an obvious place to wire into.
  const busy = false;

  // ── Avatar ──
  // Seeded from the profile's own `avatarSeed` — the value the header
  // actually displays — not a separate copy that could say something else.
  const [selectedAvatarId, setSelectedAvatarId] = useState<string>(
    () => resolveAvatar(user?.avatarSeed ?? null).id,
  );

  // ── Profile edit ──
  const [editName, setEditName] = useState('');
  const [editEmail, setEditEmail] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  // ── API Key ──
  const [apiProvider, setApiProvider] = useState<ApiProvider>(
    (apiKeyConfig?.provider as ApiProvider) || 'openai',
  );
  const [apiKeyValue, setApiKeyValue] = useState(apiKeyConfig?.key || '');
  const [apiCustomEndpoint, setApiCustomEndpoint] = useState(apiKeyConfig?.customEndpoint || '');
  const [apiModel, setApiModel] = useState(apiKeyConfig?.model || '');
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiKeyEditMode, setApiKeyEditMode] = useState(!hasApiKey);

  // The real models this key can reach, fetched from the provider on
  // demand. Replaces a free-text field prefilled from a hand-maintained
  // default that goes stale whenever a provider retires a model id.
  const modelList = useProviderModels(apiProvider, apiKeyValue, apiCustomEndpoint);

  const username = user?.username ?? 'User';
  const email = user?.email || null;

  // Init edit fields
  useEffect(() => {
    if (!isEditing) {
      setEditName(username);
      setEditEmail(email || '');
    }
  }, [username, email, isEditing]);

  // ── Avatar ──
  //
  // This used to write to a localStorage key ('neurax_account_emoji') that
  // nothing reading the header's avatar ever looked at — `updateProfile`'s
  // type didn't even accept an avatar field, so there was no path from this
  // "Save" button to the profile that's actually displayed. The toast said
  // "updated"; the header stayed exactly as it was.
  const onSaveEmoji = useCallback(() => {
    updateProfile({ avatarSeed: resolveAvatar(selectedAvatarId).seed });
    toast({ title: 'Avatar saved', description: 'Your account avatar has been updated.' });
  }, [selectedAvatarId, updateProfile, toast]);

  // ── Profile ── a local write, not a network round trip: nothing here can
  // fail the way a request can, so there's no error branch to report.
  const onSaveProfile = () => {
    updateProfile({ username: editName.trim(), email: editEmail.trim() });
    toast({ title: 'Profile saved' });
    setIsEditing(false);
  };

  // ── API Key ──
  const onSaveApiKey = () => {
    if (!apiKeyValue.trim()) {
      toast({ title: 'API key required', description: 'Please enter your API key.', variant: 'destructive' });
      return;
    }

    const newConfig: ApiKeyConfig = {
      key: apiKeyValue.trim(),
      provider: apiProvider,
      label: PROVIDERS.find(p => p.value === apiProvider)?.label ?? apiProvider,
      ...(apiProvider === 'custom' && apiCustomEndpoint.trim() ? { customEndpoint: apiCustomEndpoint.trim() } : {}),
      ...(apiModel.trim() ? { model: apiModel.trim() } : {}),
    };

    setApiKey(newConfig);
    setApiKeyEditMode(false);
    toast({ title: 'API key saved', description: `Connected to ${newConfig.label}.` });
  };

  const onClearApiKey = () => {
    clearApiKey();
    setApiKeyValue('');
    setApiKeyEditMode(true);
    toast({ title: 'API key removed' });
  };

  const onProviderChange = (value: string) => {
    const p = value as ApiProvider;
    setApiProvider(p);
    const defaults = PROVIDER_DEFAULTS[p];
    setApiModel(defaults.defaultModel);
    if (p !== 'custom') setApiCustomEndpoint('');
  };

  // ── Sign out ── clears the local profile; nothing to await.
  const onSignOut = () => {
    signOut();
    toast({ title: 'Signed out' });
    navigate('/');
  };

  // ── Not authenticated ──
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto max-w-2xl px-4 sm:px-6 py-10">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold">Account</h1>
            <Button asChild variant="outline" size="sm">
              <Link to="/">Back</Link>
            </Button>
          </div>
          <div className="mt-6 rounded-xl border bg-card/50 p-6 text-sm text-muted-foreground">
            You're not signed in.
            <Button asChild variant="link" className="px-0">
              <Link to="/">Sign in</Link>
            </Button>
          </div>
        </div>
      </div>
    );
  }



  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 py-8">

        {/* ── Header ── */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Identicon seed={resolveAvatar(selectedAvatarId).seed} size={56} />
            <div>
              <h1 className="text-2xl font-bold">{username}</h1>
              <p className="text-sm text-muted-foreground">{email}</p>
            </div>
          </div>
          <Button asChild variant="outline" size="sm">
            <Link to="/app">Back to Studio</Link>
          </Button>
        </div>

        {/* ── Tabs ── */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="w-full justify-start bg-transparent border-b border-border rounded-none h-auto p-0 mb-6 space-x-6">
            <TabsTrigger value="profile" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none pb-3 px-1 text-sm">
              <User className="w-4 h-4 mr-2" /> Profile
            </TabsTrigger>
            <TabsTrigger value="api" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none pb-3 px-1 text-sm">
              <Key className="w-4 h-4 mr-2" /> API & Agent
            </TabsTrigger>

            <TabsTrigger value="security" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none pb-3 px-1 text-sm">
              <Shield className="w-4 h-4 mr-2" /> Security
            </TabsTrigger>
          </TabsList>

          {/* ════════════════════════════════════════ PROFILE ════════════════════════════════════════ */}
          <TabsContent value="profile" className="space-y-6">
            {/* Avatar Emoji */}
            <div className="rounded-xl border bg-card p-6">
              <div className="flex items-center gap-3 mb-4">
                <Palette className="w-5 h-5 text-primary" />
                <h2 className="text-sm font-semibold">Avatar</h2>
              </div>
              <p className="text-xs text-muted-foreground mb-4">
                Generated patterns, each drawn from its own seed — nothing is fetched to display one.
              </p>
              <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-12 gap-2" role="radiogroup" aria-label="Avatar">
                {AVATAR_OPTIONS.map((avatar) => {
                  const isSelected = selectedAvatarId === avatar.id;
                  return (
                    <button
                      key={avatar.id}
                      type="button"
                      role="radio"
                      aria-checked={isSelected}
                      className={`h-12 w-12 rounded-lg border flex items-center justify-center transition-all relative group text-foreground ${
                        isSelected
                          ? 'border-primary ring-2 ring-primary/40 scale-110'
                          : 'border-border hover:scale-105'
                      }`}
                      onClick={() => setSelectedAvatarId(avatar.id)}
                      aria-label={`Select ${avatar.name}`}
                      title={avatar.name}
                    >
                      <Identicon seed={avatar.seed} size={34} />
                      <span className="absolute -top-8 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground px-2 py-1 rounded text-[10px] font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-md border">
                        {avatar.name}
                      </span>
                    </button>
                  );
                })}
              </div>
              <div className="mt-4 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  {resolveAvatar(selectedAvatarId).name}
                </span>
                <Button size="sm" onClick={onSaveEmoji} disabled={busy}>
                  <Save className="w-3.5 h-3.5 mr-1.5" /> Save Avatar
                </Button>
              </div>
            </div>

            {/* Profile Info */}
            <div className="rounded-xl border bg-card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <User className="w-5 h-5 text-primary" />
                  <h2 className="text-sm font-semibold">Personal Info</h2>
                </div>
                {!isEditing && (
                  <Button variant="outline" size="sm" onClick={() => setIsEditing(true)}>
                    <Settings className="w-3.5 h-3.5 mr-1.5" /> Edit
                  </Button>
                )}
              </div>

              {isEditing ? (
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="edit-name" className="text-xs text-muted-foreground">Username</Label>
                    <Input id="edit-name" value={editName} onChange={(e) => setEditName(e.target.value)} className="mt-1" />
                  </div>
                  <div>
                    <Label htmlFor="edit-email" className="text-xs text-muted-foreground">Email <span className="text-muted-foreground/50">(optional)</span></Label>
                    <Input id="edit-email" type="email" value={editEmail} onChange={(e) => setEditEmail(e.target.value)} className="mt-1" />
                  </div>
                  <div className="flex justify-end gap-2 pt-2">
                    <Button variant="ghost" size="sm" onClick={() => setIsEditing(false)}>
                      <X className="w-3.5 h-3.5 mr-1" /> Cancel
                    </Button>
                    <Button size="sm" onClick={onSaveProfile} disabled={busy}>
                      <Save className="w-3.5 h-3.5 mr-1" /> Save
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm">
                    <AtSign className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Username:</span>
                    <span className="font-medium">{username}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Mail className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Email:</span>
                    <span className="font-medium">{email || '—'}</span>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          {/* ════════════════════════════════════════ API & AGENT ════════════════════════════════════════ */}
          <TabsContent value="api" className="space-y-6">
            <div className="rounded-xl border bg-card p-6">
              <div className="flex items-center gap-3 mb-4">
                <Bot className="w-5 h-5 text-primary" />
                <h2 className="text-sm font-semibold">Neurax Agent — API Key</h2>
              </div>
              <p className="text-xs text-muted-foreground mb-4">
                Connect your AI provider to use Neurax Agent — the intelligent assistant that helps you design and analyze architectures.
                Your API key is stored locally and never sent to our servers.
              </p>

              {/* Provider */}
              <div className="space-y-4">
                <div>
                  <Label className="text-xs text-muted-foreground">AI Provider</Label>
                  <Select value={apiProvider} onValueChange={onProviderChange}>
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select provider" />
                    </SelectTrigger>
                    <SelectContent>
                      {PROVIDERS.map((p) => (
                        <SelectItem key={p.value} value={p.value}>
                          <span className="flex items-center gap-2">
                            <p.icon className="w-4 h-4 shrink-0" />
                            <span>{p.label}</span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* API Key */}
                {apiKeyEditMode ? (
                  <>
                    <div>
                      <Label className="text-xs text-muted-foreground">API Key</Label>
                      <div className="relative mt-1">
                        <Input
                          type={showApiKey ? 'text' : 'password'}
                          placeholder={
                            apiProvider === 'openai' ? 'sk-...' :
                            apiProvider === 'anthropic' ? 'sk-ant-...' :
                            apiProvider === 'google' ? 'AIza...' :
                            apiProvider === 'mistral' ? 'MISTRAL_...' :
                            apiProvider === 'fireworks' ? 'fw_...' :
                            apiProvider === 'deepseek' ? 'sk-...' :
                            apiProvider === 'glm' ? 'GLM key' :
                            'Enter your API key'
                          }
                          value={apiKeyValue}
                          onChange={(e) => setApiKeyValue(e.target.value)}
                          className="pr-10 font-mono text-sm"
                        />
                        <button
                          type="button"
                          className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                          onClick={() => setShowApiKey(!showApiKey)}
                          aria-label={showApiKey ? 'Hide API key' : 'Show API key'}
                        >
                          {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                    </div>

                    <div>
                      <div className="flex items-center justify-between gap-2">
                        <Label className="text-xs text-muted-foreground">Model <span className="text-muted-foreground/50">(optional)</span></Label>
                        {modelList.status === 'loading' ? (
                          <span className="text-xs text-muted-foreground">Loading models…</span>
                        ) : modelList.models.length > 0 ? (
                          <Button type="button" variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={modelList.reset}>
                            Enter manually
                          </Button>
                        ) : (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-xs"
                            disabled={!apiKeyValue.trim()}
                            onClick={modelList.load}
                          >
                            {modelList.status === 'error' ? 'Retry' : 'Load my models'}
                          </Button>
                        )}
                      </div>
                      {modelList.models.length > 0 ? (
                        <Select value={apiModel} onValueChange={setApiModel}>
                          <SelectTrigger className="mt-1">
                            <SelectValue placeholder="Choose a model" />
                          </SelectTrigger>
                          <SelectContent className="max-h-72">
                            {modelList.models.map((m) => (
                              <SelectItem key={m} value={m} className="font-mono text-xs">{m}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          placeholder={PROVIDER_DEFAULTS[apiProvider].defaultModel}
                          value={apiModel}
                          onChange={(e) => setApiModel(e.target.value)}
                          className="mt-1"
                        />
                      )}
                      {modelList.models.length > 0 ? (
                        <p className="mt-1 text-xs text-muted-foreground">
                          {modelList.models.length} model{modelList.models.length === 1 ? '' : 's'} this key can reach.
                        </p>
                      ) : null}
                      {modelList.error ? (
                        <p className="mt-1 text-xs text-destructive">{modelList.error}</p>
                      ) : null}
                    </div>

                    {apiProvider === 'custom' && (
                      <div>
                        <Label className="text-xs text-muted-foreground">API Endpoint</Label>
                        <Input
                          placeholder="https://your-endpoint.com/v1"
                          value={apiCustomEndpoint}
                          onChange={(e) => setApiCustomEndpoint(e.target.value)}
                          className="mt-1"
                        />
                      </div>
                    )}

                    <div className="flex justify-end gap-2 pt-2">
                      {hasApiKey && (
                        <Button variant="ghost" size="sm" onClick={() => { setApiKeyEditMode(false); setApiKeyValue(apiKeyConfig?.key || ''); }}>
                          <X className="w-3.5 h-3.5 mr-1" /> Cancel
                        </Button>
                      )}
                      <Button size="sm" onClick={onSaveApiKey} disabled={busy || !apiKeyValue.trim()}>
                        <Save className="w-3.5 h-3.5 mr-1" /> Save API Key
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/40 border">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium">{apiKeyConfig?.label || 'Not configured'}</div>
                        <div className="text-xs text-muted-foreground truncate font-mono">
                          {apiKeyConfig?.key
                            ? `${apiKeyConfig.key.slice(0, 8)}${'•'.repeat(Math.min(16, apiKeyConfig.key.length - 8))}`
                            : 'No API key set'}
                        </div>
                      </div>
                      {hasApiKey && (
                        <span className="flex items-center gap-1 text-[10px] text-emerald-500 font-mono uppercase">
                          <Check className="w-3 h-3" /> Active
                        </span>
                      )}
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" size="sm" onClick={() => setApiKeyEditMode(true)}>
                        <Key className="w-3.5 h-3.5 mr-1" /> Update Key
                      </Button>
                      {hasApiKey && (
                        <Button variant="outline" size="sm" className="text-destructive hover:text-destructive" onClick={onClearApiKey}>
                          <X className="w-3.5 h-3.5 mr-1" /> Remove
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-6 p-3 rounded-lg bg-primary/5 border border-primary/10">
                <p className="text-[10px] text-muted-foreground leading-relaxed">
                  <strong className="text-primary">Not sure?</strong> Your API key stays in your browser — we never see it.
                  It's used to power the Neurax Agent chat assistant directly from your browser to your chosen provider.
                </p>
              </div>
            </div>
          </TabsContent>



          {/* ════════════════════════════════════════ SECURITY ════════════════════════════════════════ */}
          <TabsContent value="security" className="space-y-6">
            {/* There is no password: this account is a local profile, not a
                credential a server checks. Nothing here to change. */}

            {/* Sign Out */}
            <div className="rounded-xl border bg-card p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <LogOut className="w-5 h-5 text-destructive" />
                  <div>
                    <h2 className="text-sm font-semibold">Sign Out</h2>
                    <p className="text-xs text-muted-foreground">Sign out of your account on this device.</p>
                  </div>
                </div>
                <Button variant="outline" size="sm" className="text-destructive hover:text-destructive" onClick={onSignOut} disabled={busy}>
                  <LogOut className="w-3.5 h-3.5 mr-1.5" /> Sign Out
                </Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

    </div>
  );
}


