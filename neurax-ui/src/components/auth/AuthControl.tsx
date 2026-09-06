import { useMemo, useState, type ComponentProps } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, LogOut, Zap, Key, Check } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { useApiKey, PROVIDER_DEFAULTS, type ApiProvider, type ApiKeyConfig } from '@/contexts/ApiKeyContext.tsx';
import {
  OpenAIIcon, AnthropicIcon, GeminiIcon, MistralIcon,
  FireworksIcon, DeepSeekIcon, GlmIcon, CustomProviderIcon,
} from '@/components/icons/ProviderIcons.tsx';
import { identiconDataUri } from '@/components/profile/Identicon.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog.tsx';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select.tsx';
import { useProviderModels } from '@/hooks/useProviderModels.ts';
import { useToast } from '@/hooks/use-toast.ts';
import { NotionistsAvatarPicker, AVATAR_OPTIONS, resolveAvatar } from '@/components/profile/NotionistsAvatarPicker.tsx';

interface AuthControlProps {
  triggerLabel?: string;
  triggerVariant?: ComponentProps<typeof Button>['variant'];
  triggerSize?: ComponentProps<typeof Button>['size'];
  triggerClassName?: string;
}

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

type SetupStep = 'profile' | 'apikey';

export function AuthControl({
  triggerLabel,
  triggerVariant,
  triggerSize,
  triggerClassName,
}: AuthControlProps) {
  const { isAuthenticated, user, signIn, signOut, avatarUrl } = useAuth();
  const { config: apiKeyConfig, isConfigured: hasApiKey, setApiKey, markSetupComplete } = useApiKey();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [open, setOpen] = useState(false);
  const [setupStep, setSetupStep] = useState<SetupStep>('profile');
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [busy] = useState(false);
  const [selectedAvatarId, setSelectedAvatarId] = useState<string>(AVATAR_OPTIONS[0].id);

  // API Key state
  const [apiProvider, setApiProvider] = useState<ApiProvider>('openai');
  const [apiKeyValue, setApiKeyValue] = useState('');
  const [apiCustomEndpoint, setApiCustomEndpoint] = useState('');
  const [apiModel, setApiModel] = useState('');

  // The real models this key can reach — see Account.tsx for why the
  // hand-maintained default this replaces could not stay correct.
  const modelList = useProviderModels(apiProvider, apiKeyValue, apiCustomEndpoint);

  const avatarSrc = useMemo(() => (user ? avatarUrl : identiconDataUri('user')), [user, avatarUrl]);
  const displayName = user?.username ?? 'User';

  // ── Reset dialog state ──
  const resetDialog = () => {
    setSetupStep('profile');
    setEmail('');
    setUsername('');
    setApiKeyValue('');
    setApiCustomEndpoint('');
    setApiModel('');
    setApiProvider('openai');
    setSelectedAvatarId(AVATAR_OPTIONS[0].id);
  };

  const closeDialog = () => {
    setOpen(false);
    resetDialog();
  };

  // ── API Key Save ──
  const onSaveApiKey = () => {
    if (!apiKeyValue.trim()) {
      toast({ title: 'API key required', description: 'Please enter your API key to use Neurax Agent.', variant: 'destructive' });
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
    markSetupComplete();

    toast({
      title: 'API key saved',
      description: `You're ready to use Neurax with ${newConfig.label}.`,
    });

    closeDialog();
    navigate('/app');
  };

  const onSkipApiKey = () => {
    markSetupComplete();
    closeDialog();
    navigate('/app');
  };

  // ── Create the local profile ────────────────────────────────
  const onCreateProfile = () => {
    // The avatar picked here used to be discarded — `signIn` always drew a
    // random seed of its own, from a completely different pool of seeds
    // than this picker's options, so the one just chosen never reached the
    // profile that's actually displayed. Passed through now, so what's
    // picked here is what appears afterward.
    signIn(
      email.trim() || 'local@neurax',
      username.trim() || undefined,
      resolveAvatar(selectedAvatarId).seed,
    );

    // Always land in the studio. An API key is only needed for the AI agent —
    // the compiler itself runs without one — so diverting to account settings
    // blocked a visitor on a requirement unrelated to what they came to do.
    closeDialog();
    if (hasApiKey) {
      toast({
        title: 'Welcome back!',
        description: `Signed in as ${username.trim() || 'Explorer'}`,
      });
    } else {
      toast({
        title: 'Ready to analyse',
        description: 'Add an API key in Account settings when you want the AI agent to design for you.',
      });
    }
    navigate('/app');
  };

  // ── Update provider and model ──
  const onProviderChange = (value: string) => {
    const p = value as ApiProvider;
    setApiProvider(p);
    const defaults = PROVIDER_DEFAULTS[p];
    setApiModel(defaults.defaultModel);
    if (p !== 'custom') {
      setApiCustomEndpoint('');
    }
  };

  // ── Authenticated state ────────────────────────────────────
  if (isAuthenticated) {
    return (
      <div className="flex items-center gap-2">
        {/* API Key indicator */}
        {hasApiKey && (
          <div className="hidden sm:flex items-center gap-1 h-8 px-2 rounded-md border border-emerald-500/20 bg-emerald-500/5 text-[10px] font-mono text-emerald-400">
            <Key className="w-3 h-3" />
            <span className="truncate max-w-[80px]">{apiKeyConfig?.label ?? 'API'}</span>
          </div>
        )}

        {/* User avatar + name */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="h-8 w-8 rounded-full overflow-hidden border border-white/20 hover:border-white/40 transition-colors flex-shrink-0"
            onClick={() => navigate('/account')}
            aria-label="Open account"
          >
            <img src={avatarSrc} alt="avatar" className="h-full w-full object-cover" />
          </button>
          <span className="hidden sm:inline text-xs text-white/60 font-medium max-w-[100px] truncate">
            {displayName}
          </span>
          <button
            type="button"
            onClick={signOut}
            className="h-7 w-7 rounded-md border border-white/10 hover:border-red-400/40 hover:bg-red-500/10 flex items-center justify-center transition-colors"
            aria-label="Sign out"
            title="Sign out"
          >
            <LogOut className="w-3.5 h-3.5 text-white/40 hover:text-red-400" />
          </button>
        </div>
      </div>
    );
  }

  // ── Dialog content: profile step ──
  const renderProfileStep = () => (
    <>
      <DialogHeader>
        <DialogTitle className="text-white text-xl">Welcome to NEURAX</DialogTitle>
        <DialogDescription className="text-white/40">
          Enter your name to start exploring the platform — nothing here ever
          leaves this device.
        </DialogDescription>
      </DialogHeader>

      <div className="space-y-4 mt-2">
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">Your Name</label>
          <Input placeholder="e.g. AI Explorer" value={username} onChange={(e) => setUsername(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
        </div>
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">Email (optional)</label>
          <Input placeholder="you@example.com" type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
        </div>
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-3 block">Choose Your Avatar</label>
          <div className="rounded-[8px] p-3 bg-white/[0.03] border border-white/[0.06]">
            <NotionistsAvatarPicker
              selectedId={selectedAvatarId}
              onSelect={setSelectedAvatarId}
            />
          </div>
        </div>
        <div className="pt-2 space-y-2">
          <Button className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold h-11" onClick={onCreateProfile}>
            <Zap className="w-4 h-4 mr-2" />
            Enter Platform
          </Button>
          <p className="text-[10px] text-center text-white/20">
            Kept on this device — no server, no password, nothing to remember.
          </p>
        </div>
      </div>
    </>
  );

  // ── Dialog content: API key setup step ──
  const renderApiKeyStep = () => (
    <>
      <DialogHeader>
        <DialogTitle className="text-white text-xl flex items-center gap-2">
          <Key className="w-5 h-5 text-emerald-400" />
          Configure AI Agent
        </DialogTitle>
        <DialogDescription className="text-white/40">
          Connect your AI provider to use Neurax Agent — the intelligent assistant that helps you design and analyze architectures.
        </DialogDescription>
      </DialogHeader>

      <div className="space-y-4 mt-2">
        {/* Provider Selector */}
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
            AI Provider
          </label>
          <Select value={apiProvider} onValueChange={onProviderChange}>
            <SelectTrigger className="bg-white/5 border-white/10 text-white">
              <SelectValue placeholder="Select provider" />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1a2e] border-white/10">
              {PROVIDERS.map((p) => (
                <SelectItem key={p.value} value={p.value} className="text-white focus:bg-white/10 focus:text-white">
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
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
            API Key
          </label>
          <Input
            type="password"
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
            className="bg-white/5 border-white/10 text-white placeholder:text-white/20 font-mono text-[12px]"
          />
        </div>

        {/* Model (optional) */}
        <div>
          <div className="flex items-center justify-between gap-2 mb-1.5">
            <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 block">
              Model <span className="text-white/20">(optional)</span>
            </label>
            {modelList.status === 'loading' ? (
              <span className="text-[10px] font-mono uppercase tracking-wider text-white/40">
                Loading models…
              </span>
            ) : modelList.models.length > 0 ? (
              <button
                type="button"
                onClick={modelList.reset}
                className="text-[10px] font-mono uppercase tracking-wider text-white/40 hover:text-white/70"
              >
                Enter manually
              </button>
            ) : (
              <button
                type="button"
                onClick={modelList.load}
                disabled={!apiKeyValue.trim()}
                className="text-[10px] font-mono uppercase tracking-wider text-emerald-400/80 hover:text-emerald-300 disabled:text-white/20 disabled:hover:text-white/20"
              >
                {modelList.status === 'error' ? 'Retry' : 'Load my models'}
              </button>
            )}
          </div>
          {modelList.models.length > 0 ? (
            <Select value={apiModel} onValueChange={setApiModel}>
              <SelectTrigger className="bg-white/5 border-white/10 text-white">
                <SelectValue placeholder="Choose a model" />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1a2e] border-white/10 max-h-72">
                {modelList.models.map((m) => (
                  <SelectItem key={m} value={m} className="text-white focus:bg-white/10 focus:text-white font-mono text-xs">
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              placeholder={PROVIDER_DEFAULTS[apiProvider].defaultModel}
              value={apiModel}
              onChange={(e) => setApiModel(e.target.value)}
              className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
            />
          )}
          {modelList.models.length > 0 ? (
            <p className="mt-1.5 text-[10px] font-mono text-white/30">
              {modelList.models.length} model{modelList.models.length === 1 ? '' : 's'} this key can reach.
            </p>
          ) : null}
          {modelList.error ? (
            <p className="mt-1.5 text-[10px] font-mono text-red-400/80">{modelList.error}</p>
          ) : null}
        </div>

        {/* Custom endpoint (only for custom provider) */}
        {apiProvider === 'custom' && (
          <div>
            <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
              API Endpoint
            </label>
            <Input
              placeholder="https://your-endpoint.com/v1"
              value={apiCustomEndpoint}
              onChange={(e) => setApiCustomEndpoint(e.target.value)}
              className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
            />
          </div>
        )}

        {/* Actions */}
        <div className="pt-4 space-y-2">
          <Button
            className="w-full bg-emerald-500 text-white hover:bg-emerald-600 font-semibold h-11"
            onClick={onSaveApiKey}
          >
            <Check className="w-4 h-4 mr-2" />
            Save & Start Using Neurax
          </Button>
          <Button
            variant="ghost"
            className="w-full text-white/30 hover:text-white/50 hover:bg-white/5"
            onClick={onSkipApiKey}
          >
            Skip for now — I'll configure later
          </Button>
        </div>

        <p className="text-[10px] text-center text-white/20 leading-relaxed">
          Your API key is stored locally and never sent to our servers.
          <br />
          Neurax Agent uses your own AI provider for intelligent assistance.
        </p>
      </div>
    </>
  );

  // ── Not authenticated ──────────────────────────────────────
  return (
    <>
      <Button
        variant={triggerVariant ?? 'default'}
        size={triggerSize ?? 'sm'}
        className={triggerClassName ?? 'bg-white/10 text-white hover:bg-white/20 border border-white/10'}
        onClick={() => {
          setSetupStep('profile');
          setOpen(true);
        }}
        disabled={busy}
      >
        <LogIn className="w-4 h-4 sm:mr-1.5" />
        <span className="hidden sm:inline">{triggerLabel ?? 'Sign in'}</span>
      </Button>

      <Dialog open={open} onOpenChange={(isOpen) => {
        if (!isOpen) {
          closeDialog();
        }
        setOpen(isOpen);
      }}>
        <DialogContent className="sm:max-w-lg bg-[#0c0c1a] border border-white/10 shadow-2xl">
          {setupStep === 'profile' ? renderProfileStep() : renderApiKeyStep()}
        </DialogContent>
      </Dialog>
    </>
  );
}
