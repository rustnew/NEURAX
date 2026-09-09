import { useMemo, useState, type ComponentProps } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, LogOut, Key, Check } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { cn } from '@/lib/utils.ts';
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
import { Identicon } from '@/components/profile/Identicon.tsx';

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
  const [showAllAvatars, setShowAllAvatars] = useState(false);

  const [setupStep, setSetupStep] = useState<SetupStep>('profile');
  const [email, setEmail] = useState('');

  const [username, setUsername] = useState('');

  /**
   * Email is now required alongside the name.
   *
   * Checked with a deliberately loose shape — something, an `@`, something, a
   * dot, something. Anything stricter rejects addresses that are perfectly
   * valid (plus-addressing, new top-level domains, non-ASCII local parts), and
   * the only way to actually know an address works is to send to it, which
   * NEURAX does not do. The check exists to catch a typo, not to be a gate.
   */
  const emailLooksValid = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email.trim());
  const canSubmit = username.trim().length > 0 && emailLooksValid;
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
            className="h-8 w-8 rounded-full overflow-hidden border border-border hover:border-primary/50 transition-colors flex-shrink-0"
            onClick={() => navigate('/account')}
            aria-label="Open account"
          >
            <img src={avatarSrc} alt="avatar" className="h-full w-full object-cover" />
          </button>
          <span className="hidden sm:inline text-xs font-medium text-foreground max-w-[100px] truncate">
            {displayName}
          </span>
          <button
            type="button"
            onClick={signOut}
            className="h-7 w-7 rounded-md border border-border hover:border-red-400/40 hover:bg-red-500/10 flex items-center justify-center transition-colors"
            aria-label="Sign out"
            title="Sign out"
          >
            <LogOut className="w-3.5 h-3.5 text-muted-foreground hover:text-red-400" />
          </button>
        </div>
      </div>
    );
  }

  // ── Dialog content: profile step ──
  const renderProfileStep = () => (
    <>
      <DialogHeader>
        <DialogTitle className="text-foreground text-xl">Create your profile</DialogTitle>
        <DialogDescription className="text-muted-foreground">
          Your name and email identify your work in NEURAX. Both are kept on this device — no
          server, no password.
        </DialogDescription>
      </DialogHeader>

      {/*
        Rebuilt around the two fields that matter.

        This screen used to open on thirty-six named avatar tiles in a
        scrolling grid, taller than everything else combined, with the name
        and email fields squeezed above it and the confirm button below the
        fold. It read as an arcade character select rather than as the first
        screen of an engineering product.

        Now the identity being created is shown once, large, beside the field
        that names it; the pattern is a single row; and the full set is one
        click away for anyone who wants to browse it.
      */}
      {/*
        `min-w-0` is load-bearing.

        Radix's DialogContent is a grid, and a grid item defaults to
        `min-width: auto` — it refuses to shrink below its content. The avatar
        row is a horizontal scroller thirty-six items wide, so without this its
        intrinsic width pushed the whole dialog past its own edge: name field,
        email field and the confirm button all ran off the right-hand side.
      */}
      <div className="space-y-5 mt-3 min-w-0">
        <div className="flex items-center gap-4">
          <div className="shrink-0 rounded-full overflow-hidden border border-border w-14 h-14 flex items-center justify-center bg-muted/30">
            <Identicon seed={resolveAvatar(selectedAvatarId).seed} size={44} />
          </div>
          <div className="flex-1 min-w-0">
            <label
              htmlFor="profile-name"
              className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground mb-1.5 block"
            >
              Your name
            </label>
            <Input
              id="profile-name"
              placeholder="e.g. Alex Moreau"
              value={username}
              autoFocus
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => {
                // Enter moves on when the form is complete, rather than
                // submitting a profile the button would refuse.
                if (e.key === 'Enter' && canSubmit) onCreateProfile();
              }}
              className="h-11 text-[15px] bg-muted/40 border-border text-foreground placeholder:text-muted-foreground/50"
            />
          </div>
        </div>

        <div>
          <label
            htmlFor="profile-email"
            className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground mb-1.5 block"
          >
            Email
          </label>
          <Input
            id="profile-email"
            placeholder="you@company.com"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && canSubmit) onCreateProfile();
            }}
            aria-invalid={email.length > 0 && !emailLooksValid}
            className={cn(
              'h-10 text-[13.5px] bg-muted/40 text-foreground placeholder:text-muted-foreground/50',
              email.length > 0 && !emailLooksValid ? 'border-destructive' : 'border-border',
            )}
          />
          {/* Only once they have typed something. Marking a field they have
              not reached yet as wrong is scolding, not helping. */}
          {email.length > 0 && !emailLooksValid ? (
            <p className="mt-1.5 text-[11px] text-destructive">
              That does not look like an email address.
            </p>
          ) : null}
        </div>

        <div>
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
              Pattern
            </span>
            <button
              type="button"
              onClick={() => setShowAllAvatars((v) => !v)}
              className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
            >
              {showAllAvatars ? 'Show fewer' : `Show all ${AVATAR_OPTIONS.length}`}
            </button>
          </div>
          <NotionistsAvatarPicker
            selectedId={selectedAvatarId}
            onSelect={setSelectedAvatarId}
            variant={showAllAvatars ? 'grid' : 'strip'}
          />
        </div>

        {/*
          Disabled has to look inactive, not broken.

          The default disabled treatment is the primary colour at reduced
          opacity, which on the brand's gold produced a large muddy brown slab
          — the biggest element on the screen, looking like a rendering fault
          rather than a control waiting for input. A neutral surface reads as
          "not yet", which is what it means.
        */}
        <Button
          className={cn(
            'w-full font-semibold h-11 transition-colors',
            canSubmit
              ? 'bg-primary text-primary-foreground hover:bg-primary/90'
              : 'bg-muted text-muted-foreground border border-border hover:bg-muted cursor-not-allowed',
          )}
          onClick={onCreateProfile}
          disabled={!canSubmit}
        >
          Continue
        </Button>
      </div>
    </>
  );

  // ── Dialog content: API key setup step ──
  const renderApiKeyStep = () => (
    <>
      <DialogHeader>
        <DialogTitle className="text-foreground text-xl flex items-center gap-2">
          <Key className="w-5 h-5 text-emerald-400" />
          Configure AI Agent
        </DialogTitle>
        <DialogDescription className="text-muted-foreground">
          Connect your AI provider to use Neurax Agent — the intelligent assistant that helps you design and analyze architectures.
        </DialogDescription>
      </DialogHeader>

      <div className="space-y-4 mt-2">
        {/* Provider Selector */}
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground mb-1.5 block">
            AI Provider
          </label>
          <Select value={apiProvider} onValueChange={onProviderChange}>
            <SelectTrigger className="bg-muted/40 border-border text-foreground">
              <SelectValue placeholder="Select provider" />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1a2e] border-border">
              {PROVIDERS.map((p) => (
                <SelectItem key={p.value} value={p.value} className="text-foreground focus:bg-accent focus:text-accent-foreground">
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
          <label className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground mb-1.5 block">
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
            className="bg-muted/40 border-border text-foreground placeholder:text-muted-foreground/50 font-mono text-[12px]"
          />
        </div>

        {/* Model (optional) */}
        <div>
          <div className="flex items-center justify-between gap-2 mb-1.5">
            <label className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground block">
              Model <span className="text-muted-foreground/60">(optional)</span>
            </label>
            {modelList.status === 'loading' ? (
              <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                Loading models…
              </span>
            ) : modelList.models.length > 0 ? (
              <button
                type="button"
                onClick={modelList.reset}
                className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground/70"
              >
                Enter manually
              </button>
            ) : (
              <button
                type="button"
                onClick={modelList.load}
                disabled={!apiKeyValue.trim()}
                className="text-[10px] font-mono uppercase tracking-wider text-emerald-400/80 hover:text-emerald-300 disabled:text-muted-foreground/60 disabled:hover:text-muted-foreground/60"
              >
                {modelList.status === 'error' ? 'Retry' : 'Load my models'}
              </button>
            )}
          </div>
          {modelList.models.length > 0 ? (
            <Select value={apiModel} onValueChange={setApiModel}>
              <SelectTrigger className="bg-muted/40 border-border text-foreground">
                <SelectValue placeholder="Choose a model" />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1a2e] border-border max-h-72">
                {modelList.models.map((m) => (
                  <SelectItem key={m} value={m} className="text-foreground focus:bg-accent focus:text-accent-foreground font-mono text-xs">
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
              className="bg-muted/40 border-border text-foreground placeholder:text-muted-foreground/50"
            />
          )}
          {modelList.models.length > 0 ? (
            <p className="mt-1.5 text-[10px] font-mono text-muted-foreground">
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
            <label className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground mb-1.5 block">
              API Endpoint
            </label>
            <Input
              placeholder="https://your-endpoint.com/v1"
              value={apiCustomEndpoint}
              onChange={(e) => setApiCustomEndpoint(e.target.value)}
              className="bg-muted/40 border-border text-foreground placeholder:text-muted-foreground/50"
            />
          </div>
        )}

        {/* Actions */}
        <div className="pt-4 space-y-2">
          <Button
            className="w-full bg-emerald-500 text-foreground hover:bg-emerald-600 font-semibold h-11"
            onClick={onSaveApiKey}
          >
            <Check className="w-4 h-4 mr-2" />
            Save & Start Using Neurax
          </Button>
          <Button
            variant="ghost"
            className="w-full text-muted-foreground hover:text-muted-foreground hover:bg-accent/60"
            onClick={onSkipApiKey}
          >
            Skip for now — I'll configure later
          </Button>
        </div>

        <p className="text-[10px] text-center text-muted-foreground/60 leading-relaxed">
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
        className={triggerClassName ?? 'bg-muted/60 text-foreground hover:bg-accent border border-border'}
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
        {/*
          The dialog takes the theme's own surface.

          It was pinned to `#0c0c1a`, a dark navy from the app's original
          single-theme days. Everything inside it — the inputs, the labels, the
          avatar tiles — had since been converted to theme tokens, so on a
          light theme the panel stayed near-black while its contents turned
          light: white fields floating on a dark card, with placeholder text
          the same colour as the field it sat in.
        */}
        <DialogContent className="sm:max-w-lg bg-card border border-border shadow-2xl">
          {setupStep === 'profile' ? renderProfileStep() : renderApiKeyStep()}
        </DialogContent>
      </Dialog>
    </>
  );
}
