import { useMemo, useState, type ComponentProps } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, Mail, Github, Gift, Zap, Sparkles, Crown, type LucideIcon } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { PLAN_CONFIGS, type PlanTier } from '@/types/plans.ts';
import { supabase } from '@/lib/supabaseClient.ts';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover.tsx';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { createBillingPortalSession, createCheckoutSession } from '@/services/neuraxApi.ts';

const SUPABASE_DISABLED = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

type AuthTab = 'password' | 'signup' | 'magic' | 'oauth';

interface AuthControlProps {
  initialTab?: AuthTab;
  triggerLabel?: string;
  triggerVariant?: ComponentProps<typeof Button>['variant'];
  triggerSize?: ComponentProps<typeof Button>['size'];
  triggerClassName?: string;
}

function randPick<T>(arr: T[]): T {
  return arr[0];
}

function stableSuffix(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i += 1) {
    h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  }
  return 1000 + (h % 9000);
}

function generateLegendMemeUsername(): string {
  const adjectives = [
    'Legendary',
    'Mythic',
    'Elite',
    'Savage',
    'Gigachad',
    'Sigma',
    'Degen',
    'Based',
    'Neural',
    'Quantum',
    'Cyber',
    'Turbo',
    'Cosmic',
  ];
  const nouns = [
    'Architect',
    'Tensor',
    'Wizard',
    'Engineer',
    'Overlord',
    'Monkey',
    'Goblin',
    'Dragon',
    'Hacker',
    'Researcher',
    'Builder',
    'Warlord',
  ];
  const suffix = stableSuffix(`${adjectives[0]}-${nouns[0]}`);
  return `${randPick(adjectives)}${randPick(nouns)}${suffix}`;
}

function defaultAvatarUrl(seed: string): string {
  const s = encodeURIComponent(seed || 'user');
  return `https://api.dicebear.com/7.x/identicon/svg?seed=${s}`;
}

const SYSTEM_AVATAR_SEEDS = Array.from({ length: 100 }, (_v, i) => `avatar-${i + 1}`);

function systemAvatarUrl(seed: string): string {
  const s = encodeURIComponent(seed);
  return `https://api.dicebear.com/7.x/bottts/svg?seed=${s}`;
}

function getSystemAvatarOptions(): string[] {
  return SYSTEM_AVATAR_SEEDS.map(systemAvatarUrl);
}

export function AuthControl({
  initialTab,
  triggerLabel,
  triggerVariant,
  triggerSize,
  triggerClassName,
}: AuthControlProps) {
  const { session, isAuthenticated } = useAuth();
  const { currentPlan, planConfig } = usePlan();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [open, setOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<AuthTab>('password');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [username, setUsername] = useState('');
  const [avatarUrl, setAvatarUrl] = useState<string | null>(() => {
    const opts = getSystemAvatarOptions();
    return opts.length > 0 ? randPick(opts) : null;
  });
  const [busy, setBusy] = useState(false);
  const [planPopoverOpen, setPlanPopoverOpen] = useState(false);

  const redirectTo = useMemo(() => {
    return `${window.location.origin}/app`;
  }, []);

  const avatarSrc = useMemo(() => {
    const m = (session?.user?.user_metadata ?? {}) as Record<string, unknown>;
    const metaUrl = typeof m.avatar_url === 'string' ? m.avatar_url : null;
    const fallback = defaultAvatarUrl(session?.user?.email ?? 'user');
    return metaUrl ?? fallback;
  }, [session]);

  const PlanIcon = useMemo<LucideIcon>(() => {
    const icons: Record<PlanTier, LucideIcon> = {
      free: Gift,
      essential: Zap,
      architect: Sparkles,
      elite: Crown,
    };
    return icons[currentPlan];
  }, [currentPlan]);

  const startCheckout = async (plan: Exclude<PlanTier, 'free'>) => {
    setBusy(true);
    try {
      const origin = window.location.origin;
      const { url } = await createCheckoutSession({
        plan,
        interval: 'year',
        success_url: `${origin}/account?checkout=success`,
        cancel_url: `${origin}/account?checkout=cancel`,
      });
      window.location.assign(url);
    } catch (e: any) {
      toast({ title: 'Checkout failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onManageBilling = async () => {
    setBusy(true);
    try {
      const { url } = await createBillingPortalSession();
      window.location.assign(url);
    } catch (e: any) {
      toast({ title: 'Billing portal failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onSignUp = async () => {
    setBusy(true);
    try {
      if (password !== confirmPassword) {
        toast({
          title: 'Passwords do not match',
          description: 'Please re-type your password.',
          variant: 'destructive',
        });
        return;
      }
      if (SUPABASE_DISABLED) {
        toast({
          title: 'Auth disabled (dev mode)',
          description: 'Supabase is disabled in this environment.',
        });
        setOpen(false);
        return;
      }

      const finalUsername = (username || generateLegendMemeUsername()).trim();
      const systemOptions = getSystemAvatarOptions();
      const finalAvatarUrl = avatarUrl || (systemOptions.length > 0 ? randPick(systemOptions) : defaultAvatarUrl(finalUsername));

      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: redirectTo,
          data: {
            username: finalUsername,
            avatar_url: finalAvatarUrl,
          },
        },
      });
      if (error) throw error;

      toast({
        title: 'Account created',
        description: 'If email confirmation is enabled, check your inbox to confirm your account.',
      });
      setOpen(false);
    } catch (e: any) {
      toast({ title: 'Sign up failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onPasswordSignIn = async () => {
    setBusy(true);
    try {
      if (SUPABASE_DISABLED) {
        toast({
          title: 'Auth disabled (dev mode)',
          description: 'Supabase is disabled in this environment.',
        });
        setOpen(false);
        return;
      }

      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      toast({ title: 'Signed in' });
      setOpen(false);
    } catch (e: any) {
      toast({ title: 'Sign in failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onMagicLink = async () => {
    setBusy(true);
    try {
      if (SUPABASE_DISABLED) {
        toast({
          title: 'Auth disabled (dev mode)',
          description: 'Supabase is disabled in this environment.',
        });
        setOpen(false);
        return;
      }

      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: { emailRedirectTo: redirectTo },
      });
      if (error) throw error;
      toast({ title: 'Magic link sent', description: 'Check your email to finish signing in.' });
      setOpen(false);
    } catch (e: any) {
      toast({ title: 'Magic link failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onOAuth = async (provider: 'google' | 'github') => {
    setBusy(true);
    try {
      if (SUPABASE_DISABLED) {
        toast({
          title: 'Auth disabled (dev mode)',
          description: 'Supabase is disabled in this environment.',
        });
        setOpen(false);
        return;
      }

      const { error } = await supabase.auth.signInWithOAuth({
        provider,
        options: { redirectTo },
      });
      if (error) throw error;
    } catch (e: any) {
      toast({ title: 'OAuth sign in failed', description: String(e?.message ?? e), variant: 'destructive' });
      setBusy(false);
    }
  };

  if (isAuthenticated) {
    return (
      <div className="flex items-center gap-2">
        <Popover open={planPopoverOpen} onOpenChange={setPlanPopoverOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              className={
                `h-8 px-2 rounded-md border border-border/60 hover:border-border transition-colors text-[10px] font-mono uppercase tracking-wider flex items-center gap-1.5 ` +
                planConfig.badge
              }
              disabled={busy}
              aria-label="Open plans"
            >
              <span className="text-current"><PlanIcon className="w-3.5 h-3.5" /></span>
              {planConfig.displayName}
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-72 p-2" align="end" alignOffset={44} sideOffset={6}>
            <div className="px-2 py-1.5">
              <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Current plan</div>
              <div className="text-sm font-semibold" style={{ color: planConfig.color }}>{planConfig.name}</div>
            </div>

            <div className="mt-2 space-y-1">
              {(Object.keys(PLAN_CONFIGS) as PlanTier[])
                .filter((tier) => tier !== currentPlan)
                .map((tier) => {
                  const cfg = PLAN_CONFIGS[tier];
                  const isPaid = tier !== 'free';
                  const Icon = tier === 'free' ? Gift : tier === 'essential' ? Zap : tier === 'architect' ? Sparkles : Crown;

                  return (
                    <button
                      key={tier}
                      type="button"
                      disabled={busy}
                      className={
                        `w-full flex items-center justify-between rounded-md border px-3 py-2 text-left transition-colors border-border/60 hover:bg-secondary/40 hover:brightness-110 ` +
                        cfg.badge
                      }
                      onClick={() => {
                        if (!isPaid) {
                          toast({ title: 'Free plan', description: 'Free plan is active by default.' });
                          setPlanPopoverOpen(false);
                          return;
                        }
                        setPlanPopoverOpen(false);
                        void startCheckout(tier);
                      }}
                      aria-label={`Upgrade to ${cfg.name}`}
                    >
                      <div className="min-w-0 flex items-center gap-2">
                        <Icon className="w-4 h-4" />
                        <div className="min-w-0">
                          <div className="text-xs font-semibold truncate" style={{ color: cfg.color }}>{cfg.displayName}</div>
                          <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground truncate">
                            {isPaid ? 'Upgrade' : 'Free'}
                          </div>
                        </div>
                      </div>
                    </button>
                  );
                })}
            </div>

            {currentPlan !== 'free' ? (
              <div className="mt-2 pt-2 border-t border-border/60">
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-border/60"
                  onClick={() => {
                    setPlanPopoverOpen(false);
                    void onManageBilling();
                  }}
                  disabled={busy}
                >
                  Manage billing
                </Button>
              </div>
            ) : null}
          </PopoverContent>
        </Popover>

        <button
          type="button"
          className="h-8 w-8 rounded-full overflow-hidden border border-border/60 hover:border-border transition-colors"
          onClick={() => navigate('/account')}
          aria-label="Open account"
          disabled={busy}
        >
          <img src={avatarSrc} alt="avatar" className="h-full w-full object-cover" />
        </button>
      </div>
    );
  }

  return (
    <>
      <Button
        variant={triggerVariant ?? 'default'}
        size={triggerSize ?? 'sm'}
        className={triggerClassName ?? 'shadow-glow-sm'}
        onClick={() => {
          setActiveTab(initialTab ?? 'password');
          setOpen(true);
        }}
      >
        <LogIn className="w-4 h-4 sm:mr-1.5" />
        <span className="hidden sm:inline">{triggerLabel ?? 'Sign in'}</span>
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{activeTab === 'signup' ? 'Sign up' : 'Sign in'}</DialogTitle>
            <DialogDescription>
              Sign in to run analysis and access plan features.
            </DialogDescription>
          </DialogHeader>

          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="password">Password</TabsTrigger>
              <TabsTrigger value="signup">Sign up</TabsTrigger>
              <TabsTrigger value="magic">Magic link</TabsTrigger>
              <TabsTrigger value="oauth">OAuth</TabsTrigger>
            </TabsList>

            <TabsContent value="password" className="space-y-3 mt-3">
              <Input
                placeholder="Email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <Input
                placeholder="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <Button className="w-full" onClick={onPasswordSignIn} disabled={busy || !email || !password}>
                <LogIn className="w-4 h-4 mr-2" />
                Sign in
              </Button>
            </TabsContent>

            <TabsContent value="signup" className="space-y-3 mt-3">
              <div className="flex gap-2">
                <Input
                  placeholder="Username (optional)"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setUsername(generateLegendMemeUsername())}
                  disabled={busy}
                >
                  Generate
                </Button>
              </div>
              <Input
                placeholder="Email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <Input
                placeholder="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <Input
                placeholder="Confirm password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />

              <div className="space-y-2">
                <div className="text-xs text-muted-foreground">Avatar (optional)</div>
                <div className="grid grid-cols-6 gap-2">
                  {getSystemAvatarOptions().map((url) => (
                    <button
                      key={url}
                      type="button"
                      onClick={() => setAvatarUrl(url)}
                      className={
                        `rounded-md border p-1 transition-colors ` +
                        (avatarUrl === url ? 'border-primary bg-primary/10' : 'border-border hover:bg-secondary/40')
                      }
                      aria-label="Select avatar"
                      disabled={busy}
                    >
                      <img src={url} alt="avatar" className="h-10 w-10" />
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const opts = getSystemAvatarOptions();
                      setAvatarUrl(opts.length > 0 ? randPick(opts) : null);
                    }}
                    disabled={busy}
                  >
                    Random
                  </Button>
                  {avatarUrl && (
                    <div className="text-xs text-muted-foreground">Selected</div>
                  )}
                </div>
              </div>

              <Button
                className="w-full"
                onClick={onSignUp}
                disabled={busy || !email || !password || !confirmPassword}
              >
                <LogIn className="w-4 h-4 mr-2" />
                Create account
              </Button>
            </TabsContent>

            <TabsContent value="magic" className="space-y-3 mt-3">
              <Input
                placeholder="Email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <Button className="w-full" onClick={onMagicLink} disabled={busy || !email}>
                <Mail className="w-4 h-4 mr-2" />
                Send magic link
              </Button>
            </TabsContent>

            <TabsContent value="oauth" className="space-y-2 mt-3">
              <Button className="w-full" variant="outline" onClick={() => onOAuth('google')} disabled={busy}>
                <Mail className="w-4 h-4 mr-2" />
                Continue with Google
              </Button>
              <Button className="w-full" variant="outline" onClick={() => onOAuth('github')} disabled={busy}>
                <Github className="w-4 h-4 mr-2" />
                Continue with GitHub
              </Button>
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>
    </>
  );
}
