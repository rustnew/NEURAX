import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Check } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { supabase } from '@/lib/supabaseClient.ts';
import { useToast } from '@/hooks/use-toast.ts';
import { PricingPage } from '@/pages/Pricing.tsx';
import { createBillingPortalSession } from '@/services/neuraxApi.ts';

const SYSTEM_AVATAR_SEEDS = Array.from({ length: 100 }, (_v, i) => `avatar-${i + 1}`);

function systemAvatarUrl(seed: string): string {
  const s = encodeURIComponent(seed);
  return `https://api.dicebear.com/7.x/bottts/svg?seed=${s}`;
}

function defaultAvatarUrl(seed: string): string {
  const s = encodeURIComponent(seed || 'user');
  return `https://api.dicebear.com/7.x/identicon/svg?seed=${s}`;
}

export default function Account() {
  const { session, isAuthenticated } = useAuth();
  const { currentPlan, planConfig } = usePlan();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [busy, setBusy] = useState(false);
  const [pricingOpen, setPricingOpen] = useState(false);

  const [selectedAvatarUrl, setSelectedAvatarUrl] = useState<string | null>(null);

  const avatarSrc = useMemo(() => {
    const m = (session?.user?.user_metadata ?? {}) as Record<string, unknown>;
    const metaUrl = typeof m.avatar_url === 'string' ? m.avatar_url : null;
    const fallback = defaultAvatarUrl(session?.user?.email ?? 'user');
    return metaUrl ?? fallback;
  }, [session]);

  const avatarOptions = useMemo(() => {
    return SYSTEM_AVATAR_SEEDS.map(systemAvatarUrl);
  }, []);

  const username = useMemo(() => {
    const m = (session?.user?.user_metadata ?? {}) as Record<string, unknown>;
    const u = typeof m.username === 'string' ? m.username : null;
    return u;
  }, [session]);

  const email = session?.user?.email ?? null;

  const onSignOut = async () => {
    setBusy(true);
    try {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      toast({ title: 'Signed out' });
      navigate('/');
    } catch (e: any) {
      toast({ title: 'Sign out failed', description: String(e?.message ?? e), variant: 'destructive' });
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
      toast({
        title: 'Billing portal failed',
        description: String(e?.message ?? e),
        variant: 'destructive',
      });
    } finally {
      setBusy(false);
    }
  };

  const onSaveAvatar = async () => {
    const nextAvatar = (selectedAvatarUrl ?? '').trim();
    if (!nextAvatar) return;

    setBusy(true);
    try {
      const { error: authErr } = await supabase.auth.updateUser({
        data: {
          avatar_url: nextAvatar,
        },
      });
      if (authErr) throw authErr;

      const userId = session?.user?.id;
      if (userId) {
        const { error: profErr } = await supabase
          .from('user_profiles')
          .update({ avatar_url: nextAvatar })
          .eq('id', userId);
        if (profErr) {
          throw profErr;
        }
      }

      toast({ title: 'Profile updated' });
    } catch (e: any) {
      toast({ title: 'Update failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto max-w-3xl px-4 sm:px-6 py-10">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold">Account</h1>
            <Button asChild variant="outline" size="sm" className="border-border/60">
              <Link to="/">Back</Link>
            </Button>
          </div>
          <div className="mt-6 rounded-xl border border-border/60 bg-card/50 p-6 text-sm text-muted-foreground">
            You’re not signed in.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-3xl px-4 sm:px-6 py-10">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">Account</h1>
          <Button asChild variant="outline" size="sm" className="border-border/60">
            <Link to="/app">Back to workspace</Link>
          </Button>
        </div>

        <div className="mt-6 rounded-xl border border-border/60 bg-card/50 p-6">
          <div className="flex items-center gap-4">
            <div className="h-14 w-14 rounded-full overflow-hidden border border-border/60">
              <img src={avatarSrc} alt="avatar" className="h-full w-full object-cover" />
            </div>
            <div className="min-w-0">
              <div className="font-semibold truncate">{username ?? 'User'}</div>
              {email ? (
                <div className="text-sm text-muted-foreground truncate">{email}</div>
              ) : null}
            </div>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-3">
            <div className="rounded-lg border border-border/60 bg-background/40 p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-xs text-muted-foreground">Current plan</div>
                  <div className="font-semibold" style={{ color: planConfig.color }}>{planConfig.name}</div>
                </div>
                <div className="flex items-center gap-2">
                  {currentPlan !== 'free' ? (
                    <Button variant="outline" className="border-border/60" onClick={onManageBilling} disabled={busy}>
                      Manage billing
                    </Button>
                  ) : null}
                  <Button variant="outline" className="border-border/60" onClick={() => setPricingOpen(true)} disabled={busy}>
                    Upgrade
                  </Button>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-border/60 bg-background/40 p-4">
              <div className="text-xs text-muted-foreground">Avatar</div>
              <div className="mt-3 grid grid-cols-6 gap-2">
                {avatarOptions.map((url) => {
                  const selected = (selectedAvatarUrl ?? avatarSrc) === url;
                  return (
                    <button
                      key={url}
                      type="button"
                      className={
                        `relative rounded-md border p-1 transition-colors ` +
                        (selected ? 'border-primary bg-primary/10' : 'border-border/60 hover:bg-secondary/40')
                      }
                      onClick={() => setSelectedAvatarUrl(url)}
                      aria-label="Select avatar"
                      disabled={busy}
                    >
                      <img src={url} alt="avatar option" className="h-10 w-10" />
                      {selected ? (
                        <span className="absolute -top-1 -right-1 rounded-full bg-primary text-primary-foreground p-0.5">
                          <Check className="w-3 h-3" />
                        </span>
                      ) : null}
                    </button>
                  );
                })}
              </div>
              <div className="mt-3 flex items-center justify-end gap-2">
                <Button
                  variant="outline"
                  className="border-border/60"
                  onClick={onSaveAvatar}
                  disabled={busy || !(selectedAvatarUrl && selectedAvatarUrl !== avatarSrc)}
                >
                  Save
                </Button>
              </div>
            </div>

            <div className="flex items-center justify-end">
              <Button variant="outline" className="border-border/60" onClick={onSignOut} disabled={busy}>
                Sign out
              </Button>
            </div>
          </div>
        </div>
      </div>

      <PricingPage
        isOpen={pricingOpen}
        onClose={() => setPricingOpen(false)}
        onSelectPlan={(plan) => {
          toast({ title: 'Plan selection', description: `Selected: ${plan}` });
          if (plan !== currentPlan) {
            toast({ title: 'Billing not connected', description: 'Hook up Stripe billing to activate upgrades.', variant: 'destructive' });
          }
          setPricingOpen(false);
        }}
      />
    </div>
  );
}
