import { useState } from 'react';
import { Check, Crown, Sparkles, Zap, ArrowRight, Gift } from 'lucide-react';
import { PLAN_CONFIGS, PlanTier } from '@/types/plans.ts';
import { cn } from '@/lib/utils.ts';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { createCheckoutSession } from '@/services/neuraxApi.ts';
import { useToast } from '@/hooks/use-toast.ts';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog.tsx';

interface PricingPageProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectPlan?: (plan: PlanTier) => void;
}

const planIcons: Record<string, React.ComponentType<{ className?: string; style?: React.CSSProperties }>> = {
  essential: Zap,
  architect: Sparkles,
  elite: Crown,
};

const PAID_PLANS: PlanTier[] = ['essential', 'architect', 'elite'];

const PLAN_DETAILS: Record<string, string[]> = {
  essential: [
    'Transformer & CNN architectures',
    'Shape validation & VRAM estimation',
    'FLOPs calculation',
    'Consumer GPU simulation (RTX 3090/4090)',
    'PyTorch & ONNX export',
    'Community support',
  ],
  architect: [
    'Everything in Essential, plus:',
    'MoE, SSM (incl. Mamba), Diffusion, GNN',
    'Gradient curvature analysis',
    'Router collapse detection',
    'Cloud GPU simulation (A100, H100, TPU)',
    'Rust/Burn & Triton kernel export',
    'Priority support',
  ],
  elite: [
    'Everything in Architect, plus:',
    'Experimental architectures (Early Access)',
    'muP scaling diagnostics',
    'Hardware bottleneck analysis',
    'Multi-node cluster simulation',
    'Megatron-LM parallelized export',
    'Real-time collaboration',
    'Dedicated support & SLA',
  ],
};

export function PricingPage({ isOpen, onClose, onSelectPlan }: PricingPageProps) {
  const [isAnnual, setIsAnnual] = useState(true);
  const { toast } = useToast();
  const [busy, setBusy] = useState(false);

  const startCheckout = async (plan: Exclude<PlanTier, 'free'>) => {
    setBusy(true);
    try {
      const origin = window.location.origin;
      const interval = isAnnual ? 'year' : 'month';

      const { url } = await createCheckoutSession({
        plan,
        interval,
        success_url: `${origin}/account?checkout=success`,
        cancel_url: `${origin}/account?checkout=cancel`,
      });

      window.location.assign(url);
    } catch (e: any) {
      toast({
        title: 'Checkout failed',
        description: String(e?.message ?? e),
        variant: 'destructive',
      });
    } finally {
      setBusy(false);
    }
  };

  const handleSelectPlan = (plan: PlanTier) => {
    if (onSelectPlan) {
      onSelectPlan(plan);
      return;
    }
    if (plan === 'free') {
      onClose();
      return;
    }
    void startCheckout(plan);
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto bg-card p-0">
        <DialogHeader className="p-6 pb-4 border-b border-border">
          <DialogTitle className="text-xl font-semibold tracking-tight">
            Choose Your Plan
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Scale your neural architecture design capabilities
          </DialogDescription>
        </DialogHeader>

        {/* Billing toggle */}
        <div className="flex items-center justify-center gap-3 py-4">
          <span className={cn(
            "text-sm transition-colors",
            !isAnnual ? "text-foreground" : "text-muted-foreground"
          )}>
            Monthly
          </span>
          <Switch
            checked={isAnnual}
            onCheckedChange={setIsAnnual}
          />
          <span className={cn(
            "text-sm transition-colors",
            isAnnual ? "text-foreground" : "text-muted-foreground"
          )}>
            Annual
          </span>
          {isAnnual && (
            <Badge variant="outline" className="ml-2 bg-emerald-500/10 text-emerald-400 border-emerald-500/30">
              Save 17%
            </Badge>
          )}
        </div>

        {/* Free plan */}
        <div className="px-6 pt-2">
          <div className="rounded-xl border border-border bg-secondary/20 p-5">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-slate-500/10">
                <Gift className="w-4 h-4 text-slate-300" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold">{PLAN_CONFIGS.free.name}</h3>
                <p className="text-[11px] text-muted-foreground">Get started with core features</p>
              </div>
              <Badge variant="outline" className="bg-slate-500/10 text-slate-300 border-slate-500/30">
                FREE
              </Badge>
            </div>

            <div className="mb-4">
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-bold">€0</span>
                <span className="text-sm text-muted-foreground">/mo</span>
              </div>
              <p className="text-[11px] text-muted-foreground mt-1">No credit card required</p>
            </div>

            <ul className="space-y-2 mb-5">
              {PLAN_CONFIGS.free.features.map((feature, idx) => (
                <li key={idx} className="flex items-start gap-2 text-xs">
                  <Check className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>

            <Button
              className="w-full"
              variant="secondary"
              onClick={() => {
                handleSelectPlan('free');
              }}
              disabled={busy}
            >
              Continue with Free
              <ArrowRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        </div>

        {/* Plans grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-6 pt-2">
          {PAID_PLANS.map((planId) => {
            const plan = PLAN_CONFIGS[planId];
            const Icon = planIcons[plan.id] || Zap;
            const price = isAnnual 
              ? Math.round(plan.price.annual / 12) 
              : plan.price.monthly;
            const isPopular = plan.id === 'architect';

            return (
              <div
                key={plan.id}
                className={cn(
                  "relative rounded-xl border p-5 transition-all",
                  isPopular 
                    ? "border-primary bg-primary/5 shadow-glow-sm" 
                    : "border-border bg-secondary/20 hover:border-primary/50"
                )}
              >
                {isPopular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <Badge className="bg-primary text-primary-foreground text-[10px] uppercase tracking-wider">
                      Most Popular
                    </Badge>
                  </div>
                )}

                <div className="flex items-center gap-2 mb-3">
                  <div 
                    className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${plan.color}20` }}
                  >
                    <Icon className="w-4 h-4" style={{ color: plan.color }} />
                  </div>
                  <h3 className="font-semibold">{plan.name}</h3>
                </div>

                <div className="mb-4">
                  <div className="flex items-baseline gap-1">
                    <span className="text-3xl font-bold">€{price}</span>
                    <span className="text-sm text-muted-foreground">/mo</span>
                  </div>
                  {isAnnual && (
                    <p className="text-[11px] text-muted-foreground mt-1">
                      Billed €{plan.price.annual}/year
                    </p>
                  )}
                </div>

                <ul className="space-y-2 mb-5">
                  {PLAN_DETAILS[plan.id].map((feature, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-xs">
                      <Check className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                      <span className={cn(
                        idx === 0 && plan.id !== 'essential' && "text-muted-foreground italic"
                      )}>
                        {feature}
                      </span>
                    </li>
                  ))}
                </ul>

                <Button
                  className={cn(
                    "w-full",
                    isPopular 
                      ? "bg-primary hover:bg-primary/90" 
                      : "bg-secondary hover:bg-secondary/80"
                  )}
                  variant={isPopular ? "default" : "secondary"}
                  onClick={() => handleSelectPlan(plan.id)}
                  disabled={busy}
                >
                  {plan.id === 'elite' ? 'Contact Sales' : 'Get Started'}
                  <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
            );
          })}
        </div>

        {/* ROI note */}
        <div className="mx-6 mb-6 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
          <p className="text-xs text-center text-amber-400/90">
            <span className="font-medium">Engineering ROI:</span> A single prevented GPU OOM or training crash can save thousands of euros in compute costs.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
