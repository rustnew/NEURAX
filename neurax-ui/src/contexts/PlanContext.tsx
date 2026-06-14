import { createContext, useContext, useEffect, useMemo, useState, ReactNode } from 'react';
import { PlanTier, PLAN_CONFIGS, PlanConfig } from '@/types/plans.ts';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { getMe, setNeuraxAccessToken } from '@/services/neuraxApi.ts';

const SUPABASE_DISABLED = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

interface PlanContextType {
  currentPlan: PlanTier;
  planConfig: PlanConfig;
  canAccess: (minPlan: PlanTier) => boolean;
}

const PlanContext = createContext<PlanContextType | undefined>(undefined);

const PLAN_ORDER: PlanTier[] = ['free', 'essential', 'architect', 'elite'];

export function PlanProvider({ children }: { children: ReactNode }) {
  const { accessToken, isAuthenticated } = useAuth();
  const [currentPlan, setCurrentPlan] = useState<PlanTier>('free');

  const planConfig = PLAN_CONFIGS[currentPlan];

  useEffect(() => {
    setNeuraxAccessToken(accessToken);
  }, [accessToken]);

  useEffect(() => {
    let mounted = true;
    if (SUPABASE_DISABLED) {
      setCurrentPlan('elite');
      return;
    }
    if (!isAuthenticated) {
      setCurrentPlan('free');
      return;
    }
    getMe()
      .then((me) => {
        if (!mounted) return;
        setCurrentPlan(me.plan);
      })
      .catch(() => {
        if (!mounted) return;
        setCurrentPlan('free');
      });
    return () => {
      mounted = false;
    };
  }, [isAuthenticated]);

  const canAccess = (minPlan: PlanTier): boolean => {
    return PLAN_ORDER.indexOf(currentPlan) >= PLAN_ORDER.indexOf(minPlan);
  };

  const value = useMemo(
    () => ({
      currentPlan,
      planConfig,
      canAccess,
    }),
    [currentPlan, planConfig],
  );

  return <PlanContext.Provider value={value}>{children}</PlanContext.Provider>;
}

export function usePlan() {
  const context = useContext(PlanContext);
  if (context === undefined) {
    throw new Error('usePlan must be used within a PlanProvider');
  }
  return context;
}
