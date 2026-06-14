import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import type { Session } from '@supabase/supabase-js';
import { supabase } from '@/lib/supabaseClient.ts';

const SUPABASE_DISABLED = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

interface AuthContextType {
  session: Session | null;
  accessToken: string | null;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);

  useEffect(() => {
    if (SUPABASE_DISABLED) {
      setSession(
        {
          access_token: 'dev-token',
          token_type: 'bearer',
          expires_in: 60 * 60 * 24 * 365,
          expires_at: Math.floor(Date.now() / 1000) + 60 * 60 * 24 * 365,
          refresh_token: 'dev-refresh',
          user: {
            id: 'dev-user',
            aud: 'authenticated',
            role: 'authenticated',
            email: 'dev@local',
            app_metadata: {},
            user_metadata: {},
            created_at: new Date().toISOString(),
          },
        } as unknown as Session,
      );
      return;
    }

    let mounted = true;

    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      setSession(data.session ?? null);
    });

    const { data: sub } = supabase.auth.onAuthStateChange((_event, next) => {
      if (!mounted) return;
      setSession(next);
    });

    return () => {
      mounted = false;
      sub.subscription.unsubscribe();
    };
  }, []);

  const value = useMemo<AuthContextType>(() => {
    const accessToken = session?.access_token ?? null;
    return {
      session,
      accessToken,
      isAuthenticated: !!accessToken,
    };
  }, [session]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
