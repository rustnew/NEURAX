import { createClient } from '@supabase/supabase-js';

const supabaseDisabled = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

if (!supabaseDisabled) {
  if (!supabaseUrl) {
    throw new Error('Missing VITE_SUPABASE_URL');
  }
  if (!supabaseAnonKey) {
    throw new Error('Missing VITE_SUPABASE_ANON_KEY');
  }
}

export const supabase = createClient(
  supabaseUrl ?? 'http://localhost:54321',
  supabaseAnonKey ?? 'dev-anon-key',
);
