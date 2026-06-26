import { useState, useEffect, useRef, useCallback } from 'react';
import { InferenceControls, buildDefaultInferenceParams } from './InferenceControls.tsx';
import { BehaviorDashboard } from './BehaviorDashboard.tsx';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { InferenceParams, InferenceReport, simulateInference } from '@/services/neuraxApi.ts';

interface InferenceIntelligenceProps {
  architectureType: ArchitectureFamily;
}

export function InferenceIntelligence({ architectureType }: InferenceIntelligenceProps) {
  const [params, setParams] = useState<InferenceParams>(() =>
    buildDefaultInferenceParams(architectureType),
  );
  const [report, setReport] = useState<InferenceReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runSimulation = useCallback(async (p: InferenceParams) => {
    setLoading(true);
    setError(null);
    try {
      const res = await simulateInference({ params: p });
      setReport(res.report);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => runSimulation(params), 800);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [params, runSimulation]);

  useEffect(() => {
    setParams(buildDefaultInferenceParams(architectureType));
  }, [architectureType]);

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden bg-background">
      <InferenceControls
        architectureType={architectureType}
        params={params}
        onParamsChange={setParams}
      />
      <BehaviorDashboard
        architectureType={architectureType}
        report={report}
        loading={loading}
        error={error}
      />
    </div>
  );
}
