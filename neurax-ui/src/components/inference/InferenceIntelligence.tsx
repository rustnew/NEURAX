import { InferenceControls } from './InferenceControls.tsx';
import { BehaviorDashboard } from './BehaviorDashboard.tsx';
import { ArchitectureFamily } from '@/types/plugins.ts';

interface InferenceIntelligenceProps {
  architectureType: ArchitectureFamily;
}

export function InferenceIntelligence({ architectureType }: InferenceIntelligenceProps) {
  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden bg-background">
      {/* Left Panel - Inference Controls */}
      <InferenceControls architectureType={architectureType} />
      
      {/* Right Panel - Behavior Dashboard */}
      <BehaviorDashboard architectureType={architectureType} />
    </div>
  );
}
