import { ArchitectureFamily } from '@/types/plugins.ts';
import { 
  TransformerPlugin, 
  MoEPlugin, 
  SSMPlugin, 
  GenerativePlugin, 
  GNNPlugin,
  RNNPlugin,
} from '@/plugins/components';
import { cn } from '@/lib/utils.ts';

interface PluginRendererProps {
  architecture: ArchitectureFamily;
  className?: string;
}

export function PluginRenderer({ architecture, className }: PluginRendererProps) {
  const renderPlugin = () => {
    switch (architecture) {
      case 'transformer':
        return <TransformerPlugin className={className} />;
      case 'moe':
        return <MoEPlugin className={className} />;
      case 'ssm':
        return <SSMPlugin className={className} />;
      case 'diffusion':
      case 'gan':
        return <GenerativePlugin className={className} />;
      case 'gnn':
        return <GNNPlugin className={className} />;
      case 'rnn':
        return <RNNPlugin className={className} />;
      case 'rl':
      case 'snn':
      case 'experimental':
      default:
        return <TransformerPlugin className={className} />;
    }
  };

  return (
    <div className={cn("transition-all duration-300 ease-in-out", className)}>
      {renderPlugin()}
    </div>
  );
}
