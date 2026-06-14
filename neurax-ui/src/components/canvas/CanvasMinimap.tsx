import { CanvasNode as CanvasNodeType, Connection } from '@/types/architecture.ts';
import { cn } from '@/lib/utils.ts';

interface CanvasMinimapProps {
  nodes: CanvasNodeType[];
  connections: Connection[];
  viewportOffset: { x: number; y: number };
  zoom: number;
  onClick: (x: number, y: number) => void;
}

export function CanvasMinimap({ nodes, connections, viewportOffset, zoom, onClick }: CanvasMinimapProps) {
  const mapWidth = 180;
  const mapHeight = 120;
  const scale = 0.06;

  // Calculate bounds
  const bounds = nodes.reduce(
    (acc, node) => ({
      minX: Math.min(acc.minX, node.x),
      minY: Math.min(acc.minY, node.y),
      maxX: Math.max(acc.maxX, node.x + 176),
      maxY: Math.max(acc.maxY, node.y + 80),
    }),
    { minX: 0, minY: 0, maxX: 1200, maxY: 800 }
  );

  const contentWidth = bounds.maxX - bounds.minX + 400;
  const contentHeight = bounds.maxY - bounds.minY + 400;
  const effectiveScale = Math.min(mapWidth / contentWidth, mapHeight / contentHeight, scale);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / effectiveScale + bounds.minX - 200;
    const y = (e.clientY - rect.top) / effectiveScale + bounds.minY - 200;
    onClick(x, y);
  };

  const handleDrag = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.buttons === 1) {
      handleClick(e);
    }
  };

  // Viewport rectangle
  const viewportX = (-viewportOffset.x / zoom - bounds.minX + 200) * effectiveScale;
  const viewportY = (-viewportOffset.y / zoom - bounds.minY + 200) * effectiveScale;
  const viewportW = (window.innerWidth / zoom) * effectiveScale;
  const viewportH = (window.innerHeight / zoom) * effectiveScale;

  if (nodes.length === 0) return null;

  return (
    <div className="absolute bottom-4 right-4 z-20 bg-card/95 backdrop-blur-md rounded-xl border border-border/50 shadow-lg p-2 group hover:shadow-xl transition-shadow">
      <svg
        width={mapWidth}
        height={mapHeight}
        className="cursor-pointer rounded-lg overflow-hidden"
        onClick={handleClick}
        onMouseMove={handleDrag}
      >
        {/* Background with subtle gradient */}
        <defs>
          <linearGradient id="minimapBg" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="hsl(var(--canvas-bg))" />
            <stop offset="100%" stopColor="hsl(var(--secondary))" stopOpacity="0.5" />
          </linearGradient>
        </defs>
        <rect width={mapWidth} height={mapHeight} fill="url(#minimapBg)" rx="6" />

        {/* Connections */}
        <g opacity="0.4">
          {connections.map((conn) => {
            const fromNode = nodes.find(n => n.id === conn.from);
            const toNode = nodes.find(n => n.id === conn.to);
            if (!fromNode || !toNode) return null;
            
            const x1 = (fromNode.x + 176 - bounds.minX + 200) * effectiveScale;
            const y1 = (fromNode.y + 40 - bounds.minY + 200) * effectiveScale;
            const x2 = (toNode.x - bounds.minX + 200) * effectiveScale;
            const y2 = (toNode.y + 40 - bounds.minY + 200) * effectiveScale;
            
            return (
              <line
                key={conn.id}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="hsl(var(--primary))"
                strokeWidth="1"
              />
            );
          })}
        </g>

        {/* Nodes */}
        {nodes.map((node) => (
          <rect
            key={node.id}
            x={(node.x - bounds.minX + 200) * effectiveScale}
            y={(node.y - bounds.minY + 200) * effectiveScale}
            width={Math.max(176 * effectiveScale, 4)}
            height={Math.max(80 * effectiveScale, 3)}
            fill="hsl(var(--primary))"
            opacity="0.7"
            rx="1"
          />
        ))}

        {/* Viewport indicator */}
        <rect
          x={Math.max(0, viewportX)}
          y={Math.max(0, viewportY)}
          width={Math.min(viewportW, mapWidth - Math.max(0, viewportX))}
          height={Math.min(viewportH, mapHeight - Math.max(0, viewportY))}
          fill="hsl(var(--foreground) / 0.05)"
          stroke="hsl(var(--foreground))"
          strokeWidth="1.5"
          opacity="0.6"
          rx="3"
          className="pointer-events-none"
        />
      </svg>
      
      {/* Label */}
      <div className="text-[9px] text-muted-foreground/50 text-center mt-1 uppercase tracking-wider">
        Overview
      </div>
    </div>
  );
}
