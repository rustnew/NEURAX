import React from 'react';

export function Logo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <svg 
      viewBox="0 0 100 100" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Background Glow */}
      <defs>
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        <linearGradient id="brandGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="hsl(var(--primary))" />
          <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0.8" />
        </linearGradient>
      </defs>

      {/* Connection Lines (Neural pathways) */}
      <path 
        d="M20 50 L80 50 M50 20 L50 80 M35 35 L65 65 M35 65 L65 35" 
        stroke="currentColor" 
        strokeWidth="1.5" 
        strokeOpacity="0.2" 
        strokeDasharray="4 4"
      />

      {/* Central "Core" Block */}
      <rect 
        x="38" y="38" width="24" height="24" rx="6" 
        fill="url(#brandGradient)"
        filter="url(#glow)"
      />
      
      {/* Satellite Blocks (Input/Output/Compute) */}
      <rect x="15" y="42" width="16" height="16" rx="4" fill="currentColor" fillOpacity="0.4" />
      <rect x="69" y="42" width="16" height="16" rx="4" fill="currentColor" fillOpacity="0.4" />
      <rect x="42" y="15" width="16" height="16" rx="4" fill="currentColor" fillOpacity="0.4" />
      <rect x="42" y="69" width="16" height="16" rx="4" fill="currentColor" fillOpacity="0.4" />

      {/* Dynamic connections to satellite blocks */}
      <path d="M31 50 H38" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M62 50 H69" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M50 31 V38" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M50 62 V69" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />

      {/* Inner detail on central block (The "X" or "N" subtle hint) */}
      <path 
        d="M44 44 L56 56 M44 56 L56 44" 
        stroke="white" 
        strokeWidth="2.5" 
        strokeLinecap="round" 
        strokeOpacity="0.9"
      />
    </svg>
  );
}
