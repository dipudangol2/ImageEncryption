import React from 'react';
import { HistogramChart } from './HistogramChart';

interface HistogramData {
  type?: string;
  channels?: {
    red?: {
      histogram?: number[];
      entropy?: number;
    };
    green?: {
      histogram?: number[];
      entropy?: number;
    };
    blue?: {
      histogram?: number[];
      entropy?: number;
    };
  };
  overall?: {
    mean_intensity?: number;
    std_intensity?: number;
  };
  entropy?: number;
}

interface HistogramPanelProps {
  title: string;
  data: HistogramData;
  overallEntropy?: number;
}

export const HistogramPanel: React.FC<HistogramPanelProps> = ({
  title,
  data,
  overallEntropy
}) => {
  if (!data || !data.channels) {
    return (
      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-foreground">{title}</h4>
        <div className="p-4 text-xs text-muted-foreground bg-muted/20 rounded">
          No histogram data available
        </div>
      </div>
    );
  }

  const { channels } = data;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-foreground">{title}</h4>
        {overallEntropy && (
          <div className="text-xs">
            <span className="text-muted-foreground">Overall Entropy: </span>
            <span className="font-medium text-foreground">{overallEntropy.toFixed(2)}</span>
          </div>
        )}
      </div>
      
      <div className="space-y-3">
        {/* Red Channel */}
        {channels.red?.histogram && (
          <HistogramChart
            data={channels.red.histogram}
            color="#ef4444"
            title="Red Channel"
            entropy={channels.red.entropy}
          />
        )}
        
        {/* Green Channel */}
        {channels.green?.histogram && (
          <HistogramChart
            data={channels.green.histogram}
            color="#22c55e"
            title="Green Channel"
            entropy={channels.green.entropy}
          />
        )}
        
        {/* Blue Channel */}
        {channels.blue?.histogram && (
          <HistogramChart
            data={channels.blue.histogram}
            color="#3b82f6"
            title="Blue Channel"
            entropy={channels.blue.entropy}
          />
        )}
      </div>

      {/* Additional Statistics */}
      {data.overall && (
        <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-border/30">
          {data.overall.mean_intensity !== undefined && (
            <div>
              <span className="text-muted-foreground">Mean:</span>
              <span className="ml-1 font-medium">{data.overall.mean_intensity.toFixed(1)}</span>
            </div>
          )}
          {data.overall.std_intensity !== undefined && (
            <div>
              <span className="text-muted-foreground">Std Dev:</span>
              <span className="ml-1 font-medium">{data.overall.std_intensity.toFixed(1)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};