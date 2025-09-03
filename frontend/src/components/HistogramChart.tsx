import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface HistogramChartProps {
  data: number[];
  color: string;
  title: string;
  entropy?: number;
}

export const HistogramChart: React.FC<HistogramChartProps> = ({
  data,
  color,
  title,
  entropy
}) => {
  // Transform histogram data for Recharts
  const chartData = data.map((value, index) => ({
    intensity: index,
    frequency: value
  }));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h5 className="text-xs font-medium text-foreground">{title}</h5>
        {entropy && (
          <span className="text-xs text-muted-foreground">
            Entropy: {entropy.toFixed(2)}
          </span>
        )}
      </div>
      <div className="h-24">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.3} />
            <XAxis 
              dataKey="intensity" 
              tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
              axisLine={false}
              tickLine={false}
              domain={[0, 255]}
            />
            <YAxis 
              tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'hsl(var(--background))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '6px',
                fontSize: '12px'
              }}
              labelStyle={{ color: 'hsl(var(--foreground))' }}
              formatter={(value: number) => [value, 'Frequency']}
              labelFormatter={(label: number) => `Intensity: ${label}`}
            />
            <Line 
              type="monotone" 
              dataKey="frequency" 
              stroke={color} 
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 2, fill: color }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};