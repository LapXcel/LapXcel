import React, { useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Box, Typography, Skeleton, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { useState } from 'react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface TelemetryDataPoint {
  timestamp: string;
  sessionTime: number;
  speedKmh: number;
  throttleInput: number;
  brakeInput: number;
  steeringInput: number;
  trackProgress: number;
  gear: number;
  rpm: number;
}

interface TelemetryChartProps {
  data: TelemetryDataPoint[];
  loading?: boolean;
  height?: number;
}

type MetricType = 'speed' | 'inputs' | 'engine' | 'all';

const TelemetryChart: React.FC<TelemetryChartProps> = ({
  data = [],
  loading = false,
  height = 300,
}) => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('speed');

  const chartData = useMemo(() => {
    if (!data.length) return null;

    const labels = data.map((point) => point.sessionTime.toFixed(1));

    const datasets = [];

    switch (selectedMetric) {
      case 'speed':
        datasets.push({
          label: 'Speed (km/h)',
          data: data.map((point) => point.speedKmh),
          borderColor: '#00ff88',
          backgroundColor: 'rgba(0, 255, 136, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          yAxisID: 'y',
        });
        break;

      case 'inputs':
        datasets.push(
          {
            label: 'Throttle',
            data: data.map((point) => point.throttleInput * 100),
            borderColor: '#00ff88',
            backgroundColor: 'rgba(0, 255, 136, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.2,
            yAxisID: 'y',
          },
          {
            label: 'Brake',
            data: data.map((point) => point.brakeInput * 100),
            borderColor: '#ff6b35',
            backgroundColor: 'rgba(255, 107, 53, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.2,
            yAxisID: 'y',
          },
          {
            label: 'Steering',
            data: data.map((point) => point.steeringInput * 100),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.2,
            yAxisID: 'y1',
          }
        );
        break;

      case 'engine':
        datasets.push(
          {
            label: 'RPM',
            data: data.map((point) => point.rpm),
            borderColor: '#ff6b35',
            backgroundColor: 'rgba(255, 107, 53, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            yAxisID: 'y',
          },
          {
            label: 'Gear',
            data: data.map((point) => point.gear),
            borderColor: '#9333ea',
            backgroundColor: 'rgba(147, 51, 234, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.1,
            yAxisID: 'y1',
            stepped: true,
          }
        );
        break;

      case 'all':
        datasets.push(
          {
            label: 'Speed (km/h)',
            data: data.map((point) => point.speedKmh),
            borderColor: '#00ff88',
            backgroundColor: 'rgba(0, 255, 136, 0.1)',
            borderWidth: 1.5,
            fill: false,
            tension: 0.4,
            yAxisID: 'y',
          },
          {
            label: 'Throttle %',
            data: data.map((point) => point.throttleInput * 100),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 1.5,
            fill: false,
            tension: 0.2,
            yAxisID: 'y1',
          },
          {
            label: 'Brake %',
            data: data.map((point) => point.brakeInput * 100),
            borderColor: '#ff6b35',
            backgroundColor: 'rgba(255, 107, 53, 0.1)',
            borderWidth: 1.5,
            fill: false,
            tension: 0.2,
            yAxisID: 'y1',
          }
        );
        break;
    }

    return {
      labels,
      datasets,
    };
  }, [data, selectedMetric]);

  const options: ChartOptions<'line'> = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index' as const,
        intersect: false,
      },
      plugins: {
        legend: {
          position: 'top' as const,
          labels: {
            color: '#ffffff',
            usePointStyle: true,
            padding: 20,
          },
        },
        tooltip: {
          backgroundColor: 'rgba(26, 26, 26, 0.95)',
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          borderColor: '#333333',
          borderWidth: 1,
          callbacks: {
            label: function (context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              if (context.parsed.y !== null) {
                if (selectedMetric === 'inputs' && label.includes('Steering')) {
                  label += context.parsed.y.toFixed(1) + '%';
                } else if (selectedMetric === 'inputs') {
                  label += context.parsed.y.toFixed(1) + '%';
                } else if (selectedMetric === 'engine' && label.includes('RPM')) {
                  label += context.parsed.y.toFixed(0);
                } else if (selectedMetric === 'engine' && label.includes('Gear')) {
                  label += context.parsed.y.toFixed(0);
                } else {
                  label += context.parsed.y.toFixed(1);
                }
              }
              return label;
            },
          },
        },
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: 'Session Time (s)',
            color: '#ffffff',
          },
          ticks: {
            color: '#b3b3b3',
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
          },
        },
        y: {
          type: 'linear' as const,
          display: true,
          position: 'left' as const,
          title: {
            display: true,
            text: selectedMetric === 'speed' ? 'Speed (km/h)' : 
                  selectedMetric === 'inputs' ? 'Input %' :
                  selectedMetric === 'engine' ? 'RPM' : 'Primary',
            color: '#ffffff',
          },
          ticks: {
            color: '#b3b3b3',
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
          },
        },
        ...(selectedMetric === 'inputs' || selectedMetric === 'engine' || selectedMetric === 'all'
          ? {
              y1: {
                type: 'linear' as const,
                display: true,
                position: 'right' as const,
                title: {
                  display: true,
                  text: selectedMetric === 'inputs' ? 'Steering %' :
                        selectedMetric === 'engine' ? 'Gear' : 'Secondary',
                  color: '#ffffff',
                },
                ticks: {
                  color: '#b3b3b3',
                },
                grid: {
                  drawOnChartArea: false,
                },
                min: selectedMetric === 'inputs' ? -100 : undefined,
                max: selectedMetric === 'inputs' ? 100 : undefined,
              },
            }
          : {}),
      },
      elements: {
        point: {
          radius: 0,
          hoverRadius: 4,
        },
      },
      animation: {
        duration: 750,
        easing: 'easeInOutQuart',
      },
    }),
    [selectedMetric]
  );

  if (loading) {
    return (
      <Box sx={{ height }}>
        <Skeleton variant="rectangular" width="100%" height={40} sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" width="100%" height={height - 60} />
      </Box>
    );
  }

  if (!data.length) {
    return (
      <Box
        sx={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'text.secondary',
        }}
      >
        <Typography variant="body1">No telemetry data available</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height }}>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'center' }}>
        <ToggleButtonGroup
          value={selectedMetric}
          exclusive
          onChange={(_, value) => value && setSelectedMetric(value)}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              color: 'text.secondary',
              borderColor: '#333',
              '&.Mui-selected': {
                backgroundColor: 'primary.main',
                color: 'black',
                '&:hover': {
                  backgroundColor: 'primary.dark',
                },
              },
            },
          }}
        >
          <ToggleButton value="speed">Speed</ToggleButton>
          <ToggleButton value="inputs">Inputs</ToggleButton>
          <ToggleButton value="engine">Engine</ToggleButton>
          <ToggleButton value="all">All</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      
      <Box sx={{ height: height - 60 }}>
        {chartData && <Line data={chartData} options={options} />}
      </Box>
    </Box>
  );
};

export default TelemetryChart;