import React, { useState } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  IconButton,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  Speed,
  Timer,
  TrendingUp,
  Settings,
  Refresh,
  PlayArrow,
  Stop,
  Analytics,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';

// Components
import TelemetryChart from '../components/charts/TelemetryChart';
import LapTimesChart from '../components/charts/LapTimesChart';
import PerformanceMetrics from '../components/dashboard/PerformanceMetrics';
import RecentSessions from '../components/dashboard/RecentSessions';
import LiveTelemetry from '../components/dashboard/LiveTelemetry';
import QuickActions from '../components/dashboard/QuickActions';

// Services
import { dashboardService } from '../services/api';

// Types
interface DashboardStats {
  totalSessions: number;
  totalDistance: number;
  bestLapTime: number;
  avgLapTime: number;
  improvementTrend: number;
  activeTraining: boolean;
}

const Dashboard: React.FC = () => {
  const [isLiveMode, setIsLiveMode] = useState(false);

  const { data: dashboardData, isLoading, error, refetch } = useQuery({
    queryKey: ['dashboard'],
    queryFn: dashboardService.getDashboardData,
    refetchInterval: isLiveMode ? 1000 : 30000, // 1s in live mode, 30s otherwise
  });

  const stats: DashboardStats = dashboardData?.stats || {
    totalSessions: 0,
    totalDistance: 0,
    bestLapTime: 0,
    avgLapTime: 0,
    improvementTrend: 0,
    activeTraining: false,
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
      },
    },
  };

  const formatLapTime = (time: number): string => {
    if (!time) return '--:--:---';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const milliseconds = Math.floor((time % 1) * 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
  };

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Failed to load dashboard data. Please try refreshing.
      </Alert>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'between', alignItems: 'center' }}>
        <Typography variant="h1" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
          LapXcel Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant={isLiveMode ? 'contained' : 'outlined'}
            startIcon={isLiveMode ? <Stop /> : <PlayArrow />}
            onClick={() => setIsLiveMode(!isLiveMode)}
            color="primary"
          >
            {isLiveMode ? 'Stop Live' : 'Live Mode'}
          </Button>
          <IconButton onClick={() => refetch()} color="primary">
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Key Performance Indicators */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div variants={itemVariants}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Speed sx={{ color: 'primary.main', mr: 1 }} />
                  <Typography variant="h6">Best Lap Time</Typography>
                </Box>
                <Typography variant="h4" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                  {formatLapTime(stats.bestLapTime)}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <TrendingUp sx={{ color: 'success.main', fontSize: 16, mr: 0.5 }} />
                  <Typography variant="caption" sx={{ color: 'success.main' }}>
                    {stats.improvementTrend > 0 ? '+' : ''}{stats.improvementTrend.toFixed(2)}s this week
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div variants={itemVariants}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Timer sx={{ color: 'secondary.main', mr: 1 }} />
                  <Typography variant="h6">Total Sessions</Typography>
                </Box>
                <Typography variant="h4" sx={{ color: 'secondary.main', fontWeight: 'bold' }}>
                  {stats.totalSessions}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {stats.totalDistance.toFixed(1)} km driven
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div variants={itemVariants}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Analytics sx={{ color: 'info.main', mr: 1 }} />
                  <Typography variant="h6">Average Lap</Typography>
                </Box>
                <Typography variant="h4" sx={{ color: 'info.main', fontWeight: 'bold' }}>
                  {formatLapTime(stats.avgLapTime)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Consistency: 92.3%
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div variants={itemVariants}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Settings sx={{ color: 'warning.main', mr: 1 }} />
                  <Typography variant="h6">Training Status</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip
                    label={stats.activeTraining ? 'Active' : 'Idle'}
                    color={stats.activeTraining ? 'success' : 'default'}
                    size="small"
                  />
                  {stats.activeTraining && (
                    <Typography variant="caption" color="text.secondary">
                      Model: SAC v2.1
                    </Typography>
                  )}
                </Box>
                {stats.activeTraining && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Progress: 67%
                    </Typography>
                    <LinearProgress variant="determinate" value={67} sx={{ mt: 0.5 }} />
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Live Telemetry */}
        <Grid item xs={12} lg={8}>
          <motion.div variants={itemVariants}>
            <Card sx={{ height: 400 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Live Telemetry
                </Typography>
                {isLiveMode ? (
                  <LiveTelemetry />
                ) : (
                  <TelemetryChart
                    data={dashboardData?.recentTelemetry || []}
                    loading={isLoading}
                  />
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} lg={4}>
          <motion.div variants={itemVariants}>
            <QuickActions />
          </motion.div>
        </Grid>

        {/* Lap Times Trend */}
        <Grid item xs={12} md={6}>
          <motion.div variants={itemVariants}>
            <Card sx={{ height: 350 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Lap Times Trend
                </Typography>
                <LapTimesChart
                  data={dashboardData?.lapTimesTrend || []}
                  loading={isLoading}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} md={6}>
          <motion.div variants={itemVariants}>
            <PerformanceMetrics
              data={dashboardData?.performanceMetrics}
              loading={isLoading}
            />
          </motion.div>
        </Grid>

        {/* Recent Sessions */}
        <Grid item xs={12}>
          <motion.div variants={itemVariants}>
            <RecentSessions
              sessions={dashboardData?.recentSessions || []}
              loading={isLoading}
            />
          </motion.div>
        </Grid>
      </Grid>
    </motion.div>
  );
};

export default Dashboard;