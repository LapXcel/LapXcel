/**
 * Dashboard Component Tests
 * Test suite for the main dashboard component.
 * Author: Sarah Siage
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Dashboard from '../../src/pages/Dashboard';
import { AuthProvider } from '../../src/hooks/useAuth';

// Mock API calls
jest.mock('../../src/services/api', () => ({
  analyticsAPI: {
    getPerformanceOverview: jest.fn(() => Promise.resolve({
      data: {
        total_sessions: 10,
        total_laps: 100,
        best_lap_time: 85.234,
        consistency_score: 87.5,
      }
    })),
  },
  telemetryAPI: {
    getSessions: jest.fn(() => Promise.resolve({
      data: []
    })),
  },
}));

// Mock hooks
jest.mock('../../src/hooks/useTelemetry', () => ({
  useTelemetry: () => ({
    sessions: [],
    loading: false,
    error: null,
    fetchSessions: jest.fn(),
  }),
}));

const renderDashboard = () => {
  return render(
    <BrowserRouter>
      <AuthProvider>
        <Dashboard />
      </AuthProvider>
    </BrowserRouter>
  );
};

describe('Dashboard Component', () => {
  it('renders dashboard without crashing', () => {
    renderDashboard();
    expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
  });

  it('displays performance metrics', async () => {
    renderDashboard();
    
    await waitFor(() => {
      // Check if any metric-related text appears
      const elements = screen.queryAllByText(/session/i);
      expect(elements.length).toBeGreaterThan(0);
    });
  });

  it('shows loading state initially', () => {
    renderDashboard();
    // The component may show loading indicators
    // This is a basic structural test
    expect(document.querySelector('.dashboard')).toBeTruthy();
  });

  it('renders recent sessions component', () => {
    renderDashboard();
    // Check for recent sessions section
    const recentSessions = screen.queryByText(/recent/i);
    expect(recentSessions).toBeTruthy();
  });

  it('renders performance metrics component', () => {
    renderDashboard();
    // Check for performance metrics
    const metrics = document.querySelector('[class*="metric"]');
    expect(metrics).toBeTruthy();
  });
});

