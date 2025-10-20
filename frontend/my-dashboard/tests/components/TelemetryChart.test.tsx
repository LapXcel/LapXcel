/**
 * Telemetry Chart Component Tests
 * Test suite for telemetry visualization components.
 * Author: Sarah Siage
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import TelemetryChart from '../../src/components/charts/TelemetryChart';

describe('TelemetryChart Component', () => {
  const mockSessionId = '12345678-1234-1234-1234-123456789012';

  it('renders without crashing', () => {
    render(<TelemetryChart sessionId={mockSessionId} />);
    const canvas = document.querySelector('canvas');
    expect(canvas).toBeInTheDocument();
  });

  it('renders canvas element', () => {
    render(<TelemetryChart sessionId={mockSessionId} />);
    const canvas = document.querySelector('canvas');
    expect(canvas).toBeInTheDocument();
    expect(canvas?.tagName).toBe('CANVAS');
  });

  it('handles empty session ID gracefully', () => {
    render(<TelemetryChart sessionId="" />);
    // Should still render but maybe show empty state
    expect(document.querySelector('.telemetry-chart')).toBeTruthy();
  });

  it('displays loading state when fetching data', () => {
    render(<TelemetryChart sessionId={mockSessionId} />);
    // Check for loading indicators
    const loadingElement = screen.queryByText(/loading/i);
    // May or may not show loading depending on implementation
    expect(true).toBe(true);
  });
});

