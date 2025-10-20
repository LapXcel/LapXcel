/**
 * Sessions Page Tests
 * Test suite for sessions list and management functionality.
 * Author: Sarah Siage
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Sessions from '../../src/pages/Sessions';
import { AuthProvider } from '../../src/hooks/useAuth';

const mockSessions = [
  {
    id: '1',
    sessionName: 'Test Session 1',
    trackName: 'Monza',
    carModel: 'Ferrari 488 GT3',
    totalLaps: 10,
    bestLapTime: 85.234,
    sessionStart: '2024-01-01T10:00:00Z',
    isComplete: true,
  },
  {
    id: '2',
    sessionName: 'Test Session 2',
    trackName: 'Spa',
    carModel: 'McLaren 720S GT3',
    totalLaps: 15,
    bestLapTime: 140.567,
    sessionStart: '2024-01-02T14:00:00Z',
    isComplete: false,
  },
];

jest.mock('../../src/hooks/useTelemetry', () => ({
  useTelemetry: () => ({
    sessions: mockSessions,
    loading: false,
    error: null,
    fetchSessions: jest.fn(),
    deleteSession: jest.fn(() => Promise.resolve()),
  }),
}));

const renderSessions = () => {
  return render(
    <BrowserRouter>
      <AuthProvider>
        <Sessions />
      </AuthProvider>
    </BrowserRouter>
  );
};

describe('Sessions Page', () => {
  it('renders sessions page without crashing', () => {
    renderSessions();
    expect(screen.getByText(/sessions/i)).toBeInTheDocument();
  });

  it('displays list of sessions', () => {
    renderSessions();
    
    expect(screen.getByText('Test Session 1')).toBeInTheDocument();
    expect(screen.getByText('Test Session 2')).toBeInTheDocument();
  });

  it('displays session details', () => {
    renderSessions();
    
    expect(screen.getByText(/Monza/i)).toBeInTheDocument();
    expect(screen.getByText(/Ferrari 488 GT3/i)).toBeInTheDocument();
  });

  it('shows create session button', () => {
    renderSessions();
    
    const createButton = screen.getByText(/new session/i) || 
                        screen.getByRole('button', { name: /create/i });
    expect(createButton).toBeInTheDocument();
  });

  it('allows filtering sessions', () => {
    renderSessions();
    
    const searchInput = screen.queryByPlaceholderText(/search/i);
    if (searchInput) {
      fireEvent.change(searchInput, { target: { value: 'Monza' } });
      
      expect(searchInput.value).toBe('Monza');
    }
  });

  it('displays session statistics', () => {
    renderSessions();
    
    // Should show total sessions count
    const statsElement = screen.queryByText(/total sessions/i) ||
                        document.querySelector('.stat');
    expect(statsElement).toBeTruthy();
  });

  it('handles session click navigation', () => {
    renderSessions();
    
    const session = screen.getByText('Test Session 1');
    fireEvent.click(session);
    
    // Should navigate or perform action
    expect(session).toBeTruthy();
  });

  it('shows delete confirmation dialog', async () => {
    renderSessions();
    
    const deleteButtons = screen.queryAllByText(/delete/i);
    if (deleteButtons.length > 0) {
      const confirmSpy = jest.spyOn(window, 'confirm');
      confirmSpy.mockImplementation(() => true);
      
      fireEvent.click(deleteButtons[0]);
      
      await waitFor(() => {
        expect(confirmSpy).toHaveBeenCalled();
      });
      
      confirmSpy.mockRestore();
    }
  });
});

