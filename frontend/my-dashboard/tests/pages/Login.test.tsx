/**
 * Login Page Tests
 * Test suite for authentication and login functionality.
 * Author: Sarah Siage
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Login from '../../src/pages/Login';
import { AuthProvider } from '../../src/hooks/useAuth';

const mockNavigate = jest.fn();

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

jest.mock('../../src/services/api', () => ({
  authAPI: {
    login: jest.fn((email, password) => {
      if (email === 'test@example.com' && password === 'TestPassword123') {
        return Promise.resolve({
          access_token: 'mock-token',
          token_type: 'bearer',
        });
      }
      return Promise.reject(new Error('Invalid credentials'));
    }),
    register: jest.fn(() => Promise.resolve({
      access_token: 'mock-token',
      token_type: 'bearer',
    })),
    getCurrentUser: jest.fn(() => Promise.resolve({
      id: '123',
      email: 'test@example.com',
      username: 'testuser',
    })),
  },
}));

const renderLogin = () => {
  return render(
    <BrowserRouter>
      <AuthProvider>
        <Login />
      </AuthProvider>
    </BrowserRouter>
  );
};

describe('Login Page', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
  });

  it('renders login form', () => {
    renderLogin();
    
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('displays LapXcel branding', () => {
    renderLogin();
    
    expect(screen.getByText(/lapxcel/i)).toBeInTheDocument();
  });

  it('allows input in email field', () => {
    renderLogin();
    
    const emailInput = screen.getByLabelText(/email/i) as HTMLInputElement;
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
    
    expect(emailInput.value).toBe('test@example.com');
  });

  it('allows input in password field', () => {
    renderLogin();
    
    const passwordInput = screen.getByLabelText(/password/i) as HTMLInputElement;
    fireEvent.change(passwordInput, { target: { value: 'password123' } });
    
    expect(passwordInput.value).toBe('password123');
  });

  it('toggles between login and registration modes', () => {
    renderLogin();
    
    const toggleButton = screen.getByText(/don't have an account/i);
    fireEvent.click(toggleButton);
    
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
  });

  it('handles successful login', async () => {
    renderLogin();
    
    const emailInput = screen.getByLabelText(/email/i);
    const passwordInput = screen.getByLabelText(/password/i);
    const submitButton = screen.getByRole('button', { name: /sign in/i });
    
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
    fireEvent.change(passwordInput, { target: { value: 'TestPassword123' } });
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    }, { timeout: 3000 });
  });

  it('displays error message on failed login', async () => {
    renderLogin();
    
    const emailInput = screen.getByLabelText(/email/i);
    const passwordInput = screen.getByLabelText(/password/i);
    const submitButton = screen.getByRole('button', { name: /sign in/i });
    
    fireEvent.change(emailInput, { target: { value: 'wrong@example.com' } });
    fireEvent.change(passwordInput, { target: { value: 'wrongpassword' } });
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      const errorMessage = screen.queryByText(/invalid|failed|error/i);
      expect(errorMessage).toBeTruthy();
    }, { timeout: 3000 });
  });
});

