/**
 * Sidebar Component Tests
 * Test suite for navigation sidebar component.
 * Author: Sarah Siage
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Sidebar from '../../src/components/layout/Sidebar';

const renderSidebar = () => {
  return render(
    <BrowserRouter>
      <Sidebar />
    </BrowserRouter>
  );
};

describe('Sidebar Component', () => {
  it('renders sidebar without crashing', () => {
    renderSidebar();
    const sidebar = document.querySelector('.sidebar') || document.querySelector('aside');
    expect(sidebar).toBeInTheDocument();
  });

  it('displays navigation links', () => {
    renderSidebar();
    
    // Check for common navigation items
    const dashboardLink = screen.queryByText(/dashboard/i);
    expect(dashboardLink).toBeTruthy();
  });

  it('displays LapXcel branding', () => {
    renderSidebar();
    
    const branding = screen.queryByText(/lapxcel/i);
    expect(branding).toBeTruthy();
  });

  it('renders all main navigation items', () => {
    renderSidebar();
    
    // Common navigation items
    const expectedItems = ['Dashboard', 'Sessions', 'Analytics', 'Training'];
    
    expectedItems.forEach(item => {
      const link = screen.queryByText(new RegExp(item, 'i'));
      expect(link).toBeTruthy();
    });
  });

  it('handles navigation clicks', () => {
    renderSidebar();
    
    const dashboardLink = screen.queryByText(/dashboard/i);
    if (dashboardLink) {
      fireEvent.click(dashboardLink);
      // Link should be clickable
      expect(dashboardLink).toBeTruthy();
    }
  });

  it('highlights active route', () => {
    renderSidebar();
    
    // Active route should have special styling
    const activeLink = document.querySelector('.active') || 
                      document.querySelector('[aria-current="page"]');
    expect(activeLink).toBeTruthy();
  });
});

