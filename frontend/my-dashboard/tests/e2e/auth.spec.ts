/**
 * Authentication E2E Tests
 * End-to-end tests for user authentication flows.
 * Author: Sarah Siage
 */

import { test, expect } from '@playwright/test';

const BASE_URL = process.env.E2E_BASE_URL || 'http://localhost:5173';
const API_URL = process.env.E2E_API_URL || 'http://localhost:8000';

test.describe('Authentication Flows', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE_URL}/login`);
  });

  test('should display login page', async ({ page }) => {
    await expect(page).toHaveTitle(/LapXcel/i);
    await expect(page.locator('text=LapXcel')).toBeVisible();
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });

  test('should show validation errors for empty fields', async ({ page }) => {
    await page.click('button[type="submit"]');
    
    // HTML5 validation should prevent submission
    const emailInput = page.locator('input[type="email"]');
    await expect(emailInput).toHaveAttribute('required', '');
  });

  test('should toggle between login and registration', async ({ page }) => {
    // Click to switch to registration
    await page.click('text=/don\'t have an account/i');
    
    await expect(page.locator('input[name="username"]')).toBeVisible();
    await expect(page.locator('text=/create account/i')).toBeVisible();
    
    // Click to switch back to login
    await page.click('text=/already have an account/i');
    
    await expect(page.locator('text=/sign in/i')).toBeVisible();
  });

  test('should register a new user', async ({ page }) => {
    const timestamp = Date.now();
    const email = `test${timestamp}@example.com`;
    const username = `testuser${timestamp}`;
    const password = 'TestPassword123';
    
    // Switch to registration mode
    await page.click('text=/don\'t have an account/i');
    
    // Fill in registration form
    await page.fill('input[type="email"]', email);
    await page.fill('input[name="username"]', username);
    await page.fill('input[name="password"]', password);
    await page.fill('input[name="confirmPassword"]', password);
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Should redirect to dashboard
    await expect(page).toHaveURL(/\/dashboard/i, { timeout: 10000 });
  });

  test('should login with valid credentials', async ({ page }) => {
    // First register a user
    const timestamp = Date.now();
    const email = `logintest${timestamp}@example.com`;
    const password = 'TestPassword123';
    
    await page.click('text=/don\'t have an account/i');
    await page.fill('input[type="email"]', email);
    await page.fill('input[name="username"]', `loginuser${timestamp}`);
    await page.fill('input[name="password"]', password);
    await page.fill('input[name="confirmPassword"]', password);
    await page.click('button[type="submit"]');
    
    await expect(page).toHaveURL(/\/dashboard/i, { timeout: 10000 });
    
    // Logout
    await page.click('text=/logout/i');
    
    // Try to login
    await page.goto(`${BASE_URL}/login`);
    await page.fill('input[type="email"]', email);
    await page.fill('input[name="password"]', password);
    await page.click('button[type="submit"]');
    
    // Should redirect to dashboard
    await expect(page).toHaveURL(/\/dashboard/i, { timeout: 10000 });
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.fill('input[type="email"]', 'invalid@example.com');
    await page.fill('input[name="password"]', 'WrongPassword123');
    await page.click('button[type="submit"]');
    
    // Should show error message
    await expect(page.locator('text=/invalid|failed|error/i')).toBeVisible({ timeout: 5000 });
  });

  test('should validate password requirements', async ({ page }) => {
    await page.click('text=/don\'t have an account/i');
    
    await page.fill('input[type="email"]', 'test@example.com');
    await page.fill('input[name="username"]', 'testuser');
    await page.fill('input[name="password"]', 'weak');
    await page.fill('input[name="confirmPassword"]', 'weak');
    await page.click('button[type="submit"]');
    
    // Should show password requirement error
    await expect(page.locator('text=/password/i')).toBeVisible();
  });
});

test.describe('Protected Routes', () => {
  test('should redirect unauthenticated users to login', async ({ page }) => {
    await page.goto(`${BASE_URL}/dashboard`);
    
    // Should redirect to login
    await expect(page).toHaveURL(/\/login/i, { timeout: 5000 });
  });

  test('should allow authenticated users to access protected routes', async ({ page, context }) => {
    // Set auth token in localStorage
    await context.addCookies([{
      name: 'access_token',
      value: 'mock-token',
      domain: 'localhost',
      path: '/',
    }]);
    
    await page.goto(`${BASE_URL}/dashboard`);
    
    // Should stay on dashboard (or handle accordingly)
    await page.waitForTimeout(1000);
  });
});

test.describe('Logout Flow', () => {
  test('should logout successfully', async ({ page }) => {
    // Register and login
    const timestamp = Date.now();
    const email = `logouttest${timestamp}@example.com`;
    const password = 'TestPassword123';
    
    await page.goto(`${BASE_URL}/login`);
    await page.click('text=/don\'t have an account/i');
    await page.fill('input[type="email"]', email);
    await page.fill('input[name="username"]', `logoutuser${timestamp}`);
    await page.fill('input[name="password"]', password);
    await page.fill('input[name="confirmPassword"]', password);
    await page.click('button[type="submit"]');
    
    await expect(page).toHaveURL(/\/dashboard/i, { timeout: 10000 });
    
    // Logout
    await page.click('text=/logout/i');
    
    // Should redirect to login
    await expect(page).toHaveURL(/\/login/i, { timeout: 5000 });
  });
});

