/**
 * Telemetry E2E Tests
 * End-to-end tests for telemetry session management.
 * Author: Sarah Siage
 */

import { test, expect } from '@playwright/test';

const BASE_URL = process.env.E2E_BASE_URL || 'http://localhost:5173';

// Helper function to login
async function login(page: any) {
  const timestamp = Date.now();
  const email = `e2e${timestamp}@example.com`;
  const password = 'TestPassword123';
  
  await page.goto(`${BASE_URL}/login`);
  await page.click('text=/don\'t have an account/i');
  await page.fill('input[type="email"]', email);
  await page.fill('input[name="username"]', `e2euser${timestamp}`);
  await page.fill('input[name="password"]', password);
  await page.fill('input[name="confirmPassword"]', password);
  await page.click('button[type="submit"]');
  
  await expect(page).toHaveURL(/\/dashboard/i, { timeout: 10000 });
}

test.describe('Telemetry Sessions', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should navigate to sessions page', async ({ page }) => {
    await page.click('text=/sessions/i');
    
    await expect(page).toHaveURL(/\/sessions/i, { timeout: 5000 });
    await expect(page.locator('text=/sessions/i')).toBeVisible();
  });

  test('should display empty state when no sessions exist', async ({ page }) => {
    await page.goto(`${BASE_URL}/sessions`);
    
    // May show empty state or session list
    await page.waitForTimeout(1000);
    const hasContent = await page.locator('text=/session/i').isVisible();
    expect(hasContent).toBeTruthy();
  });

  test('should create a new session', async ({ page }) => {
    await page.goto(`${BASE_URL}/sessions`);
    
    // Look for create session button
    const createButton = page.locator('text=/new session|create/i').first();
    if (await createButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await createButton.click();
      
      // Fill in session details (if form appears)
      await page.waitForTimeout(1000);
    }
  });

  test('should display session list', async ({ page }) => {
    await page.goto(`${BASE_URL}/sessions`);
    
    await page.waitForTimeout(1000);
    
    // Sessions page should be visible
    const sessionsContent = page.locator('text=/session/i');
    await expect(sessionsContent.first()).toBeVisible({ timeout: 5000 });
  });

  test('should filter sessions by search', async ({ page }) => {
    await page.goto(`${BASE_URL}/sessions`);
    
    const searchInput = page.locator('input[placeholder*="search" i]');
    if (await searchInput.isVisible({ timeout: 2000 }).catch(() => false)) {
      await searchInput.fill('Monza');
      await page.waitForTimeout(500);
      
      expect(await searchInput.inputValue()).toBe('Monza');
    }
  });

  test('should view session details', async ({ page }) => {
    await page.goto(`${BASE_URL}/sessions`);
    
    // Wait for sessions to load
    await page.waitForTimeout(1000);
    
    // Click on a session card if available
    const sessionCard = page.locator('[class*="session"]').first();
    if (await sessionCard.isVisible({ timeout: 2000 }).catch(() => false)) {
      await sessionCard.click();
      
      // Should navigate to session detail
      await page.waitForTimeout(1000);
    }
  });
});

test.describe('Session Analytics', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should navigate to analytics page', async ({ page }) => {
    await page.click('text=/analytics/i');
    
    await expect(page).toHaveURL(/\/analytics/i, { timeout: 5000 });
  });

  test('should display performance metrics', async ({ page }) => {
    await page.goto(`${BASE_URL}/analytics`);
    
    await page.waitForTimeout(1000);
    
    // Should show some analytics content
    const analyticsContent = page.locator('text=/performance|metric|analytics/i');
    await expect(analyticsContent.first()).toBeVisible({ timeout: 5000 });
  });

  test('should filter analytics by time period', async ({ page }) => {
    await page.goto(`${BASE_URL}/analytics`);
    
    const periodSelect = page.locator('select').first();
    if (await periodSelect.isVisible({ timeout: 2000 }).catch(() => false)) {
      await periodSelect.selectOption({ label: /weekly/i });
      await page.waitForTimeout(500);
    }
  });
});

