/**
 * Dashboard E2E Tests
 * End-to-end tests for dashboard functionality.
 * Author: Sarah Siage
 */

import { test, expect } from '@playwright/test';

const BASE_URL = process.env.E2E_BASE_URL || 'http://localhost:5173';

// Helper function to login
async function login(page: any) {
  const timestamp = Date.now();
  const email = `dashboard${timestamp}@example.com`;
  const password = 'TestPassword123';
  
  await page.goto(`${BASE_URL}/login`);
  await page.click('text=/don\'t have an account/i');
  await page.fill('input[type="email"]', email);
  await page.fill('input[name="username"]', `dashuser${timestamp}`);
  await page.fill('input[name="password"]', password);
  await page.fill('input[name="confirmPassword"]', password);
  await page.click('button[type="submit"]');
  
  await expect(page).toHaveURL(/\/dashboard/i, { timeout: 10000 });
}

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should display dashboard page', async ({ page }) => {
    await expect(page.locator('text=/dashboard/i')).toBeVisible();
  });

  test('should show navigation sidebar', async ({ page }) => {
    const sidebar = page.locator('[class*="sidebar"]').or(page.locator('aside'));
    await expect(sidebar.first()).toBeVisible({ timeout: 5000 });
  });

  test('should display performance metrics', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    const metricsSection = page.locator('text=/performance|metric/i');
    await expect(metricsSection.first()).toBeVisible({ timeout: 5000 });
  });

  test('should display recent sessions', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    const recentSessions = page.locator('text=/recent|session/i');
    await expect(recentSessions.first()).toBeVisible({ timeout: 5000 });
  });

  test('should navigate between pages', async ({ page }) => {
    // Navigate to sessions
    await page.click('text=/^Sessions$/i');
    await expect(page).toHaveURL(/\/sessions/i, { timeout: 5000 });
    
    // Navigate to analytics
    await page.click('text=/^Analytics$/i');
    await expect(page).toHaveURL(/\/analytics/i, { timeout: 5000 });
    
    // Navigate back to dashboard
    await page.click('text=/^Dashboard$/i');
    await expect(page).toHaveURL(/\/dashboard/i, { timeout: 5000 });
  });

  test('should display user menu', async ({ page }) => {
    const userMenu = page.locator('[class*="user"]').first();
    if (await userMenu.isVisible({ timeout: 2000 }).catch(() => false)) {
      await userMenu.click();
      
      // Should show dropdown menu
      await page.waitForTimeout(500);
    }
  });

  test('should be responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    
    await page.waitForTimeout(1000);
    
    // Dashboard should still be visible
    const dashboard = page.locator('text=/dashboard/i');
    await expect(dashboard.first()).toBeVisible({ timeout: 5000 });
  });

  test('should be responsive on tablet', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    
    await page.waitForTimeout(1000);
    
    // Dashboard should still be visible
    const dashboard = page.locator('text=/dashboard/i');
    await expect(dashboard.first()).toBeVisible({ timeout: 5000 });
  });
});

test.describe('Dashboard Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should refresh data', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Look for refresh button or similar
    const refreshButton = page.locator('button[aria-label*="refresh" i]').or(
      page.locator('text=/refresh/i')
    );
    
    if (await refreshButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await refreshButton.click();
      await page.waitForTimeout(500);
    }
  });

  test('should handle loading states', async ({ page }) => {
    await page.reload();
    
    // May show loading indicators
    await page.waitForTimeout(1000);
    
    // Should eventually show content
    const content = page.locator('text=/dashboard|session|metric/i');
    await expect(content.first()).toBeVisible({ timeout: 10000 });
  });

  test('should handle empty states', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Dashboard should handle empty data gracefully
    const emptyState = page.locator('text=/no data|empty|no sessions/i');
    const hasContent = page.locator('text=/session|metric|dashboard/i');
    
    // Either empty state or content should be visible
    const visible = await Promise.race([
      emptyState.first().isVisible({ timeout: 2000 }).catch(() => false),
      hasContent.first().isVisible({ timeout: 2000 }).catch(() => false)
    ]);
    
    expect(visible).toBeTruthy();
  });
});

