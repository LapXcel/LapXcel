"""
Analytics API Tests
Comprehensive test suite for analytics endpoints and performance metrics.
Author: Sarah Siage
"""

import pytest
from uuid import uuid4
from datetime import datetime, timedelta
from fastapi import status
from httpx import AsyncClient


@pytest.mark.asyncio
class TestAnalyticsAPI:
    """Test suite for analytics operations."""
    
    async def test_get_performance_overview(self, async_client: AsyncClient, auth_headers: dict):
        """Test retrieving performance overview."""
        response = await async_client.get(
            "/api/v1/analytics/performance/overview",
            params={"period": "weekly"},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "data" in data
    
    async def test_get_consistency_metrics(self, async_client: AsyncClient, auth_headers: dict):
        """Test retrieving consistency metrics."""
        response = await async_client.get(
            "/api/v1/analytics/consistency",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "data" in data
    
    async def test_get_efficiency_analysis(
        self, async_client: AsyncClient, auth_headers: dict, sample_session
    ):
        """Test efficiency analysis for a session."""
        session_id = sample_session["id"]
        
        response = await async_client.get(
            "/api/v1/analytics/efficiency",
            params={"session_id": session_id},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "data" in data
    
    async def test_compare_sessions(
        self, async_client: AsyncClient, auth_headers: dict, sample_session, sample_session_2
    ):
        """Test session comparison."""
        comparison_data = {
            "primary_session_id": sample_session["id"],
            "secondary_session_id": sample_session_2["id"],
            "comparison_type": "detailed"
        }
        
        response = await async_client.post(
            "/api/v1/analytics/compare/sessions",
            json=comparison_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "data" in data
        assert "comparison_id" in data
    
    async def test_get_leaderboard(self, async_client: AsyncClient, auth_headers: dict):
        """Test retrieving global leaderboard."""
        response = await async_client.get(
            "/api/v1/analytics/leaderboard",
            params={
                "track_name": "Monza",
                "car_model": "Ferrari 488 GT3",
                "category": "overall"
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "data" in data
    
    async def test_get_improvement_suggestions(
        self, async_client: AsyncClient, auth_headers: dict, sample_session
    ):
        """Test retrieving improvement suggestions."""
        session_id = sample_session["id"]
        
        response = await async_client.get(
            "/api/v1/analytics/improvement-suggestions",
            params={"session_id": session_id},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "data" in data
    
    async def test_get_performance_trends(self, async_client: AsyncClient, auth_headers: dict):
        """Test retrieving performance trends."""
        response = await async_client.get(
            "/api/v1/analytics/trends",
            params={"metric_type": "lap_time", "days": 30},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "data" in data
    
    async def test_clear_analytics_cache(self, async_client: AsyncClient, auth_headers: dict):
        """Test clearing analytics cache."""
        response = await async_client.delete(
            "/api/v1/analytics/cache/clear",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data


@pytest.mark.asyncio
class TestAnalyticsPermissions:
    """Test suite for analytics access control."""
    
    async def test_cannot_view_other_user_analytics(
        self, async_client: AsyncClient, auth_headers: dict, other_user_session
    ):
        """Test that users cannot view other users' analytics."""
        session_id = other_user_session["id"]
        
        response = await async_client.get(
            "/api/v1/analytics/efficiency",
            params={"session_id": session_id},
            headers=auth_headers
        )
        
        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]
    
    async def test_unauthorized_analytics_access(self, async_client: AsyncClient):
        """Test accessing analytics without authentication."""
        response = await async_client.get("/api/v1/analytics/performance/overview")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
class TestAnalyticsValidation:
    """Test suite for analytics input validation."""
    
    async def test_invalid_period(self, async_client: AsyncClient, auth_headers: dict):
        """Test invalid period parameter."""
        response = await async_client.get(
            "/api/v1/analytics/performance/overview",
            params={"period": "invalid_period"},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_invalid_metric_type(self, async_client: AsyncClient, auth_headers: dict):
        """Test invalid metric type."""
        response = await async_client.get(
            "/api/v1/analytics/trends",
            params={"metric_type": "invalid_metric"},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_invalid_session_id(self, async_client: AsyncClient, auth_headers: dict):
        """Test efficiency analysis with invalid session ID."""
        response = await async_client.get(
            "/api/v1/analytics/efficiency",
            params={"session_id": "invalid-uuid"},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Fixtures

@pytest.fixture
async def sample_session(async_client: AsyncClient, auth_headers: dict):
    """Create a sample telemetry session for testing."""
    session_data = {
        "session_name": "Test Session 1",
        "track_name": "Monza",
        "car_model": "Ferrari 488 GT3",
        "weather_conditions": "Clear",
        "track_temperature": 25.0,
        "air_temperature": 20.0
    }
    
    response = await async_client.post(
        "/api/v1/telemetry/sessions",
        json=session_data,
        headers=auth_headers
    )
    
    return response.json()


@pytest.fixture
async def sample_session_2(async_client: AsyncClient, auth_headers: dict):
    """Create a second sample telemetry session for testing."""
    session_data = {
        "session_name": "Test Session 2",
        "track_name": "Monza",
        "car_model": "Ferrari 488 GT3",
        "weather_conditions": "Clear",
        "track_temperature": 26.0,
        "air_temperature": 21.0
    }
    
    response = await async_client.post(
        "/api/v1/telemetry/sessions",
        json=session_data,
        headers=auth_headers
    )
    
    return response.json()


@pytest.fixture
async def other_user_session(async_client: AsyncClient, other_user_auth_headers: dict):
    """Create a session belonging to another user."""
    session_data = {
        "session_name": "Other User Session",
        "track_name": "Spa",
        "car_model": "Mercedes AMG GT3"
    }
    
    response = await async_client.post(
        "/api/v1/telemetry/sessions",
        json=session_data,
        headers=other_user_auth_headers
    )
    
    return response.json()

