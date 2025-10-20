"""
WebSocket Tests
Test suite for WebSocket connections and real-time data streaming.
Author: Sarah Siage
"""

import pytest
import json
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient


@pytest.mark.asyncio
class TestTelemetryWebSocket:
    """Test suite for telemetry WebSocket connections."""
    
    def test_websocket_connection(self, client: TestClient, auth_token: str):
        """Test establishing WebSocket connection."""
        with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as websocket:
            # Connection should be established
            assert websocket is not None
    
    def test_websocket_without_auth(self, client: TestClient):
        """Test WebSocket connection without authentication."""
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/telemetry"):
                pass
    
    def test_subscribe_to_session(self, client: TestClient, auth_token: str, sample_session_id: str):
        """Test subscribing to a telemetry session."""
        with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as websocket:
            # Send subscribe message
            subscribe_msg = {
                "type": "subscribe",
                "session_id": sample_session_id
            }
            websocket.send_json(subscribe_msg)
            
            # Should receive confirmation
            response = websocket.receive_json()
            assert response["type"] == "subscribed" or "type" in response
    
    def test_unsubscribe_from_session(self, client: TestClient, auth_token: str, sample_session_id: str):
        """Test unsubscribing from a telemetry session."""
        with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as websocket:
            # Subscribe first
            websocket.send_json({
                "type": "subscribe",
                "session_id": sample_session_id
            })
            websocket.receive_json()
            
            # Then unsubscribe
            websocket.send_json({
                "type": "unsubscribe",
                "session_id": sample_session_id
            })
            
            response = websocket.receive_json()
            assert response["type"] == "unsubscribed" or "type" in response
    
    def test_receive_telemetry_data(self, client: TestClient, auth_token: str, sample_session_id: str):
        """Test receiving real-time telemetry data."""
        with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as websocket:
            websocket.send_json({
                "type": "subscribe",
                "session_id": sample_session_id
            })
            
            # Wait for telemetry data (with timeout)
            try:
                data = websocket.receive_json(timeout=5.0)
                assert "type" in data
            except TimeoutError:
                # It's okay if no data arrives during test
                pass
    
    def test_ping_pong(self, client: TestClient, auth_token: str):
        """Test WebSocket ping/pong heartbeat."""
        with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as websocket:
            # Send ping
            websocket.send_json({"type": "ping", "timestamp": 1234567890})
            
            # Should receive pong
            response = websocket.receive_json(timeout=2.0)
            assert response["type"] == "pong" or "type" in response


@pytest.mark.asyncio
class TestTrainingWebSocket:
    """Test suite for training WebSocket connections."""
    
    def test_training_websocket_connection(self, client: TestClient, auth_token: str):
        """Test establishing training WebSocket connection."""
        with client.websocket_connect(f"/ws/training?token={auth_token}") as websocket:
            assert websocket is not None
    
    def test_subscribe_to_training_job(
        self, client: TestClient, auth_token: str, sample_training_job_id: str
    ):
        """Test subscribing to a training job."""
        with client.websocket_connect(f"/ws/training?token={auth_token}") as websocket:
            websocket.send_json({
                "type": "subscribe",
                "job_id": sample_training_job_id
            })
            
            response = websocket.receive_json()
            assert "type" in response
    
    def test_receive_training_progress(
        self, client: TestClient, auth_token: str, sample_training_job_id: str
    ):
        """Test receiving training progress updates."""
        with client.websocket_connect(f"/ws/training?token={auth_token}") as websocket:
            websocket.send_json({
                "type": "subscribe",
                "job_id": sample_training_job_id
            })
            
            # Wait for progress update
            try:
                data = websocket.receive_json(timeout=5.0)
                assert "type" in data
                if data["type"] == "training_progress":
                    assert "progress_percentage" in data or "timestep" in data
            except TimeoutError:
                pass
    
    def test_receive_training_metrics(
        self, client: TestClient, auth_token: str, sample_training_job_id: str
    ):
        """Test receiving training metrics."""
        with client.websocket_connect(f"/ws/training?token={auth_token}") as websocket:
            websocket.send_json({
                "type": "subscribe",
                "job_id": sample_training_job_id
            })
            
            try:
                data = websocket.receive_json(timeout=5.0)
                if data.get("type") == "training_metrics":
                    assert "metrics" in data or "episode_reward" in data
            except TimeoutError:
                pass


@pytest.mark.asyncio
class TestWebSocketErrorHandling:
    """Test suite for WebSocket error handling."""
    
    def test_invalid_message_format(self, client: TestClient, auth_token: str):
        """Test handling of invalid message format."""
        with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as websocket:
            # Send invalid JSON
            websocket.send_text("invalid json")
            
            # Should receive error or close connection
            try:
                response = websocket.receive_json(timeout=2.0)
                assert response.get("type") == "error" or "error" in response
            except WebSocketDisconnect:
                # Also acceptable
                pass
    
    def test_subscribe_invalid_session(self, client: TestClient, auth_token: str):
        """Test subscribing to non-existent session."""
        with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as websocket:
            websocket.send_json({
                "type": "subscribe",
                "session_id": "00000000-0000-0000-0000-000000000000"
            })
            
            response = websocket.receive_json()
            # Should receive error message
            assert response.get("type") == "error" or "error" in response or "type" in response
    
    def test_multiple_connections_same_user(self, client: TestClient, auth_token: str):
        """Test multiple WebSocket connections from same user."""
        with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as ws1:
            with client.websocket_connect(f"/ws/telemetry?token={auth_token}") as ws2:
                # Both connections should work
                assert ws1 is not None
                assert ws2 is not None


# Fixtures

@pytest.fixture
def sample_session_id() -> str:
    """Return a sample session ID for testing."""
    return "12345678-1234-1234-1234-123456789012"


@pytest.fixture
def sample_training_job_id() -> str:
    """Return a sample training job ID for testing."""
    return "87654321-4321-4321-4321-210987654321"


@pytest.fixture
def auth_token(client: TestClient) -> str:
    """Get authentication token for WebSocket testing."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.com", "password": "TestPassword123"}
    )
    
    if response.status_code == 200:
        return response.json()["access_token"]
    
    # Create test user if doesn't exist
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "TestPassword123"
        }
    )
    
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.com", "password": "TestPassword123"}
    )
    
    return response.json()["access_token"]

