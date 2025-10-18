"""
Pytest configuration and fixtures for LapXcel Backend API tests.
Provides database setup, authentication, and common test utilities.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.core.auth import get_current_user, create_user_tokens
from app.models.users import User, UserPreferences
from app.models.telemetry import TelemetrySession, TelemetryData, Lap
from app.models.training import TrainingJob, ModelVersion
from app.core.config import settings


# Test database URL (in-memory SQLite for fast tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False},
    echo=False
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
    autocommit=False,
)


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with TestSessionLocal() as session:
        yield session
    
    # Drop tables after test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client."""
    
    def override_get_db():
        return db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    from app.core.auth import get_password_hash
    
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpassword"),
        first_name="Test",
        last_name="User",
        display_name="Test User",
        is_active=True,
        is_verified=True,
        is_premium=False
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def premium_user(db_session: AsyncSession) -> User:
    """Create a premium test user."""
    from app.core.auth import get_password_hash
    
    user = User(
        email="premium@example.com",
        username="premiumuser",
        hashed_password=get_password_hash("premiumpassword"),
        first_name="Premium",
        last_name="User",
        display_name="Premium User",
        is_active=True,
        is_verified=True,
        is_premium=True
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def authenticated_client(client: AsyncClient, test_user: User) -> AsyncClient:
    """Create an authenticated HTTP client."""
    tokens = create_user_tokens(test_user)
    client.headers.update({"Authorization": f"Bearer {tokens['access_token']}"})
    return client


@pytest_asyncio.fixture
async def premium_client(client: AsyncClient, premium_user: User) -> AsyncClient:
    """Create an authenticated premium HTTP client."""
    tokens = create_user_tokens(premium_user)
    client.headers.update({"Authorization": f"Bearer {tokens['access_token']}"})
    return client


@pytest_asyncio.fixture
async def test_telemetry_session(db_session: AsyncSession, test_user: User) -> TelemetrySession:
    """Create a test telemetry session."""
    session = TelemetrySession(
        user_id=test_user.id,
        session_name="Test Session",
        track_name="Silverstone",
        car_model="Ferrari SF70H",
        weather_conditions="Clear",
        track_temperature=25.0,
        air_temperature=22.0,
        total_laps=5,
        best_lap_time=87.432,
        is_complete=True,
        is_valid=True
    )
    
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    
    return session


@pytest_asyncio.fixture
async def test_telemetry_data(db_session: AsyncSession, test_telemetry_session: TelemetrySession) -> List[TelemetryData]:
    """Create test telemetry data points."""
    from datetime import datetime, timedelta
    
    data_points = []
    base_time = datetime.utcnow()
    
    for i in range(100):  # 100 data points
        data_point = TelemetryData(
            session_id=test_telemetry_session.id,
            timestamp=base_time + timedelta(seconds=i * 0.1),
            session_time=i * 0.1,
            lap_time=(i % 20) * 0.1,  # Reset every 20 points (simulating laps)
            world_position_x=100.0 + i,
            world_position_y=200.0 + i * 0.5,
            world_position_z=0.0,
            velocity_x=50.0 + i * 0.1,
            velocity_y=0.0,
            velocity_z=0.0,
            speed_kmh=180.0 + i * 0.5,
            throttle_input=0.8 + (i % 10) * 0.02,
            brake_input=0.0 if i % 20 < 15 else 0.5,
            steering_input=(i % 40 - 20) * 0.01,
            track_progress=(i % 20) / 20.0,
            gear=min(6, max(1, 3 + (i % 10) // 3)),
            rpm=6000 + i * 10,
            is_on_track=True
        )
        data_points.append(data_point)
    
    db_session.add_all(data_points)
    await db_session.commit()
    
    return data_points


@pytest_asyncio.fixture
async def test_training_job(db_session: AsyncSession, test_user: User) -> TrainingJob:
    """Create a test training job."""
    job = TrainingJob(
        user_id=test_user.id,
        job_name="Test Training Job",
        job_type="rl_training",
        algorithm="SAC",
        config={
            "total_steps": 100000,
            "hyperparameters": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "batch_size": 256
            },
            "environment": {
                "frame_skip": 2,
                "reward_type": "lap_time"
            }
        },
        hyperparameters={
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256
        },
        track_name="Silverstone",
        car_model="Ferrari SF70H",
        total_steps=100000,
        status="completed",
        final_reward=150.5,
        best_reward=150.5
    )
    
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    
    return job


@pytest.fixture
def mock_redis():
    """Create a mock Redis service."""
    mock = AsyncMock()
    mock.set = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.delete = AsyncMock(return_value=1)
    mock.exists = AsyncMock(return_value=False)
    mock.ping = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_websocket_manager():
    """Create a mock WebSocket manager."""
    mock = MagicMock()
    mock.broadcast_to_session = AsyncMock()
    mock.broadcast_json_to_session = AsyncMock()
    mock.broadcast_telemetry_data = AsyncMock()
    mock.get_connection_count = MagicMock(return_value=0)
    return mock


@pytest.fixture
def sample_telemetry_data():
    """Sample telemetry data for testing."""
    return {
        "timestamp": "2023-12-01T10:00:00Z",
        "session_time": 45.5,
        "lap_time": 12.3,
        "world_position_x": 100.0,
        "world_position_y": 200.0,
        "world_position_z": 0.0,
        "velocity_x": 50.0,
        "velocity_y": 0.0,
        "velocity_z": 0.0,
        "speed_kmh": 180.0,
        "throttle_input": 0.8,
        "brake_input": 0.0,
        "steering_input": 0.1,
        "track_progress": 0.25,
        "gear": 4,
        "rpm": 6500,
        "is_on_track": True
    }


@pytest.fixture
def sample_training_config():
    """Sample training configuration for testing."""
    return {
        "job_name": "Test Training",
        "algorithm": "SAC",
        "track_name": "Silverstone",
        "car_model": "Ferrari SF70H",
        "config": {
            "total_steps": 100000,
            "hyperparameters": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99
            },
            "environment": {
                "frame_skip": 2,
                "reward_type": "lap_time",
                "max_episode_steps": 1000
            }
        }
    }


@pytest.fixture
def sample_hyperopt_config():
    """Sample hyperparameter optimization configuration."""
    return {
        "experiment_name": "Test Hyperopt",
        "algorithm": "SAC",
        "track_name": "Silverstone",
        "car_model": "Ferrari SF70H",
        "n_trials": 10,
        "search_space": {
            "learning_rate": {
                "type": "loguniform",
                "low": 1e-5,
                "high": 1e-2
            },
            "buffer_size": {
                "type": "choice",
                "choices": [50000, 100000, 200000, 500000]
            },
            "batch_size": {
                "type": "choice", 
                "choices": [64, 128, 256, 512]
            }
        }
    }


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_response_success(response):
        """Assert that a response was successful."""
        assert 200 <= response.status_code < 300, f"Response failed: {response.status_code} - {response.text}"
    
    @staticmethod
    def assert_response_error(response, expected_status=400):
        """Assert that a response returned an error."""
        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}"
    
    @staticmethod
    def assert_valid_uuid(uuid_string):
        """Assert that a string is a valid UUID."""
        from uuid import UUID
        try:
            UUID(uuid_string)
        except ValueError:
            pytest.fail(f"'{uuid_string}' is not a valid UUID")
    
    @staticmethod
    def assert_valid_timestamp(timestamp_string):
        """Assert that a string is a valid ISO timestamp."""
        from datetime import datetime
        try:
            datetime.fromisoformat(timestamp_string.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"'{timestamp_string}' is not a valid ISO timestamp")


@pytest.fixture
def test_utils():
    """Provide test utility functions."""
    return TestUtils
