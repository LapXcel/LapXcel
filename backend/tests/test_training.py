"""
Training API Tests
Comprehensive test suite for training endpoints and ML pipeline functionality.
Author: Sarah Siage
"""

import pytest
from uuid import uuid4
from datetime import datetime
from fastapi import status
from httpx import AsyncClient


@pytest.mark.asyncio
class TestTrainingJobAPI:
    """Test suite for training job operations."""
    
    async def test_create_training_job(self, async_client: AsyncClient, auth_headers: dict):
        """Test creating a new training job."""
        job_data = {
            "algorithm": "SAC",
            "environment": "AssettoCorsaEnv-v1",
            "total_timesteps": 1000000,
            "hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 1000000,
                "batch_size": 256
            },
            "description": "Test training job",
            "tags": ["test", "sac"]
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=job_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["algorithm"] == "SAC"
        assert data["status"] == "pending"
        assert "id" in data
    
    async def test_get_training_jobs(self, async_client: AsyncClient, auth_headers: dict):
        """Test retrieving list of training jobs."""
        response = await async_client.get(
            "/api/v1/training/jobs",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "jobs" in data or isinstance(data, list)
    
    async def test_get_training_job_by_id(self, async_client: AsyncClient, auth_headers: dict, sample_training_job):
        """Test retrieving a specific training job."""
        job_id = sample_training_job["id"]
        
        response = await async_client.get(
            f"/api/v1/training/jobs/{job_id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == job_id
    
    async def test_cancel_training_job(self, async_client: AsyncClient, auth_headers: dict, sample_training_job):
        """Test cancelling a training job."""
        job_id = sample_training_job["id"]
        
        response = await async_client.post(
            f"/api/v1/training/jobs/{job_id}/cancel",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "cancelled"
    
    async def test_create_job_invalid_algorithm(self, async_client: AsyncClient, auth_headers: dict):
        """Test creating job with invalid algorithm."""
        job_data = {
            "algorithm": "INVALID_ALGO",
            "environment": "AssettoCorsaEnv-v1",
            "total_timesteps": 1000000
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=job_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_create_job_unauthorized(self, async_client: AsyncClient):
        """Test creating job without authentication."""
        job_data = {
            "algorithm": "SAC",
            "environment": "AssettoCorsaEnv-v1",
            "total_timesteps": 1000000
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=job_data
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
class TestHyperparameterOptimization:
    """Test suite for hyperparameter optimization."""
    
    async def test_create_optimization_experiment(self, async_client: AsyncClient, auth_headers: dict):
        """Test creating hyperparameter optimization experiment."""
        experiment_data = {
            "algorithm": "SAC",
            "environment": "AssettoCorsaEnv-v1",
            "n_trials": 10,
            "optimization_metric": "episode_reward",
            "optimization_direction": "maximize",
            "parameter_space": {
                "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": True},
                "batch_size": {"type": "categorical", "choices": [128, 256, 512]}
            }
        }
        
        response = await async_client.post(
            "/api/v1/training/hyperopt",
            json=experiment_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["n_trials"] == 10
        assert "id" in data
    
    async def test_get_optimization_results(self, async_client: AsyncClient, auth_headers: dict, sample_optimization):
        """Test retrieving optimization results."""
        experiment_id = sample_optimization["id"]
        
        response = await async_client.get(
            f"/api/v1/training/hyperopt/{experiment_id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "best_parameters" in data or "status" in data


@pytest.mark.asyncio
class TestModelVersions:
    """Test suite for model version management."""
    
    async def test_create_model_version(self, async_client: AsyncClient, auth_headers: dict, sample_training_job):
        """Test creating a new model version."""
        version_data = {
            "training_job_id": sample_training_job["id"],
            "version_name": "v1.0.0",
            "algorithm": "SAC",
            "model_path": "/models/sac_v1.zip",
            "hyperparameters": {"learning_rate": 0.0003},
            "performance_metrics": {"episode_reward": 250.5},
            "description": "Initial model version"
        }
        
        response = await async_client.post(
            "/api/v1/training/models",
            json=version_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["version_name"] == "v1.0.0"
    
    async def test_list_model_versions(self, async_client: AsyncClient, auth_headers: dict):
        """Test listing all model versions."""
        response = await async_client.get(
            "/api/v1/training/models",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    async def test_deploy_model(self, async_client: AsyncClient, auth_headers: dict, sample_model_version):
        """Test deploying a model to production."""
        model_id = sample_model_version["id"]
        
        response = await async_client.post(
            f"/api/v1/training/models/{model_id}/deploy",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["deployment_status"] == "deployed"


@pytest.mark.asyncio
class TestTrainingMetrics:
    """Test suite for training metrics and monitoring."""
    
    async def test_get_job_metrics(self, async_client: AsyncClient, auth_headers: dict, sample_training_job):
        """Test retrieving training metrics for a job."""
        job_id = sample_training_job["id"]
        
        response = await async_client.get(
            f"/api/v1/training/jobs/{job_id}/metrics",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, (dict, list))
    
    async def test_get_training_summary(self, async_client: AsyncClient, auth_headers: dict):
        """Test retrieving training summary statistics."""
        response = await async_client.get(
            "/api/v1/training/summary",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_jobs" in data or isinstance(data, dict)


# Fixtures

@pytest.fixture
async def sample_training_job(async_client: AsyncClient, auth_headers: dict):
    """Create a sample training job for testing."""
    job_data = {
        "algorithm": "SAC",
        "environment": "AssettoCorsaEnv-v1",
        "total_timesteps": 100000,
        "description": "Sample test job"
    }
    
    response = await async_client.post(
        "/api/v1/training/jobs",
        json=job_data,
        headers=auth_headers
    )
    
    return response.json()


@pytest.fixture
async def sample_optimization(async_client: AsyncClient, auth_headers: dict):
    """Create a sample optimization experiment for testing."""
    experiment_data = {
        "algorithm": "SAC",
        "environment": "AssettoCorsaEnv-v1",
        "n_trials": 5,
        "optimization_metric": "episode_reward",
        "parameter_space": {
            "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01}
        }
    }
    
    response = await async_client.post(
        "/api/v1/training/hyperopt",
        json=experiment_data,
        headers=auth_headers
    )
    
    return response.json()


@pytest.fixture
async def sample_model_version(async_client: AsyncClient, auth_headers: dict, sample_training_job):
    """Create a sample model version for testing."""
    version_data = {
        "training_job_id": sample_training_job["id"],
        "version_name": "test-v1",
        "algorithm": "SAC",
        "model_path": "/models/test.zip",
        "hyperparameters": {},
        "performance_metrics": {"reward": 100}
    }
    
    response = await async_client.post(
        "/api/v1/training/models",
        json=version_data,
        headers=auth_headers
    )
    
    return response.json()

