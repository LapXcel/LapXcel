"""
Training API endpoints for LapXcel Backend API.
Handles ML model training, hyperparameter optimization, and model management.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload
import structlog

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.training import TrainingJob, ModelVersion, Experiment, ExperimentTrial
from app.models.users import User
from app.schemas.training import (
    TrainingJobCreate, TrainingJobResponse, TrainingJobUpdate,
    ModelVersionResponse, ExperimentCreate, ExperimentResponse,
    HyperparameterOptimizationRequest
)
from app.services.training_service import TrainingService

logger = structlog.get_logger()
router = APIRouter()


@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    job_data: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> TrainingJobResponse:
    """
    Create a new ML training job.
    The job will be queued and started automatically.
    """
    try:
        logger.info("Creating training job", 
                   user_id=str(current_user.id),
                   algorithm=job_data.algorithm,
                   job_name=job_data.job_name)
        
        # Initialize training service
        training_service = TrainingService()
        
        # Create training job
        job = await training_service.create_training_job(
            user_id=current_user.id,
            job_name=job_data.job_name,
            algorithm=job_data.algorithm,
            config=job_data.config,
            track_name=job_data.track_name,
            car_model=job_data.car_model
        )
        
        # Start training in background
        background_tasks.add_task(training_service.start_training_job, job.id)
        
        return TrainingJobResponse.from_orm(job)
        
    except Exception as e:
        logger.error("Failed to create training job", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create training job")


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def get_training_jobs(
    status: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[TrainingJobResponse]:
    """
    Get user's training jobs with optional filtering.
    """
    try:
        training_service = TrainingService()
        jobs = await training_service.get_training_jobs(
            user_id=current_user.id,
            status=status,
            limit=limit,
            offset=skip
        )
        
        return [TrainingJobResponse.from_orm(job) for job in jobs]
        
    except Exception as e:
        logger.error("Failed to retrieve training jobs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve training jobs")


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> TrainingJobResponse:
    """
    Get detailed information about a specific training job.
    """
    try:
        query = select(TrainingJob).options(
            selectinload(TrainingJob.model_versions),
            selectinload(TrainingJob.experiments)
        ).where(
            and_(
                TrainingJob.id == job_id,
                TrainingJob.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        return TrainingJobResponse.from_orm(job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get training job", job_id=str(job_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve training job")


@router.put("/jobs/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(
    job_id: UUID,
    job_update: TrainingJobUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> TrainingJobResponse:
    """
    Update training job configuration or status.
    Only certain fields can be updated depending on job status.
    """
    try:
        query = select(TrainingJob).where(
            and_(
                TrainingJob.id == job_id,
                TrainingJob.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # Update allowed fields
        update_data = job_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(job, field):
                setattr(job, field, value)
        
        job.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(job)
        
        return TrainingJobResponse.from_orm(job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update training job", job_id=str(job_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update training job")


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a running training job.
    """
    try:
        # Verify job ownership
        query = select(TrainingJob).where(
            and_(
                TrainingJob.id == job_id,
                TrainingJob.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        if job.status not in ["queued", "running"]:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")
        
        # Cancel the job
        training_service = TrainingService()
        success = await training_service.cancel_training_job(job_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to cancel job")
        
        return {"message": "Training job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel training job", job_id=str(job_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cancel training job")


@router.get("/jobs/{job_id}/progress")
async def get_training_progress(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get real-time training progress for a job.
    """
    try:
        # Verify job ownership
        query = select(TrainingJob).where(
            and_(
                TrainingJob.id == job_id,
                TrainingJob.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # Calculate progress
        progress = 0.0
        if job.total_steps and job.current_step:
            progress = (job.current_step / job.total_steps) * 100
        
        return {
            "job_id": str(job.id),
            "status": job.status,
            "progress": progress,
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "best_reward": job.best_reward,
            "estimated_completion": job.estimated_completion,
            "started_at": job.started_at,
            "updated_at": job.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get training progress", job_id=str(job_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get training progress")


@router.get("/models", response_model=List[ModelVersionResponse])
async def get_model_versions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    algorithm: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[ModelVersionResponse]:
    """
    Get user's trained model versions.
    """
    try:
        query = select(ModelVersion).join(TrainingJob).where(
            TrainingJob.user_id == current_user.id
        )
        
        if algorithm:
            query = query.where(ModelVersion.algorithm == algorithm)
        
        query = query.order_by(desc(ModelVersion.created_at)).offset(skip).limit(limit)
        
        result = await db.execute(query)
        models = result.scalars().all()
        
        return [ModelVersionResponse.from_orm(model) for model in models]
        
    except Exception as e:
        logger.error("Failed to retrieve model versions", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve model versions")


@router.get("/models/{model_id}", response_model=ModelVersionResponse)
async def get_model_version(
    model_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ModelVersionResponse:
    """
    Get detailed information about a specific model version.
    """
    try:
        query = select(ModelVersion).join(TrainingJob).where(
            and_(
                ModelVersion.id == model_id,
                TrainingJob.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model version not found")
        
        return ModelVersionResponse.from_orm(model)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model version", model_id=str(model_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve model version")


@router.post("/models/{model_id}/deploy")
async def deploy_model(
    model_id: UUID,
    deployment_environment: str = Query("staging", regex="^(staging|production)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deploy a model version to staging or production environment.
    """
    try:
        query = select(ModelVersion).join(TrainingJob).where(
            and_(
                ModelVersion.id == model_id,
                TrainingJob.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model version not found")
        
        # Update deployment status
        model.deployment_status = deployment_environment
        model.updated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info("Model deployed", 
                   model_id=str(model_id), 
                   environment=deployment_environment,
                   user_id=str(current_user.id))
        
        return {
            "message": f"Model deployed to {deployment_environment}",
            "model_id": str(model_id),
            "deployment_status": deployment_environment
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to deploy model", model_id=str(model_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to deploy model")


@router.post("/hyperopt", response_model=ExperimentResponse)
async def start_hyperparameter_optimization(
    hyperopt_request: HyperparameterOptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ExperimentResponse:
    """
    Start hyperparameter optimization experiment.
    """
    try:
        logger.info("Starting hyperparameter optimization",
                   user_id=str(current_user.id),
                   algorithm=hyperopt_request.algorithm,
                   experiment_name=hyperopt_request.experiment_name)
        
        training_service = TrainingService()
        
        experiment = await training_service.start_hyperparameter_optimization(
            user_id=current_user.id,
            experiment_name=hyperopt_request.experiment_name,
            algorithm=hyperopt_request.algorithm,
            search_space=hyperopt_request.search_space,
            n_trials=hyperopt_request.n_trials,
            track_name=hyperopt_request.track_name,
            car_model=hyperopt_request.car_model
        )
        
        return ExperimentResponse.from_orm(experiment)
        
    except Exception as e:
        logger.error("Failed to start hyperparameter optimization", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start hyperparameter optimization")


@router.get("/experiments", response_model=List[ExperimentResponse])
async def get_experiments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[ExperimentResponse]:
    """
    Get user's hyperparameter optimization experiments.
    """
    try:
        query = select(Experiment).join(TrainingJob).where(
            TrainingJob.user_id == current_user.id
        )
        
        if status:
            query = query.where(Experiment.status == status)
        
        query = query.order_by(desc(Experiment.started_at)).offset(skip).limit(limit)
        
        result = await db.execute(query)
        experiments = result.scalars().all()
        
        return [ExperimentResponse.from_orm(exp) for exp in experiments]
        
    except Exception as e:
        logger.error("Failed to retrieve experiments", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve experiments")


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ExperimentResponse:
    """
    Get detailed information about a specific experiment.
    """
    try:
        query = select(Experiment).join(TrainingJob).where(
            and_(
                Experiment.id == experiment_id,
                TrainingJob.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return ExperimentResponse.from_orm(experiment)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get experiment", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve experiment")


@router.delete("/jobs/{job_id}")
async def delete_training_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a training job and all associated data.
    Only completed, failed, or cancelled jobs can be deleted.
    """
    try:
        query = select(TrainingJob).where(
            and_(
                TrainingJob.id == job_id,
                TrainingJob.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        if job.status in ["queued", "running"]:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete running job. Cancel it first."
            )
        
        # Delete associated files
        if job.model_path and os.path.exists(job.model_path):
            import os
            import shutil
            try:
                if os.path.isfile(job.model_path):
                    os.remove(job.model_path)
                elif os.path.isdir(job.model_path):
                    shutil.rmtree(job.model_path)
            except OSError:
                logger.warning("Failed to delete model files", model_path=job.model_path)
        
        # Delete job record (cascading delete will handle related records)
        await db.delete(job)
        await db.commit()
        
        logger.info("Training job deleted", job_id=str(job_id))
        
        return {"message": "Training job deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete training job", job_id=str(job_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete training job")


@router.get("/algorithms")
async def get_supported_algorithms():
    """
    Get list of supported RL algorithms and their configurations.
    """
    algorithms = {
        "SAC": {
            "name": "Soft Actor-Critic",
            "description": "Off-policy algorithm for continuous action spaces",
            "hyperparameters": {
                "learning_rate": {"type": "float", "default": 3e-4, "range": [1e-5, 1e-2]},
                "buffer_size": {"type": "int", "default": 1000000, "range": [10000, 2000000]},
                "batch_size": {"type": "int", "default": 256, "range": [32, 512]},
                "tau": {"type": "float", "default": 0.005, "range": [0.001, 0.1]},
                "gamma": {"type": "float", "default": 0.99, "range": [0.9, 0.999]}
            }
        },
        "TQC": {
            "name": "Truncated Quantile Critics",
            "description": "Improved version of SAC with better sample efficiency",
            "hyperparameters": {
                "learning_rate": {"type": "float", "default": 3e-4, "range": [1e-5, 1e-2]},
                "buffer_size": {"type": "int", "default": 1000000, "range": [10000, 2000000]},
                "batch_size": {"type": "int", "default": 256, "range": [32, 512]},
                "n_quantiles": {"type": "int", "default": 25, "range": [5, 50]},
                "n_critics": {"type": "int", "default": 2, "range": [2, 10]}
            }
        },
        "PPO": {
            "name": "Proximal Policy Optimization",
            "description": "On-policy algorithm with good stability",
            "hyperparameters": {
                "learning_rate": {"type": "float", "default": 3e-4, "range": [1e-5, 1e-2]},
                "n_steps": {"type": "int", "default": 2048, "range": [64, 8192]},
                "batch_size": {"type": "int", "default": 64, "range": [8, 512]},
                "n_epochs": {"type": "int", "default": 10, "range": [1, 20]},
                "clip_range": {"type": "float", "default": 0.2, "range": [0.1, 0.5]}
            }
        }
    }
    
    return algorithms
