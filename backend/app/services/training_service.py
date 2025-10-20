"""
ML training service for managing reinforcement learning training jobs.
Handles job scheduling, model training, hyperparameter optimization, and model deployment.
"""

import asyncio
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func, update
from sqlalchemy.orm import selectinload
import structlog
import optuna
import mlflow
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.database import get_db_context
from app.models.training import TrainingJob, ModelVersion, Experiment, ExperimentTrial, ModelBenchmark
from app.models.users import User
from app.services.websocket_manager import WebSocketManager
from app.services.redis_service import RedisService

logger = structlog.get_logger()


class TrainingService:
    """Service for managing ML training operations"""
    
    def __init__(self, websocket_manager: Optional[WebSocketManager] = None, redis: Optional[RedisService] = None):
        self.websocket_manager = websocket_manager
        self.redis = redis
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_TRAINING_JOBS)
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Ensure model storage directory exists
        Path(settings.MODEL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.TRAINING_DATA_PATH).mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
    async def create_training_job(
        self,
        user_id: UUID,
        job_name: str,
        algorithm: str,
        config: Dict[str, Any],
        track_name: str,
        car_model: str
    ) -> TrainingJob:
        """Create a new training job"""
        try:
            async with get_db_context() as db:
                job = TrainingJob(
                    user_id=user_id,
                    job_name=job_name,
                    job_type="rl_training",
                    algorithm=algorithm,
                    config=config,
                    hyperparameters=config.get("hyperparameters", {}),
                    environment_config=config.get("environment", {}),
                    track_name=track_name,
                    car_model=car_model,
                    total_steps=config.get("total_steps", 1000000),
                    status="queued",
                    priority=config.get("priority", 5)
                )
                
                db.add(job)
                await db.commit()
                await db.refresh(job)
                
                logger.info("Training job created", 
                           job_id=str(job.id), 
                           algorithm=algorithm,
                           user_id=str(user_id))
                
                return job
                
        except Exception as e:
            logger.error("Failed to create training job", error=str(e))
            raise
    
    async def start_training_job(self, job_id: UUID) -> bool:
        """Start a training job"""
        try:
            async with get_db_context() as db:
                # Get job details
                query = select(TrainingJob).where(TrainingJob.id == job_id)
                result = await db.execute(query)
                job = result.scalar_one_or_none()
                
                if not job:
                    logger.error("Training job not found", job_id=str(job_id))
                    return False
                
                if job.status != "queued":
                    logger.warning("Job not in queued state", 
                                 job_id=str(job_id), 
                                 status=job.status)
                    return False
                
                # Update job status
                job.status = "running"
                job.started_at = datetime.utcnow()
                await db.commit()
                
                # Start training task
                task = asyncio.create_task(self._run_training_job(job))
                self.active_jobs[str(job_id)] = task
                
                logger.info("Training job started", job_id=str(job_id))
                return True
                
        except Exception as e:
            logger.error("Failed to start training job", job_id=str(job_id), error=str(e))
            return False
    
    async def _run_training_job(self, job: TrainingJob):
        """Execute the actual training job"""
        try:
            # Set up MLflow experiment
            experiment_name = f"{job.job_name}_{job.algorithm}_{job.track_name}"
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"job_{job.id}"):
                # Log hyperparameters
                mlflow.log_params(job.hyperparameters)
                mlflow.log_param("algorithm", job.algorithm)
                mlflow.log_param("track_name", job.track_name)
                mlflow.log_param("car_model", job.car_model)
                
                # Create training environment
                training_env = await self._create_training_environment(job)
                
                # Initialize model based on algorithm
                model = await self._initialize_model(job, training_env)
                
                # Set up callbacks for progress tracking
                callback = TrainingCallback(
                    job_id=job.id,
                    websocket_manager=self.websocket_manager,
                    redis=self.redis
                )
                
                # Run training
                model.learn(
                    total_timesteps=job.total_steps,
                    callback=callback,
                    log_interval=100
                )
                
                # Save trained model
                model_path = await self._save_model(job, model)
                
                # Create model version
                model_version = await self._create_model_version(job, model_path, callback.best_reward)
                
                # Update job status
                await self._complete_training_job(job.id, model_version.id, callback.best_reward)
                
                # Log final metrics
                mlflow.log_metric("final_reward", callback.best_reward)
                mlflow.log_metric("training_steps", job.total_steps)
                mlflow.log_artifact(model_path)
                
                logger.info("Training job completed successfully", job_id=str(job.id))
                
        except Exception as e:
            logger.error("Training job failed", job_id=str(job.id), error=str(e))
            await self._fail_training_job(job.id, str(e))
        finally:
            # Clean up
            if str(job.id) in self.active_jobs:
                del self.active_jobs[str(job.id)]
    
    async def _create_training_environment(self, job: TrainingJob):
        """Create training environment based on job configuration"""
        # Import the existing VAE environment
        from src.envs.vae_env import ACVAEEnv
        from src.vae.vae import VAE
        from src.config import Z_SIZE, FRAME_SKIP, MIN_THROTTLE, MAX_THROTTLE
        
        # Load VAE model
        vae = VAE(z_size=Z_SIZE)
        vae_model_path = os.path.join(settings.MODEL_STORAGE_PATH, "vae_trained_params.pkl")
        if os.path.exists(vae_model_path):
            vae.load_model(vae_model_path)
        
        # Create environment
        env = ACVAEEnv(
            frame_skip=FRAME_SKIP,
            vae=vae,
            min_throttle=MIN_THROTTLE,
            max_throttle=MAX_THROTTLE,
            verbose=False
        )
        
        return env
    
    async def _initialize_model(self, job: TrainingJob, env):
        """Initialize RL model based on algorithm"""
        algorithm = job.algorithm.lower()
        hyperparams = job.hyperparameters
        
        if algorithm == "sac":
            from stable_baselines3 import SAC
            model = SAC(
                policy="MlpPolicy",
                env=env,
                device="cuda" if self._has_gpu() else "cpu",
                verbose=1,
                tensorboard_log=os.path.join(settings.MODEL_STORAGE_PATH, "tensorboard"),
                **hyperparams
            )
        elif algorithm == "tqc":
            from sbx import TQC
            model = TQC(
                policy="MlpPolicy",
                env=env,
                device="cuda" if self._has_gpu() else "cpu",
                verbose=1,
                tensorboard_log=os.path.join(settings.MODEL_STORAGE_PATH, "tensorboard"),
                **hyperparams
            )
        elif algorithm == "ppo":
            from stable_baselines3 import PPO
            model = PPO(
                policy="MlpPolicy",
                env=env,
                device="cuda" if self._has_gpu() else "cpu",
                verbose=1,
                tensorboard_log=os.path.join(settings.MODEL_STORAGE_PATH, "tensorboard"),
                **hyperparams
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return model
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def _save_model(self, job: TrainingJob, model) -> str:
        """Save trained model to disk"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{job.algorithm}_{job.track_name}_{timestamp}"
        model_path = os.path.join(settings.MODEL_STORAGE_PATH, model_filename)
        
        # Save model
        model.save(model_path)
        
        # Save additional metadata
        metadata = {
            "job_id": str(job.id),
            "algorithm": job.algorithm,
            "track_name": job.track_name,
            "car_model": job.car_model,
            "hyperparameters": job.hyperparameters,
            "training_steps": job.total_steps,
            "created_at": datetime.utcnow().isoformat()
        }
        
        metadata_path = f"{model_path}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    async def _create_model_version(self, job: TrainingJob, model_path: str, best_reward: float) -> ModelVersion:
        """Create a model version record"""
        async with get_db_context() as db:
            # Get next version number
            version_query = select(func.max(ModelVersion.version_number)).where(
                ModelVersion.training_job_id == job.id
            )
            result = await db.execute(version_query)
            max_version = result.scalar() or 0
            next_version = max_version + 1
            
            # Create model version
            model_version = ModelVersion(
                training_job_id=job.id,
                version=f"v{next_version}.0.0",
                version_number=next_version,
                is_latest=True,
                model_name=f"{job.algorithm}_{job.track_name}",
                algorithm=job.algorithm,
                evaluation_score=best_reward,
                training_steps=job.total_steps,
                model_file_path=model_path,
                deployment_status="not_deployed"
            )
            
            # Mark previous versions as not latest
            await db.execute(
                update(ModelVersion)
                .where(
                    and_(
                        ModelVersion.training_job_id == job.id,
                        ModelVersion.id != model_version.id
                    )
                )
                .values(is_latest=False)
            )
            
            db.add(model_version)
            await db.commit()
            await db.refresh(model_version)
            
            return model_version
    
    async def _complete_training_job(self, job_id: UUID, model_version_id: UUID, final_reward: float):
        """Mark training job as completed"""
        async with get_db_context() as db:
            query = select(TrainingJob).where(TrainingJob.id == job_id)
            result = await db.execute(query)
            job = result.scalar_one()
            
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.final_reward = final_reward
            job.best_reward = final_reward
            
            await db.commit()
    
    async def _fail_training_job(self, job_id: UUID, error_message: str):
        """Mark training job as failed"""
        async with get_db_context() as db:
            query = select(TrainingJob).where(TrainingJob.id == job_id)
            result = await db.execute(query)
            job = result.scalar_one_or_none()
            
            if job:
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                job.error_message = error_message
                
                await db.commit()
    
    async def cancel_training_job(self, job_id: UUID) -> bool:
        """Cancel a running training job"""
        try:
            # Cancel the task if it's running
            job_id_str = str(job_id)
            if job_id_str in self.active_jobs:
                task = self.active_jobs[job_id_str]
                task.cancel()
                del self.active_jobs[job_id_str]
            
            # Update job status in database
            async with get_db_context() as db:
                query = select(TrainingJob).where(TrainingJob.id == job_id)
                result = await db.execute(query)
                job = result.scalar_one_or_none()
                
                if job:
                    job.status = "cancelled"
                    job.completed_at = datetime.utcnow()
                    await db.commit()
            
            logger.info("Training job cancelled", job_id=str(job_id))
            return True
            
        except Exception as e:
            logger.error("Failed to cancel training job", job_id=str(job_id), error=str(e))
            return False
    
    async def get_training_jobs(
        self,
        user_id: Optional[UUID] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TrainingJob]:
        """Get training jobs with optional filtering"""
        async with get_db_context() as db:
            query = select(TrainingJob).options(
                selectinload(TrainingJob.model_versions)
            )
            
            if user_id:
                query = query.where(TrainingJob.user_id == user_id)
            
            if status:
                query = query.where(TrainingJob.status == status)
            
            query = query.order_by(desc(TrainingJob.created_at)).offset(offset).limit(limit)
            
            result = await db.execute(query)
            return result.scalars().all()
    
    async def start_hyperparameter_optimization(
        self,
        user_id: UUID,
        experiment_name: str,
        algorithm: str,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        track_name: str = "default",
        car_model: str = "default"
    ) -> Experiment:
        """Start hyperparameter optimization experiment"""
        try:
            async with get_db_context() as db:
                # Create base training job
                base_config = {
                    "total_steps": 100000,  # Shorter for hyperopt
                    "hyperparameters": {},
                    "environment": {}
                }
                
                training_job = await self.create_training_job(
                    user_id=user_id,
                    job_name=f"hyperopt_{experiment_name}",
                    algorithm=algorithm,
                    config=base_config,
                    track_name=track_name,
                    car_model=car_model
                )
                
                # Create experiment
                experiment = Experiment(
                    training_job_id=training_job.id,
                    experiment_name=experiment_name,
                    experiment_type="hyperopt",
                    search_space=search_space,
                    optimization_metric="reward",
                    optimization_direction="maximize",
                    total_trials=n_trials,
                    status="running"
                )
                
                db.add(experiment)
                await db.commit()
                await db.refresh(experiment)
                
                # Start optimization task
                task = asyncio.create_task(
                    self._run_hyperparameter_optimization(experiment)
                )
                self.active_jobs[f"hyperopt_{experiment.id}"] = task
                
                logger.info("Hyperparameter optimization started", 
                           experiment_id=str(experiment.id))
                
                return experiment
                
        except Exception as e:
            logger.error("Failed to start hyperparameter optimization", error=str(e))
            raise
    
    async def _run_hyperparameter_optimization(self, experiment: Experiment):
        """Run hyperparameter optimization using Optuna"""
        try:
            study = optuna.create_study(
                direction=experiment.optimization_direction,
                study_name=experiment.experiment_name
            )
            
            def objective(trial):
                # This would be implemented to run a training trial
                # with the suggested hyperparameters
                return self._run_optimization_trial(trial, experiment)
            
            # Run optimization
            study.optimize(objective, n_trials=experiment.total_trials)
            
            # Update experiment with results
            await self._complete_hyperparameter_optimization(experiment.id, study)
            
        except Exception as e:
            logger.error("Hyperparameter optimization failed", 
                        experiment_id=str(experiment.id), 
                        error=str(e))
            await self._fail_hyperparameter_optimization(experiment.id, str(e))
    
    def _run_optimization_trial(self, trial, experiment: Experiment) -> float:
        """Run a single optimization trial"""
        # This is a simplified placeholder
        # In a real implementation, this would:
        # 1. Suggest hyperparameters based on the search space
        # 2. Create a training job with those hyperparameters
        # 3. Run training for a shorter duration
        # 4. Return the evaluation metric
        
        # For now, return a random reward
        import random
        return random.uniform(-100, 100)
    
    async def _complete_hyperparameter_optimization(self, experiment_id: UUID, study):
        """Complete hyperparameter optimization"""
        async with get_db_context() as db:
            query = select(Experiment).where(Experiment.id == experiment_id)
            result = await db.execute(query)
            experiment = result.scalar_one()
            
            experiment.status = "completed"
            experiment.completed_at = datetime.utcnow()
            experiment.best_value = study.best_value
            experiment.best_parameters = study.best_params
            experiment.completed_trials = len(study.trials)
            
            await db.commit()
    
    async def _fail_hyperparameter_optimization(self, experiment_id: UUID, error_message: str):
        """Mark hyperparameter optimization as failed"""
        async with get_db_context() as db:
            query = select(Experiment).where(Experiment.id == experiment_id)
            result = await db.execute(query)
            experiment = result.scalar_one_or_none()
            
            if experiment:
                experiment.status = "failed"
                experiment.completed_at = datetime.utcnow()
                
                await db.commit()


class TrainingCallback:
    """Callback for tracking training progress and broadcasting updates"""
    
    def __init__(self, job_id: UUID, websocket_manager: Optional[WebSocketManager] = None, redis: Optional[RedisService] = None):
        self.job_id = job_id
        self.websocket_manager = websocket_manager
        self.redis = redis
        self.best_reward = float('-inf')
        self.episode_count = 0
        self.step_count = 0
        
    def on_step(self) -> bool:
        """Called on each training step"""
        self.step_count += 1
        
        # Broadcast progress every 1000 steps
        if self.step_count % 1000 == 0:
            asyncio.create_task(self._broadcast_progress())
        
        return True
    
    def on_episode_end(self, episode_reward: float):
        """Called at the end of each episode"""
        self.episode_count += 1
        
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            asyncio.create_task(self._broadcast_best_reward())
    
    async def _broadcast_progress(self):
        """Broadcast training progress"""
        if self.websocket_manager:
            progress_data = {
                "job_id": str(self.job_id),
                "step_count": self.step_count,
                "episode_count": self.episode_count,
                "best_reward": self.best_reward,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.websocket_manager.broadcast_training_update(
                str(self.job_id),
                progress_data
            )
    
    async def _broadcast_best_reward(self):
        """Broadcast new best reward"""
        if self.websocket_manager:
            reward_data = {
                "job_id": str(self.job_id),
                "best_reward": self.best_reward,
                "episode_count": self.episode_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.websocket_manager.broadcast_training_update(
                str(self.job_id),
                reward_data
            )
