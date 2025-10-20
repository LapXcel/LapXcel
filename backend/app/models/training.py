"""
Training and ML model management models for LapXcel Backend API
Defines database schemas for training jobs, model versions, and experiments.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class TrainingJob(Base):
    """
    Represents ML model training jobs.
    Tracks training progress, hyperparameters, and results.
    """
    __tablename__ = "training_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Job identification
    job_name = Column(String(255), nullable=False)
    job_type = Column(String(50), nullable=False)  # rl_training, hyperopt, evaluation
    algorithm = Column(String(50), nullable=False)  # SAC, TQC, PPO, etc.
    
    # Training configuration
    config = Column(JSON, nullable=False)  # Complete training configuration
    hyperparameters = Column(JSON)  # Model hyperparameters
    environment_config = Column(JSON)  # Environment settings
    
    # Data sources
    training_data_source = Column(String(255))  # Path or identifier for training data
    validation_data_source = Column(String(255))
    track_name = Column(String(255))
    car_model = Column(String(255))
    
    # Job status
    status = Column(String(50), default="queued")  # queued, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)  # 0.0 to 100.0
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer)
    
    # Resource allocation
    gpu_id = Column(String(50))
    memory_limit_gb = Column(Integer)
    cpu_cores = Column(Integer)
    priority = Column(Integer, default=5)  # 1 (highest) to 10 (lowest)
    
    # Timing
    queued_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    estimated_completion = Column(DateTime)
    
    # Results
    final_reward = Column(Float)
    best_reward = Column(Float)
    convergence_step = Column(Integer)
    training_metrics = Column(JSON)  # Loss curves, rewards, etc.
    
    # Output
    model_path = Column(String(500))
    logs_path = Column(String(500))
    tensorboard_path = Column(String(500))
    artifacts_path = Column(String(500))
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="training_jobs")
    model_versions = relationship("ModelVersion", back_populates="training_job", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="training_job", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_training_jobs_user_status', 'user_id', 'status'),
        Index('ix_training_jobs_status_priority', 'status', 'priority'),
        Index('ix_training_jobs_created_at', 'created_at'),
    )


class ModelVersion(Base):
    """
    Represents different versions of trained models.
    Tracks model performance, metadata, and deployment status.
    """
    __tablename__ = "model_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False)
    
    # Version information
    version = Column(String(50), nullable=False)  # e.g., "1.0.0", "2.1.3"
    version_number = Column(Integer, nullable=False)  # Incremental version
    is_latest = Column(Boolean, default=False)
    
    # Model metadata
    model_name = Column(String(255), nullable=False)
    algorithm = Column(String(50), nullable=False)
    model_size_mb = Column(Float)
    parameter_count = Column(Integer)
    
    # Performance metrics
    evaluation_score = Column(Float)
    lap_time_improvement = Column(Float)  # Seconds improved vs baseline
    consistency_score = Column(Float)
    stability_score = Column(Float)
    
    # Training details
    training_steps = Column(Integer)
    training_duration_hours = Column(Float)
    convergence_achieved = Column(Boolean, default=False)
    overfitting_detected = Column(Boolean, default=False)
    
    # Deployment information
    deployment_status = Column(String(50), default="not_deployed")  # not_deployed, staging, production
    deployment_url = Column(String(500))
    health_check_url = Column(String(500))
    
    # File paths
    model_file_path = Column(String(500), nullable=False)
    config_file_path = Column(String(500))
    metadata_file_path = Column(String(500))
    
    # Validation results
    validation_metrics = Column(JSON)
    test_results = Column(JSON)
    benchmark_comparisons = Column(JSON)
    
    # Usage tracking
    inference_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="model_versions")
    
    # Indexes
    __table_args__ = (
        Index('ix_model_versions_training_job', 'training_job_id'),
        Index('ix_model_versions_version', 'version'),
        Index('ix_model_versions_is_latest', 'is_latest'),
        Index('ix_model_versions_deployment_status', 'deployment_status'),
    )


class Experiment(Base):
    """
    Represents ML experiments for hyperparameter optimization.
    Tracks different parameter combinations and their results.
    """
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False)
    
    # Experiment identification
    experiment_name = Column(String(255), nullable=False)
    experiment_type = Column(String(50), default="hyperopt")  # hyperopt, grid_search, random_search
    
    # Hyperparameter configuration
    hyperparameters = Column(JSON, nullable=False)
    search_space = Column(JSON)  # Definition of parameter search space
    
    # Optimization details
    optimization_metric = Column(String(100), default="reward")
    optimization_direction = Column(String(10), default="maximize")  # maximize, minimize
    
    # Trial information
    total_trials = Column(Integer)
    completed_trials = Column(Integer, default=0)
    failed_trials = Column(Integer, default=0)
    pruned_trials = Column(Integer, default=0)
    
    # Best results
    best_trial_id = Column(UUID(as_uuid=True))
    best_value = Column(Float)
    best_parameters = Column(JSON)
    
    # Current status
    status = Column(String(50), default="running")  # running, completed, failed, stopped
    progress = Column(Float, default=0.0)
    
    # Resource usage
    total_compute_hours = Column(Float, default=0.0)
    estimated_remaining_hours = Column(Float)
    
    # Results and analysis
    results_summary = Column(JSON)
    parameter_importance = Column(JSON)
    optimization_history = Column(JSON)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="experiments")
    trials = relationship("ExperimentTrial", back_populates="experiment", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_experiments_training_job', 'training_job_id'),
        Index('ix_experiments_status', 'status'),
        Index('ix_experiments_best_value', 'best_value'),
    )


class ExperimentTrial(Base):
    """
    Individual trials within an experiment.
    Records specific parameter combinations and their outcomes.
    """
    __tablename__ = "experiment_trials"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    
    # Trial identification
    trial_number = Column(Integer, nullable=False)
    
    # Parameters tested
    parameters = Column(JSON, nullable=False)
    
    # Results
    objective_value = Column(Float)
    intermediate_values = Column(JSON)  # Values at different steps
    final_metrics = Column(JSON)
    
    # Trial status
    status = Column(String(50), default="running")  # running, completed, failed, pruned
    pruned_at_step = Column(Integer)
    
    # Execution details
    duration_seconds = Column(Float)
    compute_resources = Column(JSON)
    
    # Error information
    error_message = Column(Text)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="trials")
    
    # Indexes
    __table_args__ = (
        Index('ix_experiment_trials_experiment', 'experiment_id'),
        Index('ix_experiment_trials_trial_number', 'trial_number'),
        Index('ix_experiment_trials_objective_value', 'objective_value'),
    )


class ModelBenchmark(Base):
    """
    Benchmark results for model performance comparison.
    Stores standardized test results across different models and configurations.
    """
    __tablename__ = "model_benchmarks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False)
    
    # Benchmark configuration
    benchmark_name = Column(String(255), nullable=False)
    track_name = Column(String(255), nullable=False)
    car_model = Column(String(255), nullable=False)
    weather_conditions = Column(String(100))
    
    # Performance metrics
    average_lap_time = Column(Float, nullable=False)
    best_lap_time = Column(Float, nullable=False)
    consistency_score = Column(Float)  # Lower is better
    completion_rate = Column(Float)    # Percentage of laps completed without incidents
    
    # Detailed metrics
    sector_times = Column(JSON)        # Average sector times
    speed_metrics = Column(JSON)       # Top speed, average speed, etc.
    efficiency_metrics = Column(JSON)  # Fuel consumption, tire wear
    
    # Comparison data
    human_baseline_time = Column(Float)
    improvement_vs_baseline = Column(Float)  # Seconds faster/slower
    percentile_rank = Column(Float)    # Among all benchmarked models
    
    # Test configuration
    number_of_laps = Column(Integer, default=10)
    random_seed = Column(Integer)
    test_duration_minutes = Column(Float)
    
    # Timestamps
    benchmark_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model_version = relationship("ModelVersion")
    
    # Indexes
    __table_args__ = (
        Index('ix_model_benchmarks_model_version', 'model_version_id'),
        Index('ix_model_benchmarks_track_car', 'track_name', 'car_model'),
        Index('ix_model_benchmarks_lap_time', 'average_lap_time'),
    )
