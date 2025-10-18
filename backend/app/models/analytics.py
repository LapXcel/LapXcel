"""
Analytics and caching models for LapXcel Backend API
Defines database schemas for performance analytics, caching, and comparison results.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class AnalyticsCache(Base):
    """
    Caches computed analytics results to improve performance.
    Stores pre-calculated metrics and insights for quick retrieval.
    """
    __tablename__ = "analytics_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Cache identification
    cache_key = Column(String(255), unique=True, nullable=False)
    cache_type = Column(String(50), nullable=False)  # lap_analysis, sector_comparison, etc.
    
    # Data source
    session_id = Column(UUID(as_uuid=True), ForeignKey("telemetry_sessions.id"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Cache metadata
    parameters = Column(JSON)  # Parameters used to generate this cache entry
    data_version = Column(String(50))  # Version of the data/algorithm used
    
    # Cached results
    result_data = Column(JSON, nullable=False)
    summary_stats = Column(JSON)
    
    # Cache management
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_valid = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("TelemetrySession")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('ix_analytics_cache_key', 'cache_key'),
        Index('ix_analytics_cache_type_user', 'cache_type', 'user_id'),
        Index('ix_analytics_cache_expires_at', 'expires_at'),
        Index('ix_analytics_cache_last_accessed', 'last_accessed'),
    )


class PerformanceMetric(Base):
    """
    Stores aggregated performance metrics for users, tracks, and cars.
    Used for leaderboards, comparisons, and statistical analysis.
    """
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric identification
    metric_type = Column(String(50), nullable=False)  # lap_time, sector_time, consistency, etc.
    aggregation_level = Column(String(50), nullable=False)  # user, track, car, global
    
    # Context
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    track_name = Column(String(255))
    car_model = Column(String(255))
    
    # Time period
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly, all_time
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Metric values
    value = Column(Float, nullable=False)
    count = Column(Integer, default=1)  # Number of data points
    min_value = Column(Float)
    max_value = Column(Float)
    avg_value = Column(Float)
    std_deviation = Column(Float)
    
    # Percentiles
    percentile_25 = Column(Float)
    percentile_50 = Column(Float)  # Median
    percentile_75 = Column(Float)
    percentile_90 = Column(Float)
    percentile_95 = Column(Float)
    percentile_99 = Column(Float)
    
    # Ranking information
    rank = Column(Integer)
    total_participants = Column(Integer)
    percentile_rank = Column(Float)
    
    # Additional metadata
    conditions = Column(JSON)  # Weather, track conditions, etc.
    sample_size = Column(Integer)
    confidence_interval = Column(JSON)
    
    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('ix_performance_metrics_type_level', 'metric_type', 'aggregation_level'),
        Index('ix_performance_metrics_user_track_car', 'user_id', 'track_name', 'car_model'),
        Index('ix_performance_metrics_period', 'period_type', 'period_start', 'period_end'),
        Index('ix_performance_metrics_value', 'value'),
        Index('ix_performance_metrics_rank', 'rank'),
    )


class ComparisonResult(Base):
    """
    Stores results of performance comparisons between sessions, users, or models.
    Used for detailed analysis and improvement recommendations.
    """
    __tablename__ = "comparison_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Comparison identification
    comparison_type = Column(String(50), nullable=False)  # session_vs_session, user_vs_user, etc.
    comparison_name = Column(String(255))
    
    # Source data
    primary_session_id = Column(UUID(as_uuid=True), ForeignKey("telemetry_sessions.id"))
    secondary_session_id = Column(UUID(as_uuid=True), ForeignKey("telemetry_sessions.id"))
    primary_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    secondary_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Comparison parameters
    comparison_criteria = Column(JSON)  # What aspects to compare
    weight_factors = Column(JSON)       # Importance weights for different metrics
    
    # Overall results
    overall_score = Column(Float)       # -100 to +100, negative means primary is worse
    confidence_level = Column(Float)    # Statistical confidence in the result
    significance = Column(String(20))   # low, medium, high, very_high
    
    # Detailed comparison results
    lap_time_delta = Column(Float)      # Seconds difference
    sector_deltas = Column(JSON)        # Delta for each sector
    consistency_delta = Column(Float)   # Difference in consistency scores
    efficiency_delta = Column(Float)    # Difference in efficiency scores
    
    # Breakdown by category
    braking_comparison = Column(JSON)   # Braking performance analysis
    cornering_comparison = Column(JSON) # Cornering performance analysis
    acceleration_comparison = Column(JSON) # Acceleration performance analysis
    
    # Improvement opportunities
    primary_strengths = Column(JSON)    # What primary does better
    primary_weaknesses = Column(JSON)   # What primary needs to improve
    specific_recommendations = Column(JSON) # Actionable improvement suggestions
    
    # Statistical analysis
    sample_sizes = Column(JSON)         # Number of laps/data points used
    statistical_tests = Column(JSON)    # Results of statistical significance tests
    outliers_removed = Column(Integer)  # Number of outlier data points excluded
    
    # Visualization data
    chart_data = Column(JSON)           # Pre-computed data for charts
    heatmap_data = Column(JSON)         # Track heatmap comparison data
    
    # Metadata
    analysis_version = Column(String(50)) # Version of comparison algorithm
    processing_time_ms = Column(Integer)  # Time taken to compute comparison
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    primary_session = relationship("TelemetrySession", foreign_keys=[primary_session_id])
    secondary_session = relationship("TelemetrySession", foreign_keys=[secondary_session_id])
    primary_user = relationship("User", foreign_keys=[primary_user_id])
    secondary_user = relationship("User", foreign_keys=[secondary_user_id])
    
    # Indexes
    __table_args__ = (
        Index('ix_comparison_results_type', 'comparison_type'),
        Index('ix_comparison_results_sessions', 'primary_session_id', 'secondary_session_id'),
        Index('ix_comparison_results_users', 'primary_user_id', 'secondary_user_id'),
        Index('ix_comparison_results_overall_score', 'overall_score'),
        Index('ix_comparison_results_created_at', 'created_at'),
    )


class GlobalLeaderboard(Base):
    """
    Global leaderboard entries for track/car combinations.
    Maintains rankings across all users for competitive analysis.
    """
    __tablename__ = "global_leaderboards"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Leaderboard context
    track_name = Column(String(255), nullable=False)
    car_model = Column(String(255), nullable=False)
    category = Column(String(50), default="overall")  # overall, sector_1, sector_2, sector_3, etc.
    
    # User and performance
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("telemetry_sessions.id"), nullable=False)
    lap_id = Column(UUID(as_uuid=True), ForeignKey("laps.id"))
    
    # Performance data
    time_value = Column(Float, nullable=False)  # Lap time or sector time in seconds
    rank = Column(Integer, nullable=False)
    
    # Additional metrics
    consistency_score = Column(Float)
    efficiency_score = Column(Float)
    difficulty_rating = Column(Float)  # Track/car difficulty when this time was set
    
    # Conditions when record was set
    weather_conditions = Column(String(100))
    track_temperature = Column(Float)
    air_temperature = Column(Float)
    track_grip_level = Column(Float)
    
    # Record metadata
    is_current_record = Column(Boolean, default=True)
    previous_record_time = Column(Float)
    improvement_margin = Column(Float)  # Seconds improved over previous record
    
    # Verification and validity
    is_verified = Column(Boolean, default=False)
    verification_method = Column(String(50))  # manual, automated, community
    is_disputed = Column(Boolean, default=False)
    
    # Timestamps
    record_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    session = relationship("TelemetrySession")
    lap = relationship("Lap")
    
    # Indexes
    __table_args__ = (
        Index('ix_global_leaderboards_track_car_category', 'track_name', 'car_model', 'category'),
        Index('ix_global_leaderboards_rank', 'rank'),
        Index('ix_global_leaderboards_time_value', 'time_value'),
        Index('ix_global_leaderboards_user', 'user_id'),
        Index('ix_global_leaderboards_current', 'is_current_record'),
    )
