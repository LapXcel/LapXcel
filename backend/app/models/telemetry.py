"""
Telemetry Data Models
Database models for telemetry sessions, data points, laps, and analytics.
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, JSON, Index, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class TelemetrySession(Base):
    """
    Represents a racing session with telemetry data collection.
    Links to a user and contains multiple laps and data points.
    """
    __tablename__ = "telemetry_sessions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session metadata
    session_name = Column(String(255), nullable=False)
    track_name = Column(String(255), nullable=False)
    car_model = Column(String(255), nullable=False)
    
    # Environmental data
    weather_conditions = Column(String(100))
    track_temperature = Column(Float)
    air_temperature = Column(Float)
    
    # Session statistics (updated as data comes in)
    total_laps = Column(Integer, default=0)
    best_lap_time = Column(Float)
    average_lap_time = Column(Float)
    total_distance_km = Column(Float)
    session_duration_seconds = Column(Float)
    
    # Status flags
    is_complete = Column(Boolean, default=False)
    is_valid = Column(Boolean, default=True)
    
    # Additional data
    game_version = Column(String(50))
    setup_data = Column(JSON)  # Car setup information
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    session_start = Column(DateTime)
    session_end = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="telemetry_sessions")
    laps = relationship("Lap", back_populates="session", cascade="all, delete-orphan")
    telemetry_data = relationship("TelemetryData", back_populates="session", cascade="all, delete-orphan")
    analytics = relationship("SessionAnalytics", back_populates="session", uselist=False, cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_telemetry_sessions_user_id', 'user_id'),
        Index('ix_telemetry_sessions_track_car', 'track_name', 'car_model'),
        Index('ix_telemetry_sessions_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TelemetrySession {self.session_name} - {self.track_name}>"


class TelemetryData(Base):
    """
    Individual telemetry data points captured at high frequency (100Hz typical).
    Represents instantaneous vehicle state at a specific moment.
    """
    __tablename__ = "telemetry_data"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    session_id = Column(UUID(as_uuid=True), ForeignKey("telemetry_sessions.id", ondelete="CASCADE"), nullable=False)
    lap_id = Column(UUID(as_uuid=True), ForeignKey("laps.id", ondelete="SET NULL"))
    
    # Timing
    timestamp = Column(DateTime, nullable=False)
    session_time = Column(Float, nullable=False)  # Seconds since session start
    lap_time = Column(Float)  # Seconds since lap start
    
    # Speed and motion
    speed_kmh = Column(Float)
    speed_mph = Column(Float)
    velocity_x = Column(Float)
    velocity_y = Column(Float)
    velocity_z = Column(Float)
    
    # Driver inputs (normalized 0.0 to 1.0)
    throttle_input = Column(Float)
    brake_input = Column(Float)
    clutch_input = Column(Float)
    steering_input = Column(Float)  # -1.0 (full left) to 1.0 (full right)
    
    # Vehicle state
    gear = Column(Integer)
    rpm = Column(Integer)
    engine_temperature = Column(Float)
    
    # Physics and dynamics
    g_force_lateral = Column(Float)
    g_force_longitudinal = Column(Float)
    g_force_vertical = Column(Float)
    slip_angle = Column(Float)
    
    # Position and orientation
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    
    # Rotation (for track mapping)
    yaw = Column(Float)
    pitch = Column(Float)
    roll = Column(Float)
    
    # Tyre data (FL = Front Left, FR = Front Right, RL = Rear Left, RR = Rear Right)
    tyre_temp_fl = Column(Float)
    tyre_temp_fr = Column(Float)
    tyre_temp_rl = Column(Float)
    tyre_temp_rr = Column(Float)
    
    tyre_pressure_fl = Column(Float)
    tyre_pressure_fr = Column(Float)
    tyre_pressure_rl = Column(Float)
    tyre_pressure_rr = Column(Float)
    
    tyre_wear_fl = Column(Float)
    tyre_wear_fr = Column(Float)
    tyre_wear_rl = Column(Float)
    tyre_wear_rr = Column(Float)
    
    # Brake data
    brake_temp_fl = Column(Float)
    brake_temp_fr = Column(Float)
    brake_temp_rl = Column(Float)
    brake_temp_rr = Column(Float)
    
    # Suspension
    suspension_travel_fl = Column(Float)
    suspension_travel_fr = Column(Float)
    suspension_travel_rl = Column(Float)
    suspension_travel_rr = Column(Float)
    
    # Fuel
    fuel_level = Column(Float)
    
    # Relationships
    session = relationship("TelemetrySession", back_populates="telemetry_data")
    lap = relationship("Lap", back_populates="telemetry_data")
    
    # Indexes for performance (critical for time-series queries)
    __table_args__ = (
        Index('ix_telemetry_data_session_id', 'session_id'),
        Index('ix_telemetry_data_session_time', 'session_id', 'session_time'),
        Index('ix_telemetry_data_lap_id', 'lap_id'),
        Index('ix_telemetry_data_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<TelemetryData session={self.session_id} time={self.session_time}>"


class Lap(Base):
    """
    Represents a single lap within a racing session.
    Contains timing, validity, and performance information.
    """
    __tablename__ = "laps"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    session_id = Column(UUID(as_uuid=True), ForeignKey("telemetry_sessions.id", ondelete="CASCADE"), nullable=False)
    
    # Lap identification
    lap_number = Column(Integer, nullable=False)
    
    # Timing (all times in seconds)
    lap_time = Column(Float)  # Total lap time
    sector_1_time = Column(Float)
    sector_2_time = Column(Float)
    sector_3_time = Column(Float)
    
    # Speed statistics
    max_speed_kmh = Column(Float)
    avg_speed_kmh = Column(Float)
    min_speed_kmh = Column(Float)
    top_gear = Column(Integer)
    
    # Performance metrics
    fuel_used = Column(Float)
    avg_throttle = Column(Float)
    avg_brake = Column(Float)
    
    # G-force statistics
    max_g_lateral = Column(Float)
    max_g_longitudinal = Column(Float)
    
    # Tyre statistics
    avg_tyre_temp = Column(Float)
    max_tyre_temp = Column(Float)
    
    # Validity flags
    is_valid = Column(Boolean, default=True)
    is_best_lap = Column(Boolean, default=False)
    is_in_lap = Column(Boolean, default=False)  # In-lap (pit entry)
    is_out_lap = Column(Boolean, default=False)  # Out-lap (pit exit)
    
    # Track limits and incidents
    cuts = Column(Integer, default=0)
    off_track_incidents = Column(Integer, default=0)
    yellow_flags = Column(Integer, default=0)
    
    # Weather conditions during lap
    weather = Column(String(50))
    track_temp_at_lap = Column(Float)
    air_temp_at_lap = Column(Float)
    
    # Timestamps
    lap_start_time = Column(DateTime)
    lap_end_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Additional data
    notes = Column(Text)
    
    # Relationships
    session = relationship("TelemetrySession", back_populates="laps")
    telemetry_data = relationship("TelemetryData", back_populates="lap")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_laps_session_id', 'session_id'),
        Index('ix_laps_session_lap_number', 'session_id', 'lap_number'),
        Index('ix_laps_best_lap', 'session_id', 'is_best_lap'),
    )
    
    def __repr__(self):
        return f"<Lap {self.lap_number} - {self.lap_time}s>"


class SessionAnalytics(Base):
    """
    Computed analytics and insights for a telemetry session.
    Contains performance scores, metrics, and AI-generated recommendations.
    """
    __tablename__ = "session_analytics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys (one-to-one with session)
    session_id = Column(UUID(as_uuid=True), ForeignKey("telemetry_sessions.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    # Performance scores (0-100 scale)
    consistency_score = Column(Float)
    efficiency_score = Column(Float)
    smoothness_score = Column(Float)
    racecraft_score = Column(Float)
    overall_score = Column(Float)
    
    # Statistical measures
    lap_time_std_dev = Column(Float)
    lap_time_variance = Column(Float)
    sector_consistency = Column(JSON)  # Consistency by sector
    improvement_rate = Column(Float)  # Lap time improvement trend
    
    # Driving style analysis
    aggressive_braking_percentage = Column(Float)
    throttle_smoothness = Column(Float)
    steering_smoothness = Column(Float)
    corner_entry_consistency = Column(Float)
    corner_exit_consistency = Column(Float)
    
    # Comparisons and benchmarks
    theoretical_best_lap = Column(Float)  # Best sectors combined
    gap_to_best = Column(Float)  # Gap to theoretical best
    percentile_rank = Column(Float)  # Global ranking
    similar_sessions_avg = Column(Float)  # Average of similar sessions
    
    # Improvement potential
    potential_time_gain = Column(Float)  # Estimated possible improvement
    bottleneck_areas = Column(JSON)  # Areas with most time to gain
    
    # Insights and recommendations (AI-generated)
    strengths = Column(JSON)  # List of strengths
    weaknesses = Column(JSON)  # List of weaknesses  
    recommendations = Column(JSON)  # List of actionable recommendations
    
    # Detailed analytics
    brake_points_analysis = Column(JSON)  # Braking point consistency
    racing_line_analysis = Column(JSON)  # Racing line adherence
    throttle_application_analysis = Column(JSON)  # Throttle technique
    
    # Fuel and tire management
    fuel_consumption_rate = Column(Float)
    tyre_degradation_rate = Column(Float)
    predicted_stint_length = Column(Float)  # Laps before pit stop needed
    
    # Metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    calculation_version = Column(String(50))  # Track analytics algorithm version
    calculation_duration_ms = Column(Integer)  # Time taken to compute
    
    # Relationships
    session = relationship("TelemetrySession", back_populates="analytics")
    
    # Indexes
    __table_args__ = (
        Index('ix_session_analytics_session_id', 'session_id'),
        Index('ix_session_analytics_overall_score', 'overall_score'),
    )
    
    def __repr__(self):
        return f"<SessionAnalytics session={self.session_id} score={self.overall_score}>"


