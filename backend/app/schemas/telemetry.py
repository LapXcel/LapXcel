"""
Pydantic schemas for telemetry data validation and serialization.
Defines request/response models for the telemetry API endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, validator


class TelemetrySessionBase(BaseModel):
    """Base schema for telemetry sessions"""
    session_name: str = Field(..., min_length=1, max_length=255)
    track_name: str = Field(..., min_length=1, max_length=255)
    car_model: str = Field(..., min_length=1, max_length=255)
    weather_conditions: Optional[str] = Field(None, max_length=100)
    track_temperature: Optional[float] = Field(None, ge=-50, le=100)
    air_temperature: Optional[float] = Field(None, ge=-50, le=100)
    game_version: Optional[str] = Field(None, max_length=50)
    setup_data: Optional[Dict[str, Any]] = None
    notes: Optional[str] = Field(None, max_length=1000)


class TelemetrySessionCreate(TelemetrySessionBase):
    """Schema for creating a new telemetry session"""
    pass


class TelemetrySessionUpdate(BaseModel):
    """Schema for updating telemetry session"""
    session_name: Optional[str] = Field(None, min_length=1, max_length=255)
    is_complete: Optional[bool] = None
    is_valid: Optional[bool] = None
    session_end: Optional[datetime] = None
    total_duration: Optional[float] = Field(None, ge=0)
    total_laps: Optional[int] = Field(None, ge=0)
    best_lap_time: Optional[float] = Field(None, gt=0)
    notes: Optional[str] = Field(None, max_length=1000)


class TelemetrySessionResponse(TelemetrySessionBase):
    """Schema for telemetry session responses"""
    id: UUID
    user_id: UUID
    session_start: datetime
    session_end: Optional[datetime]
    total_duration: Optional[float]
    total_laps: int
    best_lap_time: Optional[float]
    is_complete: bool
    is_valid: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TelemetryDataPoint(BaseModel):
    """Individual telemetry data point"""
    timestamp: datetime
    session_time: float = Field(..., ge=0)
    lap_time: Optional[float] = Field(None, ge=0)
    
    # Position and movement
    world_position_x: float
    world_position_y: float
    world_position_z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    speed_kmh: float = Field(..., ge=0, le=500)
    
    # Vehicle dynamics
    acceleration_x: Optional[float] = None
    acceleration_y: Optional[float] = None
    acceleration_z: Optional[float] = None
    heading: Optional[float] = Field(None, ge=0, le=6.28)  # 0 to 2π radians
    pitch: Optional[float] = None
    roll: Optional[float] = None
    
    # Driver inputs
    throttle_input: float = Field(..., ge=0, le=1)
    brake_input: float = Field(..., ge=0, le=1)
    steering_input: float = Field(..., ge=-1, le=1)
    clutch_input: Optional[float] = Field(None, ge=0, le=1)
    gear: Optional[int] = Field(None, ge=-1, le=10)
    
    # Engine and drivetrain
    rpm: Optional[int] = Field(None, ge=0, le=20000)
    engine_temperature: Optional[float] = Field(None, ge=0, le=200)
    fuel_level: Optional[float] = Field(None, ge=0, le=100)
    fuel_consumption_rate: Optional[float] = Field(None, ge=0)
    
    # Tires
    tire_temperature_fl: Optional[float] = Field(None, ge=0, le=200)
    tire_temperature_fr: Optional[float] = Field(None, ge=0, le=200)
    tire_temperature_rl: Optional[float] = Field(None, ge=0, le=200)
    tire_temperature_rr: Optional[float] = Field(None, ge=0, le=200)
    tire_pressure_fl: Optional[float] = Field(None, ge=0, le=5)
    tire_pressure_fr: Optional[float] = Field(None, ge=0, le=5)
    tire_pressure_rl: Optional[float] = Field(None, ge=0, le=5)
    tire_pressure_rr: Optional[float] = Field(None, ge=0, le=5)
    tire_wear_fl: Optional[float] = Field(None, ge=0, le=100)
    tire_wear_fr: Optional[float] = Field(None, ge=0, le=100)
    tire_wear_rl: Optional[float] = Field(None, ge=0, le=100)
    tire_wear_rr: Optional[float] = Field(None, ge=0, le=100)
    
    # Track information
    track_progress: float = Field(..., ge=0, le=1)
    track_grip: Optional[float] = Field(None, ge=0, le=1)
    is_on_track: bool = True
    
    # Assists and systems
    abs_active: bool = False
    traction_control_active: bool = False
    drs_available: bool = False
    drs_active: bool = False
    ers_deployment: Optional[float] = Field(None, ge=0, le=100)
    ers_recovery: Optional[float] = Field(None, ge=0, le=100)


class TelemetryDataCreate(TelemetryDataPoint):
    """Schema for creating telemetry data"""
    lap_id: Optional[UUID] = None


class TelemetryDataBatch(BaseModel):
    """Schema for batch telemetry data ingestion"""
    data_points: List[TelemetryDataPoint] = Field(..., min_items=1, max_items=10000)
    
    @validator('data_points')
    def validate_chronological_order(cls, v):
        """Ensure data points are in chronological order"""
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].session_time < v[i-1].session_time:
                    raise ValueError("Data points must be in chronological order")
        return v


class TelemetryDataResponse(TelemetryDataPoint):
    """Schema for telemetry data responses"""
    id: UUID
    session_id: UUID
    lap_id: Optional[UUID]
    
    class Config:
        from_attributes = True


class LapBase(BaseModel):
    """Base schema for lap data"""
    lap_number: int = Field(..., ge=1)
    lap_time: float = Field(..., gt=0)
    sector_1_time: Optional[float] = Field(None, gt=0)
    sector_2_time: Optional[float] = Field(None, gt=0)
    sector_3_time: Optional[float] = Field(None, gt=0)
    is_valid: bool = True
    invalidation_reason: Optional[str] = Field(None, max_length=255)
    max_speed: Optional[float] = Field(None, ge=0)
    avg_speed: Optional[float] = Field(None, ge=0)
    fuel_consumed: Optional[float] = Field(None, ge=0)


class LapCreate(LapBase):
    """Schema for creating lap data"""
    lap_start_time: datetime
    lap_end_time: datetime
    
    @validator('lap_end_time')
    def validate_lap_times(cls, v, values):
        """Ensure lap end time is after start time"""
        if 'lap_start_time' in values and v <= values['lap_start_time']:
            raise ValueError("Lap end time must be after start time")
        return v


class LapResponse(LapBase):
    """Schema for lap data responses"""
    id: UUID
    session_id: UUID
    is_personal_best: bool
    lap_start_time: datetime
    lap_end_time: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


class SessionAnalyticsResponse(BaseModel):
    """Schema for session analytics responses"""
    id: UUID
    session_id: UUID
    
    # Performance scores
    consistency_score: Optional[float] = Field(None, ge=0, le=100)
    efficiency_score: Optional[float] = Field(None, ge=0, le=100)
    smoothness_score: Optional[float] = Field(None, ge=0, le=100)
    overall_score: Optional[float] = Field(None, ge=0, le=100)
    
    # Time analysis
    theoretical_best_time: Optional[float]
    time_loss_braking: Optional[float]
    time_loss_cornering: Optional[float]
    time_loss_acceleration: Optional[float]
    
    # Driving style metrics
    avg_throttle_application: Optional[float]
    avg_brake_pressure: Optional[float]
    steering_smoothness: Optional[float]
    
    # Sector comparisons (vs personal best)
    sector_1_delta: Optional[float]
    sector_2_delta: Optional[float]
    sector_3_delta: Optional[float]
    
    # Improvement suggestions
    suggestions: Optional[List[Dict[str, Any]]]
    priority_areas: Optional[List[str]]
    
    # Ranking information
    percentile_rank: Optional[float] = Field(None, ge=0, le=100)
    similar_drivers: Optional[List[Dict[str, Any]]]
    
    calculated_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TelemetryComparisonRequest(BaseModel):
    """Schema for telemetry comparison requests"""
    primary_session_id: UUID
    secondary_session_id: Optional[UUID] = None
    comparison_type: str = Field(..., regex="^(session_vs_session|session_vs_best|session_vs_theoretical)$")
    metrics: List[str] = Field(default=["lap_time", "sector_times", "consistency"])
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate comparison metrics"""
        valid_metrics = [
            "lap_time", "sector_times", "consistency", "efficiency",
            "braking", "cornering", "acceleration", "tire_usage"
        ]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}")
        return v


class TelemetryComparisonResponse(BaseModel):
    """Schema for telemetry comparison responses"""
    comparison_id: UUID
    comparison_type: str
    primary_session_id: UUID
    secondary_session_id: Optional[UUID]
    
    # Overall comparison
    overall_score: float = Field(..., ge=-100, le=100)  # Negative means primary is worse
    confidence_level: float = Field(..., ge=0, le=1)
    
    # Detailed deltas
    lap_time_delta: Optional[float]
    sector_deltas: Optional[Dict[str, float]]
    consistency_delta: Optional[float]
    
    # Category breakdowns
    braking_comparison: Optional[Dict[str, Any]]
    cornering_comparison: Optional[Dict[str, Any]]
    acceleration_comparison: Optional[Dict[str, Any]]
    
    # Improvement suggestions
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[Dict[str, Any]]
    
    created_at: datetime
    
    class Config:
        from_attributes = True
