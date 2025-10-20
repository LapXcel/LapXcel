"""
User and authentication models for LapXcel Backend API
Defines database schemas for user management, profiles, and preferences.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class User(Base):
    """
    User model for authentication and profile management.
    Stores user credentials, preferences, and racing profile information.
    """
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Authentication
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile information
    first_name = Column(String(100))
    last_name = Column(String(100))
    display_name = Column(String(150))
    avatar_url = Column(String(500))
    bio = Column(Text)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    
    # Racing profile
    preferred_units = Column(String(20), default="metric")  # metric or imperial
    skill_level = Column(String(20), default="beginner")    # beginner, intermediate, advanced, expert
    favorite_tracks = Column(JSON)  # List of track names
    favorite_cars = Column(JSON)    # List of car models
    racing_style = Column(String(50))  # aggressive, smooth, consistent, etc.
    
    # Preferences
    privacy_settings = Column(JSON)
    notification_settings = Column(JSON)
    dashboard_layout = Column(JSON)
    
    # Statistics
    total_sessions = Column(Integer, default=0)
    total_distance_km = Column(Float, default=0.0)
    total_time_hours = Column(Float, default=0.0)
    personal_bests = Column(JSON)  # Track/car combinations with best times
    
    # Subscription and billing
    subscription_tier = Column(String(20), default="free")  # free, premium, pro
    subscription_expires = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    last_activity = Column(DateTime)
    
    # Relationships
    telemetry_sessions = relationship("TelemetrySession", back_populates="user", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.username
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self.is_active and not self.is_deleted
    
    @property
    def is_deleted(self) -> bool:
        """Check if user account is deleted (soft delete)"""
        return not self.is_active


class APIKey(Base):
    """
    API keys for programmatic access to the LapXcel API.
    Allows users to integrate with external tools and scripts.
    """
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Key information
    name = Column(String(100), nullable=False)  # User-defined name for the key
    key_hash = Column(String(255), nullable=False, unique=True)  # Hashed API key
    key_prefix = Column(String(20), nullable=False)  # First few characters for display
    
    # Permissions and restrictions
    permissions = Column(JSON)  # List of allowed operations
    rate_limit = Column(Integer, default=1000)  # Requests per hour
    ip_whitelist = Column(JSON)  # Allowed IP addresses
    
    # Usage tracking
    total_requests = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.name}, user_id={self.user_id})>"


class UserSession(Base):
    """
    User session tracking for security and analytics.
    Stores information about user login sessions and activity.
    """
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session information
    session_token = Column(String(255), unique=True, nullable=False)
    refresh_token = Column(String(255), unique=True)
    
    # Device and location
    device_info = Column(JSON)  # Browser, OS, device type
    ip_address = Column(String(45))  # IPv4 or IPv6
    location = Column(JSON)  # Country, city, timezone
    user_agent = Column(Text)
    
    # Session status
    is_active = Column(Boolean, default=True)
    is_mobile = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow)
    logout_at = Column(DateTime)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"


class UserPreferences(Base):
    """
    Detailed user preferences and settings.
    Stores customization options for the dashboard and application behavior.
    """
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    
    # Dashboard preferences
    default_track = Column(String(255))
    default_car = Column(String(255))
    dashboard_theme = Column(String(20), default="dark")  # dark, light, auto
    chart_colors = Column(JSON)  # Custom color scheme
    
    # Data visualization preferences
    preferred_units = Column(JSON)  # Speed, temperature, distance units
    telemetry_smoothing = Column(Float, default=0.1)  # Data smoothing factor
    chart_refresh_rate = Column(Integer, default=100)  # Milliseconds
    
    # Analysis preferences
    comparison_reference = Column(String(50), default="personal_best")  # personal_best, theoretical, other_drivers
    analysis_depth = Column(String(20), default="standard")  # basic, standard, advanced
    auto_analysis = Column(Boolean, default=True)
    
    # Notification preferences
    email_notifications = Column(Boolean, default=True)
    push_notifications = Column(Boolean, default=True)
    training_complete_alerts = Column(Boolean, default=True)
    personal_best_alerts = Column(Boolean, default=True)
    weekly_summary = Column(Boolean, default=True)
    
    # Privacy preferences
    profile_visibility = Column(String(20), default="friends")  # public, friends, private
    data_sharing = Column(Boolean, default=False)
    anonymous_analytics = Column(Boolean, default=True)
    
    # Performance preferences
    data_retention_days = Column(Integer, default=365)
    auto_cleanup = Column(Boolean, default=True)
    compression_level = Column(String(20), default="standard")  # low, standard, high
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<UserPreferences(id={self.id}, user_id={self.user_id})>"

