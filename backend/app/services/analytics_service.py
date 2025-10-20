"""
Analytics Service for LapXcel Backend API
Provides comprehensive analytics, performance calculations, and comparison logic.
Author: Colby Todd
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
import structlog

from app.models.telemetry import TelemetrySession, TelemetryData, Lap, SessionAnalytics
from app.models.analytics import PerformanceMetric, ComparisonResult, AnalyticsCache
from app.models.users import User

logger = structlog.get_logger()


class AnalyticsService:
    """
    Comprehensive analytics service for telemetry data analysis.
    Provides performance metrics, comparisons, and AI-driven insights.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_performance_overview(
        self,
        user_id: UUID,
        track_name: Optional[str] = None,
        car_model: Optional[str] = None,
        period: str = "weekly"
    ) -> Dict[str, Any]:
        """
        Get comprehensive performance overview with aggregated metrics.
        """
        # Calculate date range based on period
        end_date = datetime.utcnow()
        if period == "daily":
            start_date = end_date - timedelta(days=1)
        elif period == "weekly":
            start_date = end_date - timedelta(days=7)
        elif period == "monthly":
            start_date = end_date - timedelta(days=30)
        else:  # all_time
            start_date = datetime(2020, 1, 1)
        
        # Build query
        query = select(TelemetrySession).where(
            and_(
                TelemetrySession.user_id == user_id,
                TelemetrySession.session_start >= start_date,
                TelemetrySession.is_valid == True
            )
        )
        
        if track_name:
            query = query.where(TelemetrySession.track_name == track_name)
        if car_model:
            query = query.where(TelemetrySession.car_model == car_model)
        
        result = await self.db.execute(query)
        sessions = result.scalars().all()
        
        if not sessions:
            return {
                "total_sessions": 0,
                "total_laps": 0,
                "message": "No sessions found for the specified criteria"
            }
        
        # Aggregate statistics
        total_sessions = len(sessions)
        total_laps = sum(s.total_laps for s in sessions)
        best_lap_time = min((s.best_lap_time for s in sessions if s.best_lap_time), default=None)
        
        # Get all laps for more detailed analysis
        lap_times = []
        for session in sessions:
            laps_query = select(Lap).where(
                and_(
                    Lap.session_id == session.id,
                    Lap.is_valid == True
                )
            )
            laps_result = await self.db.execute(laps_query)
            session_laps = laps_result.scalars().all()
            lap_times.extend([lap.lap_time for lap in session_laps])
        
        # Calculate statistics
        avg_lap_time = np.mean(lap_times) if lap_times else None
        lap_time_std = np.std(lap_times) if lap_times else None
        consistency_score = self._calculate_consistency_score(lap_times) if lap_times else 0
        
        # Calculate improvement trend
        improvement_trend = self._calculate_improvement_trend(sessions)
        
        return {
            "total_sessions": total_sessions,
            "total_laps": total_laps,
            "best_lap_time": best_lap_time,
            "average_lap_time": float(avg_lap_time) if avg_lap_time else None,
            "lap_time_std_dev": float(lap_time_std) if lap_time_std else None,
            "consistency_score": consistency_score,
            "improvement_trend": improvement_trend,
            "period": period,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "tracks_driven": len(set(s.track_name for s in sessions)),
            "cars_used": len(set(s.car_model for s in sessions))
        }
    
    async def calculate_consistency(
        self,
        user_id: UUID,
        session_id: Optional[UUID] = None,
        track_name: Optional[str] = None,
        car_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate detailed consistency metrics for lap times.
        """
        # Build query for sessions
        query = select(TelemetrySession).where(
            and_(
                TelemetrySession.user_id == user_id,
                TelemetrySession.is_valid == True
            )
        )
        
        if session_id:
            query = query.where(TelemetrySession.id == session_id)
        if track_name:
            query = query.where(TelemetrySession.track_name == track_name)
        if car_model:
            query = query.where(TelemetrySession.car_model == car_model)
        
        result = await self.db.execute(query)
        sessions = result.scalars().all()
        
        all_lap_times = []
        sector_1_times = []
        sector_2_times = []
        sector_3_times = []
        
        for session in sessions:
            laps_query = select(Lap).where(
                and_(
                    Lap.session_id == session.id,
                    Lap.is_valid == True
                )
            )
            laps_result = await self.db.execute(laps_query)
            laps = laps_result.scalars().all()
            
            for lap in laps:
                all_lap_times.append(lap.lap_time)
                if lap.sector_1_time:
                    sector_1_times.append(lap.sector_1_time)
                if lap.sector_2_time:
                    sector_2_times.append(lap.sector_2_time)
                if lap.sector_3_time:
                    sector_3_times.append(lap.sector_3_time)
        
        if not all_lap_times:
            return {
                "consistency_score": 0,
                "message": "No valid laps found"
            }
        
        # Calculate consistency metrics
        lap_time_mean = np.mean(all_lap_times)
        lap_time_std = np.std(all_lap_times)
        coefficient_of_variation = (lap_time_std / lap_time_mean) * 100 if lap_time_mean > 0 else 0
        
        # Consistency score: lower CV = higher consistency
        consistency_score = max(0, 100 - (coefficient_of_variation * 10))
        
        return {
            "consistency_score": round(consistency_score, 2),
            "total_laps_analyzed": len(all_lap_times),
            "lap_time_statistics": {
                "mean": round(lap_time_mean, 3),
                "std_deviation": round(lap_time_std, 3),
                "coefficient_of_variation": round(coefficient_of_variation, 2),
                "best": round(min(all_lap_times), 3),
                "worst": round(max(all_lap_times), 3),
                "range": round(max(all_lap_times) - min(all_lap_times), 3)
            },
            "sector_consistency": {
                "sector_1": self._calculate_sector_consistency(sector_1_times),
                "sector_2": self._calculate_sector_consistency(sector_2_times),
                "sector_3": self._calculate_sector_consistency(sector_3_times)
            }
        }
    
    async def analyze_efficiency(self, session_id: UUID) -> Dict[str, Any]:
        """
        Analyze driving efficiency for a session.
        """
        # Get telemetry data
        query = select(TelemetryData).where(
            TelemetryData.session_id == session_id
        ).order_by(TelemetryData.session_time)
        
        result = await self.db.execute(query)
        telemetry_points = result.scalars().all()
        
        if not telemetry_points:
            return {
                "efficiency_score": 0,
                "message": "No telemetry data found"
            }
        
        # Calculate efficiency metrics
        throttle_efficiency = self._calculate_throttle_efficiency(telemetry_points)
        brake_efficiency = self._calculate_brake_efficiency(telemetry_points)
        cornering_efficiency = self._calculate_cornering_efficiency(telemetry_points)
        fuel_efficiency = self._calculate_fuel_efficiency(telemetry_points)
        
        # Overall efficiency score (weighted average)
        overall_efficiency = (
            throttle_efficiency * 0.3 +
            brake_efficiency * 0.3 +
            cornering_efficiency * 0.3 +
            fuel_efficiency * 0.1
        )
        
        return {
            "efficiency_score": round(overall_efficiency, 2),
            "throttle_efficiency": round(throttle_efficiency, 2),
            "brake_efficiency": round(brake_efficiency, 2),
            "cornering_efficiency": round(cornering_efficiency, 2),
            "fuel_efficiency": round(fuel_efficiency, 2),
            "data_points_analyzed": len(telemetry_points),
            "recommendations": self._generate_efficiency_recommendations(
                throttle_efficiency,
                brake_efficiency,
                cornering_efficiency,
                fuel_efficiency
            )
        }
    
    async def compare_sessions(
        self,
        primary_session_id: UUID,
        secondary_session_id: UUID,
        comparison_type: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Compare two telemetry sessions in detail.
        """
        # Get both sessions
        primary = await self.db.get(TelemetrySession, primary_session_id)
        secondary = await self.db.get(TelemetrySession, secondary_session_id)
        
        if not primary or not secondary:
            raise ValueError("One or both sessions not found")
        
        # Get laps for both sessions
        primary_laps = await self._get_session_laps(primary_session_id)
        secondary_laps = await self._get_session_laps(secondary_session_id)
        
        # Calculate deltas
        lap_time_delta = (primary.best_lap_time or 0) - (secondary.best_lap_time or 0)
        
        # Get session analytics
        primary_analytics = await self._get_or_calculate_analytics(primary_session_id)
        secondary_analytics = await self._get_or_calculate_analytics(secondary_session_id)
        
        consistency_delta = (
            (primary_analytics.get("consistency_score", 0) or 0) -
            (secondary_analytics.get("consistency_score", 0) or 0)
        )
        
        efficiency_delta = (
            (primary_analytics.get("efficiency_score", 0) or 0) -
            (secondary_analytics.get("efficiency_score", 0) or 0)
        )
        
        # Calculate overall score (-100 to +100)
        overall_score = self._calculate_comparison_score(
            lap_time_delta,
            consistency_delta,
            efficiency_delta
        )
        
        return {
            "overall_score": round(overall_score, 2),
            "lap_time_delta": round(lap_time_delta, 3),
            "consistency_delta": round(consistency_delta, 2),
            "efficiency_delta": round(efficiency_delta, 2),
            "primary_session": {
                "id": str(primary.id),
                "best_lap_time": primary.best_lap_time,
                "total_laps": primary.total_laps,
                "track": primary.track_name,
                "car": primary.car_model
            },
            "secondary_session": {
                "id": str(secondary.id),
                "best_lap_time": secondary.best_lap_time,
                "total_laps": secondary.total_laps,
                "track": secondary.track_name,
                "car": secondary.car_model
            },
            "significance": self._determine_significance(overall_score),
            "recommendations": self._generate_comparison_recommendations(
                lap_time_delta,
                consistency_delta,
                efficiency_delta
            )
        }
    
    async def generate_improvement_suggestions(
        self,
        session_id: UUID
    ) -> Dict[str, Any]:
        """
        Generate AI-powered improvement suggestions.
        """
        analytics = await self._get_or_calculate_analytics(session_id)
        
        suggestions = []
        priority_areas = []
        
        # Analyze consistency
        consistency_score = analytics.get("consistency_score", 0)
        if consistency_score < 70:
            suggestions.append({
                "area": "consistency",
                "priority": "high",
                "message": "Focus on maintaining consistent lap times",
                "recommendation": "Practice hitting the same braking points and turn-in points every lap"
            })
            priority_areas.append("consistency")
        
        # Analyze efficiency
        efficiency_score = analytics.get("efficiency_score", 0)
        if efficiency_score < 70:
            suggestions.append({
                "area": "efficiency",
                "priority": "high",
                "message": "Improve driving efficiency",
                "recommendation": "Work on smoother throttle and brake inputs"
            })
            priority_areas.append("efficiency")
        
        return {
            "suggestions": suggestions,
            "priority_areas": priority_areas,
            "overall_performance_score": round((consistency_score + efficiency_score) / 2, 2)
        }
    
    async def calculate_trends(
        self,
        user_id: UUID,
        metric_type: str,
        track_name: Optional[str] = None,
        car_model: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate performance trends over time.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        query = select(TelemetrySession).where(
            and_(
                TelemetrySession.user_id == user_id,
                TelemetrySession.session_start >= start_date,
                TelemetrySession.is_valid == True
            )
        ).order_by(TelemetrySession.session_start)
        
        if track_name:
            query = query.where(TelemetrySession.track_name == track_name)
        if car_model:
            query = query.where(TelemetrySession.car_model == car_model)
        
        result = await self.db.execute(query)
        sessions = result.scalars().all()
        
        trend_data = []
        for session in sessions:
            value = None
            if metric_type == "lap_time":
                value = session.best_lap_time
            # Add more metric types as needed
            
            if value:
                trend_data.append({
                    "date": session.session_start.isoformat(),
                    "value": value,
                    "session_id": str(session.id)
                })
        
        return {
            "metric_type": metric_type,
            "data_points": trend_data,
            "trend": self._calculate_linear_trend(trend_data) if trend_data else "insufficient_data"
        }
    
    # Private helper methods
    
    def _calculate_consistency_score(self, lap_times: List[float]) -> float:
        """Calculate consistency score from lap times."""
        if len(lap_times) < 2:
            return 0
        
        mean = np.mean(lap_times)
        std = np.std(lap_times)
        cv = (std / mean) * 100 if mean > 0 else 0
        
        return max(0, 100 - (cv * 10))
    
    def _calculate_improvement_trend(self, sessions: List[TelemetrySession]) -> str:
        """Calculate overall improvement trend."""
        if len(sessions) < 2:
            return "insufficient_data"
        
        sorted_sessions = sorted(sessions, key=lambda s: s.session_start)
        lap_times = [s.best_lap_time for s in sorted_sessions if s.best_lap_time]
        
        if len(lap_times) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(lap_times))
        slope = np.polyfit(x, lap_times, 1)[0]
        
        if slope < -0.1:
            return "improving"
        elif slope > 0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_sector_consistency(self, sector_times: List[float]) -> Dict[str, Any]:
        """Calculate consistency for a specific sector."""
        if not sector_times:
            return {"score": 0, "message": "No data"}
        
        mean = np.mean(sector_times)
        std = np.std(sector_times)
        
        return {
            "score": round(max(0, 100 - (std / mean * 1000)), 2),
            "mean": round(mean, 3),
            "std_deviation": round(std, 3)
        }
    
    def _calculate_throttle_efficiency(self, telemetry: List[TelemetryData]) -> float:
        """Calculate throttle application efficiency."""
        smooth_applications = sum(1 for t in telemetry if 0.8 <= t.throttle_input <= 1.0)
        return (smooth_applications / len(telemetry)) * 100 if telemetry else 0
    
    def _calculate_brake_efficiency(self, telemetry: List[TelemetryData]) -> float:
        """Calculate braking efficiency."""
        effective_braking = sum(1 for t in telemetry if t.brake_input > 0.7)
        return (effective_braking / len(telemetry)) * 100 if telemetry else 0
    
    def _calculate_cornering_efficiency(self, telemetry: List[TelemetryData]) -> float:
        """Calculate cornering efficiency."""
        smooth_corners = sum(1 for t in telemetry if abs(t.steering_input) < 0.5)
        return (smooth_corners / len(telemetry)) * 100 if telemetry else 0
    
    def _calculate_fuel_efficiency(self, telemetry: List[TelemetryData]) -> float:
        """Calculate fuel efficiency."""
        # Simple fuel efficiency based on fuel consumption rate
        avg_consumption = np.mean([t.fuel_consumption_rate for t in telemetry if t.fuel_consumption_rate])
        return max(0, 100 - (avg_consumption * 10)) if avg_consumption else 80
    
    def _generate_efficiency_recommendations(
        self,
        throttle: float,
        brake: float,
        cornering: float,
        fuel: float
    ) -> List[str]:
        """Generate efficiency improvement recommendations."""
        recommendations = []
        
        if throttle < 70:
            recommendations.append("Apply throttle more smoothly and progressively")
        if brake < 70:
            recommendations.append("Brake earlier and more smoothly")
        if cornering < 70:
            recommendations.append("Use smoother steering inputs through corners")
        if fuel < 70:
            recommendations.append("Focus on fuel management and efficiency")
        
        return recommendations
    
    async def _get_session_laps(self, session_id: UUID) -> List[Lap]:
        """Get all laps for a session."""
        query = select(Lap).where(
            and_(
                Lap.session_id == session_id,
                Lap.is_valid == True
            )
        )
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def _get_or_calculate_analytics(self, session_id: UUID) -> Dict[str, Any]:
        """Get or calculate session analytics."""
        # Try to get cached analytics
        query = select(SessionAnalytics).where(SessionAnalytics.session_id == session_id)
        result = await self.db.execute(query)
        analytics = result.scalars().first()
        
        if analytics:
            return {
                "consistency_score": analytics.consistency_score,
                "efficiency_score": analytics.efficiency_score,
                "overall_score": analytics.overall_score
            }
        
        # Calculate if not cached
        consistency = await self.calculate_consistency(
            user_id=UUID('00000000-0000-0000-0000-000000000000'),  # Will be filtered by session
            session_id=session_id
        )
        efficiency = await self.analyze_efficiency(session_id)
        
        return {
            "consistency_score": consistency.get("consistency_score", 0),
            "efficiency_score": efficiency.get("efficiency_score", 0),
            "overall_score": (consistency.get("consistency_score", 0) + efficiency.get("efficiency_score", 0)) / 2
        }
    
    def _calculate_comparison_score(
        self,
        lap_delta: float,
        consistency_delta: float,
        efficiency_delta: float
    ) -> float:
        """Calculate overall comparison score."""
        # Negative lap_delta is better (faster)
        lap_score = -lap_delta * 10
        
        # Combine all scores with weights
        return (lap_score * 0.6 + consistency_delta * 0.2 + efficiency_delta * 0.2)
    
    def _determine_significance(self, score: float) -> str:
        """Determine statistical significance of comparison."""
        abs_score = abs(score)
        if abs_score < 5:
            return "low"
        elif abs_score < 15:
            return "medium"
        elif abs_score < 30:
            return "high"
        else:
            return "very_high"
    
    def _generate_comparison_recommendations(
        self,
        lap_delta: float,
        consistency_delta: float,
        efficiency_delta: float
    ) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []
        
        if lap_delta > 0.5:
            recommendations.append("Focus on reducing lap times through better racing line")
        if consistency_delta < -10:
            recommendations.append("Work on maintaining more consistent lap times")
        if efficiency_delta < -10:
            recommendations.append("Improve driving efficiency with smoother inputs")
        
        return recommendations
    
    def _calculate_linear_trend(self, data_points: List[Dict]) -> str:
        """Calculate linear trend from data points."""
        if len(data_points) < 2:
            return "insufficient_data"
        
        values = [p["value"] for p in data_points]
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope < -0.05:
            return "improving"
        elif slope > 0.05:
            return "declining"
        else:
            return "stable"

