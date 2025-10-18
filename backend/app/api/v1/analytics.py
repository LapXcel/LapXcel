"""
Analytics API endpoints for LapXcel Backend API
Provides comprehensive analytics, performance metrics, and comparison capabilities.
Author: Colby Todd
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import structlog

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.users import User
from app.models.telemetry import TelemetrySession, Lap
from app.models.analytics import PerformanceMetric, ComparisonResult, GlobalLeaderboard, AnalyticsCache
from app.services.analytics_service import AnalyticsService

logger = structlog.get_logger()
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/performance/overview")
async def get_performance_overview(
    user_id: Optional[UUID] = None,
    track_name: Optional[str] = None,
    car_model: Optional[str] = None,
    period: str = Query("weekly", regex="^(daily|weekly|monthly|all_time)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive performance overview with metrics and trends.
    Returns aggregated statistics across sessions.
    """
    try:
        analytics_service = AnalyticsService(db)
        
        # Use current user if no user_id specified
        target_user_id = user_id or current_user.id
        
        # Authorization: users can only view their own data unless admin
        if target_user_id != current_user.id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view other users' analytics"
            )
        
        overview = await analytics_service.get_performance_overview(
            user_id=target_user_id,
            track_name=track_name,
            car_model=car_model,
            period=period
        )
        
        return {
            "status": "success",
            "data": overview,
            "period": period,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error fetching performance overview", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch performance overview: {str(e)}"
        )


@router.get("/consistency")
async def get_consistency_metrics(
    session_id: Optional[UUID] = None,
    track_name: Optional[str] = None,
    car_model: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate consistency metrics for user's laps.
    Analyzes lap time variance, sector consistency, and performance stability.
    """
    try:
        analytics_service = AnalyticsService(db)
        
        consistency_data = await analytics_service.calculate_consistency(
            user_id=current_user.id,
            session_id=session_id,
            track_name=track_name,
            car_model=car_model
        )
        
        return {
            "status": "success",
            "data": consistency_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error calculating consistency metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate consistency metrics: {str(e)}"
        )


@router.get("/efficiency")
async def get_efficiency_analysis(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze driving efficiency including fuel usage, tire management, and racing line optimization.
    """
    try:
        analytics_service = AnalyticsService(db)
        
        # Verify session ownership
        session = await db.get(TelemetrySession, session_id)
        if not session or session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or unauthorized"
            )
        
        efficiency_data = await analytics_service.analyze_efficiency(session_id)
        
        return {
            "status": "success",
            "data": efficiency_data,
            "session_id": str(session_id),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error analyzing efficiency", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze efficiency: {str(e)}"
        )


@router.post("/compare/sessions")
async def compare_sessions(
    primary_session_id: UUID,
    secondary_session_id: UUID,
    comparison_type: str = Query("detailed", regex="^(quick|detailed|advanced)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Compare two telemetry sessions in detail.
    Provides lap-by-lap comparison, sector analysis, and improvement recommendations.
    """
    try:
        analytics_service = AnalyticsService(db)
        
        # Verify both sessions belong to user
        primary_session = await db.get(TelemetrySession, primary_session_id)
        secondary_session = await db.get(TelemetrySession, secondary_session_id)
        
        if not primary_session or not secondary_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both sessions not found"
            )
        
        if primary_session.user_id != current_user.id or secondary_session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to compare these sessions"
            )
        
        comparison = await analytics_service.compare_sessions(
            primary_session_id=primary_session_id,
            secondary_session_id=secondary_session_id,
            comparison_type=comparison_type
        )
        
        # Save comparison result
        comparison_result = ComparisonResult(
            comparison_type="session_vs_session",
            primary_session_id=primary_session_id,
            secondary_session_id=secondary_session_id,
            primary_user_id=current_user.id,
            secondary_user_id=current_user.id,
            **comparison
        )
        db.add(comparison_result)
        await db.commit()
        
        return {
            "status": "success",
            "data": comparison,
            "comparison_id": str(comparison_result.id),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error comparing sessions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare sessions: {str(e)}"
        )


@router.get("/leaderboard")
async def get_leaderboard(
    track_name: str,
    car_model: str,
    category: str = Query("overall", regex="^(overall|sector_1|sector_2|sector_3)$"),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """
    Get global leaderboard for specific track and car combination.
    """
    try:
        query = select(GlobalLeaderboard).where(
            and_(
                GlobalLeaderboard.track_name == track_name,
                GlobalLeaderboard.car_model == car_model,
                GlobalLeaderboard.category == category,
                GlobalLeaderboard.is_current_record == True
            )
        ).order_by(GlobalLeaderboard.rank).limit(limit)
        
        result = await db.execute(query)
        leaderboard_entries = result.scalars().all()
        
        return {
            "status": "success",
            "data": {
                "track": track_name,
                "car": car_model,
                "category": category,
                "entries": [
                    {
                        "rank": entry.rank,
                        "user_id": str(entry.user_id),
                        "time": entry.time_value,
                        "session_id": str(entry.session_id),
                        "consistency_score": entry.consistency_score,
                        "efficiency_score": entry.efficiency_score,
                        "record_date": entry.record_date.isoformat(),
                        "weather": entry.weather_conditions,
                        "is_verified": entry.is_verified
                    }
                    for entry in leaderboard_entries
                ],
                "total_entries": len(leaderboard_entries)
            }
        }
        
    except Exception as e:
        logger.error("Error fetching leaderboard", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch leaderboard: {str(e)}"
        )


@router.get("/improvement-suggestions")
async def get_improvement_suggestions(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get AI-powered improvement suggestions based on session analysis.
    Compares with best practices and provides actionable recommendations.
    """
    try:
        analytics_service = AnalyticsService(db)
        
        # Verify session ownership
        session = await db.get(TelemetrySession, session_id)
        if not session or session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or unauthorized"
            )
        
        suggestions = await analytics_service.generate_improvement_suggestions(session_id)
        
        return {
            "status": "success",
            "data": suggestions,
            "session_id": str(session_id),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating improvement suggestions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate improvement suggestions: {str(e)}"
        )


@router.get("/trends")
async def get_performance_trends(
    metric_type: str = Query(..., regex="^(lap_time|consistency|efficiency|speed)$"),
    track_name: Optional[str] = None,
    car_model: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get performance trends over time for specified metrics.
    """
    try:
        analytics_service = AnalyticsService(db)
        
        trends = await analytics_service.calculate_trends(
            user_id=current_user.id,
            metric_type=metric_type,
            track_name=track_name,
            car_model=car_model,
            days=days
        )
        
        return {
            "status": "success",
            "data": trends,
            "metric": metric_type,
            "period_days": days,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error calculating performance trends", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate performance trends: {str(e)}"
        )


@router.delete("/cache/clear")
async def clear_analytics_cache(
    cache_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Clear analytics cache for the current user.
    Optionally specify cache_type to clear specific cache entries.
    """
    try:
        query = select(AnalyticsCache).where(
            AnalyticsCache.user_id == current_user.id
        )
        
        if cache_type:
            query = query.where(AnalyticsCache.cache_type == cache_type)
        
        result = await db.execute(query)
        cache_entries = result.scalars().all()
        
        count = 0
        for entry in cache_entries:
            await db.delete(entry)
            count += 1
        
        await db.commit()
        
        return {
            "status": "success",
            "message": f"Cleared {count} cache entries",
            "cache_type": cache_type or "all"
        }
        
    except Exception as e:
        logger.error("Error clearing analytics cache", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear analytics cache: {str(e)}"
        )

