"""
Telemetry API endpoints for LapXcel Backend API
Handles telemetry data ingestion, retrieval, and real-time streaming.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.orm import selectinload
import structlog
import json

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.telemetry import TelemetrySession, TelemetryData, Lap, SessionAnalytics
from app.models.users import User
from app.schemas.telemetry import (
    TelemetrySessionCreate, TelemetrySessionResponse, TelemetrySessionUpdate,
    TelemetryDataCreate, TelemetryDataResponse, TelemetryDataBatch,
    LapCreate, LapResponse, SessionAnalyticsResponse
)
from app.services.telemetry_service import TelemetryService
from app.services.analytics_service import AnalyticsService
from app.services.websocket_manager import WebSocketManager

logger = structlog.get_logger()
router = APIRouter()


@router.post("/sessions", response_model=TelemetrySessionResponse)
async def create_telemetry_session(
    session_data: TelemetrySessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> TelemetrySessionResponse:
    """
    Create a new telemetry recording session.
    This endpoint is called when starting a new racing session in Assetto Corsa.
    """
    try:
        logger.info("Creating new telemetry session", 
                   user_id=str(current_user.id), 
                   track=session_data.track_name,
                   car=session_data.car_model)
        
        # Create new session
        session = TelemetrySession(
            user_id=current_user.id,
            session_name=session_data.session_name,
            track_name=session_data.track_name,
            car_model=session_data.car_model,
            weather_conditions=session_data.weather_conditions,
            track_temperature=session_data.track_temperature,
            air_temperature=session_data.air_temperature,
            game_version=session_data.game_version,
            setup_data=session_data.setup_data,
            notes=session_data.notes
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        logger.info("Telemetry session created", session_id=str(session.id))
        
        return TelemetrySessionResponse.from_orm(session)
        
    except Exception as e:
        logger.error("Failed to create telemetry session", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.get("/sessions", response_model=List[TelemetrySessionResponse])
async def get_user_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    track_name: Optional[str] = Query(None),
    car_model: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[TelemetrySessionResponse]:
    """
    Retrieve user's telemetry sessions with optional filtering.
    Supports pagination and filtering by track, car, and date range.
    """
    try:
        # Build query with filters
        query = select(TelemetrySession).where(TelemetrySession.user_id == current_user.id)
        
        if track_name:
            query = query.where(TelemetrySession.track_name.ilike(f"%{track_name}%"))
        
        if car_model:
            query = query.where(TelemetrySession.car_model.ilike(f"%{car_model}%"))
        
        if start_date:
            query = query.where(TelemetrySession.created_at >= start_date)
        
        if end_date:
            query = query.where(TelemetrySession.created_at <= end_date)
        
        # Add ordering and pagination
        query = query.order_by(desc(TelemetrySession.created_at)).offset(skip).limit(limit)
        
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        return [TelemetrySessionResponse.from_orm(session) for session in sessions]
        
    except Exception as e:
        logger.error("Failed to retrieve sessions", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@router.get("/sessions/{session_id}", response_model=TelemetrySessionResponse)
async def get_session_details(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> TelemetrySessionResponse:
    """
    Get detailed information about a specific telemetry session.
    Includes lap data and basic analytics.
    """
    try:
        query = select(TelemetrySession).options(
            selectinload(TelemetrySession.laps),
            selectinload(TelemetrySession.analytics)
        ).where(
            and_(
                TelemetrySession.id == session_id,
                TelemetrySession.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return TelemetrySessionResponse.from_orm(session)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session details", session_id=str(session_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


@router.put("/sessions/{session_id}", response_model=TelemetrySessionResponse)
async def update_session(
    session_id: UUID,
    session_update: TelemetrySessionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> TelemetrySessionResponse:
    """
    Update telemetry session information.
    Used to mark sessions as complete or update metadata.
    """
    try:
        query = select(TelemetrySession).where(
            and_(
                TelemetrySession.id == session_id,
                TelemetrySession.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update fields
        update_data = session_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(session, field, value)
        
        session.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(session)
        
        logger.info("Session updated", session_id=str(session_id))
        
        return TelemetrySessionResponse.from_orm(session)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update session", session_id=str(session_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update session")


@router.post("/sessions/{session_id}/data/batch")
async def ingest_telemetry_batch(
    session_id: UUID,
    batch_data: TelemetryDataBatch,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest a batch of telemetry data points.
    Optimized for high-frequency data ingestion from Assetto Corsa.
    """
    try:
        # Verify session ownership
        query = select(TelemetrySession).where(
            and_(
                TelemetrySession.id == session_id,
                TelemetrySession.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Use service for efficient batch processing
        telemetry_service = TelemetryService(db)
        processed_count = await telemetry_service.ingest_batch(session_id, batch_data.data_points)
        
        logger.info("Telemetry batch ingested", 
                   session_id=str(session_id), 
                   count=processed_count)
        
        return {"processed_count": processed_count, "status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to ingest telemetry batch", 
                    session_id=str(session_id), 
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to ingest telemetry data")


@router.get("/sessions/{session_id}/data")
async def get_telemetry_data(
    session_id: UUID,
    lap_number: Optional[int] = Query(None),
    start_time: Optional[float] = Query(None),
    end_time: Optional[float] = Query(None),
    downsample: Optional[int] = Query(None, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve telemetry data for a session.
    Supports filtering by lap and time range, with optional downsampling for performance.
    """
    try:
        # Verify session access
        session_query = select(TelemetrySession).where(
            and_(
                TelemetrySession.id == session_id,
                TelemetrySession.user_id == current_user.id
            )
        )
        
        result = await db.execute(session_query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Use service for efficient data retrieval
        telemetry_service = TelemetryService(db)
        data = await telemetry_service.get_session_data(
            session_id=session_id,
            lap_number=lap_number,
            start_time=start_time,
            end_time=end_time,
            downsample_factor=downsample
        )
        
        return {"data": data, "count": len(data)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve telemetry data", 
                    session_id=str(session_id), 
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve telemetry data")


@router.get("/sessions/{session_id}/laps", response_model=List[LapResponse])
async def get_session_laps(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[LapResponse]:
    """
    Get all laps for a specific session with detailed timing information.
    """
    try:
        # Verify session access
        session_query = select(TelemetrySession).where(
            and_(
                TelemetrySession.id == session_id,
                TelemetrySession.user_id == current_user.id
            )
        )
        
        result = await db.execute(session_query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get laps
        laps_query = select(Lap).where(Lap.session_id == session_id).order_by(Lap.lap_number)
        result = await db.execute(laps_query)
        laps = result.scalars().all()
        
        return [LapResponse.from_orm(lap) for lap in laps]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve laps", session_id=str(session_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve laps")


@router.get("/sessions/{session_id}/analytics", response_model=SessionAnalyticsResponse)
async def get_session_analytics(
    session_id: UUID,
    recalculate: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> SessionAnalyticsResponse:
    """
    Get comprehensive analytics for a telemetry session.
    Includes performance insights, improvement suggestions, and comparisons.
    """
    try:
        # Verify session access
        session_query = select(TelemetrySession).where(
            and_(
                TelemetrySession.id == session_id,
                TelemetrySession.user_id == current_user.id
            )
        )
        
        result = await db.execute(session_query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Use analytics service
        analytics_service = AnalyticsService(db)
        analytics = await analytics_service.get_session_analytics(
            session_id=session_id,
            recalculate=recalculate
        )
        
        return SessionAnalyticsResponse.from_orm(analytics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session analytics", 
                    session_id=str(session_id), 
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@router.post("/sessions/{session_id}/export")
async def export_session_data(
    session_id: UUID,
    format: str = Query("csv", regex="^(csv|json|zip)$"),
    include_analytics: bool = Query(True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Export telemetry session data in various formats.
    Supports CSV, JSON, and compressed ZIP formats.
    """
    try:
        # Verify session access
        session_query = select(TelemetrySession).where(
            and_(
                TelemetrySession.id == session_id,
                TelemetrySession.user_id == current_user.id
            )
        )
        
        result = await db.execute(session_query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Use service for data export
        telemetry_service = TelemetryService(db)
        export_data = await telemetry_service.export_session(
            session_id=session_id,
            format=format,
            include_analytics=include_analytics
        )
        
        # Return as streaming response
        filename = f"lapxcel_session_{session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        return StreamingResponse(
            export_data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export session data", 
                    session_id=str(session_id), 
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to export session data")


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a telemetry session and all associated data.
    This operation cannot be undone.
    """
    try:
        # Verify session ownership
        query = select(TelemetrySession).where(
            and_(
                TelemetrySession.id == session_id,
                TelemetrySession.user_id == current_user.id
            )
        )
        
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete session (cascading delete will handle related data)
        await db.delete(session)
        await db.commit()
        
        logger.info("Session deleted", session_id=str(session_id))
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session", session_id=str(session_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.post("/upload")
async def upload_telemetry_file(
    file: UploadFile = File(...),
    session_name: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload telemetry data from a file (CSV, JSON, or ZIP).
    Automatically parses and creates sessions from uploaded data.
    """
    try:
        if file.size > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        allowed_types = ['.csv', '.json', '.zip']
        file_ext = '.' + file.filename.split('.')[-1].lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Use service for file processing
        telemetry_service = TelemetryService(db)
        result = await telemetry_service.process_uploaded_file(
            file=file,
            user_id=current_user.id,
            session_name=session_name
        )
        
        logger.info("Telemetry file uploaded", 
                   filename=file.filename,
                   user_id=str(current_user.id),
                   sessions_created=len(result.get('sessions', [])))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload telemetry file", 
                    filename=file.filename, 
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process uploaded file")
