"""
Telemetry service for efficient data processing and analysis.
Handles high-frequency data ingestion, processing, and optimization.
"""

import asyncio
import csv
import json
import zipfile
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator
from uuid import UUID
from io import StringIO, BytesIO

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func, text
from sqlalchemy.orm import selectinload
from fastapi import UploadFile
import pandas as pd
import numpy as np
import structlog

from app.models.telemetry import TelemetrySession, TelemetryData, Lap, SessionAnalytics
from app.schemas.telemetry import TelemetryDataPoint
from app.services.redis_service import RedisService

logger = structlog.get_logger()


class TelemetryService:
    """Service for telemetry data processing and management"""
    
    def __init__(self, db: AsyncSession, redis: Optional[RedisService] = None):
        self.db = db
        self.redis = redis
        
    async def ingest_batch(self, session_id: UUID, data_points: List[TelemetryDataPoint]) -> int:
        """
        Efficiently ingest a batch of telemetry data points.
        Uses bulk insert for performance optimization.
        """
        try:
            # Convert Pydantic models to database models
            db_objects = []
            for point in data_points:
                db_object = TelemetryData(
                    session_id=session_id,
                    **point.dict()
                )
                db_objects.append(db_object)
            
            # Bulk insert for efficiency
            self.db.add_all(db_objects)
            await self.db.commit()
            
            # Cache latest data point for real-time streaming
            if self.redis and data_points:
                latest_point = data_points[-1]
                await self.redis.set(
                    f"telemetry:latest:{session_id}",
                    latest_point.json(),
                    expire=300  # 5 minutes
                )
            
            logger.info("Telemetry batch ingested", 
                       session_id=str(session_id), 
                       count=len(data_points))
            
            return len(data_points)
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Failed to ingest telemetry batch", 
                        session_id=str(session_id), 
                        error=str(e))
            raise
    
    async def get_session_data(
        self, 
        session_id: UUID,
        lap_number: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        downsample_factor: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve telemetry data with optional filtering and downsampling.
        Optimized for large datasets and real-time visualization.
        """
        try:
            # Build base query
            query = select(TelemetryData).where(TelemetryData.session_id == session_id)
            
            # Add lap filter
            if lap_number is not None:
                # Get lap ID first
                lap_query = select(Lap.id).where(
                    and_(Lap.session_id == session_id, Lap.lap_number == lap_number)
                )
                result = await self.db.execute(lap_query)
                lap_id = result.scalar_one_or_none()
                if lap_id:
                    query = query.where(TelemetryData.lap_id == lap_id)
            
            # Add time range filters
            if start_time is not None:
                query = query.where(TelemetryData.session_time >= start_time)
            if end_time is not None:
                query = query.where(TelemetryData.session_time <= end_time)
            
            # Order by time
            query = query.order_by(TelemetryData.session_time)
            
            # Apply downsampling if requested
            if downsample_factor and downsample_factor > 1:
                # Use modulo on row number for uniform sampling
                query = query.filter(
                    text(f"ROW_NUMBER() OVER (ORDER BY session_time) % {downsample_factor} = 0")
                )
            
            # Execute query
            result = await self.db.execute(query)
            data_points = result.scalars().all()
            
            # Convert to dictionaries for JSON serialization
            return [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'session_time': point.session_time,
                    'lap_time': point.lap_time,
                    'speed_kmh': point.speed_kmh,
                    'throttle_input': point.throttle_input,
                    'brake_input': point.brake_input,
                    'steering_input': point.steering_input,
                    'world_position_x': point.world_position_x,
                    'world_position_y': point.world_position_y,
                    'world_position_z': point.world_position_z,
                    'track_progress': point.track_progress,
                    'gear': point.gear,
                    'rpm': point.rpm,
                    'is_on_track': point.is_on_track
                }
                for point in data_points
            ]
            
        except Exception as e:
            logger.error("Failed to retrieve session data", 
                        session_id=str(session_id), 
                        error=str(e))
            raise
    
    async def process_lap_data(self, session_id: UUID) -> List[Lap]:
        """
        Process raw telemetry data to identify and create lap records.
        Analyzes track progress to detect lap boundaries and calculate lap times.
        """
        try:
            # Get all telemetry data for the session, ordered by time
            query = select(TelemetryData).where(
                TelemetryData.session_id == session_id
            ).order_by(TelemetryData.session_time)
            
            result = await self.db.execute(query)
            data_points = result.scalars().all()
            
            if not data_points:
                return []
            
            laps = []
            current_lap_start = None
            current_lap_data = []
            lap_number = 1
            
            for i, point in enumerate(data_points):
                # Detect lap start/finish (track progress crossing 0/1 boundary)
                if i > 0:
                    prev_progress = data_points[i-1].track_progress
                    curr_progress = point.track_progress
                    
                    # Lap completed when progress goes from high (>0.9) to low (<0.1)
                    if prev_progress > 0.9 and curr_progress < 0.1:
                        if current_lap_start is not None and current_lap_data:
                            # Calculate lap time and metrics
                            lap_time = point.session_time - current_lap_start.session_time
                            
                            # Calculate sector times (approximate thirds)
                            sector_times = self._calculate_sector_times(current_lap_data)
                            
                            # Calculate lap metrics
                            max_speed = max(p.speed_kmh for p in current_lap_data)
                            avg_speed = sum(p.speed_kmh for p in current_lap_data) / len(current_lap_data)
                            
                            # Create lap record
                            lap = Lap(
                                session_id=session_id,
                                lap_number=lap_number,
                                lap_time=lap_time,
                                sector_1_time=sector_times.get('sector_1'),
                                sector_2_time=sector_times.get('sector_2'),
                                sector_3_time=sector_times.get('sector_3'),
                                max_speed=max_speed,
                                avg_speed=avg_speed,
                                is_valid=self._is_lap_valid(current_lap_data),
                                lap_start_time=current_lap_start.timestamp,
                                lap_end_time=point.timestamp
                            )
                            
                            laps.append(lap)
                            lap_number += 1
                        
                        # Start new lap
                        current_lap_start = point
                        current_lap_data = [point]
                    else:
                        if current_lap_start is not None:
                            current_lap_data.append(point)
                else:
                    # First data point starts the first lap
                    current_lap_start = point
                    current_lap_data = [point]
            
            # Save laps to database
            if laps:
                self.db.add_all(laps)
                await self.db.commit()
                
                # Update session with lap count and best lap time
                session_query = select(TelemetrySession).where(
                    TelemetrySession.id == session_id
                )
                result = await self.db.execute(session_query)
                session = result.scalar_one()
                
                valid_laps = [lap for lap in laps if lap.is_valid]
                if valid_laps:
                    best_lap = min(valid_laps, key=lambda l: l.lap_time)
                    session.best_lap_time = best_lap.lap_time
                    session.best_lap_id = best_lap.id
                
                session.total_laps = len(laps)
                await self.db.commit()
            
            logger.info("Processed lap data", 
                       session_id=str(session_id), 
                       laps_found=len(laps))
            
            return laps
            
        except Exception as e:
            logger.error("Failed to process lap data", 
                        session_id=str(session_id), 
                        error=str(e))
            raise
    
    def _calculate_sector_times(self, lap_data: List[TelemetryData]) -> Dict[str, float]:
        """Calculate sector times based on track progress"""
        if not lap_data:
            return {}
        
        sector_times = {}
        sector_boundaries = [0.33, 0.66, 1.0]
        current_sector = 1
        sector_start_time = lap_data[0].session_time
        
        for point in lap_data:
            if point.track_progress >= sector_boundaries[current_sector - 1]:
                sector_time = point.session_time - sector_start_time
                sector_times[f'sector_{current_sector}'] = sector_time
                
                if current_sector < 3:
                    current_sector += 1
                    sector_start_time = point.session_time
                else:
                    break
        
        return sector_times
    
    def _is_lap_valid(self, lap_data: List[TelemetryData]) -> bool:
        """Determine if a lap is valid based on track limits and other factors"""
        if not lap_data:
            return False
        
        # Check for track limit violations
        off_track_count = sum(1 for point in lap_data if not point.is_on_track)
        off_track_ratio = off_track_count / len(lap_data)
        
        # Lap is invalid if more than 5% of time was spent off track
        if off_track_ratio > 0.05:
            return False
        
        # Check for minimum lap time (avoid outliers)
        lap_duration = lap_data[-1].session_time - lap_data[0].session_time
        if lap_duration < 30:  # Minimum 30 seconds for a valid lap
            return False
        
        return True
    
    async def export_session(
        self, 
        session_id: UUID, 
        format: str = "csv",
        include_analytics: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """
        Export session data in specified format.
        Supports streaming for large datasets.
        """
        try:
            # Get session info
            session_query = select(TelemetrySession).options(
                selectinload(TelemetrySession.laps),
                selectinload(TelemetrySession.analytics) if include_analytics else None
            ).where(TelemetrySession.id == session_id)
            
            result = await self.db.execute(session_query)
            session = result.scalar_one()
            
            if format == "csv":
                async for chunk in self._export_csv(session_id, session):
                    yield chunk
            elif format == "json":
                async for chunk in self._export_json(session_id, session, include_analytics):
                    yield chunk
            elif format == "zip":
                async for chunk in self._export_zip(session_id, session, include_analytics):
                    yield chunk
            
        except Exception as e:
            logger.error("Failed to export session", 
                        session_id=str(session_id), 
                        format=format, 
                        error=str(e))
            raise
    
    async def _export_csv(self, session_id: UUID, session: TelemetrySession) -> AsyncGenerator[bytes, None]:
        """Export telemetry data as CSV"""
        # Stream CSV data in chunks
        query = select(TelemetryData).where(
            TelemetryData.session_id == session_id
        ).order_by(TelemetryData.session_time)
        
        # CSV header
        header = [
            'timestamp', 'session_time', 'lap_time', 'speed_kmh',
            'throttle_input', 'brake_input', 'steering_input',
            'world_position_x', 'world_position_y', 'world_position_z',
            'track_progress', 'gear', 'rpm', 'is_on_track'
        ]
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(header)
        
        # Yield header
        yield output.getvalue().encode()
        
        # Stream data in chunks
        chunk_size = 1000
        offset = 0
        
        while True:
            chunk_query = query.offset(offset).limit(chunk_size)
            result = await self.db.execute(chunk_query)
            data_points = result.scalars().all()
            
            if not data_points:
                break
            
            output = StringIO()
            writer = csv.writer(output)
            
            for point in data_points:
                writer.writerow([
                    point.timestamp.isoformat(),
                    point.session_time,
                    point.lap_time,
                    point.speed_kmh,
                    point.throttle_input,
                    point.brake_input,
                    point.steering_input,
                    point.world_position_x,
                    point.world_position_y,
                    point.world_position_z,
                    point.track_progress,
                    point.gear,
                    point.rpm,
                    point.is_on_track
                ])
            
            yield output.getvalue().encode()
            offset += chunk_size
    
    async def _export_json(
        self, 
        session_id: UUID, 
        session: TelemetrySession,
        include_analytics: bool
    ) -> AsyncGenerator[bytes, None]:
        """Export session data as JSON"""
        # Build complete data structure
        export_data = {
            'session': {
                'id': str(session.id),
                'name': session.session_name,
                'track': session.track_name,
                'car': session.car_model,
                'start_time': session.session_start.isoformat(),
                'total_laps': session.total_laps,
                'best_lap_time': session.best_lap_time
            },
            'laps': [],
            'telemetry_data': []
        }
        
        # Add laps
        for lap in session.laps:
            export_data['laps'].append({
                'lap_number': lap.lap_number,
                'lap_time': lap.lap_time,
                'sector_1_time': lap.sector_1_time,
                'sector_2_time': lap.sector_2_time,
                'sector_3_time': lap.sector_3_time,
                'is_valid': lap.is_valid,
                'max_speed': lap.max_speed,
                'avg_speed': lap.avg_speed
            })
        
        # Add analytics if requested
        if include_analytics and session.analytics:
            analytics = session.analytics[0]  # Assuming latest analytics
            export_data['analytics'] = {
                'consistency_score': analytics.consistency_score,
                'efficiency_score': analytics.efficiency_score,
                'smoothness_score': analytics.smoothness_score,
                'overall_score': analytics.overall_score,
                'suggestions': analytics.suggestions,
                'priority_areas': analytics.priority_areas
            }
        
        # Stream telemetry data
        yield json.dumps(export_data, indent=2)[:100].encode()  # Start of JSON
        
        # Note: Full JSON streaming implementation would be more complex
        # This is a simplified version for demonstration
    
    async def _export_zip(
        self, 
        session_id: UUID, 
        session: TelemetrySession,
        include_analytics: bool
    ) -> AsyncGenerator[bytes, None]:
        """Export session data as ZIP archive"""
        # Create ZIP file in memory
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add session metadata
            session_info = {
                'session_name': session.session_name,
                'track_name': session.track_name,
                'car_model': session.car_model,
                'session_start': session.session_start.isoformat(),
                'total_laps': session.total_laps,
                'best_lap_time': session.best_lap_time
            }
            zip_file.writestr('session_info.json', json.dumps(session_info, indent=2))
            
            # Add CSV data (simplified - would normally stream this)
            csv_data = "timestamp,session_time,speed_kmh,throttle_input,brake_input\n"
            zip_file.writestr('telemetry_data.csv', csv_data)
            
            if include_analytics and session.analytics:
                analytics_data = {
                    'overall_score': session.analytics[0].overall_score,
                    'suggestions': session.analytics[0].suggestions
                }
                zip_file.writestr('analytics.json', json.dumps(analytics_data, indent=2))
        
        zip_buffer.seek(0)
        yield zip_buffer.getvalue()
    
    async def process_uploaded_file(
        self, 
        file: UploadFile, 
        user_id: UUID,
        session_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded telemetry file and create session(s).
        Supports CSV, JSON, and ZIP formats.
        """
        try:
            file_ext = file.filename.split('.')[-1].lower()
            content = await file.read()
            
            if file_ext == 'csv':
                return await self._process_csv_upload(content, user_id, session_name)
            elif file_ext == 'json':
                return await self._process_json_upload(content, user_id, session_name)
            elif file_ext == 'zip':
                return await self._process_zip_upload(content, user_id, session_name)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error("Failed to process uploaded file", 
                        filename=file.filename, 
                        error=str(e))
            raise
    
    async def _process_csv_upload(
        self, 
        content: bytes, 
        user_id: UUID,
        session_name: Optional[str]
    ) -> Dict[str, Any]:
        """Process CSV file upload"""
        # Implementation would parse CSV and create telemetry session
        # This is a placeholder for the actual implementation
        return {"message": "CSV processing not yet implemented", "sessions": []}
    
    async def _process_json_upload(
        self, 
        content: bytes, 
        user_id: UUID,
        session_name: Optional[str]
    ) -> Dict[str, Any]:
        """Process JSON file upload"""
        # Implementation would parse JSON and create telemetry session
        return {"message": "JSON processing not yet implemented", "sessions": []}
    
    async def _process_zip_upload(
        self, 
        content: bytes, 
        user_id: UUID,
        session_name: Optional[str]
    ) -> Dict[str, Any]:
        """Process ZIP file upload"""
        # Implementation would extract and process ZIP contents
        return {"message": "ZIP processing not yet implemented", "sessions": []}
