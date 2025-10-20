"""
WebSocket connection manager for real-time telemetry streaming.
Handles WebSocket connections, message broadcasting, and connection lifecycle.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
import structlog

logger = structlog.get_logger()


class WebSocketManager:
    """
    Manages WebSocket connections for real-time telemetry streaming.
    Supports grouping connections by session and broadcasting to specific groups.
    """
    
    def __init__(self):
        # Active connections grouped by session/room
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a WebSocket connection and add it to the session group"""
        try:
            await websocket.accept()
            
            # Add to session group
            if session_id not in self.active_connections:
                self.active_connections[session_id] = []
            
            self.active_connections[session_id].append(websocket)
            
            # Store metadata
            self.connection_metadata[websocket] = {
                "session_id": session_id,
                "connected_at": asyncio.get_event_loop().time()
            }
            
            logger.info("WebSocket connection established", 
                       session_id=session_id,
                       total_connections=len(self.connection_metadata))
                       
        except Exception as e:
            logger.error("Failed to establish WebSocket connection", 
                        session_id=session_id, 
                        error=str(e))
            raise
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection from the session group"""
        try:
            # Remove from session group
            if session_id in self.active_connections:
                if websocket in self.active_connections[session_id]:
                    self.active_connections[session_id].remove(websocket)
                
                # Clean up empty session groups
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            
            # Remove metadata
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
            
            logger.info("WebSocket connection closed", 
                       session_id=session_id,
                       total_connections=len(self.connection_metadata))
                       
        except Exception as e:
            logger.error("Error during WebSocket disconnect", 
                        session_id=session_id, 
                        error=str(e))
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(message)
        except WebSocketDisconnect:
            # Connection already closed
            self._cleanup_disconnected_websocket(websocket)
        except Exception as e:
            logger.error("Failed to send personal message", error=str(e))
            self._cleanup_disconnected_websocket(websocket)
    
    async def send_json_to_connection(self, data: Dict[str, Any], websocket: WebSocket):
        """Send JSON data to a specific WebSocket connection"""
        try:
            await websocket.send_json(data)
        except WebSocketDisconnect:
            self._cleanup_disconnected_websocket(websocket)
        except Exception as e:
            logger.error("Failed to send JSON message", error=str(e))
            self._cleanup_disconnected_websocket(websocket)
    
    async def broadcast_to_session(self, session_id: str, message: str):
        """Broadcast a message to all connections in a session"""
        if session_id not in self.active_connections:
            return
        
        connections_to_remove = []
        
        for websocket in self.active_connections[session_id]:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                connections_to_remove.append(websocket)
            except Exception as e:
                logger.error("Failed to broadcast message", 
                           session_id=session_id, 
                           error=str(e))
                connections_to_remove.append(websocket)
        
        # Clean up disconnected connections
        for websocket in connections_to_remove:
            self.disconnect(websocket, session_id)
    
    async def broadcast_json_to_session(self, session_id: str, data: Dict[str, Any]):
        """Broadcast JSON data to all connections in a session"""
        if session_id not in self.active_connections:
            return
        
        connections_to_remove = []
        
        for websocket in self.active_connections[session_id]:
            try:
                await websocket.send_json(data)
            except WebSocketDisconnect:
                connections_to_remove.append(websocket)
            except Exception as e:
                logger.error("Failed to broadcast JSON", 
                           session_id=session_id, 
                           error=str(e))
                connections_to_remove.append(websocket)
        
        # Clean up disconnected connections
        for websocket in connections_to_remove:
            self.disconnect(websocket, session_id)
    
    async def broadcast_telemetry_data(self, session_id: str, telemetry_data: Dict[str, Any]):
        """
        Broadcast telemetry data to all connections in a session.
        Optimized for high-frequency data streaming.
        """
        if session_id not in self.active_connections:
            return
        
        # Format telemetry message
        message = {
            "type": "telemetry_update",
            "session_id": session_id,
            "timestamp": telemetry_data.get("timestamp"),
            "data": telemetry_data
        }
        
        await self.broadcast_json_to_session(session_id, message)
    
    async def broadcast_lap_completed(self, session_id: str, lap_data: Dict[str, Any]):
        """Broadcast lap completion event to session connections"""
        message = {
            "type": "lap_completed",
            "session_id": session_id,
            "lap_data": lap_data
        }
        
        await self.broadcast_json_to_session(session_id, message)
    
    async def broadcast_training_update(self, training_id: str, update_data: Dict[str, Any]):
        """Broadcast training progress updates"""
        session_id = f"training_{training_id}"
        
        message = {
            "type": "training_update",
            "training_id": training_id,
            "data": update_data
        }
        
        await self.broadcast_json_to_session(session_id, message)
    
    def _cleanup_disconnected_websocket(self, websocket: WebSocket):
        """Clean up a disconnected WebSocket from all groups"""
        if websocket in self.connection_metadata:
            session_id = self.connection_metadata[websocket]["session_id"]
            self.disconnect(websocket, session_id)
    
    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """Get the number of active connections, optionally for a specific session"""
        if session_id:
            return len(self.active_connections.get(session_id, []))
        return len(self.connection_metadata)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of sessions with active connections"""
        return list(self.active_connections.keys())
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get detailed connection information for monitoring"""
        return {
            "total_connections": len(self.connection_metadata),
            "active_sessions": len(self.active_connections),
            "sessions": {
                session_id: len(connections) 
                for session_id, connections in self.active_connections.items()
            }
        }


class TelemetryStreamer:
    """
    High-level interface for streaming telemetry data.
    Handles data formatting and streaming optimization.
    """
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self._streaming_sessions: Dict[str, bool] = {}
        self._stream_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_streaming(self, session_id: str, data_source):
        """Start streaming telemetry data for a session"""
        if session_id in self._streaming_sessions:
            logger.warning("Streaming already active for session", session_id=session_id)
            return
        
        self._streaming_sessions[session_id] = True
        
        # Start streaming task
        task = asyncio.create_task(
            self._stream_telemetry_data(session_id, data_source)
        )
        self._stream_tasks[session_id] = task
        
        logger.info("Started telemetry streaming", session_id=session_id)
    
    async def stop_streaming(self, session_id: str):
        """Stop streaming telemetry data for a session"""
        if session_id not in self._streaming_sessions:
            return
        
        # Stop streaming
        self._streaming_sessions[session_id] = False
        
        # Cancel task if exists
        if session_id in self._stream_tasks:
            task = self._stream_tasks[session_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._stream_tasks[session_id]
        
        logger.info("Stopped telemetry streaming", session_id=session_id)
    
    async def _stream_telemetry_data(self, session_id: str, data_source):
        """Stream telemetry data at regular intervals"""
        try:
            while self._streaming_sessions.get(session_id, False):
                # Get latest telemetry data
                telemetry_data = await data_source.get_latest_data(session_id)
                
                if telemetry_data:
                    await self.websocket_manager.broadcast_telemetry_data(
                        session_id, 
                        telemetry_data
                    )
                
                # Stream at 10 Hz (100ms intervals)
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.info("Telemetry streaming cancelled", session_id=session_id)
        except Exception as e:
            logger.error("Error in telemetry streaming", 
                        session_id=session_id, 
                        error=str(e))
        finally:
            if session_id in self._streaming_sessions:
                del self._streaming_sessions[session_id]
