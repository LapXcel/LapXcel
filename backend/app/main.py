"""
LapXcel Backend API
Main FastAPI application with telemetry processing, ML model management, and real-time analytics.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import structlog

from app.core.config import settings
from app.core.database import init_db, get_db
from app.api.v1 import telemetry, training, analytics, auth
from app.services.websocket_manager import WebSocketManager
from app.services.redis_service import RedisService
from app.middleware.logging import LoggingMiddleware

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('websocket_connections_active', 'Active WebSocket connections')

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting LapXcel Backend API", version=settings.VERSION)
    
    # Initialize database
    await init_db()
    
    # Initialize Redis connection
    redis_service = RedisService()
    await redis_service.connect()
    app.state.redis = redis_service
    
    # Initialize WebSocket manager
    app.state.websocket_manager = WebSocketManager()
    
    logger.info("Backend API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LapXcel Backend API")
    await redis_service.disconnect()
    logger.info("Backend API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LapXcel API",
    description="Advanced sim racing telemetry optimization and analytics platform",
    version=settings.VERSION,
    docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include API routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(telemetry.router, prefix="/api/v1/telemetry", tags=["telemetry"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "message": "LapXcel Backend API",
        "version": settings.VERSION,
        "status": "healthy"
    }


@app.get("/health")
async def health_check(db=Depends(get_db)) -> Dict[str, Any]:
    """Comprehensive health check"""
    try:
        # Test database connection
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        db_status = "unhealthy"
    
    try:
        # Test Redis connection
        redis_service = app.state.redis
        await redis_service.ping()
        redis_status = "healthy"
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        redis_status = "unhealthy"
    
    health_status = {
        "status": "healthy" if all([db_status == "healthy", redis_status == "healthy"]) else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {
            "database": db_status,
            "redis": redis_status,
            "websockets": "healthy"
        }
    }
    
    if health_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@app.websocket("/ws/telemetry/{session_id}")
async def websocket_telemetry_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time telemetry streaming"""
    websocket_manager = app.state.websocket_manager
    
    await websocket_manager.connect(websocket, session_id)
    ACTIVE_CONNECTIONS.inc()
    
    try:
        logger.info("WebSocket connection established", session_id=session_id)
        
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Echo back for now (can be enhanced for bidirectional communication)
            await websocket_manager.send_personal_message(f"Echo: {data}", websocket)
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed", session_id=session_id)
        websocket_manager.disconnect(websocket, session_id)
        ACTIVE_CONNECTIONS.dec()
    except Exception as e:
        logger.error("WebSocket error", session_id=session_id, error=str(e))
        websocket_manager.disconnect(websocket, session_id)
        ACTIVE_CONNECTIONS.dec()


@app.websocket("/ws/training/{training_id}")
async def websocket_training_endpoint(websocket: WebSocket, training_id: str):
    """WebSocket endpoint for real-time training progress"""
    websocket_manager = app.state.websocket_manager
    
    await websocket_manager.connect(websocket, f"training_{training_id}")
    ACTIVE_CONNECTIONS.inc()
    
    try:
        logger.info("Training WebSocket connection established", training_id=training_id)
        
        while True:
            data = await websocket.receive_text()
            # Handle training-specific messages
            await websocket_manager.send_personal_message(f"Training update: {data}", websocket)
            
    except WebSocketDisconnect:
        logger.info("Training WebSocket connection closed", training_id=training_id)
        websocket_manager.disconnect(websocket, f"training_{training_id}")
        ACTIVE_CONNECTIONS.dec()
    except Exception as e:
        logger.error("Training WebSocket error", training_id=training_id, error=str(e))
        websocket_manager.disconnect(websocket, f"training_{training_id}")
        ACTIVE_CONNECTIONS.dec()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_config=None  # Use structlog instead
    )
