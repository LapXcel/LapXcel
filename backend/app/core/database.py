"""
Database Configuration and Session Management
Provides database connection, session management, and base model class.
"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Create declarative base for models
Base = declarative_base()

# Global engine and session factory
_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    """
    Get or create the database engine.
    Uses connection pooling for better performance.
    """
    global _engine
    
    if _engine is None:
        # Convert postgres:// to postgresql+asyncpg:// for async support
        database_url = settings.DATABASE_URL
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        _engine = create_async_engine(
            database_url,
            echo=settings.DEBUG,  # SQL logging in debug mode
            future=True,
            pool_pre_ping=True,  # Test connections before use
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            poolclass=QueuePool if not settings.TESTING else NullPool,
        )
        
        logger.info(
            "Database engine created",
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW
        )
    
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Get or create the session factory."""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        logger.info("Session factory created")
    
    return _SessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.
    
    Usage in FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            # Use db here
            pass
    
    Yields:
        AsyncSession: Database session
    """
    SessionLocal = get_session_factory()
    
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """
    Context manager for getting database sessions outside of FastAPI.
    
    Usage:
        async with get_db_context() as db:
            result = await db.execute(query)
    """
    SessionLocal = get_session_factory()
    
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database - create all tables.
    Call this on application startup.
    
    Note: In production, use Alembic migrations instead.
    This is primarily for development and testing.
    """
    engine = get_engine()
    
    # Import all models so they're registered with Base
    # This ensures all tables are created
    from app.models import users, telemetry, analytics, training  # noqa: F401
    
    logger.info("Initializing database tables")
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created successfully")


async def drop_db() -> None:
    """
    Drop all database tables.
    WARNING: This will delete all data!
    Only use in development/testing.
    """
    engine = get_engine()
    
    # Import all models
    from app.models import users, telemetry, analytics, training  # noqa: F401
    
    logger.warning("Dropping all database tables")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.warning("All database tables dropped")


async def close_db() -> None:
    """
    Close database connections and dispose of engine.
    Call this on application shutdown.
    """
    global _engine, _SessionLocal
    
    if _engine is not None:
        logger.info("Closing database connections")
        await _engine.dispose()
        _engine = None
        _SessionLocal = None
        logger.info("Database connections closed")


# Database health check
async def check_db_connection() -> bool:
    """
    Check if database connection is healthy.
    
    Returns:
        bool: True if connection is healthy, False otherwise
    """
    try:
        async with get_db_context() as db:
            await db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False


# Connection pool event listeners for monitoring
@event.listens_for(QueuePool, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log new database connections"""
    logger.debug("Database connection established")


@event.listens_for(QueuePool, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log when connection is checked out from pool"""
    logger.debug("Database connection checked out from pool")


@event.listens_for(QueuePool, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Log when connection is returned to pool"""
    logger.debug("Database connection returned to pool")


# Database utilities
class DatabaseError(Exception):
    """Base exception for database errors"""
    pass


class ConnectionError(DatabaseError):
    """Database connection error"""
    pass


class TransactionError(DatabaseError):
    """Database transaction error"""
    pass


# Transaction decorator for atomic operations
def atomic_transaction(func):
    """
    Decorator for functions that need atomic database transactions.
    
    Usage:
        @atomic_transaction
        async def create_user_and_session(db: AsyncSession, user_data, session_data):
            # All operations here are atomic
            user = User(**user_data)
            db.add(user)
            await db.flush()  # Get user.id
            
            session = TelemetrySession(user_id=user.id, **session_data)
            db.add(session)
            # If any error occurs, all changes are rolled back
    """
    async def wrapper(*args, **kwargs):
        # Extract db session from args or kwargs
        db = kwargs.get('db') or (args[0] if args else None)
        
        if not isinstance(db, AsyncSession):
            raise ValueError("First argument must be AsyncSession")
        
        async with db.begin():
            return await func(*args, **kwargs)
    
    return wrapper


# Query result pagination helper
class Pagination:
    """Helper class for paginating query results"""
    
    def __init__(self, items: list, total: int, page: int, per_page: int):
        self.items = items
        self.total = total
        self.page = page
        self.per_page = per_page
        self.pages = (total + per_page - 1) // per_page if per_page > 0 else 0
        self.has_prev = page > 1
        self.has_next = page < self.pages
        self.prev_page = page - 1 if self.has_prev else None
        self.next_page = page + 1 if self.has_next else None


async def paginate_query(query, page: int = 1, per_page: int = 50):
    """
    Paginate a SQLAlchemy query.
    
    Args:
        query: SQLAlchemy query object
        page: Page number (1-indexed)
        per_page: Items per page
    
    Returns:
        Pagination: Pagination object with results
    """
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 50
    
    # Get total count
    total_query = query.with_only_columns([func.count()]).order_by(None)
    total = await total_query.scalar()
    
    # Get paginated results
    offset = (page - 1) * per_page
    items = await query.offset(offset).limit(per_page).all()
    
    return Pagination(items, total, page, per_page)


# Bulk operations helper
class BulkOperations:
    """Helper class for efficient bulk database operations"""
    
    @staticmethod
    async def bulk_insert(db: AsyncSession, model_class, data_list: list, chunk_size: int = 1000):
        """
        Efficiently insert multiple records in chunks.
        
        Args:
            db: Database session
            model_class: SQLAlchemy model class
            data_list: List of dictionaries with model data
            chunk_size: Number of records per chunk
        
        Returns:
            int: Number of records inserted
        """
        total_inserted = 0
        
        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i:i + chunk_size]
            objects = [model_class(**data) for data in chunk]
            db.add_all(objects)
            await db.flush()
            total_inserted += len(chunk)
            
            logger.debug(f"Bulk inserted {len(chunk)} records", model=model_class.__name__)
        
        await db.commit()
        logger.info(f"Bulk insert completed", total=total_inserted, model=model_class.__name__)
        
        return total_inserted
    
    @staticmethod
    async def bulk_update(db: AsyncSession, model_class, updates: list):
        """
        Efficiently update multiple records.
        
        Args:
            db: Database session
            model_class: SQLAlchemy model class
            updates: List of dicts with 'id' and fields to update
        
        Returns:
            int: Number of records updated
        """
        for update_data in updates:
            record_id = update_data.pop('id')
            await db.execute(
                model_class.__table__.update()
                .where(model_class.id == record_id)
                .values(**update_data)
            )
        
        await db.commit()
        logger.info(f"Bulk update completed", count=len(updates), model=model_class.__name__)
        
        return len(updates)


# Export commonly used items
__all__ = [
    'Base',
    'get_db',
    'get_db_context',
    'init_db',
    'drop_db',
    'close_db',
    'check_db_connection',
    'DatabaseError',
    'ConnectionError',
    'TransactionError',
    'atomic_transaction',
    'Pagination',
    'paginate_query',
    'BulkOperations',
]


