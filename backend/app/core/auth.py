"""
Authentication and authorization utilities for LapXcel Backend API.
Handles JWT tokens, password hashing, and user authentication.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from app.core.config import settings
from app.core.database import get_db
from app.models.users import User

logger = structlog.get_logger()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handling
security = HTTPBearer()


class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Custom authorization error"""
    def __init__(self, detail: str = "Not enough permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: The data to encode in the token
        expires_delta: Token expiration time (default: 30 minutes)
    
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(user_id: UUID) -> str:
    """
    Create a refresh token for token renewal.
    
    Args:
        user_id: User ID to encode in the token
    
    Returns:
        JWT refresh token string
    """
    data = {
        "sub": str(user_id),
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=30)  # 30 days expiration
    }
    
    encoded_jwt = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload
    
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Check if token has expired
        exp = payload.get("exp")
        if exp is None:
            raise AuthenticationError("Token missing expiration")
        
        if datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise AuthenticationError("Token has expired")
        
        return payload
        
    except JWTError as e:
        logger.warning("JWT verification failed", error=str(e))
        raise AuthenticationError("Invalid token")


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email address"""
    try:
        query = select(User).where(User.email == email)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    except Exception as e:
        logger.error("Failed to get user by email", email=email, error=str(e))
        return None


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[User]:
    """Get user by ID"""
    try:
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    except Exception as e:
        logger.error("Failed to get user by ID", user_id=str(user_id), error=str(e))
        return None


async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """
    Authenticate a user with email and password.
    
    Args:
        db: Database session
        email: User email
        password: Plain text password
    
    Returns:
        User object if authentication successful, None otherwise
    """
    user = await get_user_by_email(db, email)
    
    if not user:
        return None
    
    if not user.is_active:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    # Update last login time
    user.last_login = datetime.utcnow()
    await db.commit()
    
    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        db: Database session
    
    Returns:
        Current user object
    
    Raises:
        AuthenticationError: If authentication fails
    """
    try:
        # Verify token
        payload = verify_token(credentials.credentials)
        
        # Get user ID from token
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise AuthenticationError("Token missing user ID")
        
        try:
            user_id = UUID(user_id_str)
        except ValueError:
            raise AuthenticationError("Invalid user ID in token")
        
        # Get user from database
        user = await get_user_by_id(db, user_id)
        if user is None:
            raise AuthenticationError("User not found")
        
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        
        # Update last activity
        user.last_activity = datetime.utcnow()
        await db.commit()
        
        return user
        
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise AuthenticationError("Authentication failed")


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user (additional check for user status).
    
    Args:
        current_user: Current user from get_current_user
    
    Returns:
        Active user object
    
    Raises:
        AuthenticationError: If user is inactive
    """
    if not current_user.is_active:
        raise AuthenticationError("User account is disabled")
    
    return current_user


async def get_current_premium_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current user and verify premium status.
    
    Args:
        current_user: Current user from get_current_user
    
    Returns:
        Premium user object
    
    Raises:
        AuthorizationError: If user doesn't have premium access
    """
    if not current_user.is_premium:
        raise AuthorizationError("Premium subscription required")
    
    return current_user


def require_permissions(*required_permissions: str):
    """
    Decorator to require specific permissions for an endpoint.
    
    Args:
        required_permissions: List of required permissions
    
    Returns:
        Dependency function
    """
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        # This is a placeholder for permission checking
        # In a full implementation, you'd check user roles/permissions
        # For now, we'll just return the user
        return current_user
    
    return permission_checker


class TokenData:
    """Token data model for type hints"""
    def __init__(self, user_id: Optional[UUID] = None, email: Optional[str] = None):
        self.user_id = user_id
        self.email = email


def create_user_tokens(user: User) -> Dict[str, str]:
    """
    Create access and refresh tokens for a user.
    
    Args:
        user: User object
    
    Returns:
        Dictionary containing access_token and refresh_token
    """
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email},
        expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token = create_refresh_token(user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
    }


async def refresh_access_token(refresh_token: str, db: AsyncSession) -> Dict[str, str]:
    """
    Refresh an access token using a refresh token.
    
    Args:
        refresh_token: Refresh token string
        db: Database session
    
    Returns:
        New token data
    
    Raises:
        AuthenticationError: If refresh token is invalid
    """
    try:
        # Verify refresh token
        payload = verify_token(refresh_token)
        
        # Check token type
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type")
        
        # Get user
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise AuthenticationError("Token missing user ID")
        
        user_id = UUID(user_id_str)
        user = await get_user_by_id(db, user_id)
        
        if user is None or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Create new tokens
        return create_user_tokens(user)
        
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise AuthenticationError("Token refresh failed")


def validate_password_strength(password: str) -> bool:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
    
    Returns:
        True if password meets requirements
    """
    # Basic password requirements
    if len(password) < 8:
        return False
    
    # Check for at least one uppercase letter
    if not any(c.isupper() for c in password):
        return False
    
    # Check for at least one lowercase letter
    if not any(c.islower() for c in password):
        return False
    
    # Check for at least one digit
    if not any(c.isdigit() for c in password):
        return False
    
    return True


async def create_user_account(
    db: AsyncSession,
    email: str,
    username: str,
    password: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None
) -> User:
    """
    Create a new user account.
    
    Args:
        db: Database session
        email: User email
        username: Username
        password: Plain text password
        first_name: Optional first name
        last_name: Optional last name
    
    Returns:
        Created user object
    
    Raises:
        ValueError: If validation fails
        HTTPException: If user already exists
    """
    # Validate password
    if not validate_password_strength(password):
        raise ValueError("Password does not meet strength requirements")
    
    # Check if user already exists
    existing_user = await get_user_by_email(db, email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username is taken
    username_query = select(User).where(User.username == username)
    result = await db.execute(username_query)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    
    user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        first_name=first_name,
        last_name=last_name,
        display_name=first_name or username,
        is_active=True,
        is_verified=False  # Email verification would be implemented separately
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    logger.info("User account created", user_id=str(user.id), email=email, username=username)
    
    return user

