"""
Authentication API endpoints for LapXcel Backend API.
Handles user registration, login, logout, and token management.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, validator
import structlog

from app.core.database import get_db
from app.core.auth import (
    authenticate_user,
    create_user_account,
    create_user_tokens,
    refresh_access_token,
    get_current_user,
    AuthenticationError
)
from app.models.users import User

logger = structlog.get_logger()
router = APIRouter()
security = HTTPBearer()


# Request/Response Models
class UserRegistrationRequest(BaseModel):
    """User registration request schema"""
    email: EmailStr
    username: str
    password: str
    first_name: str = None
    last_name: str = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(v) > 50:
            raise ValueError('Username must be less than 50 characters long')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if len(v) > 128:
            raise ValueError('Password must be less than 128 characters long')
        return v


class UserLoginRequest(BaseModel):
    """User login request schema"""
    email: EmailStr
    password: str


class TokenRefreshRequest(BaseModel):
    """Token refresh request schema"""
    refresh_token: str


class UserResponse(BaseModel):
    """User response schema"""
    id: str
    email: str
    username: str
    first_name: str = None
    last_name: str = None
    display_name: str = None
    is_active: bool
    is_verified: bool
    is_premium: bool
    created_at: datetime
    last_login: datetime = None
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserResponse


class LoginResponse(BaseModel):
    """Login response schema"""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserResponse
    message: str = "Login successful"


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegistrationRequest,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Register a new user account.
    Creates a new user and returns authentication tokens.
    """
    try:
        logger.info("User registration attempt", email=user_data.email, username=user_data.username)
        
        # Create user account
        user = await create_user_account(
            db=db,
            email=user_data.email,
            username=user_data.username,
            password=user_data.password,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        
        # Generate tokens
        tokens = create_user_tokens(user)
        
        # Create response
        user_response = UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_premium=user.is_premium,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
        logger.info("User registration successful", user_id=str(user.id))
        
        return TokenResponse(
            **tokens,
            user=user_response
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("User registration validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("User registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=LoginResponse)
async def login_user(
    login_data: UserLoginRequest,
    db: AsyncSession = Depends(get_db)
) -> LoginResponse:
    """
    Authenticate user and return access tokens.
    """
    try:
        logger.info("User login attempt", email=login_data.email)
        
        # Authenticate user
        user = await authenticate_user(db, login_data.email, login_data.password)
        
        if not user:
            logger.warning("Login failed - invalid credentials", email=login_data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Generate tokens
        tokens = create_user_tokens(user)
        
        # Create response
        user_response = UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_premium=user.is_premium,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
        logger.info("User login successful", user_id=str(user.id))
        
        return LoginResponse(
            **tokens,
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("User login failed", email=login_data.email, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Refresh access token using refresh token.
    """
    try:
        logger.info("Token refresh attempt")
        
        # Refresh tokens
        tokens = await refresh_access_token(token_data.refresh_token, db)
        
        # Get user info for response (extract from new access token)
        from app.core.auth import verify_token, get_user_by_id
        from uuid import UUID
        
        payload = verify_token(tokens["access_token"])
        user_id = UUID(payload["sub"])
        user = await get_user_by_id(db, user_id)
        
        if not user:
            raise AuthenticationError("User not found")
        
        user_response = UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_premium=user.is_premium,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
        logger.info("Token refresh successful", user_id=str(user.id))
        
        return TokenResponse(
            **tokens,
            user=user_response
        )
        
    except AuthenticationError as e:
        logger.warning("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Logout current user.
    In a full implementation, this would invalidate the token.
    """
    try:
        logger.info("User logout", user_id=str(current_user.id))
        
        # In a full implementation, you would:
        # 1. Add the token to a blacklist
        # 2. Remove any active sessions
        # 3. Clear any cached user data
        
        return {"message": "Logout successful"}
        
    except Exception as e:
        logger.error("Logout failed", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    """
    Get current user information.
    """
    try:
        return UserResponse(
            id=str(current_user.id),
            email=current_user.email,
            username=current_user.username,
            first_name=current_user.first_name,
            last_name=current_user.last_name,
            display_name=current_user.display_name,
            is_active=current_user.is_active,
            is_verified=current_user.is_verified,
            is_premium=current_user.is_premium,
            created_at=current_user.created_at,
            last_login=current_user.last_login
        )
        
    except Exception as e:
        logger.error("Failed to get user info", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )


@router.post("/verify-token")
async def verify_token_endpoint(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verify if the current token is valid.
    """
    return {
        "valid": True,
        "user_id": str(current_user.id),
        "email": current_user.email,
        "username": current_user.username,
        "is_active": current_user.is_active
    }


@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Change user password.
    """
    try:
        from app.core.auth import verify_password, get_password_hash, validate_password_strength
        
        # Verify old password
        if not verify_password(old_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        if not validate_password_strength(new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password does not meet strength requirements"
            )
        
        # Update password
        current_user.hashed_password = get_password_hash(new_password)
        await db.commit()
        
        logger.info("Password changed successfully", user_id=str(current_user.id))
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change failed", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

