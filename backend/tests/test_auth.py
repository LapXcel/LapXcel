"""
Tests for authentication endpoints and functionality.
Covers user registration, login, token management, and security.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.users import User
from app.core.auth import verify_password, create_access_token, verify_token


class TestUserRegistration:
    """Test user registration functionality."""
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, client: AsyncClient, test_utils):
        """Test successful user registration."""
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "SecurePass123",
            "first_name": "New",
            "last_name": "User"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        test_utils.assert_response_success(response)
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        assert data["user"]["email"] == user_data["email"]
        assert data["user"]["username"] == user_data["username"]
        assert data["user"]["first_name"] == user_data["first_name"]
        assert data["user"]["last_name"] == user_data["last_name"]
        assert data["token_type"] == "bearer"
        
        test_utils.assert_valid_uuid(data["user"]["id"])
    
    @pytest.mark.asyncio
    async def test_register_user_duplicate_email(self, client: AsyncClient, test_user: User):
        """Test registration with duplicate email."""
        user_data = {
            "email": test_user.email,
            "username": "different_username",
            "password": "SecurePass123"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Email already registered" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_register_user_duplicate_username(self, client: AsyncClient, test_user: User):
        """Test registration with duplicate username."""
        user_data = {
            "email": "different@example.com",
            "username": test_user.username,
            "password": "SecurePass123"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Username already taken" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_register_user_weak_password(self, client: AsyncClient):
        """Test registration with weak password."""
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "weak"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_register_user_invalid_email(self, client: AsyncClient):
        """Test registration with invalid email format."""
        user_data = {
            "email": "not-an-email",
            "username": "newuser",
            "password": "SecurePass123"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 422  # Validation error


class TestUserLogin:
    """Test user login functionality."""
    
    @pytest.mark.asyncio
    async def test_login_success(self, client: AsyncClient, test_user: User, test_utils):
        """Test successful user login."""
        login_data = {
            "email": test_user.email,
            "password": "testpassword"
        }
        
        response = await client.post("/api/v1/auth/login", json=login_data)
        
        test_utils.assert_response_success(response)
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        assert data["user"]["email"] == test_user.email
        assert data["user"]["username"] == test_user.username
        assert data["token_type"] == "bearer"
        assert data["message"] == "Login successful"
    
    @pytest.mark.asyncio
    async def test_login_invalid_email(self, client: AsyncClient):
        """Test login with invalid email."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "anypassword"
        }
        
        response = await client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "Invalid email or password" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_login_invalid_password(self, client: AsyncClient, test_user: User):
        """Test login with invalid password."""
        login_data = {
            "email": test_user.email,
            "password": "wrongpassword"
        }
        
        response = await client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "Invalid email or password" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_login_inactive_user(self, client: AsyncClient, db_session: AsyncSession):
        """Test login with inactive user account."""
        from app.core.auth import get_password_hash
        
        # Create inactive user
        inactive_user = User(
            email="inactive@example.com",
            username="inactiveuser",
            hashed_password=get_password_hash("password"),
            is_active=False
        )
        db_session.add(inactive_user)
        await db_session.commit()
        
        login_data = {
            "email": inactive_user.email,
            "password": "password"
        }
        
        response = await client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401


class TestTokenManagement:
    """Test token creation, validation, and refresh."""
    
    @pytest.mark.asyncio
    async def test_verify_token_valid(self, authenticated_client: AsyncClient, test_utils):
        """Test token verification with valid token."""
        response = await authenticated_client.post("/api/v1/auth/verify-token")
        
        test_utils.assert_response_success(response)
        data = response.json()
        
        assert data["valid"] is True
        assert "user_id" in data
        assert "email" in data
        assert "username" in data
        test_utils.assert_valid_uuid(data["user_id"])
    
    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, client: AsyncClient):
        """Test token verification with invalid token."""
        client.headers.update({"Authorization": "Bearer invalid_token"})
        
        response = await client.post("/api/v1/auth/verify-token")
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, authenticated_client: AsyncClient, test_user: User, test_utils):
        """Test getting current user information."""
        response = await authenticated_client.get("/api/v1/auth/me")
        
        test_utils.assert_response_success(response)
        data = response.json()
        
        assert data["email"] == test_user.email
        assert data["username"] == test_user.username
        assert data["first_name"] == test_user.first_name
        assert data["last_name"] == test_user.last_name
        test_utils.assert_valid_uuid(data["id"])
        test_utils.assert_valid_timestamp(data["created_at"])
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, client: AsyncClient, test_user: User, test_utils):
        """Test successful token refresh."""
        from app.core.auth import create_refresh_token
        
        refresh_token = create_refresh_token(test_user.id)
        refresh_data = {"refresh_token": refresh_token}
        
        response = await client.post("/api/v1/auth/refresh", json=refresh_data)
        
        test_utils.assert_response_success(response)
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, client: AsyncClient):
        """Test token refresh with invalid refresh token."""
        refresh_data = {"refresh_token": "invalid_refresh_token"}
        
        response = await client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == 401


class TestPasswordManagement:
    """Test password change functionality."""
    
    @pytest.mark.asyncio
    async def test_change_password_success(self, authenticated_client: AsyncClient):
        """Test successful password change."""
        password_data = {
            "old_password": "testpassword",
            "new_password": "NewSecurePass123"
        }
        
        response = await authenticated_client.post(
            "/api/v1/auth/change-password",
            params=password_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Password changed successfully"
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_old_password(self, authenticated_client: AsyncClient):
        """Test password change with wrong old password."""
        password_data = {
            "old_password": "wrongpassword",
            "new_password": "NewSecurePass123"
        }
        
        response = await authenticated_client.post(
            "/api/v1/auth/change-password",
            params=password_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Current password is incorrect" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_change_password_weak_new_password(self, authenticated_client: AsyncClient):
        """Test password change with weak new password."""
        password_data = {
            "old_password": "testpassword",
            "new_password": "weak"
        }
        
        response = await authenticated_client.post(
            "/api/v1/auth/change-password",
            params=password_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "does not meet strength requirements" in data["detail"]


class TestLogout:
    """Test logout functionality."""
    
    @pytest.mark.asyncio
    async def test_logout_success(self, authenticated_client: AsyncClient):
        """Test successful logout."""
        response = await authenticated_client.post("/api/v1/auth/logout")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Logout successful"
    
    @pytest.mark.asyncio
    async def test_logout_unauthenticated(self, client: AsyncClient):
        """Test logout without authentication."""
        response = await client.post("/api/v1/auth/logout")
        
        assert response.status_code == 401


class TestAuthUtilities:
    """Test authentication utility functions."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "TestPassword123"
        hashed = verify_password.__globals__['get_password_hash'](password)
        
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)
    
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification."""
        from uuid import uuid4
        
        user_id = uuid4()
        token_data = {"sub": str(user_id), "email": "test@example.com"}
        
        token = create_access_token(token_data)
        assert token is not None
        
        decoded = verify_token(token)
        assert decoded["sub"] == str(user_id)
        assert decoded["email"] == "test@example.com"
    
    def test_invalid_token_verification(self):
        """Test verification of invalid tokens."""
        with pytest.raises(Exception):  # Should raise AuthenticationError
            verify_token("invalid_token")
    
    def test_expired_token_verification(self):
        """Test verification of expired tokens."""
        from datetime import datetime, timedelta
        
        # Create token that expires immediately
        expired_data = {
            "sub": "test_user",
            "exp": datetime.utcnow() - timedelta(minutes=1)
        }
        
        from jose import jwt
        from app.core.config import settings
        
        expired_token = jwt.encode(expired_data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        with pytest.raises(Exception):  # Should raise AuthenticationError
            verify_token(expired_token)
