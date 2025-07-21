from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel, EmailStr
import uuid
from datetime import timedelta

from ..core.database import get_db
from ..core.auth import (
    get_password_hash, verify_password, create_access_token, create_refresh_token,
    verify_token, verify_firebase_token, get_or_create_user_from_firebase,
    AuthenticationError, get_current_user
)
from ..models.database import User as UserModel
from ..models.schemas import User
from ..core.logging import logger
from ..core.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()


# Request/Response Models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class FirebaseLoginRequest(BaseModel):
    id_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    user: User
    tokens: TokenResponse


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: RegisterRequest,
    db: Session = Depends(get_db)
):
    """Register a new user with email and password"""
    
    try:
        # Check if user already exists
        existing_user = db.query(UserModel).filter(UserModel.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = UserModel(
            email=user_data.email,
            name=user_data.name,
            hashed_password=hashed_password,
            preferences={}
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create tokens
        access_token = create_access_token(
            data={"sub": str(new_user.user_id), "email": new_user.email}
        )
        refresh_token = create_refresh_token(
            data={"sub": str(new_user.user_id)}
        )
        
        logger.info(f"New user registered: {user_data.email}")
        
        return UserResponse(
            user=User(
                user_id=new_user.user_id,
                email=new_user.email,
                name=new_user.name,
                created_at=new_user.created_at,
                updated_at=new_user.updated_at,
                is_active=new_user.is_active,
                preferences=new_user.preferences
            ),
            tokens=TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=settings.access_token_expire_minutes * 60
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=UserResponse)
async def login(
    user_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login with email and password"""
    
    try:
        # Find user
        user = db.query(UserModel).filter(UserModel.email == user_data.email).first()
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        # Verify password
        if not user.hashed_password or not verify_password(user_data.password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")
        
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        
        # Create tokens
        access_token = create_access_token(
            data={"sub": str(user.user_id), "email": user.email}
        )
        refresh_token = create_refresh_token(
            data={"sub": str(user.user_id)}
        )
        
        logger.info(f"User logged in: {user.email}")
        
        return UserResponse(
            user=User(
                user_id=user.user_id,
                email=user.email,
                name=user.name,
                created_at=user.created_at,
                updated_at=user.updated_at,
                is_active=user.is_active,
                preferences=user.preferences
            ),
            tokens=TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=settings.access_token_expire_minutes * 60
            )
        )
        
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/firebase-login", response_model=UserResponse)
async def firebase_login(
    firebase_data: FirebaseLoginRequest,
    db: Session = Depends(get_db)
):
    """Login with Firebase ID token"""
    
    try:
        # Verify Firebase token
        firebase_user = await verify_firebase_token(firebase_data.id_token)
        
        # Get or create user
        user_model = await get_or_create_user_from_firebase(firebase_user, db)
        
        # Create JWT tokens
        access_token = create_access_token(
            data={"sub": str(user_model.user_id), "email": user_model.email}
        )
        refresh_token = create_refresh_token(
            data={"sub": str(user_model.user_id)}
        )
        
        logger.info(f"Firebase user logged in: {user_model.email}")
        
        return UserResponse(
            user=User(
                user_id=user_model.user_id,
                email=user_model.email,
                name=user_model.name,
                created_at=user_model.created_at,
                updated_at=user_model.updated_at,
                is_active=user_model.is_active,
                preferences=user_model.preferences
            ),
            tokens=TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=settings.access_token_expire_minutes * 60
            )
        )
        
    except Exception as e:
        logger.error(f"Firebase login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase authentication failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token"""
    
    try:
        # Verify refresh token
        payload = verify_token(token_data.refresh_token, "refresh")
        user_id = payload.get("sub")
        
        if not user_id:
            raise AuthenticationError("Invalid refresh token")
        
        # Verify user still exists and is active
        user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Create new access token
        access_token = create_access_token(
            data={"sub": str(user.user_id), "email": user.email}
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=token_data.refresh_token,  # Keep the same refresh token
            expires_in=settings.access_token_expire_minutes * 60
        )
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user)
):
    """Logout user (client should discard tokens)"""
    
    logger.info(f"User logged out: {current_user.email}")
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    
    return current_user


@router.get("/verify")
async def verify_token_endpoint(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Verify if a token is valid"""
    
    try:
        user = await get_current_user(credentials, db)
        return {
            "valid": True,
            "user_id": str(user.user_id),
            "email": user.email
        }
    except Exception:
        return {"valid": False}
