import firebase_admin
from firebase_admin import credentials, auth
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from ..core.config import settings
from ..core.database import get_db
from ..models.database import User as UserModel
from ..models.schemas import User
from ..core.logging import logger


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Security
security = HTTPBearer()

# Firebase Admin initialization
firebase_app = None


def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global firebase_app
    if firebase_app is None and settings.firebase_credentials_path:
        try:
            cred = credentials.Certificate(settings.firebase_credentials_path)
            firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            firebase_app = None


class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Custom authorization error"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != token_type:
            raise JWTError("Invalid token type")
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


async def verify_firebase_token(id_token: str) -> Dict[str, Any]:
    """Verify Firebase ID token"""
    if not firebase_app:
        raise AuthenticationError("Firebase not configured")
    
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
        raise AuthenticationError("Invalid Firebase token")


async def get_or_create_user_from_firebase(firebase_user: Dict[str, Any], db: Session) -> UserModel:
    """Get or create user from Firebase authentication"""
    email = firebase_user.get("email")
    name = firebase_user.get("name", firebase_user.get("email", "Unknown"))
    firebase_uid = firebase_user.get("uid")
    
    if not email:
        raise AuthenticationError("Email not found in Firebase token")
    
    # Check if user exists
    user = db.query(UserModel).filter(UserModel.email == email).first()
    
    if not user:
        # Create new user
        user = UserModel(
            email=email,
            name=name,
            preferences={"firebase_uid": firebase_uid}
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created new user: {email}")
    else:
        # Update Firebase UID if not present
        if "firebase_uid" not in user.preferences:
            user.preferences = {**user.preferences, "firebase_uid": firebase_uid}
            db.commit()
    
    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    
    try:
        # Try JWT token first
        payload = verify_token(token, "access")
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("User ID not found in token")
        
        user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
        if not user:
            raise AuthenticationError("User not found")
        
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        
        return User.from_orm(user)
        
    except AuthenticationError:
        # If JWT fails, try Firebase token
        try:
            firebase_user = await verify_firebase_token(token)
            user_model = await get_or_create_user_from_firebase(firebase_user, db)
            return User.from_orm(user_model)
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError("Invalid authentication credentials")


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise AuthenticationError("User account is disabled")
    return current_user


def check_workspace_permission(
    user: User,
    workspace_owner_id: str,
    required_permission: str = "read"
) -> bool:
    """Check if user has permission to access workspace"""
    # For now, only owner has access
    # TODO: Implement role-based permissions for collaborative workspaces
    return str(user.user_id) == workspace_owner_id


async def verify_workspace_access(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    permission: str = "read"
) -> bool:
    """Verify user has access to workspace"""
    from ..models.database import Workspace as WorkspaceModel
    
    workspace = db.query(WorkspaceModel).filter(
        WorkspaceModel.workspace_id == workspace_id
    ).first()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    if not check_workspace_permission(current_user, str(workspace.owner_id), permission):
        raise AuthorizationError("Insufficient permissions to access this workspace")
    
    return True


class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 15):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests = {}
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limit"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        # Clean old entries
        if user_id in self.requests:
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if req_time > window_start
            ]
        else:
            self.requests[user_id] = []
        
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[user_id].append(now)
        return True


# Rate limiter instances
query_rate_limiter = RateLimiter(max_requests=50, window_minutes=15)
upload_rate_limiter = RateLimiter(max_requests=10, window_minutes=60)


def check_rate_limit(rate_limiter: RateLimiter, user: User):
    """Dependency to check rate limits"""
    if not rate_limiter.is_allowed(str(user.user_id)):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )


# Initialize Firebase on module import
initialize_firebase()
