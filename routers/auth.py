import os
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase credentials not found in environment variables")

# Models
class UserCreate(BaseModel):
    email: str
    password: str
    username: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Helper functions for Supabase auth
async def authenticate_user(email: str, password: str):
    """Authenticate user with Supabase"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
                json={"email": email, "password": password},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Authentication failed: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Error during authentication: {str(e)}")
        return None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {token}"
                }
            )
            
            if response.status_code == 200:
                user_data = response.json()
                return user_data
            else:
                raise credentials_exception
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}")
        raise credentials_exception

# Routes
@router.post("/register", response_model=Token)
async def register_user(user: UserCreate):
    """Register a new user"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SUPABASE_URL}/auth/v1/signup",
                json={
                    "email": user.email,
                    "password": user.password,
                    "data": {"username": user.username}
                },
                headers={
                    "apikey": SUPABASE_KEY,
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                auth_response = await authenticate_user(user.email, user.password)
                if auth_response:
                    return {
                        "access_token": auth_response["access_token"],
                        "token_type": "bearer"
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Registration successful but login failed"
                    )
            else:
                logger.error(f"Registration failed: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.text
                )
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user and return access token"""
    auth_response = await authenticate_user(form_data.username, form_data.password)
    
    if not auth_response:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "access_token": auth_response["access_token"],
        "token_type": "bearer"
    }

@router.get("/me")
async def read_users_me(current_user = Depends(get_current_user)):
    """Get current user information"""
    return current_user