import os
import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
from routers.auth import get_current_user
from celery_worker import celery_app
from core.manager import manager

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# HeyGen API configuration
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_API_URL = os.getenv("HEYGEN_API_URL", "https://api.heygen.com/v1")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase credentials not found in environment variables")

if not HEYGEN_API_KEY:
    logger.error("HeyGen API key not found in environment variables")

# Models
class VideoCreateRequest(BaseModel):
    project_id: str
    avatar_id: Optional[str] = None
    voice_id: Optional[str] = None
    resolution: Optional[str] = "720p"

class VideoResponse(BaseModel):
    task_id: str
    message: str

# Routes
@router.post("/{project_id}/create", response_model=VideoResponse)
async def create_video(
    project_id: str,
    request: VideoCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a video using HeyGen API"""
    try:
        # Check if project exists and belongs to user
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{project_id}"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
            
            if response.status_code != 200 or not response.json():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            project = response.json()[0]
            if project["user_id"] != current_user["id"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this project"
                )
        
        # Check if generated content exists
        async with httpx.AsyncClient() as client:
            content_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/generated_contents",
                params={"project_id": f"eq.{project_id}"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
            
            if content_response.status_code != 200 or not content_response.json():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No generated content available for video creation"
                )
        
        # Update project status
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{project_id}"},
                json={"status": "creating_video"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json"
                }
            )
        
        # Start Celery task for video creation
        task = celery_app.send_task(
            "app.tasks.create_video",
            args=[
                project_id,
                current_user["id"],
                request.avatar_id,
                request.voice_id,
                request.resolution
            ],
            retry=True,
            retry_policy={
                "max_retries": 5,  # More retries for external API
                "interval_start": 60,  # Start with 1 minute delay
                "interval_step": 120,  # Increase by 2 minutes each retry
                "interval_max": 600,  # Max 10 minutes between retries
            }
        )
        
        # Send real-time update
        background_tasks.add_task(
            manager.send_update,
            current_user["id"],
            {
                "type": "video_creation_started",
                "project_id": project_id,
                "task_id": task.id
            }
        )
        
        return {
            "task_id": task.id,
            "message": "Video creation started"
        }
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating video: {str(e)}"
        )

@router.get("/{project_id}")
async def get_video_status(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get video status for a project"""
    try:
        # Check if project exists and belongs to user
        async with httpx.AsyncClient() as client:
            project_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{project_id}"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
            
            if project_response.status_code != 200 or not project_response.json():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            project = project_response.json()[0]
            if project["user_id"] != current_user["id"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this project"
                )
        
        # Get video information
        async with httpx.AsyncClient() as client:
            video_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/videos",
                params={"project_id": f"eq.{project_id}"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
            
            if video_response.status_code != 200 or not video_response.json():
                return {"status": "not_started", "message": "No video has been created for this project"}
            
            video = video_response.json()[0]
            return {
                "status": video["status"],
                "video_url": video.get("video_url"),
                "created_at": video["created_at"],
                "updated_at": video.get("updated_at")
            }
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting video status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting video status: {str(e)}"
        )