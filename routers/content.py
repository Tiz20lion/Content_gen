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

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase credentials not found in environment variables")

# Models
class ContentExtractRequest(BaseModel):
    project_id: str
    youtube_url: Optional[str] = None
    website_url: Optional[str] = None

class ContentGenerateRequest(BaseModel):
    project_id: str

class ContentResponse(BaseModel):
    task_id: str
    message: str

# Routes
@router.post("/youtube", response_model=ContentResponse)
async def extract_youtube_transcript(
    request: ContentExtractRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Extract transcript from YouTube video"""
    if not request.youtube_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="YouTube URL is required"
        )
    
    try:
        # Check if project exists and belongs to user
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{request.project_id}"},
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
        
        # Update project with YouTube URL if not already set
        if not project.get("youtube_url"):
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/projects",
                    params={"id": f"eq.{request.project_id}"},
                    json={"youtube_url": request.youtube_url},
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json"
                    }
                )
        
        # Start Celery task for transcript extraction
        task = celery_app.send_task(
            "app.tasks.extract_youtube_transcript",
            args=[request.project_id, request.youtube_url, current_user["id"]],
            retry=True,
            retry_policy={
                "max_retries": 3,
                "interval_start": 0,
                "interval_step": 0.2,
                "interval_max": 0.5,
            }
        )
        
        # Send real-time update
        background_tasks.add_task(
            manager.send_update,
            current_user["id"],
            {
                "type": "transcript_extraction_started",
                "project_id": request.project_id,
                "task_id": task.id
            }
        )
        
        return {
            "task_id": task.id,
            "message": "YouTube transcript extraction started"
        }
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error extracting YouTube transcript: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting YouTube transcript: {str(e)}"
        )

@router.post("/web", response_model=ContentResponse)
async def crawl_website(
    request: ContentExtractRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Crawl website content"""
    if not request.website_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Website URL is required"
        )
    
    try:
        # Check if project exists and belongs to user
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{request.project_id}"},
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
        
        # Update project with website URL if not already set
        if not project.get("website_url"):
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/projects",
                    params={"id": f"eq.{request.project_id}"},
                    json={"website_url": request.website_url},
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json"
                    }
                )
        
        # Start Celery task for website crawling
        task = celery_app.send_task(
            "app.tasks.crawl_website",
            args=[request.project_id, request.website_url, current_user["id"]],
            retry=True,
            retry_policy={
                "max_retries": 3,
                "interval_start": 0,
                "interval_step": 0.2,
                "interval_max": 0.5,
            }
        )
        
        # Send real-time update
        background_tasks.add_task(
            manager.send_update,
            current_user["id"],
            {
                "type": "website_crawling_started",
                "project_id": request.project_id,
                "task_id": task.id
            }
        )
        
        return {
            "task_id": task.id,
            "message": "Website crawling started"
        }
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error crawling website: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error crawling website: {str(e)}"
        )

@router.post("/generate", response_model=ContentResponse)
async def generate_content(
    request: ContentGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate content using LangGraph AI pipeline"""
    try:
        # Check if project exists and belongs to user
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{request.project_id}"},
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
        
        # Check if transcript and website content exist
        async with httpx.AsyncClient() as client:
            transcript_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/transcripts",
                params={"project_id": f"eq.{request.project_id}"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
            
            website_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/website_contents",
                params={"project_id": f"eq.{request.project_id}"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
            
            if (transcript_response.status_code != 200 or not transcript_response.json()) and \
               (website_response.status_code != 200 or not website_response.json()):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No transcript or website content available for content generation"
                )
        
        # Update project status
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{request.project_id}"},
                json={"status": "generating"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json"
                }
            )
        
        # Start Celery task for content generation
        task = celery_app.send_task(
            "app.tasks.generate_content",
            args=[request.project_id, current_user["id"]],
            retry=True,
            retry_policy={
                "max_retries": 3,
                "interval_start": 0,
                "interval_step": 0.2,
                "interval_max": 0.5,
            }
        )
        
        # Send real-time update
        background_tasks.add_task(
            manager.send_update,
            current_user["id"],
            {
                "type": "content_generation_started",
                "project_id": request.project_id,
                "task_id": task.id
            }
        )
        
        return {
            "task_id": task.id,
            "message": "Content generation started"
        }
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content: {str(e)}"
        )