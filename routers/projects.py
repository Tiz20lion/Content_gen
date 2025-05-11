import os
import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import uuid
from datetime import datetime
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
class ProjectCreate(BaseModel):
    name: str
    youtube_url: Optional[str] = None
    website_url: Optional[str] = None

class Project(BaseModel):
    id: str
    user_id: str
    name: str
    youtube_url: Optional[str] = None
    website_url: Optional[str] = None
    created_at: datetime
    status: str = "pending"

# Helper functions for Supabase database operations
async def create_project_in_db(project_data: Dict[str, Any]):
    """Create a new project in Supabase"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SUPABASE_URL}/rest/v1/projects",
                json=project_data,
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
            )
            
            if response.status_code == 201:
                return response.json()[0]
            else:
                logger.error(f"Project creation failed: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        return None

async def get_projects_for_user(user_id: str):
    """Get all projects for a user"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"user_id": f"eq.{user_id}", "order": "created_at.desc"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get projects: {response.text}")
                return []
    except Exception as e:
        logger.error(f"Error getting projects: {str(e)}")
        return []

async def get_project_by_id(project_id: str):
    """Get a project by ID"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{project_id}"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
            
            if response.status_code == 200 and response.json():
                return response.json()[0]
            else:
                return None
    except Exception as e:
        logger.error(f"Error getting project: {str(e)}")
        return None

async def update_project_status(project_id: str, status: str):
    """Update project status"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{project_id}"},
                json={"status": status},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
            )
            
            if response.status_code == 200:
                return response.json()[0]
            else:
                logger.error(f"Failed to update project status: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Error updating project status: {str(e)}")
        return None

# Routes
@router.post("/", response_model=Project)
async def create_project(
    project: ProjectCreate,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new project and trigger content extraction"""
    try:
        # Create project in database
        project_id = str(uuid.uuid4())
        project_data = {
            "id": project_id,
            "user_id": current_user["id"],
            "name": project.name,
            "youtube_url": project.youtube_url,
            "website_url": project.website_url,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }
        
        db_project = await create_project_in_db(project_data)
        if not db_project:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create project"
            )
        
        # Trigger content extraction tasks
        if project.youtube_url or project.website_url:
            task = celery_app.send_task(
                "app.tasks.extract_content",
                args=[project_id, project.youtube_url, project.website_url, current_user["id"]]
            )
            logger.info(f"Content extraction task started: {task.id}")
            
            # Send real-time update
            background_tasks.add_task(
                manager.send_update,
                current_user["id"],
                {"type": "project_created", "project_id": project_id}
            )
        
        return Project(**db_project)
    
    except Exception as e:
        logger.error(f"Error in create_project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[Project])
async def get_projects(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get all projects for the current user"""
    projects = await get_projects_for_user(current_user["id"])
    return [Project(**project) for project in projects]

@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a project by ID"""
    project = await get_project_by_id(project_id)
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    if project["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this project"
        )
    
    return Project(**project)

@router.post("/{project_id}/generate")
async def generate_content(
    project_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Trigger content generation for a project"""
    project = await get_project_by_id(project_id)
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    if project["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this project"
        )
    
    # Update project status
    updated_project = await update_project_status(project_id, "generating")
    if not updated_project:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update project status"
        )
    
    # Trigger content generation task
    task = celery_app.send_task(
        "app.tasks.generate_content",
        args=[project_id, current_user["id"]]
    )
    logger.info(f"Content generation task started: {task.id}")
    
    # Send real-time update
    background_tasks.add_task(
        manager.send_update,
        current_user["id"],
        {"type": "generation_started", "project_id": project_id}
    )
    
    return {"message": "Content generation started", "task_id": task.id}