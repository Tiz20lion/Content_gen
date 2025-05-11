import os
import logging
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Content Generation Platform",
    description="API for extracting YouTube transcripts, crawling websites, and generating content using AI",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers after app initialization to avoid circular imports
from routers import auth, projects, content, videos

# Register routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(projects.router, prefix="/projects", tags=["Projects"])
app.include_router(content.router, prefix="/content", tags=["Content"])
app.include_router(videos.router, prefix="/videos", tags=["Videos"], methods=["POST", "GET"], response_model_include=["task_id", "message"])

# Import WebSocket manager
from core.manager import manager

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

@app.get("/")
async def root():
    return {"message": "YouTube Content Generation Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import pathlib
    import sys
    
    # Add the src directory to Python path
    root_path = pathlib.Path(__file__).parent.parent
    sys.path.append(str(root_path))
    
    # Run the application
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(root_path / "src")]
    )