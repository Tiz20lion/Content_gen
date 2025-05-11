from redis import Redis

# Initialize Redis MCP Server
redis_client = Redis(host='localhost', port=6379, db=0)

# Replace Celery tasks with Redis MCP Server tasks
import os
import logging
from celery import Celery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery app
celery_app = Celery(
    "youtube_content_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks",  # Include task modules
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit for tasks
    worker_max_tasks_per_child=200,  # Restart worker after 200 tasks
    broker_connection_retry_on_startup=True,
    # Retry settings
    task_acks_late=True,  # Tasks are acknowledged after execution
    task_reject_on_worker_lost=True,  # Reject tasks if worker is lost
    # Rate limiting
    task_default_rate_limit="10/m",  # Default rate limit for tasks
    # Exponential backoff for retries
    task_default_retry_delay=60,  # 1 minute initial delay
    task_max_retries=5,  # Maximum 5 retries
)

# Define task routes
celery_app.conf.task_routes = {
    "app.tasks.extract_content": {"queue": "extraction"},
    "app.tasks.generate_content": {"queue": "generation"},
    "app.tasks.create_video": {"queue": "video"},
}

# Define periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-old-tasks": {
        "task": "app.tasks.cleanup_old_tasks",
        "schedule": 86400.0,  # Run daily
    },
}

if __name__ == "__main__":
    celery_app.start()