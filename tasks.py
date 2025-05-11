import os
import logging
import time
import json
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from celery import Task
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from datetime import datetime, timedelta
from app.celery_worker import celery_app
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from CONFIG import (
    MAX_CONCURRENT_REQUESTS,
    CHUNK_SIZE,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    MAX_PAGES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# HeyGen API configuration
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_API_URL = os.getenv("HEYGEN_API_URL", "https://api.heygen.com/v1")

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")
SITE_TITLE = os.getenv("SITE_TITLE", "YouTube Content Generation Platform")

# Base task class with error handling and retry logic
class BaseTask(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': MAX_RETRIES}
    retry_backoff = True
    retry_backoff_max = 600  # 10 minutes max
    retry_jitter = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        # Try to update project status on failure
        try:
            if args and len(args) >= 2:
                project_id = args[0]
                user_id = args[1]
                self.update_project_status(project_id, "failed")
                self.send_error_notification(user_id, project_id, str(exc))
        except Exception as e:
            logger.error(f"Error updating project status on failure: {e}")
    
    def update_project_status(self, project_id: str, status: str):
        try:
            httpx.patch(
                f"{SUPABASE_URL}/rest/v1/projects",
                params={"id": f"eq.{project_id}"},
                json={"status": status},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json"
                }
            )
        except Exception as e:
            logger.error(f"Error updating project status: {e}")
    
    def send_error_notification(self, user_id: str, project_id: str, error_message: str):
        try:
            # This would typically send a WebSocket message or create a notification in the database
            # For now, we'll just log it
            logger.info(f"Error notification for user {user_id}, project {project_id}: {error_message}")
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

# Helper class for content cleaning
class ContentCleaner:
    @staticmethod
    def clean_html_content(content: str) -> str:
        """Clean HTML content by removing unwanted elements and normalizing text."""
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Remove extra whitespace and normalize line breaks
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Remove specific unwanted patterns
        patterns_to_remove = [
            r'var\(--.+?\)',  # CSS variable patterns
            r'\.drop-zone-draggable\s+.*?{.*?}',  # Drop zone CSS
            r'@media.*?}',  # Media queries
            r'#.*?{.*?}',  # ID-based CSS rules
            r'\.[a-zA-Z-]+\s*{[^}]*}',  # Class-based CSS rules
            r'undefined',  # Remove undefined values
            r'\s+px',  # Remove px units
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        return text.strip()

    @staticmethod
    def chunk_content(content: str, chunk_size: int = 1000) -> List[str]:
        """Split content into manageable chunks."""
        words = content.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

# Task for extracting YouTube transcript
@celery_app.task(base=BaseTask, bind=True, name="app.tasks.extract_youtube_transcript")
def extract_youtube_transcript(self, project_id: str, youtube_url: str, user_id: str):
    """Extract transcript from YouTube video and store in Supabase"""
    logger.info(f"Extracting transcript for project {project_id}, URL: {youtube_url}")
    
    try:
        # Extract video ID from URL
        match = re.search(r"[?&]v=([^&]+)", youtube_url)
        if not match:
            raise ValueError("Invalid YouTube URL format")
        
        video_id = match.group(1)
        
        # Get transcript from YouTube API
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([snippet["text"] for snippet in transcript])
        
        # Clean and chunk transcript
        cleaner = ContentCleaner()
        cleaned_text = cleaner.clean_html_content(transcript_text)
        chunks = cleaner.chunk_content(cleaned_text, CHUNK_SIZE)
        
        # Store in Supabase
        response = httpx.post(
            f"{SUPABASE_URL}/rest/v1/transcripts",
            json={
                "project_id": project_id,
                "transcript_text": cleaned_text,
                "chunks": json.dumps(chunks),
                "created_at": datetime.utcnow().isoformat()
            },
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }
        )
        
        if response.status_code != 201:
            raise Exception(f"Failed to store transcript: {response.text}")
        
        # Update project status
        self.update_project_status(project_id, "transcript_extracted")
        
        logger.info(f"Transcript extraction completed for project {project_id}")
        return {"status": "success", "project_id": project_id}
    
    except Exception as e:
        logger.error(f"Error extracting transcript: {str(e)}")
        # Update project status
        self.update_project_status(project_id, "transcript_failed")
        raise

# Task for crawling website content
@celery_app.task(base=BaseTask, bind=True, name="app.tasks.crawl_website")
def crawl_website(self, project_id: str, website_url: str, user_id: str):
    """Crawl website content and store in Supabase"""
    logger.info(f"Crawling website for project {project_id}, URL: {website_url}")
    
    try:
        # Set up async crawler
        browser_config = BrowserConfig(
            headless=True,
            ignore_https_errors=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.NEVER,
            timeout=REQUEST_TIMEOUT
        )
        
        # Run the async crawler in a synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_crawler():
            crawler = AsyncWebCrawler(browser_config)
            
            # Try to get URLs from sitemap first
            sitemap_url = urljoin(website_url, '/sitemap.xml')
            urls_to_crawl = set()
            
            try:
                result = await crawler.arun(url=sitemap_url, config=crawler_config)
                if result.success and result.html:
                    soup = BeautifulSoup(result.html, 'xml')
                    urls = {loc.text for loc in soup.find_all('loc')}
                    base_domain = urlparse(website_url).netloc
                    urls_to_crawl = {url for url in urls if urlparse(url).netloc == base_domain}
            except Exception as e:
                logger.warning(f"Could not fetch sitemap: {e}")
            
            # If no URLs from sitemap, discover by crawling
            if not urls_to_crawl:
                discovered_urls = set()
                to_visit = {website_url}
                visited = set()
                base_domain = urlparse(website_url).netloc
                
                while to_visit and len(discovered_urls) < MAX_PAGES:
                    current_url = to_visit.pop()
                    if current_url in visited:
                        continue
                    
                    try:
                        result = await crawler.arun(url=current_url, config=crawler_config)
                        visited.add(current_url)
                        discovered_urls.add(current_url)
                        
                        if result.success and result.html:
                            soup = BeautifulSoup(result.html, 'html.parser')
                            for link in soup.find_all('a', href=True):
                                url = urljoin(website_url, link['href'])
                                if (urlparse(url).netloc == base_domain and 
                                    url not in visited and 
                                    not url.endswith(('.pdf', '.jpg', '.png', '.gif'))):
                                    to_visit.add(url)
                    except Exception as e:
                        logger.warning(f"Error crawling {current_url}: {e}")
                
                urls_to_crawl = discovered_urls
            
            # Limit the number of URLs to crawl
            urls_to_crawl = list(urls_to_crawl)[:MAX_PAGES]
            
            # Crawl the URLs in parallel
            all_content = ""
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            
            async def crawl_url(url):
                async with semaphore:
                    try:
                        result = await crawler.arun(url=url, config=crawler_config)
                        if result.success and result.html:
                            return result.html
                    except Exception as e:
                        logger.warning(f"Error crawling {url}: {e}")
                    return ""
            
            tasks = [crawl_url(url) for url in urls_to_crawl]
            results = await asyncio.gather(*tasks)
            
            # Combine and clean content
            all_content = "\n\n".join([content for content in results if content])
            cleaner = ContentCleaner()
            cleaned_content = cleaner.clean_html_content(all_content)
            chunks = cleaner.chunk_content(cleaned_content, CHUNK_SIZE)
            
            await crawler.aclose()
            return cleaned_content, chunks
        
        cleaned_content, chunks = loop.run_until_complete(run_crawler())
        loop.close()
        
        # Store in Supabase
        response = httpx.post(
            f"{SUPABASE_URL}/rest/v1/website_contents",
            json={
                "project_id": project_id,
                "url": website_url,
                "content": cleaned_content,
                "chunks": json.dumps(chunks),
                "created_at": datetime.utcnow().isoformat()
            },
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }
        )
        
        if response.status_code != 201:
            raise Exception(f"Failed to store website content: {response.text}")
        
        # Update project status
        self.update_project_status(project_id, "website_crawled")
        
        logger.info(f"Website crawling completed for project {project_id}")
        return {"status": "success", "project_id": project_id}
    
    except Exception as e:
        logger.error(f"Error crawling website: {str(e)}")
        # Update project status
        self.update_project_status(project_id, "website_crawl_failed")
        raise

# Task for generating content using LangGraph
@celery_app.task(base=BaseTask, bind=True, name="app.tasks.generate_content")
def generate_content(self, project_id: str, user_id: str):
    """Generate content using LangGraph AI pipeline"""
    logger.info(f"Generating content for project {project_id}")
    
    try:
        # Get project information
        project_response = httpx.get(
            f"{SUPABASE_URL}/rest/v1/projects",
            params={"id": f"eq.{project_id}"},
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
        )
        
        if project_response.status_code != 200 or not project_response.json():
            raise Exception("Project not found")
        
        project = project_response.json()[0]
        
        # Get transcript if available
        transcript_response = httpx.get(
            f"{SUPABASE_URL}/rest/v1/transcripts",
            params={"project_id": f"eq.{project_id}"},
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
        )
        
        transcript = None
        if transcript_response.status_code == 200 and transcript_response.json():
            transcript = transcript_response.json()[0]
        
        # Get website content if available
        website_response = httpx.get(
            f"{SUPABASE_URL}/rest/v1/website_contents",
            params={"project_id": f"eq.{project_id}"},
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
        )
        
        website_content = None
        if website_response.status_code == 200 and website_response.json():
            website_content = website_response.json()[0]
        
        if not transcript and not website_content:
            raise Exception("No transcript or website content available for content generation")
        
        # Initialize OpenAI client for OpenRouter
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_TITLE,
            }
        )
        
        # Prepare prompt based on available content
        prompt = "Generate engaging content based on the following information:\n\n"
        
        if transcript:
            prompt += f"YouTube Transcript:\n{transcript['transcript_text']}\n\n"
        
        if website_content:
            prompt += f"Website Content:\n{website_content['content']}\n\n"
        
        prompt += "\nPlease create a well-structured, informative, and engaging article that combines insights from both sources."
        
        # Generate content using OpenRouter
        response = client.chat.completions.create(
            model="openai/gpt-4-turbo",  # Use appropriate model
            messages=[
                {"role": "system", "content": "You are a content creation assistant that generates high-quality, engaging content based on YouTube transcripts and website information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        generated_content = response.choices[0].message.content
        
        # Store generated content in Supabase
        content_response = httpx.post(
            f"{SUPABASE_URL}/rest/v1/generated_contents",
            json={
                "project_id": project_id,
                "content": generated_content,
                "created_at": datetime.utcnow().isoformat()
            },
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }
        )
        
        if content_response.status_code != 201:
            raise Exception(f"Failed to store generated content: {content_response.text}")
        
        # Update project status
        self.update_project_status(project_id, "content_generated")
        
        logger.info(f"Content generation completed for project {project_id}")
        return {"status": "success", "project_id": project_id}
    
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        # Update project status
        self.update_project_status(project_id, "content_generation_failed")
        raise

# Task for creating video using HeyGen API
@celery_app.task(base=BaseTask, bind=True, name="app.tasks.create_video")
def create_video(self, project_id: str, user_id: str, avatar_id: Optional[str] = None, voice_id: Optional[str] = None, resolution: str = "720p"):
    """Create video using HeyGen API"""
    logger.info(f"Creating video for project {project_id}")
    
    try:
        # Get generated content
        content_response = httpx.get(
            f"{SUPABASE_URL}/rest/v1/generated_contents",
            params={"project_id": f"eq.{project_id}"},
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
        )
        
        if content_response.status_code != 200 or not content_response.json():
            raise Exception("No generated content available for video creation")
        
        content = content_response.json()[0]
        
        # Default avatar and voice if not provided
        if not avatar_id:
            avatar_id = "default_avatar"  # Replace with actual default avatar ID
        
        if not voice_id:
            voice_id = "default_voice"  # Replace with actual default voice ID
        
        # Prepare script for HeyGen
        script = content["content"]
        
        # Call HeyGen API to create video
        # This is a placeholder for the actual HeyGen API call
        # In a real implementation, you would use their API documentation
        heygen_response = httpx.post(
            f"{HEYGEN_API_URL}/videos",
            json={
                "script": script,
                "avatar_id": avatar_id,
                "voice_id": voice_id,
                "resolution": resolution
            },
            headers={
                "Authorization": f"Bearer {HEYGEN_API_KEY}",
                "Content-Type": "application/json"
            }
        )
        
        if heygen_response.status_code not in (200, 201, 202):
            raise Exception(f"Failed to create video with HeyGen: {heygen_response.text}")
        
        heygen_data = heygen_response.json()
        video_id = heygen_data.get("id")
        
        # Store video information in Supabase
        video_response = httpx.post(
            f"{SUPABASE_URL}/rest/v1/videos",
            json={
                "project_id": project_id,
                "heygen_video_id": video_id,
                "status": "processing",
                "created_at": datetime.utcnow().isoformat()
            },
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }
        )
        
        if video_response.status_code != 201:
            raise Exception(f"Failed to store video information: {video_response.text}")
        
        # Update project status
        self.update_project_status(project_id, "video_processing")
        
        # Poll HeyGen API for video status
        max_polls = 30  # Maximum number of status checks
        poll_interval = 60  # Seconds between checks
        
        for i in range(max_polls):
            time.sleep(poll_interval)
            
            status_response = httpx.get(
                f"{HEYGEN_API_URL}/videos/{video_id}",
                headers={
                    "Authorization": f"Bearer {HEYGEN_API_KEY}"
                }
            )
            
            if status_response.status_code != 200:
                logger.warning(f"Failed to get video status: {status_response.text}")
                continue
            
            status_data = status_response.json()
            video_status = status_data.get("status")
            
            if video_status == "completed":
                video_url = status_data.get("video_url")
                
                # Update video information in Supabase
                update_response = httpx.patch(
                    f"{SUPABASE_URL}/rest/v1/videos",
                    params={"project_id": f"eq.{project_id}"},
                    json={
                        "status": "completed",
                        "video_url": video_url,
                        "updated_at": datetime.utcnow().isoformat()
                    },
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                
                if update_response.status_code != 200:
                    logger.error(f"Failed to update video status: {update_response.text}")
                
                # Update project status
                self.update_project_status(project_id, "completed")
                
                logger.info(f"Video creation completed for project {project_id}")
                return {"status": "success", "project_id": project_id, "video_url": video_url}
            
            elif video_status == "failed":
                # Update video status in Supabase
                update_response = httpx.patch(
                    f"{SUPABASE_URL}/rest/v1/videos",
                    params={"project_id": f"eq.{project_id}"},
                    json={
                        "status": "failed",
                        "updated_at": datetime.utcnow().isoformat()
                    },
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                
                # Update project status
                self.update_project_status(project_id, "video_failed")
                
                raise Exception("Video creation failed in HeyGen")
        
        # If we've reached here, the video is still processing after max polls
        logger.warning(f"Video still processing after {max_polls} checks for project {project_id}")
        return {"status": "processing", "project_id": project_id}
    
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        # Update project status
        self.update_project_status(project_id, "video_failed")
        raise

# Task for cleaning up old tasks
@celery_app.task(name="app.tasks.cleanup_old_tasks")
def cleanup_old_tasks():
    """Clean up old tasks and temporary files"""
    logger.info("Cleaning up old tasks")
    
    try:
        # Get tasks older than 30 days
        cutoff_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
        
        # This is a placeholder for actual cleanup logic
        # In a real implementation, you would clean up temporary files and database records
        
        logger.info("Cleanup completed successfully")
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise