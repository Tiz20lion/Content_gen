import os
import logging
from typing import Dict, Any, List, TypedDict, Annotated, Literal, Union, cast
import json
from datetime import datetime
from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")
SITE_TITLE = os.getenv("SITE_TITLE", "YouTube Content Generation Platform")

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_TITLE,
    }
)

# Define state types
class ContentState(TypedDict):
    transcript: str
    website_content: str
    cleaned_content: str
    generated_content: str
    quality_score: float
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    errors: List[str]
    status: str

# Node functions
def transcript_node(state: ContentState) -> ContentState:
    """Process transcript data"""
    try:
        logger.info("Processing transcript data")
        # In a real implementation, this would process the transcript
        # For now, we'll just pass it through
        return state
    except Exception as e:
        logger.error(f"Error in transcript node: {str(e)}")
        state["errors"].append(f"Transcript processing error: {str(e)}")
        state["status"] = "error"
        return state

def crawl_node(state: ContentState) -> ContentState:
    """Process website content data"""
    try:
        logger.info("Processing website content data")
        # In a real implementation, this would process the website content
        # For now, we'll just pass it through
        return state
    except Exception as e:
        logger.error(f"Error in crawl node: {str(e)}")
        state["errors"].append(f"Website content processing error: {str(e)}")
        state["status"] = "error"
        return state

def clean_node(state: ContentState) -> ContentState:
    """Clean and preprocess data"""
    try:
        logger.info("Cleaning and preprocessing data")
        # Combine transcript and website content
        combined_content = ""
        
        if state.get("transcript"):
            combined_content += f"Transcript:\n{state['transcript']}\n\n"
        
        if state.get("website_content"):
            combined_content += f"Website Content:\n{state['website_content']}"
        
        state["cleaned_content"] = combined_content
        state["status"] = "cleaned"
        return state
    except Exception as e:
        logger.error(f"Error in clean node: {str(e)}")
        state["errors"].append(f"Content cleaning error: {str(e)}")
        state["status"] = "error"
        return state

def generate_node(state: ContentState) -> ContentState:
    """Generate content using LLM"""
    try:
        logger.info("Generating content using LLM")
        
        # Prepare system message
        system_message = SystemMessage(
            content="You are a content creation assistant that generates high-quality, engaging content based on YouTube transcripts and website information."
        )
        
        # Prepare user message with cleaned content
        user_message = HumanMessage(
            content=f"Generate engaging content based on the following information:\n\n{state['cleaned_content']}\n\nPlease create a well-structured, informative, and engaging article that combines insights from both sources."
        )
        
        # Add messages to state
        if "messages" not in state:
            state["messages"] = []
        
        state["messages"].append(system_message)
        state["messages"].append(user_message)
        
        # Generate content using OpenRouter
        response = client.chat.completions.create(
            model="openai/gpt-4-turbo",  # Use appropriate model
            messages=[
                {"role": "system", "content": system_message.content},
                {"role": "user", "content": user_message.content}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        generated_content = response.choices[0].message.content
        
        # Add AI message to state
        ai_message = AIMessage(content=generated_content)
        state["messages"].append(ai_message)
        
        # Store generated content
        state["generated_content"] = generated_content
        state["status"] = "generated"
        
        return state
    except Exception as e:
        logger.error(f"Error in generate node: {str(e)}")
        state["errors"].append(f"Content generation error: {str(e)}")
        state["status"] = "error"
        return state

def evaluate_node(state: ContentState) -> ContentState:
    """Evaluate content quality"""
    try:
        logger.info("Evaluating content quality")
        
        # Prepare evaluation prompt
        evaluation_prompt = f"""Please evaluate the quality of the following content on a scale of 0 to 1, where 0 is poor quality and 1 is excellent quality. Consider factors like coherence, informativeness, engagement, and readability.\n\nContent to evaluate:\n{state['generated_content']}\n\nProvide only a numeric score between 0 and 1."""
        
        # Generate evaluation using OpenRouter
        response = client.chat.completions.create(
            model="openai/gpt-4-turbo",  # Use appropriate model
            messages=[
                {"role": "system", "content": "You are a content quality evaluator. You provide numeric scores between 0 and 1."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3,
            max_tokens=10
        )
        
        # Extract score from response
        score_text = response.choices[0].message.content.strip()
        try:
            # Try to extract a numeric score
            score = float(score_text)
            # Ensure score is between 0 and 1
            score = max(0, min(1, score))
        except ValueError:
            # If parsing fails, assign a default score
            logger.warning(f"Could not parse quality score: {score_text}. Using default.")
            score = 0.5
        
        state["quality_score"] = score
        return state
    except Exception as e:
        logger.error(f"Error in evaluate node: {str(e)}")
        state["errors"].append(f"Content evaluation error: {str(e)}")
        state["quality_score"] = 0.5  # Default score on error
        return state

# Define conditional routing
def should_regenerate(state: ContentState) -> Literal["regenerate", "complete"]:
    """Decide whether to regenerate content based on quality score"""
    # If there was an error, don't regenerate
    if state.get("status") == "error":
        return "complete"
    
    # If quality score is below threshold, regenerate
    if state.get("quality_score", 0) < 0.7:
        # Check if we've already tried regenerating too many times
        regeneration_count = len([msg for msg in state.get("messages", []) if isinstance(msg, AIMessage)])
        if regeneration_count >= 3:
            return "complete"  # Give up after 3 attempts
        return "regenerate"
    
    return "complete"

def regenerate_node(state: ContentState) -> ContentState:
    """Regenerate content with feedback"""
    try:
        logger.info("Regenerating content with feedback")
        
        # Prepare feedback message
        feedback_message = HumanMessage(
            content=f"The previous content received a quality score of {state['quality_score']}, which is below our threshold. Please regenerate the content with the following improvements:\n\n1. Ensure better coherence between paragraphs\n2. Add more specific details from the source material\n3. Improve the overall structure with clear sections\n4. Make the content more engaging for the reader"
        )
        
        # Add feedback message to state
        state["messages"].append(feedback_message)
        
        # Generate improved content
        messages_for_api = [
            {"role": "system", "content": "You are a content creation assistant that generates high-quality, engaging content based on YouTube transcripts and website information."},
        ]
        
        # Convert state messages to API format
        for msg in state["messages"]:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue
            
            messages_for_api.append({"role": role, "content": msg.content})
        
        # Generate content using OpenRouter
        response = client.chat.completions.create(
            model="openai/gpt-4-turbo",  # Use appropriate model
            messages=messages_for_api,
            temperature=0.7,
            max_tokens=2000
        )
        
        regenerated_content = response.choices[0].message.content
        
        # Add AI message to state
        ai_message = AIMessage(content=regenerated_content)
        state["messages"].append(ai_message)
        
        # Update generated content
        state["generated_content"] = regenerated_content
        state["status"] = "regenerated"
        
        return state
    except Exception as e:
        logger.error(f"Error in regenerate node: {str(e)}")
        state["errors"].append(f"Content regeneration error: {str(e)}")
        state["status"] = "error"
        return state

# Create the graph
def create_content_generation_graph():
    """Create the LangGraph for content generation"""
    # Create a new graph
    builder = StateGraph(ContentState)
    
    # Add nodes
    builder.add_node("transcript", transcript_node)
    builder.add_node("crawl", crawl_node)
    builder.add_node("clean", clean_node)
    builder.add_node("generate", generate_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("regenerate", regenerate_node)
    
    # Add edges
    builder.add_edge("transcript", "crawl")
    builder.add_edge("crawl", "clean")
    builder.add_edge("clean", "generate")
    builder.add_edge("generate", "evaluate")
    
    # Add conditional edges
    builder.add_conditional_edges(
        "evaluate",
        should_regenerate,
        {
            "regenerate": "regenerate",
            "complete": END
        }
    )
    
    # Add edge from regenerate back to evaluate
    builder.add_edge("regenerate", "evaluate")
    
    # Set the entry point
    builder.set_entry_point("transcript")
    
    # Create checkpointer for state persistence
    checkpointer = SqliteSaver("./langgraph_checkpoints.db")
    
    # Compile the graph
    return builder.compile(checkpointer=checkpointer)

# Function to run the graph
async def run_content_generation(transcript: str, website_content: str, project_id: str) -> Dict[str, Any]:
    """Run the content generation graph"""
    try:
        # Initialize state
        initial_state = ContentState(
            transcript=transcript,
            website_content=website_content,
            cleaned_content="",
            generated_content="",
            quality_score=0.0,
            messages=[],
            errors=[],
            status="started"
        )
        
        # Create graph
        graph = create_content_generation_graph()
        
        # Run the graph with persistence
        # The checkpoint_id allows resuming from failures
        checkpoint_id = f"project_{project_id}_{datetime.utcnow().isoformat()}"
        result = await graph.ainvoke(initial_state, checkpoint_id=checkpoint_id)
        
        # Return the final state
        return {
            "project_id": project_id,
            "generated_content": result["generated_content"],
            "quality_score": result["quality_score"],
            "status": result["status"],
            "errors": result["errors"]
        }
    except Exception as e:
        logger.error(f"Error running content generation graph: {str(e)}")
        return {
            "project_id": project_id,
            "generated_content": "",
            "quality_score": 0.0,
            "status": "error",
            "errors": [f"Graph execution error: {str(e)}"] 
        }