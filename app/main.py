"""
FastAPI application for Member QnA Service.

This module provides REST API endpoints for the question-answering system.
"""

import logging
from fastapi import FastAPI, HTTPException

from app.schemas import (
    QuestionRequest,
    AnswerResponse,
    HealthResponse,
    CacheClearResponse,
)
from langgraph_agent import QnAAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Member QnA Service",
    description="A question-answering service for member data using LangGraph",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize the QnA agent (singleton)
agent = QnAAgent()


@app.get("/")
async def root():
    """
    Root endpoint with service information.
    
    Returns:
        Service information including available endpoints
    """
    return {
        "service": "Member QnA Service",
        "version": "2.0.0",
        "description": "LangGraph-based question-answering system for member data",
        "endpoints": {
            "/ask": "POST - Ask a question about member data",
            "/health": "GET - Health check",
            "/warmup": "POST - Pre-load FAISS index (reduces first request latency)",
            "/clear-cache": "POST - Clear member data cache",
            "/docs": "GET - Interactive API documentation (Swagger)",
            "/redoc": "GET - Alternative API documentation (ReDoc)",
        },
        "performance": {
            "typical_latency": "1-3 seconds per request",
            "first_request": "2-4 seconds (loads FAISS index)",
            "tip": "Call /warmup after deployment to pre-load the index"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the service
    """
    return HealthResponse(status="healthy")


@app.post("/warmup")
async def warmup():
    """
    Warmup endpoint to pre-load the FAISS index into memory.
    
    Call this endpoint after deployment to avoid cold start latency
    on the first real user request.
    
    Returns:
        Status message indicating warmup completion
    """
    try:
        logger.info("Warming up agent (loading FAISS index)...")
        # Trigger a simple question to load the index
        agent.ask("warmup")
        logger.info("Warmup completed successfully")
        return {
            "status": "success",
            "message": "Agent warmed up successfully. FAISS index loaded into memory."
        }
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        return {
            "status": "partial",
            "message": f"Warmup completed with warnings: {str(e)}"
        }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a natural language question about member data.
    
    This endpoint uses a LangGraph agent to:
    1. Fetch relevant member data from the API
    2. Extract context related to the question
    3. Generate an accurate answer using LLM
    
    Performance characteristics:
    - First request: ~2-3 seconds (loads FAISS index from disk)
    - Subsequent requests: ~1-2 seconds (uses cached index)
    - Main latency: OpenAI API call (~1-2 seconds)
    
    Memory optimized for 512MB environments (Render starter pack).
    
    Args:
        request: QuestionRequest containing the question
        
    Returns:
        AnswerResponse with the generated answer
        
    Raises:
        HTTPException: If question is empty or processing fails
    """
    import time
    import gc
    
    start_time = time.time()
    
    try:
        logger.info(f"Received question: {request.question}")
        
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Get answer from agent
        answer = agent.ask(request.question)
        
        # MEMORY OPTIMIZATION: Force garbage collection after request
        gc.collect()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated answer in {elapsed_time:.2f}s: {answer[:100]}...")
        
        return AnswerResponse(answer=answer)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        # Force GC even on error to clean up partial state
        import gc
        gc.collect()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/clear-cache", response_model=CacheClearResponse)
async def clear_cache():
    """
    Clear the member data cache to force refresh from API.
    
    Use this endpoint when the member data has been updated
    and you want to fetch the latest information.
    
    Returns:
        CacheClearResponse with operation status
        
    Raises:
        HTTPException: If cache clearing fails
    """
    try:
        agent.clear_cache()
        logger.info("Cache cleared successfully")
        return CacheClearResponse(
            status="success",
            message="Cache cleared successfully"
        )
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

