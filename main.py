from fastapi import FastAPI, Header, Depends, HTTPException, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import os
import time
import asyncio
from contextlib import asynccontextmanager

from processor import load_and_index_document
from model import query_document

# Startup/shutdown handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown

app = FastAPI(
    title="Advanced RAG System",
    description="Optimized RAG system with rank fusion",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth setup
api_key_header = APIKeyHeader(name="Authorization")

def verify_token(authorization: str = Depends(api_key_header)):
    """Verify API token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")
    return authorization

# Request/Response models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]
    
    @validator('questions')
    def questions_not_empty(cls, v):
        if not v:
            raise ValueError('Questions list cannot be empty')
        if len(v) > 20:
            raise ValueError('Maximum 20 questions allowed')
        return v
    
    @validator('documents')
    def documents_valid_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Documents must be a valid URL')
        return v

class QueryResponse(BaseModel):
    status: str
    answers: Optional[List[str]] = None
    message: Optional[str] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time()
    )

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Depends(verify_token)
):
    """Main query processing endpoint with optimized RAG"""
    start_time = time.time()
    
    try:
        # Load and index document
        # chunks = await asyncio.to_thread(load_and_index_document, request.documents)
        
        # Process questions SEQUENTIALLY - one at a time
        results = []
        
        for i, question in enumerate(request.questions, 1):
            print(f"\nQuestion {i}: {question}")
            
            try:
                # Process ONE question at a time and wait for complete response
                answer = await asyncio.to_thread(query_document, question)
                
                print(f"Answer {i}: {answer}")
                
                results.append(answer)
                
            except Exception as qe:
                error_msg = f"Error processing question '{question}': {str(qe)}"
                print(f"Error: {error_msg}")
                results.append(error_msg)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            status="success",
            answers=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Internal error: {str(e)}"
        
        return QueryResponse(
            status="error",
            message=error_msg,
            processing_time=processing_time
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )