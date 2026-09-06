from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from agents.orchestrator import Orchestrator
from agents.llm_agent import PromptTooLargeError
import uvicorn
import logging
import json
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Headers that may carry credentials and must never be logged
SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie", "proxy-authorization", "x-api-key"}

def _safe_headers(headers) -> dict:
    """Return headers with credential-bearing values redacted."""
    return {
        k: ("<redacted>" if k.lower() in SENSITIVE_HEADERS else v)
        for k, v in headers.items()
    }

class AnalysisRequest(BaseModel):
    symbol: str
    days: int = 1
    # Evaluation aid: return the exact excerpts that went into the prompt.
    # Off by default so production responses are unchanged.
    include_prompt_context: bool = False

class AnalysisResponse(BaseModel):
    stock_data: dict
    news_articles: list
    stock_data_source: Optional[dict] = None
    articles_retrieved: int
    articles_used: int
    articles_used_indices: list
    summary: str
    timestamp: str
    # Omitted from the response unless include_prompt_context was requested.
    prompt_context: Optional[List[str]] = None

app = FastAPI(
    title="Stock News AI Agent",
    description="AI-powered stock news analysis and summarization system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing orchestrator...")
    await orchestrator.initialize()
    logger.info("Orchestrator initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Cleaning up resources...")
    await orchestrator.cleanup()
    logger.info("Cleanup completed")

@app.post("/analyze", response_model=AnalysisResponse, response_model_exclude_none=True)
async def analyze_stock(request: AnalysisRequest, raw_request: Request):
    try:
        # Log the incoming request
        logger.debug(f"Request headers: {_safe_headers(raw_request.headers)}")
        logger.debug(f"Received request body: {(await raw_request.body()).decode()}")
        logger.info(f"Parsed request: symbol={request.symbol}, days={request.days}")

        # Validate request data
        if not request.symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")
        if request.days < 1:
            raise HTTPException(status_code=400, detail="Days must be greater than 0")

        result = await orchestrator.process({
            "symbol": request.symbol,
            "days": request.days,
            "include_prompt_context": request.include_prompt_context,
            "timestamp": datetime.now().isoformat()
        })
        
        # Validate response data
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid response format")
        
        logger.info("Successfully processed request")
        return result
    except HTTPException:
        # Already carries an intended status code — don't remap it to a 500 below.
        raise
    except PromptTooLargeError as e:
        # Budgeting could not make the articles fit. Say so explicitly instead
        # of surfacing a llama-cpp ValueError.
        logger.error(f"Prompt budget exceeded: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    ) 