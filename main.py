from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime
from agents.orchestrator import Orchestrator
import uvicorn
import logging
import json
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisRequest(BaseModel):
    symbol: str
    days: int = 1

class AnalysisResponse(BaseModel):
    stock_data: dict
    news_articles: list
    summary: str
    timestamp: str

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
    await orchestrator.cleanup()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest, raw_request: Request):
    try:
        # Log the incoming request
        body = await raw_request.body()
        logger.info(f"Request headers: {raw_request.headers}")
        logger.info(f"Received request body: {body.decode()}")
        logger.info(f"Parsed request: symbol={request.symbol}, days={request.days}")

        # Validate request data
        if not request.symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")
        if request.days < 1:
            raise HTTPException(status_code=400, detail="Days must be greater than 0")

        result = await orchestrator.process({
            "symbol": request.symbol,
            "days": request.days,
            "timestamp": datetime.now().isoformat()
        })
        
        # Validate response data
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid response format")
        
        logger.info("Successfully processed request")
        return result
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