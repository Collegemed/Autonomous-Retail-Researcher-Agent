"""
Autonomous Retail Researcher - FastAPI Backend
Main application entry point with all routes and middleware.
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from database import init_db, save_query, get_history, get_cached_result, save_cache
from agents import ResearchOrchestrator
from rag import RAGMemory

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("retail-researcher")

# ─── Lifespan (startup/shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Retail Researcher API...")
    init_db()
    app.state.rag = RAGMemory()
    app.state.orchestrator = ResearchOrchestrator(app.state.rag)
    logger.info("All systems initialized.")
    yield
    logger.info("Shutting down...")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Retail Researcher API",
    description="Multi-agent retail research system with RAG memory",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str
    use_cache: bool = True

class ResearchResponse(BaseModel):
    query: str
    result: dict
    cached: bool
    execution_time_seconds: float

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "Autonomous Retail Researcher"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """
    Main research endpoint.
    Runs multi-agent pipeline: Research → Analysis → Summary → Storage
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 chars).")

    start = time.time()

    # Check cache first
    if req.use_cache:
        cached = get_cached_result(query)
        if cached:
            logger.info(f"Cache hit for query: {query[:60]}...")
            return ResearchResponse(
                query=query,
                result=cached,
                cached=True,
                execution_time_seconds=round(time.time() - start, 3),
            )

    try:
        logger.info(f"Running research pipeline for: {query[:60]}...")
        result = await app.state.orchestrator.run(query)

        elapsed = round(time.time() - start, 3)
        result["execution_time"] = elapsed

        # Persist to DB and cache
        save_query(query, result)
        save_cache(query, result)

        # Store in RAG vector memory
        app.state.rag.store(query, result.get("summary", ""))

        logger.info(f"Research completed in {elapsed}s")
        return ResearchResponse(
            query=query,
            result=result,
            cached=False,
            execution_time_seconds=elapsed,
        )

    except TimeoutError as e:
        logger.error(f"Timeout: {e}")
        raise HTTPException(status_code=504, detail=f"Research timed out: {str(e)}")
    except Exception as e:
        logger.exception(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@app.get("/history")
async def history(limit: int = 20, offset: int = 0):
    """Returns paginated query history from the database."""
    try:
        rows = get_history(limit=limit, offset=offset)
        return {"history": rows, "count": len(rows)}
    except Exception as e:
        logger.exception(f"History fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """Clears the query cache (for testing purposes)."""
    from database import clear_cache as db_clear
    db_clear()
    return {"status": "cache cleared"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)