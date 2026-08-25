"""
Syllabus Slayer AI Learning Platform — FastAPI Backend
Production-ready server with AI/ML integration
"""

import logging
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from routes import auth, files, ai_features, progress
from utils.database import connect_db, close_db
from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("🚀 Syllabus Slayer backend starting up…")
    await connect_db()
    logger.info("✅ MongoDB connected")
    yield
    logger.info("🛑 Syllabus Slayer backend shutting down…")
    await close_db()


app = FastAPI(
    title="Syllabus Slayer AI Learning Platform",
    description="Production-grade backend for the Syllabus Slayer AI study workspace",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware ──────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timer(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{ms:.1f}ms"
    logger.debug("%s %s → %s (%.1fms)", request.method, request.url.path, response.status_code, ms)
    return response


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth.router,         prefix="/auth",     tags=["Authentication"])
app.include_router(files.router,        prefix="",          tags=["File Upload"])
app.include_router(ai_features.router,  prefix="",          tags=["AI Features"])
app.include_router(progress.router,     prefix="",          tags=["Progress"])


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "Syllabus Slayer API", "version": "1.0.0"}


# ── Global error handler ───────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
