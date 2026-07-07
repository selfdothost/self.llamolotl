"""
FastAPI server for controlling training jobs (HF Trainer + DeepSpeed + PEFT).
Runs on port 8093, manages job lifecycle and progress monitoring.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .state import (
    API_VERSION,
    _ensure_dirs,
    _load_jobs,
    _load_pipeline_tasks,
    _poll_jobs,
    _try_start_next_pending,
)
from .routers import jobs, models, pipeline, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown."""
    _ensure_dirs()
    _load_jobs()
    _load_pipeline_tasks()
    _try_start_next_pending()
    poll_task = asyncio.create_task(_poll_jobs())
    yield
    poll_task.cancel()


app = FastAPI(title="Training API", version=API_VERSION, lifespan=lifespan)

app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(pipeline.router)
app.include_router(system.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8093, log_level="info")
