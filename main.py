from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
from ai_agent_service import analyze_error, process_query
from git_repo_manager import GitRepoManager
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced XCode AI Coding Assistant",
    version="2.0.0",
    description="AI-powered coding assistant with collaborative DeepSeek + Gemini analysis"
)

# Configure templates
templates = Jinja2Templates(directory="templates")

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for CSS and JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize GitRepoManager
repo_manager = GitRepoManager()

# Health check endpoint
@app.get("/api/health")
async def health_check():
    status = {
        "status": "healthy",
        "repositories": len(repo_manager.get_repositories()),
        "total_files": repo_manager.get_total_files(),
        "context_files": repo_manager.get_context_files(),
        "critical_files": repo_manager.get_critical_files(),
        "last_sync": repo_manager.get_last_sync_time()
    }
    return JSONResponse(status)

# Repository management endpoints
@app.get("/api/repositories")
async def get_repositories():
    repos = repo_manager.get_repositories()
    return JSONResponse({"repositories": repos})

@app.post("/api/repositories/add")
async def add_repository(background_tasks: BackgroundTasks, name: str, url: str, branch: str = "main", access_token: str = None, sync_interval: int = 300):
    repo_data = {"name": name, "url": url, "branch": branch, "access_token": access_token, "sync_interval": sync_interval}
    background_tasks.add_task(repo_manager.add_repository, **repo_data)
    return JSONResponse({"success": True, "message": f"Repository {name} queued for addition"})

@app.post("/api/repositories/sync")
async def sync_repositories():
    repo_manager.sync_repositories()
    return JSONResponse({"success": True, "message": "Sync initiated"})

# Xcode error analysis endpoint
@app.post("/api/xcode/analyze-error")
async def analyze_xcode_error(background_tasks: BackgroundTasks, error_message: str, use_deepseek: bool = False, force_sync: bool = False):
    job_id = await analyze_error(error_message, use_deepseek, force_sync)
    return JSONResponse({"success": True, "job_id": job_id})

# General query endpoint
@app.post("/api/query")
async def submit_query(background_tasks: BackgroundTasks, query: str, use_deepseek: bool = False):
    job_id = await process_query(query, use_deepseek)
    return JSONResponse({"success": True, "job_id": job_id})

# Job status endpoint
@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    status = await asyncio.get_event_loop().run_in_executor(None, lambda: {"status": "completed", "result": {"message": "Sample result"}})  # Placeholder
    return JSONResponse(status)

# Root endpoint to serve index.html
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)