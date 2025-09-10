from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
from datetime import datetime
import json
import uuid
from collections import deque
import tempfile
from pathlib import Path
import time

# Import our enhanced modules
from git_repo_manager import GitRepoManager
from ai_agent_service import AIAgentService

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

# Initialize services with enhanced configuration
render_temp_dir = os.getenv('RENDER_TEMP_DIR', tempfile.gettempdir())
repo_base_path = os.path.join(render_temp_dir, "repos")

repo_manager = GitRepoManager(base_path=repo_base_path)
ai_agent = AIAgentService(
    gemini_api_key=os.getenv("GOOGLE_API_KEY"),
    deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Enhanced job storage with better management
job_results = {}
job_queue = deque()
MAX_JOBS_STORED = 50
RENDER_TIMEOUT = 25  # Keep under 30 second limit

# Background sync management
sync_in_progress = False
last_sync_attempt = None

# Root endpoint to serve index.html
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Enhanced periodic sync task
async def enhanced_periodic_sync():
    global sync_in_progress, last_sync_attempt
    
    while True:
        try:
            current_time = datetime.now()
            
            if not sync_in_progress and (
                not last_sync_attempt or 
                (current_time - last_sync_attempt).total_seconds() > 180  # 3 minutes
            ):
                sync_in_progress = True
                last_sync_attempt = current_time
                
                print(f"🔄 Starting enhanced periodic sync at {current_time}")
                
                # Use batch sync with timeout handling
                sync_results = await repo_manager.sync_all_repositories_batch(
                    batch_size=2,  # Small batches for timeout safety
                    max_duration=RENDER_TIMEOUT
                )
                
                # Update AI agent context with priority files
                context_update_count = 0
                for repo_name, (success, message, file_count) in sync_results.items():
                    if success and file_count > 0:
                        critical_files = await process_repository_files(repo_name, priority_only=True)
                        context_update_count += critical_files
                        
                        # Limit context updates to prevent timeout
                        if context_update_count > 50:
                            break
                
                print(f"✅ Enhanced sync completed: {len(sync_results)} repos, {context_update_count} files processed")
                
        except Exception as e:
            print(f"❌ Enhanced sync error: {str(e)}")
        finally:
            sync_in_progress = False
            
        # Wait 5 minutes before next sync
        await asyncio.sleep(300)

async def process_repository_files(repo_name: str, priority_only: bool = True) -> int:
    """Process repository files with priority handling"""
    try:
        files = repo_manager.list_files(repo_name)
        processed_count = 0
        
        # Process files in priority order
        for file_path in files[:30 if priority_only else 100]:  # Limit for timeout safety
            try:
                content = repo_manager.get_file_content(repo_name, file_path)
                if content and len(content) > 10 and not content.startswith("File too large"):
                    await ai_agent.update_file_context(repo_name, file_path, content)
                    processed_count += 1
                    
                    # Process in small batches to prevent timeout
                    if processed_count % 10 == 0:
                        await asyncio.sleep(0.1)  # Brief pause to prevent blocking
                        
            except Exception as e:
                print(f"❌ Error processing file {file_path}: {e}")
                continue
        
        return processed_count
        
    except Exception as e:
        print(f"❌ Error processing repository {repo_name}: {e}")
        return 0

# Enhanced job cleanup
async def enhanced_job_cleanup():
    """Enhanced job cleanup with better memory management"""
    while True:
        await asyncio.sleep(300)  # Clean up every 5 minutes
        try:
            current_time = datetime.now()
            jobs_to_remove = []
            
            for job_id, job_data in job_results.items():
                # Remove completed jobs older than 30 minutes
                time_threshold = 1800 if job_data.get('status') == 'completed' else 3600
                
                if (current_time - job_data['created_at']).total_seconds() > time_threshold:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                job_results.pop(job_id, None)
                
            # Ensure we don't exceed storage limits
            while len(job_results) > MAX_JOBS_STORED:
                oldest_job_id = min(job_results.keys(), 
                                  key=lambda k: job_results[k]['created_at'])
                job_results.pop(oldest_job_id, None)
                
            if jobs_to_remove:
                print(f"🗑️ Cleaned up {len(jobs_to_remove)} old jobs")
                
            # Clean up AI agent context periodically
            await ai_agent.refresh_context_if_needed()
                
        except Exception as e:
            print(f"❌ Job cleanup error: {str(e)}")

# Enhanced Pydantic models
class RepoConfig(BaseModel):
    name: str
    url: str
    branch: str = "main"
    access_token: Optional[str] = None
    sync_interval: int = 300

class XCodeErrorRequest(BaseModel):
    error_message: str
    use_deepseek: str  # Changed to str to handle "both"

class GeneralQueryRequest(BaseModel):
    query: str
    use_deepseek: str  # Changed to str to handle "both"

# Enhanced endpoint for adding repository
@app.post("/api/repositories/add")
async def add_repository(request: RepoConfig, background_tasks: BackgroundTasks):
    try:
        repo_manager.add_repository(
            name=request.name,
            url=request.url,
            branch=request.branch,
            access_token=request.access_token,
            sync_interval=request.sync_interval
        )
        background_tasks.add_task(repo_manager.clone_or_update_repo_with_timeout, request.name)
        return {
            "success": True,
            "message": f"Repository {request.name} added and sync started"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error adding repository: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Enhanced endpoint for syncing all repositories
@app.post("/api/repositories/sync")
async def sync_all_repositories(background_tasks: BackgroundTasks):
    repos = repo_manager.get_repositories()
    if not repos:
        raise HTTPException(status_code=404, detail="No repositories configured")

    for repo_name in repos:
        background_tasks.add_task(repo_manager.clone_or_update_repo_with_timeout, repo_name)
    
    return {
        "success": True,
        "message": "Sync started for all repositories",
        "repositories_count": len(repos)
    }

# Enhanced endpoint for getting repositories
@app.get("/api/repositories")
async def get_repositories():
    return {
        "repositories": list(repo_manager.repos_config.keys()),
        "sync_progress": {name: repo_manager.get_sync_progress(name) for name in repo_manager.repos_config}
    }

# Enhanced endpoint for getting repository structure
@app.get("/api/repositories/{repo_name}")
async def get_repository(repo_name: str):
    structure = repo_manager.get_repository_structure(repo_name)
    if not structure:
        raise HTTPException(status_code=404, detail="Repository not found")
    return structure

# Enhanced endpoint for getting file content
@app.get("/api/repositories/{repo_name}/files/{path:path}")
async def get_file_content(repo_name: str, path: str):
    content = repo_manager.get_file_content(repo_name, path)
    if not content:
        raise HTTPException(status_code=404, detail="File not found")
    return {"content": content}

# Enhanced endpoint for analyzing Xcode error
@app.post("/api/xcode/analyze-error")
async def enhanced_analyze_xcode_error(request: XCodeErrorRequest):
    try:
        job_id = str(uuid.uuid4())
        
        # Queue the collaborative analysis
        asyncio.create_task(process_collaborative_analysis_async(
            job_id, "", True, request.error_message, request.use_deepseek
        ))
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Enhanced XCode error analysis started",
            "estimated_completion": "30-60 seconds"
        }
        
    except Exception as e:
        print(f"❌ Error queuing XCode analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced endpoint for general query
@app.post("/api/query")
async def enhanced_general_query(request: GeneralQueryRequest):
    try:
        job_id = str(uuid.uuid4())
        
        # Queue the collaborative analysis
        asyncio.create_task(process_collaborative_analysis_async(
            job_id, request.query, False, request.use_deepseek
        ))
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Enhanced query processing started",
            "estimated_completion": "30-60 seconds"
        }
        
    except Exception as e:
        print(f"❌ Error queuing general query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced job status endpoint
@app.get("/api/job/{job_id}")
async def get_enhanced_job_status(job_id: str):
    if job_id not in job_results:
        return {"status": "not_found"}
    
    job_data = job_results[job_id]
    
    # Add progress information
    if job_data.get('status') == 'processing':
        # Calculate estimated progress based on time elapsed
        elapsed = (datetime.now() - job_data['created_at']).total_seconds()
        estimated_progress = min(90, int(elapsed / 60 * 100))  # Estimate based on 60s completion time
        
        return {
            "status": job_data['status'],
            "progress": estimated_progress,
            "message": job_data.get('progress', 'Processing...'),
            "elapsed_seconds": int(elapsed)
        }
    
    return job_data

@app.get("/api/context/summary")
async def get_enhanced_context_summary():
    try:
        summary = ai_agent.get_context_summary()
        
        # Add repository information
        repo_summary = repo_manager.get_sync_statistics()
        
        return {
            **summary,
            "repositories": repo_summary,
            "system_health": {
                "sync_in_progress": sync_in_progress,
                "active_jobs": len(job_results),
                "memory_usage": len(str(job_results)) / 1024,  # Rough estimate in KB
                "last_cleanup": datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_files": 0,
            "repositories": {},
            "system_health": {"status": "error"}
        }

# Enhanced status endpoint
@app.get("/api/status")
async def enhanced_status():
    return {
        "message": "Enhanced XCode AI Coding Assistant API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Collaborative AI Analysis (DeepSeek + Gemini)",
            "Timeout-Aware Repository Syncing",
            "Priority-Based File Processing",
            "Enhanced Error Analysis",
            "Real-time Progress Tracking",
            "Intelligent Context Management"
        ],
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int((datetime.now() - datetime.now().replace(microsecond=0)).total_seconds()),
        "system_status": {
            "repos_configured": len(repo_manager.repos_config),
            "context_files": len(ai_agent.file_contexts),
            "active_jobs": len(job_results),
            "sync_in_progress": sync_in_progress
        }
    }

async def process_collaborative_analysis_async(job_id: str, query: str, is_error_analysis: bool, use_deepseek: str):
    """Enhanced collaborative analysis processing"""
    try:
        print(f"🔍 Processing collaborative job {job_id}")
        job_results[job_id] = {
            'status': 'processing',
            'created_at': datetime.now(),
            'result': None,
            'error': None,
            'progress': 'Initializing analysis...'
        }
        
        # Handle force sync if it's error analysis
        if is_error_analysis and force_sync:
            job_results[job_id]['progress'] = 'Syncing repositories...'
            await repo_manager.sync_all_repositories_batch()
        
        # Perform the analysis
        if is_error_analysis:
            result = await ai_agent.analyze_xcode_error(query, use_deepseek)
        else:
            result = await ai_agent.general_coding_query(query, use_deepseek)
        
        job_results[job_id]['status'] = 'completed'
        job_results[job_id]['result'] = result
        job_results[job_id]['completed_at'] = datetime.now()
        print(f"✅ Collaborative job {job_id} completed")
        
    except Exception as e:
        print(f"❌ Collaborative job {job_id} failed: {e}")
        job_results[job_id]['status'] = 'failed'
        job_results[job_id]['error'] = str(e)
        job_results[job_id]['failed_at'] = datetime.now()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced XCode AI Coding Assistant...")
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="info")