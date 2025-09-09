from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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

# Enhanced initialization
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

# Enhanced background sync task
async def enhanced_periodic_sync():
    """Enhanced background sync with timeout awareness"""
    global sync_in_progress, last_sync_attempt
    
    while True:
        try:
            current_time = datetime.now()
            
            # Only sync if not already in progress and enough time has passed
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
    use_deepseek: Any = "both"  # "both", "deepseek", "gemini"
    force_sync: bool = False

class GeneralQueryRequest(BaseModel):
    query: str
    use_deepseek: Any = "both"
    force_sync: bool = False

class FileRequest(BaseModel):
    repo_name: str
    file_path: str

# Enhanced startup event
@app.on_event("startup")
async def enhanced_startup():
    """Enhanced startup with better error handling"""
    print("🚀 Starting Enhanced XCode AI Assistant...")
    
    # Debug information
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"💾 Temp directory: {tempfile.gettempdir()}")
    print(f"🗂️ Repository path: {repo_base_path}")
    print(f"⏱️ Render timeout limit: {RENDER_TIMEOUT}s")
    
    # Verify API keys
    gemini_key = os.getenv("GOOGLE_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not gemini_key:
        print("⚠️ WARNING: GOOGLE_API_KEY not found")
    else:
        print("✅ Gemini API key configured")
        
    if not deepseek_key:
        print("⚠️ WARNING: DEEPSEEK_API_KEY not found")
    else:
        print("✅ DeepSeek API key configured")
    
    # Test filesystem access
    test_path = Path(repo_base_path)
    try:
        test_path.mkdir(exist_ok=True, parents=True)
        test_file = test_path / "startup_test.txt"
        test_file.write_text(f"Enhanced startup test - {datetime.now()}")
        print(f"✅ Filesystem access verified: {test_file}")
        test_file.unlink()  # Clean up
    except Exception as e:
        print(f"❌ Filesystem access error: {e}")
        print("⚠️ Repository management may have limited functionality")
    
    # Start enhanced background tasks
    asyncio.create_task(enhanced_periodic_sync())
    asyncio.create_task(enhanced_job_cleanup())
    
    print("🎉 Enhanced XCode AI Assistant ready!")

# Enhanced API endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_enhanced_interface():
    """Serve the enhanced web interface using template"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/api/health")
async def enhanced_health_check():
    """Enhanced health check with detailed metrics"""
    try:
        # Calculate detailed statistics
        total_files = 0
        critical_files = 0
        healthy_repos = 0
        
        for repo_name in repo_manager.repos_config:
            files = repo_manager.list_files(repo_name)
            total_files += len(files)
            critical_files += sum(1 for f in files if f.endswith(('.swift', '.m', '.h')))
            
            health = repo_manager.get_repository_health(repo_name)
            if health["status"] == "healthy":
                healthy_repos += 1
        
        # Get sync statistics
        sync_stats = repo_manager.get_sync_statistics()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "repositories": len(repo_manager.repos_config),
            "healthy_repositories": healthy_repos,
            "context_files": len(ai_agent.file_contexts),
            "total_files": total_files,
            "critical_files": critical_files,
            "active_jobs": len(job_results),
            "last_sync": sync_stats.get("last_successful_sync"),
            "sync_in_progress": sync_in_progress,
            "performance": {
                "avg_sync_time": sync_stats.get("performance", {}).get("total_sync_time", 0),
                "files_per_repo": sync_stats.get("performance", {}).get("files_per_repo", 0),
                "memory_usage_mb": len(str(job_results)) / 1024,  # Rough estimate
            }
        }
    except Exception as e:
        print(f"❌ Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repositories")
async def list_enhanced_repositories():
    """List repositories with enhanced metadata"""
    try:
        repos = []
        for name, config in repo_manager.repos_config.items():
            structure = repo_manager.get_repository_structure(name)
            repos.append({
                "name": name,
                "url": config["url"],
                "branch": config.get("branch", "main"),
                "total_files": structure.get("total_files", 0),
                "critical_files": structure.get("critical_files", 0),
                "important_files": structure.get("important_files", 0),
                "status": structure.get("status", "unknown"),
                "last_sync": structure.get("last_sync", "Never"),
                "sync_progress": structure.get("sync_progress", {}),
                "performance_metrics": structure.get("performance_metrics", {})
            })
        return {"repositories": repos}
    except Exception as e:
        print(f"❌ Repository listing error: {e}")
        return {"repositories": []}

@app.post("/api/repositories/add")
async def add_enhanced_repository(repo_config: RepoConfig):
    """Add repository with enhanced validation and processing"""
    try:
        print(f"➕ Adding enhanced repository: {repo_config.name}")
        
        # Enhanced validation
        if not repo_config.name.strip():
            raise HTTPException(status_code=400, detail="Repository name cannot be empty")
        
        if not repo_config.url.strip():
            raise HTTPException(status_code=400, detail="Repository URL cannot be empty")
        
        # Add repository to manager
        repo_manager.add_repository(
            name=repo_config.name,
            url=repo_config.url,
            branch=repo_config.branch,
            access_token=repo_config.access_token,
            sync_interval=repo_config.sync_interval
        )
        
        # Perform initial sync with timeout protection
        success, message, file_count = await repo_manager.clone_or_update_repo_with_timeout(
            repo_config.name, 
            max_duration=RENDER_TIMEOUT
        )
        
        if success:
            # Process critical files immediately for context
            context_files = await process_repository_files(repo_config.name, priority_only=True)
            
            return {
                "success": True,
                "message": f"Repository {repo_config.name} added successfully",
                "files_loaded": file_count,
                "context_files": context_files,
                "sync_message": message
            }
        else:
            return {
                "success": False,
                "message": f"Repository added but sync had issues: {message}",
                "files_loaded": file_count
            }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error adding repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/repositories/sync")
async def enhanced_repository_sync(background_tasks: BackgroundTasks):
    """Enhanced repository sync with timeout handling"""
    try:
        global sync_in_progress
        
        if sync_in_progress:
            return {
                "message": "Sync already in progress", 
                "status": "already_running"
            }
        
        sync_in_progress = True
        
        # Start batch sync in background
        background_tasks.add_task(perform_enhanced_sync)
        
        return {
            "message": "Enhanced batch sync started", 
            "status": "started",
            "estimated_duration": "30-60 seconds"
        }
    except Exception as e:
        sync_in_progress = False
        print(f"❌ Error starting enhanced sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def perform_enhanced_sync():
    """Perform enhanced sync in background"""
    global sync_in_progress
    
    try:
        print("🔄 Starting enhanced background sync...")
        
        # Use batch sync with timeout handling
        sync_results = await repo_manager.sync_all_repositories_batch(
            batch_size=3,
            max_duration=RENDER_TIMEOUT
        )
        
        # Update context with priority files
        total_context_updates = 0
        for repo_name, (success, message, file_count) in sync_results.items():
            if success:
                context_updates = await process_repository_files(repo_name, priority_only=True)
                total_context_updates += context_updates
        
        print(f"✅ Enhanced background sync completed: {len(sync_results)} repos, {total_context_updates} context updates")
        
    except Exception as e:
        print(f"❌ Enhanced background sync error: {e}")
    finally:
        sync_in_progress = False

@app.get("/api/repositories/{repo_name}/files")
async def get_enhanced_repository_files(repo_name: str):
    """Get repository files with enhanced metadata"""
    try:
        if repo_name not in repo_manager.repos_config:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        structure = repo_manager.get_repository_structure(repo_name)
        files = repo_manager.list_files(repo_name)
        
        # Enhanced file information
        file_details = []
        for file_path in files[:100]:  # Limit for performance
            content = repo_manager.get_file_content(repo_name, file_path)
            file_size = len(content) if content else 0
            
            file_details.append({
                "path": file_path,
                "size": file_size,
                "extension": Path(file_path).suffix.lower(),
                "type": "critical" if file_path.endswith(('.swift', '.m', '.h')) else "important" if file_path.endswith(('.py', '.js', '.json')) else "other",
                "preview": content[:200] + "..." if content and len(content) > 200 else content
            })
        
        return {
            "repository": repo_name,
            "total_files": len(files),
            "files": file_details,
            "file_types": list(set(f["extension"] for f in file_details if f["extension"])),
            "critical_files": len([f for f in file_details if f["type"] == "critical"]),
            "important_files": len([f for f in file_details if f["type"] == "important"]),
            "repository_health": repo_manager.get_repository_health(repo_name)
        }
            
    except Exception as e:
        print(f"❌ Error listing repository files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/repositories/file-content")
async def get_enhanced_file_content(request: FileRequest):
    """Get file content with enhanced handling"""
    try:
        content = repo_manager.get_file_content(request.repo_name, request.file_path)
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "repo_name": request.repo_name,
            "file_path": request.file_path,
            "content": content,
            "size": len(content),
            "type": "critical" if request.file_path.endswith(('.swift', '.m', '.h')) else "important" if request.file_path.endswith(('.py', '.js', '.json')) else "other",
            "language": Path(request.file_path).suffix.lower().lstrip('.')
        }
            
    except Exception as e:
        print(f"❌ Error getting file content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced job processing functions
async def process_collaborative_analysis_async(job_id: str, query: str, is_error_analysis: bool, error_message: str = None):
    """Enhanced collaborative analysis processing"""
    try:
        print(f"🤖 Processing collaborative analysis job {job_id}")
        job_results[job_id] = {
            'status': 'processing',
            'created_at': datetime.now(),
            'result': None,
            'error': None,
            'progress': 'Initializing collaborative analysis...'
        }
        
        # Use enhanced collaborative analysis
        if is_error_analysis:
            result = await ai_agent.collaborative_analysis("", True, error_message)
        else:
            result = await ai_agent.collaborative_analysis(query, False)
        
        job_results[job_id]['status'] = 'completed'
        job_results[job_id]['result'] = result
        job_results[job_id]['completed_at'] = datetime.now()
        print(f"✅ Collaborative analysis job {job_id} completed")
        
    except Exception as e:
        print(f"❌ Collaborative analysis job {job_id} failed: {e}")
        job_results[job_id]['status'] = 'failed'
        job_results[job_id]['error'] = str(e)
        job_results[job_id]['failed_at'] = datetime.now()

async def process_single_model_async(job_id: str, query: str, use_deepseek: bool, is_error_analysis: bool):
    """Enhanced single model processing"""
    try:
        print(f"🔍 Processing single model job {job_id} ({'DeepSeek' if use_deepseek else 'Gemini'})")
        job_results[job_id] = {
            'status': 'processing',
            'created_at': datetime.now(),
            'result': None,
            'error': None,
            'progress': f'Processing with {"DeepSeek" if use_deepseek else "Gemini"}...'
        }
        
        if is_error_analysis:
            result = await ai_agent.analyze_xcode_error(query, use_deepseek)
        else:
            result = await ai_agent.general_coding_query(query, use_deepseek)
        
        job_results[job_id]['status'] = 'completed'
        job_results[job_id]['result'] = result
        job_results[job_id]['completed_at'] = datetime.now()
        print(f"✅ Single model job {job_id} completed")
        
    except Exception as e:
        print(f"❌ Single model job {job_id} failed: {e}")
        job_results[job_id]['status'] = 'failed'
        job_results[job_id]['error'] = str(e)
        job_results[job_id]['failed_at'] = datetime.now()

@app.post("/api/xcode/analyze-error")
async def enhanced_analyze_xcode_error(request: XCodeErrorRequest):
    """Enhanced XCode error analysis"""
    try:
        print(f"🚫 XCode error analysis requested: {request.error_message[:100]}...")
        
        job_id = str(uuid.uuid4())
        
        if request.use_deepseek == "both":
            # Use collaborative processing
            asyncio.create_task(process_collaborative_analysis_async(
                job_id, "", True, request.error_message
            ))
        else:
            # Use single model processing
            use_deepseek = request.use_deepseek == "deepseek" if isinstance(request.use_deepseek, str) else bool(request.use_deepseek)
            asyncio.create_task(process_single_model_async(
                job_id, request.error_message, use_deepseek, True
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

@app.post("/api/query")
async def enhanced_general_query(request: GeneralQueryRequest):
    """Enhanced general coding query"""
    try:
        print(f"💬 General query requested: {request.query[:100]}...")
        
        job_id = str(uuid.uuid4())
        
        if request.use_deepseek == "both":
            # Use collaborative processing
            asyncio.create_task(process_collaborative_analysis_async(
                job_id, request.query, False
            ))
        else:
            # Use single model processing
            use_deepseek = request.use_deepseek == "deepseek" if isinstance(request.use_deepseek, str) else bool(request.use_deepseek)
            asyncio.create_task(process_single_model_async(
                job_id, request.query, use_deepseek, False
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

@app.get("/api/job/{job_id}")
async def get_enhanced_job_status(job_id: str):
    """Enhanced job status with progress tracking"""
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
    
    return job_results[job_id]

@app.get("/api/context/summary")
async def get_enhanced_context_summary():
    """Enhanced context summary"""
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
    """Enhanced API status"""
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

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced XCode AI Coding Assistant...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")