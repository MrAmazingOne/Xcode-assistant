from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
from datetime import datetime
import json

# Import our custom modules
from git_repo_manager import GitRepoManager
from ai_agent_service import AIAgentService

app = FastAPI(title="XCode AI Coding Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
repo_manager = GitRepoManager()
ai_agent = AIAgentService(
    gemini_api_key=os.getenv("GOOGLE_API_KEY"),
    deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Background task for periodic syncing
async def periodic_sync():
    """Background task to sync repositories periodically"""
    while True:
        try:
            await repo_manager.sync_all_repositories()
            
            # Update AI agent context with changed files
            for repo_name in repo_manager.repos_config:
                changed_files = repo_manager.get_changed_files(repo_name)
                for file_path in changed_files:
                    content = repo_manager.get_file_content(repo_name, file_path)
                    if content:
                        await ai_agent.update_file_context(repo_name, file_path, content)
            
            print(f"Sync completed at {datetime.now()}")
            
        except Exception as e:
            print(f"Sync error: {str(e)}")
            
        # Wait 5 minutes before next sync
        await asyncio.sleep(300)

# Pydantic models
class RepoConfig(BaseModel):
    name: str
    url: str
    branch: str = "main"
    access_token: Optional[str] = None
    sync_interval: int = 300

class XCodeErrorRequest(BaseModel):
    error_message: str
    use_deepseek: bool = True
    force_sync: bool = False

class GeneralQueryRequest(BaseModel):
    query: str
    use_deepseek: bool = True
    force_sync: bool = False

class FileRequest(BaseModel):
    repo_name: str
    file_path: str

# Web Interface Routes
@app.get("/")
async def serve_web_interface():
    """Serve the main web interface"""
    return FileResponse('static/index.html')

@app.get("/dashboard")
async def serve_dashboard():
    """Alternative route for the dashboard"""
    return FileResponse('static/index.html')

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Start background sync task"""
    asyncio.create_task(periodic_sync())

@app.get("/api/status")
async def root():
    """API status endpoint"""
    return {
        "message": "XCode AI Coding Assistant API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/repositories/add")
async def add_repository(repo_config: RepoConfig):
    """Add a new repository to monitor"""
    try:
        repo_manager.add_repository(
            name=repo_config.name,
            url=repo_config.url,
            branch=repo_config.branch,
            access_token=repo_config.access_token,
            sync_interval=repo_config.sync_interval
        )
        
        # Initial clone/sync
        success = await repo_manager.clone_or_update_repo(repo_config.name)
        
        if success:
            # Load all files into AI context
            files = repo_manager.list_files(repo_config.name, 
                extensions=['.swift', '.m', '.h', '.py', '.js', '.json', '.plist'])
            
            for file_path in files:
                content = repo_manager.get_file_content(repo_config.name, file_path)
                if content:
                    await ai_agent.update_file_context(repo_config.name, file_path, content)
            
            return {
                "success": True,
                "message": f"Repository {repo_config.name} added successfully",
                "files_loaded": len(files)
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to sync repository")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/repositories")
async def list_repositories():
    """List all configured repositories"""
    repos = []
    for name, config in repo_manager.repos_config.items():
        structure = repo_manager.get_repository_structure(name)
        repos.append({
            "name": name,
            "url": config["url"],
            "branch": config["branch"],
            "sync_interval": config["sync_interval"],
            "last_sync": repo_manager.last_sync.get(name, "Never"),
            "total_files": structure.get("total_files", 0)
        })
    
    return {"repositories": repos}

@app.post("/api/repositories/sync")
async def sync_repositories(background_tasks: BackgroundTasks):
    """Manually trigger repository sync"""
    background_tasks.add_task(repo_manager.sync_all_repositories)
    return {"message": "Sync started in background"}

@app.get("/api/repositories/{repo_name}/files")
async def list_repository_files(repo_name: str):
    """List files in a specific repository"""
    if repo_name not in repo_manager.repos_config:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    return repo_manager.get_repository_structure(repo_name)

@app.post("/api/files/content")
async def get_file_content(file_request: FileRequest):
    """Get content of a specific file"""
    content = repo_manager.get_file_content(file_request.repo_name, file_request.file_path)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "repo_name": file_request.repo_name,
        "file_path": file_request.file_path,
        "content": content
    }

@app.post("/api/xcode/analyze-error")
async def analyze_xcode_error(request: XCodeErrorRequest):
    """Analyze XCode error and provide solution"""
    try:
        if request.force_sync:
            await repo_manager.sync_all_repositories()
        
        result = await ai_agent.analyze_xcode_error(
            request.error_message, 
            use_deepseek=request.use_deepseek
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def general_query(request: GeneralQueryRequest):
    """Handle general coding queries"""
    try:
        if request.force_sync:
            await repo_manager.sync_all_repositories()
        
        result = await ai_agent.general_coding_query(
            request.query,
            use_deepseek=request.use_deepseek
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/context/summary")
async def get_context_summary():
    """Get summary of current AI context"""
    return ai_agent.get_context_summary()

@app.get("/api/conversation/history")
async def get_conversation_history(limit: int = 10):
    """Get recent conversation history"""
    return ai_agent.get_conversation_history(limit)

@app.delete("/api/context/clear")
async def clear_context():
    """Clear AI context (useful for memory management)"""
    ai_agent.clear_context()
    return {"message": "Context cleared successfully"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "repositories": len(repo_manager.repos_config),
        "context_files": len(ai_agent.file_contexts)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)