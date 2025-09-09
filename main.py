from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
from git_repo_manager import GitRepoManager  # Enhanced version from previous artifact
from ai_agent_service import AIAgentService    # Enhanced version from previous artifact

app = FastAPI(
    title="Enhanced XCode AI Coding Assistant", 
    version="2.0.0",
    description="AI-powered coding assistant with collaborative DeepSeek + Gemini analysis"
)

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
    """Serve the enhanced web interface embedded directly"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XCode AI Coding Assistant - Enhanced</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }

        .panel h2 {
            color: #5a67d8;
            margin-bottom: 20px;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-bar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 500;
            padding: 10px;
            background: rgba(90, 103, 216, 0.1);
            border-radius: 8px;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #f56565;
            animation: pulse 2s infinite;
        }

        .status-dot.connected {
            background: #48bb78;
        }

        .status-dot.syncing {
            background: #ed8936;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #4a5568;
        }

        .form-group input,
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.2s;
        }

        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #5a67d8;
            box-shadow: 0 0 0 3px rgba(90, 103, 216, 0.1);
        }

        .form-group textarea {
            resize: vertical;
            min-height: 120px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', Consolas, monospace;
            font-size: 13px;
        }

        .btn {
            background: linear-gradient(45deg, #5a67d8, #667eea);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .btn:hover::before {
            left: 100%;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(90, 103, 216, 0.4);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .btn.secondary {
            background: linear-gradient(45deg, #4a5568, #718096);
        }

        .btn.success {
            background: linear-gradient(45deg, #38a169, #48bb78);
        }

        .btn.danger {
            background: linear-gradient(45deg, #e53e3e, #f56565);
        }

        .response-section {
            grid-column: 1 / -1;
            margin-top: 20px;
        }

        .response-tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            background: rgba(74, 85, 104, 0.1);
            padding: 5px;
            border-radius: 10px;
        }

        .tab-btn {
            background: transparent;
            color: #4a5568;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
            flex: 1;
        }

        .tab-btn.active {
            background: #5a67d8;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(90, 103, 216, 0.3);
        }

        .tab-content {
            display: none;
            background: #1a202c;
            border-radius: 12px;
            position: relative;
            overflow: hidden;
        }

        .tab-content.active {
            display: block;
        }

        .response-content {
            color: #e2e8f0;
            padding: 20px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', Consolas, monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 600px;
            overflow-y: auto;
            position: relative;
        }

        .copy-btn {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(74, 85, 104, 0.8);
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.3s;
            backdrop-filter: blur(10px);
        }

        .copy-btn:hover {
            background: rgba(90, 103, 216, 0.8);
            transform: translateY(-1px);
        }

        .file-browser {
            max-height: 400px;
            overflow-y: auto;
            background: #f7fafc;
            border-radius: 10px;
            border: 2px solid #e2e8f0;
        }

        .file-browser-header {
            padding: 15px;
            background: #5a67d8;
            color: white;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
            border-radius: 8px 8px 0 0;
        }

        .file-item {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 1px solid #e2e8f0;
            font-size: 13px;
            font-family: 'SF Mono', Monaco, monospace;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .file-item:hover {
            background: linear-gradient(90deg, rgba(90, 103, 216, 0.1), rgba(102, 126, 234, 0.1));
            transform: translateX(5px);
        }

        .file-item .file-icon {
            margin-right: 8px;
            font-size: 16px;
        }

        .file-item .file-size {
            color: #666;
            font-size: 11px;
        }

        .repo-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }

        .stat-item {
            background: rgba(90, 103, 216, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-value {
            font-size: 20px;
            font-weight: bold;
            color: #5a67d8;
        }

        .stat-label {
            font-size: 12px;
            color: #666;
            margin-top: 2px;
        }

        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #5a67d8;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .progress-bar {
            background: rgba(226, 232, 240, 0.3);
            border-radius: 10px;
            overflow: hidden;
            height: 8px;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #5a67d8, #667eea);
            border-radius: 10px;
            transition: width 0.3s ease;
            position: relative;
        }

        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .code-file-container {
            background: #2d3748;
            border-radius: 10px;
            margin: 15px 0;
            overflow: hidden;
            border: 1px solid #4a5568;
        }

        .code-file-header {
            background: #4a5568;
            color: #e2e8f0;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            font-weight: 600;
        }

        .code-file-content {
            background: #1a202c;
            color: #e2e8f0;
            padding: 20px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            line-height: 1.5;
            overflow-x: auto;
            white-space: pre;
            max-height: 500px;
            overflow-y: auto;
        }

        .copy-file-btn {
            background: #5a67d8;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            margin-left: auto;
        }

        .copy-file-btn:hover {
            background: #4c51bf;
        }

        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }

        .notification.show {
            transform: translateX(0);
        }

        .notification.success {
            background: #48bb78;
        }

        .notification.error {
            background: #f56565;
        }

        .notification.info {
            background: #5a67d8;
        }

        .model-comparison {
            background: #2d3748;
            border-radius: 12px;
            margin: 20px 0;
            overflow: hidden;
        }

        .model-comparison details {
            border-bottom: 1px solid #4a5568;
        }

        .model-comparison details:last-child {
            border-bottom: none;
        }

        .model-comparison summary {
            cursor: pointer;
            padding: 15px 20px;
            background: #4a5568;
            color: #e2e8f0;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background 0.3s;
        }

        .model-comparison summary:hover {
            background: #5a67d8;
        }

        .model-comparison details[open] summary {
            background: #5a67d8;
        }

        .model-content {
            padding: 20px;
            background: #1a202c;
            color: #e2e8f0;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .sync-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(90, 103, 216, 0.1);
            border-radius: 8px;
            margin: 10px 0;
        }

        .sync-status.syncing {
            background: rgba(237, 137, 54, 0.1);
            border-left: 4px solid #ed8936;
        }

        .sync-status.success {
            background: rgba(72, 187, 120, 0.1);
            border-left: 4px solid #48bb78;
        }

        .sync-status.error {
            background: rgba(245, 101, 101, 0.1);
            border-left: 4px solid #f56565;
        }

        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .status-bar {
                grid-template-columns: 1fr;
                gap: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛠️ XCode AI Coding Assistant</h1>
            <p>Enhanced with DeepSeek & Gemini Collaborative Intelligence</p>
        </div>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot" id="statusDot"></div>
                <span>Server: <strong id="serverStatus">Connecting...</strong></span>
            </div>
            <div class="status-item">
                <span>📂 Repos: <strong id="repoCount">0</strong></span>
                <span>📱 Critical: <strong id="criticalFiles">0</strong></span>
            </div>
            <div class="status-item">
                <span>📄 Total Files: <strong id="totalFiles">0</strong></span>
                <span>🧠 Context: <strong id="contextFiles">0</strong></span>
            </div>
            <div class="status-item">
                <span>🕒 Last Sync: <strong id="lastSync">Never</strong></span>
                <div class="progress-bar" id="syncProgress" style="display: none;">
                    <div class="progress-fill" id="syncProgressFill" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <div class="main-grid">
            <!-- Repository Management Panel -->
            <div class="panel">
                <h2>📁 Repository Management</h2>
                
                <div class="form-group">
                    <label for="repoName">Repository Name:</label>
                    <input type="text" id="repoName" placeholder="ReResell-frontend">
                </div>
                
                <div class="form-group">
                    <label for="repoUrl">Git URL:</label>
                    <input type="text" id="repoUrl" placeholder="https://github.com/MrAmazingOne/ReResell-frontend.git">
                </div>
                
                <div class="form-group">
                    <label for="repoBranch">Branch:</label>
                    <input type="text" id="repoBranch" value="main" placeholder="main">
                </div>
                
                <div class="form-group">
                    <label for="accessToken">Access Token:</label>
                    <input type="password" id="accessToken" placeholder="ghp_xxxxxxxxxxxx">
                </div>
                
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button class="btn" onclick="addRepository()">
                        <span>➕</span> Add Repository
                    </button>
                    
                    <button class="btn secondary" onclick="syncRepositories()">
                        <span>🔄</span> Sync All
                    </button>
                    
                    <button class="btn danger" onclick="clearContext()">
                        <span>🗑️</span> Clear Context
                    </button>
                </div>

                <div class="repo-stats" id="repoStats">
                    <!-- Repository statistics will be populated here -->
                </div>

                <div class="file-browser" id="fileBrowser">
                    <div class="file-browser-header">
                        <span>📁</span> Repository Files
                    </div>
                    <div style="text-align: center; color: #666; padding: 40px;">
                        Add a repository to view files
                    </div>
                </div>
            </div>

            <!-- XCode Error Analysis Panel -->
            <div class="panel">
                <h2>🚫 XCode Error Analysis</h2>
                
                <div class="form-group">
                    <label for="xcodeError">XCode Error Message:</label>
                    <textarea id="xcodeError" placeholder="Paste your XCode error message here..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="aiModel">AI Model:</label>
                    <select id="aiModel">
                        <option value="both">🤖 Collaborative (DeepSeek + Gemini)</option>
                        <option value="deepseek">⚡ DeepSeek (Coding Expert)</option>
                        <option value="gemini">🧠 Gemini (Reasoning)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="forceSync" style="width: auto; margin-right: 8px;">
                        Force repository sync before analysis
                    </label>
                </div>
                
                <button class="btn" onclick="analyzeError()">
                    <span>🔍</span> Analyze Error
                </button>
            </div>

            <!-- General Query Panel -->
            <div class="panel">
                <h2>💬 General Coding Query</h2>
                
                <div class="form-group">
                    <label for="generalQuery">Your Question:</label>
                    <textarea id="generalQuery" placeholder="Ask anything about your code..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="queryModel">AI Model:</label>
                    <select id="queryModel">
                        <option value="both">🤖 Collaborative (DeepSeek + Gemini)</option>
                        <option value="deepseek">⚡ DeepSeek (Coding Expert)</option>
                        <option value="gemini">🧠 Gemini (Reasoning)</option>
                    </select>
                </div>
                
                <button class="btn" onclick="submitQuery()">
                    <span>🤖</span> Ask AI
                </button>
            </div>

            <!-- Enhanced Response Section -->
            <div class="panel response-section">
                <h2>📋 AI Response</h2>
                
                <div class="response-tabs">
                    <button class="tab-btn active" onclick="switchTab('collaborative')">🤖 Collaborative</button>
                    <button class="tab-btn" onclick="switchTab('files')">📄 Code Files</button>
                    <button class="tab-btn" onclick="switchTab('individual')">🔍 Individual Models</button>
                    <button class="tab-btn" onclick="switchTab('raw')">📝 Raw Response</button>
                </div>
                
                <!-- Collaborative Analysis Tab -->
                <div id="collaborative-content" class="tab-content active">
                    <div class="response-content" id="collaborativeResponse">
                        <button class="copy-btn" onclick="copyToClipboard('collaborativeResponse')">📋 Copy All</button>
                        🚀 Ready for enhanced AI analysis with collaborative intelligence!
                        
                        Upload your repositories and start analyzing XCode errors with both DeepSeek and Gemini working together.
                    </div>
                </div>
                
                <!-- Code Files Tab -->
                <div id="files-content" class="tab-content">
                    <div id="codeFilesContainer" style="padding: 20px;">
                        <p style="text-align: center; color: #666; margin: 40px 0;">
                            Code files will appear here after analysis
                        </p>
                    </div>
                </div>
                
                <!-- Individual Models Tab -->
                <div id="individual-content" class="tab-content">
                    <div class="model-comparison" id="modelComparison">
                        <p style="text-align: center; color: #666; margin: 40px 0;">
                            Individual model analyses will appear here
                        </p>
                    </div>
                </div>
                
                <!-- Raw Response Tab -->
                <div id="raw-content" class="tab-content">
                    <div class="response-content" id="rawResponse">
                        <button class="copy-btn" onclick="copyToClipboard('rawResponse')">📋 Copy</button>
                        Raw JSON response will appear here...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Notification System -->
    <div id="notification" class="notification"></div>

    <script>
        const API_BASE = window.location.origin;
        
        let repositories = [];
        let currentResponse = null;
        let currentJobId = null;
        let statusUpdateInterval = null;

        // Enhanced initialization
        async function init() {
            console.log('🚀 Initializing Enhanced XCode AI Assistant...');
            console.log('🔗 API Base URL:', API_BASE);
            
            await checkServerStatus();
            await loadRepositories();
            
            // Set up real-time status updates
            statusUpdateInterval = setInterval(updateStatus, 15000); // Every 15 seconds
            
            showNotification('🚀 XCode AI Assistant Ready!', 'success');
        }

        // Enhanced server status checking
        async function checkServerStatus() {
            try {
                const response = await fetch(`${API_BASE}/api/health`);
                if (response.ok) {
                    const data = await response.json();
                    updateStatusDisplay(data);
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (error) {
                document.getElementById('serverStatus').textContent = 'Disconnected ❌';
                document.getElementById('statusDot').classList.remove('connected', 'syncing');
                console.error('❌ Server connection failed:', error);
            }
        }

        function updateStatusDisplay(data) {
            document.getElementById('serverStatus').textContent = 'Connected ✅';
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusDot').classList.remove('syncing');
            
            document.getElementById('repoCount').textContent = data.repositories || 0;
            document.getElementById('totalFiles').textContent = data.total_files || 0;
            document.getElementById('contextFiles').textContent = data.context_files || 0;
            document.getElementById('criticalFiles').textContent = data.critical_files || 0;
            
            // Update last sync time
            if (data.last_sync) {
                const lastSync = new Date(data.last_sync);
                const timeDiff = Date.now() - lastSync.getTime();
                const minutes = Math.floor(timeDiff / 60000);
                
                if (minutes < 1) {
                    document.getElementById('lastSync').textContent = 'Just now';
                } else if (minutes < 60) {
                    document.getElementById('lastSync').textContent = `${minutes}m ago`;
                } else {
                    const hours = Math.floor(minutes / 60);
                    document.getElementById('lastSync').textContent = `${hours}h ago`;
                }
            }
        }

        // Enhanced repository loading
        async function loadRepositories() {
            try {
                const response = await fetch(`${API_BASE}/api/repositories`);
                if (response.ok) {
                    const data = await response.json();
                    repositories = data.repositories || [];
                    updateFileTree();
                    updateRepositoryStats();
                }
            } catch (error) {
                console.error('❌ Failed to load repositories:', error);
                showNotification('Failed to load repositories', 'error');
            }
        }

        function updateRepositoryStats() {
            const statsContainer = document.getElementById('repoStats');
            
            if (repositories.length === 0) {
                statsContainer.innerHTML = '';
                return;
            }

            const totalFiles = repositories.reduce((sum, repo) => sum + (repo.total_files || 0), 0);
            const criticalFiles = repositories.reduce((sum, repo) => sum + (repo.critical_files || 0), 0);
            const healthyRepos = repositories.filter(repo => repo.status === 'healthy').length;

            statsContainer.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${repositories.length}</div>
                    <div class="stat-label">Total Repos</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${healthyRepos}</div>
                    <div class="stat-label">Healthy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${totalFiles}</div>
                    <div class="stat-label">Total Files</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${criticalFiles}</div>
                    <div class="stat-label">Swift/ObjC</div>
                </div>
            `;
        }

        // Enhanced repository addition
        async function addRepository() {
            const name = document.getElementById('repoName').value.trim();
            const url = document.getElementById('repoUrl').value.trim();
            const branch = document.getElementById('repoBranch').value.trim() || 'main';
            const token = document.getElementById('accessToken').value.trim();

            if (!name || !url) {
                showNotification('Repository name and URL are required', 'error');
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/api/repositories/add`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        name: name,
                        url: url,
                        branch: branch,
                        access_token: token,
                        sync_interval: 300
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    showNotification(`Repository ${name} added successfully!`, 'success');
                    document.getElementById('repoName').value = '';
                    document.getElementById('repoUrl').value = '';
                    document.getElementById('accessToken').value = '';
                    await loadRepositories();
                } else {
                    showNotification(result.message, 'error');
                }
            } catch (error) {
                console.error('❌ Error adding repository:', error);
                showNotification('Failed to add repository', 'error');
            }
        }

        async function syncRepositories() {
            try {
                showNotification('🔄 Syncing repositories...', 'info');
                const response = await fetch(`${API_BASE}/api/repositories/sync`, {
                    method: 'POST'
                });
                const result = await response.json();
                showNotification(result.message, 'success');
                await loadRepositories();
            } catch (error) {
                console.error('❌ Error syncing repositories:', error);
                showNotification('Failed to sync repositories', 'error');
            }
        }

        async function clearContext() {
            try {
                showNotification('🗑️ Clearing context...', 'info');
                // Add API call to clear context if needed
                await loadRepositories();
                showNotification('Context cleared successfully', 'success');
            } catch (error) {
                console.error('❌ Error clearing context:', error);
                showNotification('Failed to clear context', 'error');
            }
        }

        async function analyzeError() {
            const errorMessage = document.getElementById('xcodeError').value.trim();
            const useDeepseek = document.getElementById('aiModel').value;
            const forceSync = document.getElementById('forceSync').checked;

            if (!errorMessage) {
                showNotification('Please enter an error message', 'error');
                return;
            }

            try {
                showNotification('🔍 Analyzing error...', 'info');
                const response = await fetch(`${API_BASE}/api/xcode/analyze-error`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        error_message: errorMessage,
                        use_deepseek: useDeepseek,
                        force_sync: forceSync
                    })
                });

                const result = await response.json();
                currentJobId = result.job_id;
                showNotification('Analysis started! Tracking progress...', 'success');
                trackJobProgress();
            } catch (error) {
                console.error('❌ Error analyzing error:', error);
                showNotification('Failed to analyze error', 'error');
            }
        }

        async function submitQuery() {
            const query = document.getElementById('generalQuery').value.trim();
            const useDeepseek = document.getElementById('queryModel').value;

            if (!query) {
                showNotification('Please enter a query', 'error');
                return;
            }

            try {
                showNotification('🤖 Processing query...', 'info');
                const response = await fetch(`${API_BASE}/api/query`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        use_deepseek: useDeepseek
                    })
                });

                const result = await response.json();
                currentJobId = result.job_id;
                showNotification('Query processing started!', 'success');
                trackJobProgress();
            } catch (error) {
                console.error('❌ Error submitting query:', error);
                showNotification('Failed to submit query', 'error');
            }
        }

        async function trackJobProgress() {
            if (!currentJobId) return;

            const checkProgress = async () => {
                try {
                    const response = await fetch(`${API_BASE}/api/job/${currentJobId}`);
                    const status = await response.json();
                    
                    if (status.status === 'completed') {
                        currentResponse = status.result;
                        displayResponse(currentResponse);
                        showNotification('✅ Analysis completed!', 'success');
                    } else if (status.status === 'failed') {
                        showNotification('❌ Analysis failed', 'error');
                    } else {
                        // Still processing, check again in 2 seconds
                        setTimeout(checkProgress, 2000);
                    }
                } catch (error) {
                    console.error('❌ Error checking job progress:', error);
                    setTimeout(checkProgress, 2000);
                }
            };

            await checkProgress();
        }

        function displayResponse(response) {
            if (response.collaborative_analysis) {
                document.getElementById('collaborativeResponse').textContent = response.collaborative_analysis;
            }
            
            if (response.deepseek_analysis) {
                // Display individual model analyses
            }
            
            if (response.gemini_analysis) {
                // Display individual model analyses
            }
            
            if (response.code_sections) {
                displayCodeFiles(response.code_sections);
            }
            
            document.getElementById('rawResponse').textContent = JSON.stringify(response, null, 2);
        }

        function displayCodeFiles(codeSections) {
            const container = document.getElementById('codeFilesContainer');
            container.innerHTML = '';
            
            for (const [filename, code] of Object.entries(codeSections)) {
                const fileElement = document.createElement('div');
                fileElement.className = 'code-file-container';
                fileElement.innerHTML = `
                    <div class="code-file-header">
                        <span>📄 ${filename}</span>
                        <button class="copy-file-btn" onclick="copyCodeToClipboard('${filename}')">📋 Copy</button>
                    </div>
                    <div class="code-file-content" id="code-${filename}">
                        ${escapeHtml(code)}
                    </div>
                `;
                container.appendChild(fileElement);
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Deactivate all buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Activate selected tab
            document.getElementById(`${tabName}-content`).classList.add('active');
            
            // Activate selected button
            document.querySelector(`.tab-btn[onclick="switchTab('${tabName}')"]`).classList.add('active');
        }

        function copyToClipboard(elementId) {
            const element = document.getElementById(elementId);
            const text = element.textContent || element.innerText;
            navigator.clipboard.writeText(text).then(() => {
                showNotification('📋 Copied to clipboard!', 'success');
            }).catch(err => {
                console.error('❌ Failed to copy:', err);
                showNotification('❌ Failed to copy', 'error');
            });
        }

        function copyCodeToClipboard(filename) {
            const codeElement = document.getElementById(`code-${filename}`);
            const text = codeElement.textContent;
            navigator.clipboard.writeText(text).then(() => {
                showNotification(`📋 ${filename} copied!`, 'success');
            }).catch(err => {
                console.error('❌ Failed to copy code:', err);
                showNotification('❌ Failed to copy code', 'error');
            });
        }

        function showNotification(message, type) {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = `notification ${type} show`;
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }

        function updateFileTree() {
            const fileBrowser = document.getElementById('fileBrowser');
            
            if (repositories.length === 0) {
                fileBrowser.innerHTML = `
                    <div class="file-browser-header">
                        <span>📁</span> Repository Files
                    </div>
                    <div style="text-align: center; color: #666; padding: 40px;">
                        Add a repository to view files
                    </div>
                `;
                return;
            }
            
            // For now, just show a simple message
            fileBrowser.innerHTML = `
                <div class="file-browser-header">
                    <span>📁</span> Repository Files (${repositories.length} repos)
                </div>
                <div style="text-align: center; color: #666; padding: 20px;">
                    File browser functionality coming soon!
                </div>
            `;
        }

        // Initialize the application
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
""")

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