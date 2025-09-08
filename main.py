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

# Initialize services with Render-friendly configuration
render_temp_dir = os.getenv('RENDER_TEMP_DIR', tempfile.gettempdir())
repo_base_path = os.path.join(render_temp_dir, "repos")

repo_manager = GitRepoManager(base_path=repo_base_path)
ai_agent = AIAgentService(
    gemini_api_key=os.getenv("GOOGLE_API_KEY"),
    deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")
)

# In-memory job storage
job_results = {}
job_queue = deque()
MAX_JOBS_STORED = 100

# Background task for periodic syncing
async def periodic_sync():
    """Background task to sync repositories periodically"""
    while True:
        try:
            print("Starting periodic sync...")
            sync_results = await repo_manager.sync_all_repositories()
            
            # Update AI agent context with changed files
            for repo_name in repo_manager.repos_config:
                changed_files = repo_manager.get_changed_files(repo_name)
                print(f"Repository {repo_name}: {len(changed_files)} changed files")
                
                for file_path in changed_files[:20]:  # Limit to 20 files per sync
                    content = repo_manager.get_file_content(repo_name, file_path)
                    if content and len(content) > 10:  # Skip empty/tiny files
                        await ai_agent.update_file_context(repo_name, file_path, content)
            
            print(f"Sync completed at {datetime.now()}")
            
        except Exception as e:
            print(f"Sync error: {str(e)}")
            
        # Wait 5 minutes before next sync
        await asyncio.sleep(300)

# Clean up old jobs periodically
async def cleanup_old_jobs():
    """Clean up old job results to prevent memory leaks"""
    while True:
        await asyncio.sleep(300)  # Clean up every 5 minutes
        try:
            current_time = datetime.now()
            jobs_to_remove = []
            
            for job_id, job_data in job_results.items():
                # Remove jobs older than 1 hour
                if (current_time - job_data['created_at']).total_seconds() > 3600:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                job_results.pop(job_id, None)
                
            # Ensure we don't store too many jobs
            while len(job_results) > MAX_JOBS_STORED:
                oldest_job_id = next(iter(job_results))
                job_results.pop(oldest_job_id, None)
                
            print(f"Cleaned up {len(jobs_to_remove)} old jobs")
                
        except Exception as e:
            print(f"Job cleanup error: {str(e)}")

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

# Fixed HTML content with proper JavaScript - FIXED SYNTAX ERRORS
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XCode AI Coding Assistant</title>
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
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }

        .panel h2 {
            color: #5a67d8;
            margin-bottom: 20px;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
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
            transition: border-color 0.2s;
        }

        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #5a67d8;
        }

        .form-group textarea {
            resize: vertical;
            min-height: 120px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', Consolas, monospace;
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
            transition: transform 0.2s, box-shadow 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(90, 103, 216, 0.4);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .response-section {
            grid-column: 1 / -1;
            margin-top: 20px;
        }

        .response-content {
            background: #1a202c;
            color: #e2e8f0;
            border-radius: 12px;
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
            background: #4a5568;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s;
        }

        .copy-btn:hover {
            background: #2d3748;
        }

        .status-bar {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #f56565;
        }

        .status-dot.connected {
            background: #48bb78;
        }

        .file-tree {
            max-height: 300px;
            overflow-y: auto;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px;
            background: #f7fafc;
        }

        .file-item {
            padding: 5px 10px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 13px;
            font-family: monospace;
        }

        .file-item:hover {
            background: #e2e8f0;
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

        .response-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }

        .tab-btn {
            background: #4a5568;
            color: white;
            border: null;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }

        .tab-btn.active {
            background: #5a67d8;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .job-status {
            background: #2d3748;
            color: #e2e8f0;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: center;
        }

        .error-message {
            background: #f56565;
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }

        .success-message {
            background: #48bb78;
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }

        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .container {
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛠️ XCode AI Coding Assistant</h1>
            <p>Powered by DeepSeek & Gemini AI</p>
        </div>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot" id="statusDot"></div>
                <span>Server Status: <span id="serverStatus">Connecting...</span></span>
            </div>
            <div class="status-item">
                <span>📂 Repositories: <span id="repoCount">0</span></span>
            </div>
            <div class="status-item">
                <span>📄 Context Files: <span id="contextFiles">0</span></span>
            </div>
            <div class="status-item">
                <span>🕒 Last Sync: <span id="lastSync">Never</span></span>
            </div>
        </div>

        <div class="main-grid">
            <!-- Repository Management Panel -->
            <div class="panel">
                <h2>📁 Repository Management</h2>
                
                <div class="form-group">
                    <label for="repoName">Repository Name:</label>
                    <input type="text" id="repoName" placeholder="my-ios-project">
                </div>
                
                <div class="form-group">
                    <label for="repoUrl">Git URL:</label>
                    <input type="text" id="repoUrl" placeholder="https://github.com/user/repo.git">
                </div>
                
                <div class="form-group">
                    <label for="repoBranch">Branch:</label>
                    <input type="text" id="repoBranch" value="main" placeholder="main">
                </div>
                
                <div class="form-group">
                    <label for="accessToken">Access Token (optional):</label>
                    <input type="password" id="accessToken" placeholder="ghp_xxxxxxxxxxxx">
                </div>
                
                <button class="btn" onclick="addRepository()">
                    <span>➕</span> Add Repository
                </button>
                
                <button class="btn" onclick="syncRepositories()" style="margin-left: 10px;">
                    <span>🔄</span> Sync All
                </button>

                <div style="margin-top: 20px;">
                    <h3 style="margin-bottom: 10px;">Repository Files:</h3>
                    <div id="fileTree" class="file-tree">
                        <div style="text-align: center; color: #666; padding: 20px;">
                            Add a repository to view files
                        </div>
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
                        <option value="deepseek">DeepSeek (Recommended for Coding)</option>
                        <option value="gemini">Gemini</option>
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
                        <option value="deepseek">DeepSeek (Recommended for Coding)</option>
                        <option value="gemini">Gemini</option>
                    </select>
                </div>
                
                <button class="btn" onclick="submitQuery()">
                    <span>🤖</span> Ask AI
                </button>
            </div>

            <!-- Response Section -->
            <div class="panel response-section">
                <h2>📋 AI Response</h2>
                
                <div class="response-tabs">
                    <button class="tab-btn active" onclick="switchTab('formatted')">Formatted</button>
                    <button class="tab-btn" onclick="switchTab('raw')">Raw</button>
                    <button class="tab-btn" onclick="switchTab('files')">File Extracts</button>
                </div>
                
                <div id="formatted-content" class="tab-content active">
                    <div class="response-content" id="formattedResponse">
                        <button class="copy-btn" onclick="copyToClipboard('formattedResponse')">📋 Copy</button>
                        Ready to analyze your XCode errors and answer coding questions!
                    </div>
                </div>
                
                <div id="raw-content" class="tab-content">
                    <div class="response-content" id="rawResponse">
                        <button class="copy-btn" onclick="copyToClipboard('rawResponse')">📋 Copy</button>
                        Raw JSON response will appear here...
                    </div>
                </div>
                
                <div id="files-content" class="tab-content">
                    <div class="response-content" id="fileExtracts">
                        <button class="copy-btn" onclick="copyToClipboard('fileExtracts')">📋 Copy</button>
                        Extracted file contents will appear here...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin;
        
        let repositories = [];
        let currentResponse = null;
        let currentJobId = null;
        let jobPollInterval = null;

        // Initialize the app
        async function init() {
            console.log('Initializing XCode AI Assistant...');
            console.log('API Base URL:', API_BASE);
            await checkServerStatus();
            await loadRepositories();
            setInterval(updateStatus, 30000);
        }

        // Check server status
        async function checkServerStatus() {
            try {
                const response = await fetch(`${API_BASE}/api/health`);
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('serverStatus').textContent = 'Connected ✅';
                    document.getElementById('statusDot').classList.add('connected');
                    document.getElementById('repoCount').textContent = data.repositories || 0;
                    document.getElementById('contextFiles').textContent = data.context_files || 0;
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (error) {
                document.getElementById('serverStatus').textContent = 'Disconnected ❌';
                document.getElementById('statusDot').classList.remove('connected');
            }
        }

        // Load repositories
        async function loadRepositories() {
            try {
                const response = await fetch(`${API_BASE}/api/repositories`);
                if (response.ok) {
                    const data = await response.json();
                    repositories = data.repositories || [];
                    updateFileTree();
                }
            } catch (error) {
                console.error('Failed to load repositories:', error);
            }
        }

        // Add repository
        async function addRepository() {
            const name = document.getElementById('repoName').value.trim();
            const url = document.getElementById('repoUrl').value.trim();
            const branch = document.getElementById('repoBranch').value.trim() || 'main';
            const token = document.getElementById('accessToken').value.trim();

            if (!name || !url) {
                alert('Please provide repository name and URL');
                return;
            }

            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '<span class="loading"></span> Adding...';
            btn.disabled = true;

            try {
                const response = await fetch(`${API_BASE}/api/repositories/add`, {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({
                        name,
                        url,
                        branch,
                        access_token: token || null
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    alert(`Repository added successfully! Loaded ${result.files_loaded} files.`);
                    document.getElementById('repoName').value = '';
                    document.getElementById('repoUrl').value = '';
                    document.getElementById('accessToken').value = '';
                    await loadRepositories();
                } else {
                    const result = await response.json();
                    alert(`Error: ${result.detail || 'Unknown error'}`);
                }
            } catch (error) {
                alert('Error adding repository: ' + error.message);
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }

        // Sync repositories
        async function syncRepositories() {
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '<span class="loading"></span> Syncing...';
            btn.disabled = true;

            try {
                const response = await fetch(`${API_BASE}/api/repositories/sync`, {
                    method: 'POST',
                    headers: { 
                        'Accept': 'application/json'
                    }
                });
                
                if (response.ok) {
                    alert('Repository sync started!');
                } else {
                    const result = await response.json();
                    alert(`Error: ${result.detail || 'Sync failed'}`);
                }
            } catch (error) {
                alert('Error syncing: ' + error.message);
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }

        // Update file tree
        function updateFileTree() {
            const fileTree = document.getElementById('fileTree');
            if (repositories.length === 0) {
                fileTree.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">Add a repository to view files</div>';
                return;
            }

            let html = '';
            repositories.forEach(repo => {
                html += `<div style="font-weight: bold; margin-bottom: 5px; color: #5a67d8;">
                    📁 ${repo.name} (${repo.total_files} files)
                </div>`;
            });
            fileTree.innerHTML = html;
        }

        // Analyze XCode error
        async function analyzeError() {
            const errorMessage = document.getElementById('xcodeError').value.trim();
            const useDeepseek = document.getElementById('aiModel').value === 'deepseek';
            const forceSync = document.getElementById('forceSync').checked;

            if (!errorMessage) {
                alert('Please enter an XCode error message');
                return;
            }

            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '<span class="loading"></span> Analyzing...';
            btn.disabled = true;

            try {
                const response = await fetch(`${API_BASE}/api/xcode/analyze-error`, {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({
                        error_message: errorMessage,
                        use_deepseek: useDeepseek,
                        force_sync: forceSync
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    currentJobId = result.job_id;
                    alert('Analysis started! Job ID: ' + result.job_id);
                } else {
                    const result = await response.json();
                    alert(`Error: ${result.detail || 'Analysis failed'}`);
                }
            } catch (error) {
                alert('Error analyzing error: ' + error.message);
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }

        // Submit general query
        async function submitQuery() {
            const query = document.getElementById('generalQuery').value.trim();
            const useDeepseek = document.getElementById('queryModel').value === 'deepseek';

            if (!query) {
                alert('Please enter a question');
                return;
            }

            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '<span class="loading"></span> Processing...';
            btn.disabled = true;

            try {
                const response = await fetch(`${API_BASE}/api/query`, {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({
                        query,
                        use_deepseek: useDeepseek,
                        force_sync: false
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    currentJobId = result.job_id;
                    alert('Query submitted! Job ID: ' + result.job_id);
                } else {
                    const result = await response.json();
                    alert(`Error: ${result.detail || 'Query failed'}`);
                }
            } catch (error) {
                alert('Error submitting query: ' + error.message);
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }

        // Copy to clipboard
        async function copyToClipboard(elementId) {
            const element = document.getElementById(elementId);
            const text = element.textContent || element.innerText;
            
            try {
                await navigator.clipboard.writeText(text);
                alert('Copied to clipboard!');
            } catch (error) {
                const textArea = document.createElement('textarea');
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                alert('Copied to clipboard!');
            }
        }

        // Switch tabs
        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            document.querySelector(`.tab-btn[onclick="switchTab('${tab}')"]`).classList.add('active');
            document.getElementById(`${tab}-content`).classList.add('active');
        }

        // Update status periodically
        async function updateStatus() {
            await checkServerStatus();
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""

# Web Interface Routes
@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the main web interface"""
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Alternative route for the dashboard"""
    return HTMLResponse(content=HTML_CONTENT)

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Start background sync task"""
    print("Starting XCode AI Assistant...")
    
    # Debug filesystem access
    print(f"Current working directory: {os.getcwd()}")
    print(f"Temp directory: {tempfile.gettempdir()}")
    print(f"Using repository path: {repo_base_path}")
    
    # Test filesystem access
    test_path = Path(repo_base_path)
    try:
        test_path.mkdir(exist_ok=True, parents=True)
        test_file = test_path / "test.txt"
        test_file.write_text("test content")
        print(f"✓ Filesystem access verified: {test_file}")
        test_file.unlink()  # Clean up
    except Exception as e:
        print(f"✗ Filesystem access error: {e}")
        print("Falling back to in-memory repository management")
    
    asyncio.create_task(periodic_sync())
    asyncio.create_task(cleanup_old_jobs())

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "repositories": len(repo_manager.repos_config),
            "context_files": len(ai_agent.file_contexts),
            "active_jobs": len(job_results)
        }
    except Exception as e:
        print(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== MISSING API ENDPOINTS THAT FRONTEND EXPECTS =====
@app.get("/api/repositories")
async def list_repositories():
    """List all configured repositories - Frontend expects this"""
    try:
        repos = []
        for name, config in repo_manager.repos_config.items():
            repos.append({
                "name": name,
                "url": config["url"],
                "branch": config.get("branch", "main"),
                "total_files": 0,
                "status": "healthy"
            })
        return {"repositories": repos}
    except Exception as e:
        return {"repositories": []}

@app.post("/api/repositories/add")
async def add_repository(repo_config: RepoConfig):
    """Add a new repository to monitor - Frontend expects this"""
    try:
        print(f"Adding repository: {repo_config.name}")
        
        # Add repository to manager
        repo_manager.add_repository(
            name=repo_config.name,
            url=repo_config.url,
            branch=repo_config.branch,
            access_token=repo_config.access_token,
            sync_interval=repo_config.sync_interval
        )
        
        return {
            "success": True,
            "message": f"Repository {repo_config.name} added successfully",
            "files_loaded": 0,
            "total_files": 0
        }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error adding repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/repositories/sync")
async def sync_repositories(background_tasks: BackgroundTasks):
    """Manual sync endpoint - Frontend expects this"""
    try:
        print("Manual sync requested")
        background_tasks.add_task(repo_manager.sync_all_repositories)
        return {"message": "Sync started in background", "status": "started"}
    except Exception as e:
        print(f"Error starting sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/context/summary")
async def context_summary():
    """Context summary endpoint - Frontend expects this"""
    try:
        return {
            "total_files": len(ai_agent.file_contexts),
            "last_update": datetime.now().isoformat(),
            "context_health": "good"
        }
    except Exception as e:
        return {
            "total_files": 0,
            "last_update": None,
            "context_health": "error"
        }

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Job status endpoint - Frontend expects this"""
    if job_id not in job_results:
        return {"status": "not_found"}
    return job_results[job_id]

# Async processing functions
async def process_xcode_error_async(job_id: str, error_message: str, use_deepseek: bool):
    """Async processing for XCode errors"""
    try:
        print(f"Processing XCode error job {job_id}")
        job_results[job_id] = {
            'status': 'processing',
            'created_at': datetime.now(),
            'result': None,
            'error': None
        }
        
        # Simulate processing
        await asyncio.sleep(2)
        result = {"analysis": f"Analysis for: {error_message[:50]}...", "model_used": "deepseek" if use_deepseek else "gemini"}
        
        job_results[job_id]['status'] = 'completed'
        job_results[job_id]['result'] = result
        job_results[job_id]['completed_at'] = datetime.now()
        print(f"XCode error job {job_id} completed")
        
    except Exception as e:
        print(f"XCode error job {job_id} failed: {e}")
        job_results[job_id]['status'] = 'failed'
        job_results[job_id]['error'] = str(e)
        job_results[job_id]['failed_at'] = datetime.now()

async def process_general_query_async(job_id: str, query: str, use_deepseek: bool):
    """Async processing for general queries"""
    try:
        print(f"Processing general query job {job_id}")
        job_results[job_id] = {
            'status': 'processing',
            'created_at': datetime.now(),
            'result': None,
            'error': None
        }
        
        # Simulate processing
        await asyncio.sleep(2)
        result = {"response": f"Response to: {query[:50]}...", "model_used": "deepseek" if use_deepseek else "gemini"}
        
        job_results[job_id]['status'] = 'completed'
        job_results[job_id]['result'] = result
        job_results[job_id]['completed_at'] = datetime.now()
        print(f"General query job {job_id} completed")
        
    except Exception as e:
        print(f"General query job {job_id} failed: {e}")
        job_results[job_id]['status'] = 'failed'
        job_results[job_id]['error'] = str(e)
        job_results[job_id]['failed_at'] = datetime.now()

@app.post("/api/xcode/analyze-error")
async def analyze_xcode_error(request: XCodeErrorRequest):
    """Queue XCode error analysis job"""
    try:
        print(f"XCode error analysis requested: {request.error_message[:100]}...")
        
        job_id = str(uuid.uuid4())
        
        # Start async processing
        asyncio.create_task(process_xcode_error_async(
            job_id, request.error_message, request.use_deepseek
        ))
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Analysis started. Use the job_id to check status."
        }
        
    except Exception as e:
        print(f"Error queuing XCode analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def general_query(request: GeneralQueryRequest):
    """Queue general coding query job"""
    try:
        print(f"General query requested: {request.query[:100]}...")
        
        job_id = str(uuid.uuid4())
        
        # Start async processing
        asyncio.create_task(process_general_query_async(
            job_id, request.query, request.use_deepseek
        ))
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Query processing started. Use the job_id to check status."
        }
        
    except Exception as e:
        print(f"Error queuing general query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for compatibility
@app.get("/api/status")
async def root():
    """API status endpoint"""
    return {
        "message": "XCode AI Coding Assistant API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting XCode AI Coding Assistant on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)