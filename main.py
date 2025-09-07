from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

# HTML content as string
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
            border: none;
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
            <h1>🍎 XCode AI Coding Assistant</h1>
            <p>Powered by DeepSeek & Gemini AI</p>
        </div>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot"></div>
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
                    <textarea id="xcodeError" placeholder="Paste your XCode error message here...

Example:
error: use of unresolved identifier 'someVariable'
  --> MyViewController.swift:45:12
   |
45 |     print(someVariable)
   |            ^^^^^^^^^^^
   |"></textarea>
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
                    <textarea id="generalQuery" placeholder="Ask anything about your code...

Examples:
- How can I optimize this function?
- Add error handling to my network request
- Convert this to use async/await
- Add unit tests for this class"></textarea>
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
                        
                        Tips:
                        • Add your repositories first
                        • Paste complete error messages for best results  
                        • The AI has access to all your project files for context
                        • Use "Force Sync" if you've made recent changes
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
        const API_BASE = window.location.origin + '/api'; // Updated to use /api prefix
        
        let repositories = [];
        let currentResponse = null;

        // Initialize the app
        async function init() {
            await checkServerStatus();
            await loadRepositories();
            setInterval(updateStatus, 30000); // Update every 30 seconds
        }

        // Check server status
        async function checkServerStatus() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                document.getElementById('serverStatus').textContent = 'Connected ✅';
                document.getElementById('repoCount').textContent = data.repositories || 0;
                document.getElementById('contextFiles').textContent = data.context_files || 0;
            } catch (error) {
                document.getElementById('serverStatus').textContent = 'Disconnected ❌';
                console.error('Server status check failed:', error);
            }
        }

        // Load repositories
        async function loadRepositories() {
            try {
                const response = await fetch(`${API_BASE}/repositories`);
                const data = await response.json();
                repositories = data.repositories;
                updateFileTree();
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
                const response = await fetch(`${API_BASE}/repositories/add`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name,
                        url,
                        branch,
                        access_token: token || null
                    })
                });

                const result = await response.json();
                
                if (response.ok) {
                    alert(`Repository added successfully! Loaded ${result.files_loaded} files.`);
                    // Clear form
                    document.getElementById('repoName').value = '';
                    document.getElementById('repoUrl').value = '';
                    document.getElementById('accessToken').value = '';
                    await loadRepositories();
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                alert(`Error adding repository: ${error.message}`);
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
                const response = await fetch(`${API_BASE}/repositories/sync`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    alert('Repository sync started!');
                    setTimeout(updateStatus, 2000);
                } else {
                    alert('Error starting sync');
                }
            } catch (error) {
                alert(`Error syncing: ${error.message}`);
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }

        // Update file tree
        function updateFileTree() {
            const fileTree = document.getElementById('fileTree');
            
            if (repositories.length === 0) {
                fileTree.innerHTML = `<div style="text-align: center; color: #666; padding: 20px;">
                    Add a repository to view files
                </div>`;
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
                const response = await fetch(`${API_BASE}/xcode/analyze-error`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        error_message: errorMessage,
                        use_deepseek: useDeepseek,
                        force_sync: forceSync
                    })
                });

                const result = await response.json();
                
                if (response.ok) {
                    displayResponse(result);
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                alert(`Error analyzing error: ${error.message}`);
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
            btn.innerHTML = '<span class="loading"></span> Thinking...';
            btn.disabled = true;

            try {
                const response = await fetch(`${API_BASE}/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        use_deepseek: useDeepseek,
                        force_sync: false
                    })
                });

                const result = await response.json();
                
                if (response.ok) {
                    displayResponse(result);
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                alert(`Error submitting query: {error.message}`);
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }

        // Display response
        function displayResponse(result) {
            currentResponse = result;
            
            // Formatted response
            const formattedResponse = document.getElementById('formattedResponse');
            formattedResponse.innerHTML = `<button class="copy-btn" onclick="copyToClipboard('formattedResponse')">📋 Copy</button>${formatResponse(result.analysis || result.response)}`;
            
            // Raw response
            const rawResponse = document.getElementById('rawResponse');
            rawResponse.innerHTML = `<button class="copy-btn" onclick="copyToClipboard('rawResponse')">📋 Copy</button>${JSON.stringify(result, null, 2)}`;
            
            // Extract files from response
            extractFiles(result.analysis || result.response);
            
            // Switch to formatted tab
            switchTab('formatted');
        }

        // Format response for better readability
        function formatResponse(text) {
            // Add syntax highlighting and formatting
            return text
                .replace(/```(\w+)?\n([\s\S]*?)\n```/g, '<div style="background: #2d3748; padding: 15px; border-radius: 8px; margin: 10px 0; overflow-x: auto;"><pre style="margin: 0; color: #e2e8f0;">$2</pre></div>')
                .replace(/\*\*(.*?)\*\*/g, '<strong style="color: #5a67d8;">$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/\n/g, '<br>');
        }

        // Extract file contents from AI response
        function extractFiles(text) {
            const fileExtracts = document.getElementById('fileExtracts');
            const fileMatches = text.match(/```\w*\n([\s\S]*?)\n```/g);
            
            if (fileMatches) {
                let extractedContent = '<button class="copy-btn" onclick="copyToClipboard(\'fileExtracts\')">📋 Copy All Files</button>\n\n';
                fileMatches.forEach((match, index) => {
                    const content = match.replace(/```\w*\n/, '').replace(/\n```$/, '');
                    extractedContent += `// File ${index + 1}\n${content}\n\n${'='.repeat(50)}\n\n`;
                });
                fileExtracts.innerHTML = extractedContent;
            } else {
                fileExtracts.innerHTML = '<button class="copy-btn" onclick="copyToClipboard(\'fileExtracts\')">📋 Copy</button>No file contents extracted from response.';
            }
        }

        // Switch tabs
        function switchTab(tab) {
            // Remove active class from all tabs
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            // Add active class to selected tab
            document.querySelector(`.tab-btn[onclick="switchTab('${tab}')"]`).classList.add('active');
            document.getElementById(`${tab}-content`).classList.add('active');
        }

        // Copy to clipboard
        async function copyToClipboard(elementId) {
            const element = document.getElementById(elementId);
            const text = element.textContent || element.innerText;
            
            try {
                await navigator.clipboard.writeText(text);
                
                // Visual feedback
                const btn = element.querySelector('.copy-btn');
                const originalText = btn.textContent;
                btn.textContent = '✅ Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 2000);
            } catch (error) {
                console.error('Copy failed:', error);
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
            }
        }

        // Update status periodically
        async function updateStatus() {
            await checkServerStatus();
            
            try {
                const contextResponse = await fetch(`${API_BASE}/context/summary`);
                const contextData = await contextResponse.json();
                document.getElementById('contextFiles').textContent = contextData.total_files || 0;
                
                if (contextData.last_update) {
                    const lastUpdate = new Date(contextData.last_update);
                    document.getElementById('lastSync').textContent = lastUpdate.toLocaleTimeString();
                }
            } catch (error) {
                console.error('Failed to update context status:', error);
            }
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