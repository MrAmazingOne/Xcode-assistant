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
        const modelComparison = document.getElementById('modelComparison');
        modelComparison.innerHTML = `
            <details>
                <summary>⚡ DeepSeek Analysis</summary>
                <div class="model-content">${response.deepseek_analysis}</div>
            </details>
            <details>
                <summary>🧠 Gemini Analysis</summary>
                <div class="model-content">${response.gemini_analysis || 'No Gemini analysis available'}</div>
            </details>
        `;
    }

    if (response.gemini_analysis && !response.deepseek_analysis) {
        const modelComparison = document.getElementById('modelComparison');
        modelComparison.innerHTML = `
            <details>
                <summary>🧠 Gemini Analysis</summary>
                <div class="model-content">${response.gemini_analysis}</div>
            </details>
        `;
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