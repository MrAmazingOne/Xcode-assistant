import os
import git
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import shutil
import hashlib
import fnmatch
import tempfile
import time

class GitRepoManager:
    def __init__(self, base_path: str = None):
        # Use Render's temp dir or system temp dir
        if base_path is None:
            render_temp = os.getenv('RENDER_TEMP_DIR', tempfile.gettempdir())
            base_path = os.path.join(render_temp, "repos")
        
        self.base_path = Path(base_path)
        
        # Create directory with proper permissions
        try:
            self.base_path.mkdir(exist_ok=True, parents=True)
            print(f"✅ Using repository directory: {self.base_path}")
        except PermissionError:
            # Fallback to a different directory if we can't create this one
            fallback_path = Path(tempfile.gettempdir()) / "xcode_repos"
            fallback_path.mkdir(exist_ok=True, parents=True)
            self.base_path = fallback_path
            print(f"⚠️ Using fallback repository directory: {self.base_path}")
        
        self.repos_config = {}
        self.last_sync = {}
        self.file_hashes = {}
        self.sync_locks = {}  # Prevent concurrent syncs
        self.sync_progress = {}  # Track sync progress
        
        # Enhanced file filtering configuration
        self.critical_extensions = {
            '.swift', '.m', '.h', '.mm'  # iOS/macOS native files - highest priority
        }
        
        self.important_extensions = {
            '.py', '.js', '.ts', '.json',  # Backend/config
            '.plist', '.xml', '.yaml', '.yml',  # Config files
            '.md', '.txt', '.gitignore'  # Documentation
        }
        
        self.exclude_patterns = {
            '*.xcworkspace/*', '*.xcodeproj/*', '.git/*', 
            'node_modules/*', '__pycache__/*', '*.pyc', '.DS_Store',
            'build/*', 'Build/*', 'DerivedData/*', '.build/*',
            '*.framework/*', '*.a', '*.so', '*.dylib',
            'Pods/*', 'Carthage/*', 'Package.resolved',
            '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico',  # Images
            '*.pdf', '*.zip', '*.tar.gz'  # Binaries
        }
        
        # File size limits based on importance
        self.file_size_limits = {
            'critical': 500 * 1024,    # 500KB for Swift/ObjC files
            'important': 200 * 1024,   # 200KB for other code files
            'regular': 50 * 1024       # 50KB for config files
        }
        
        # GitHub API integration for faster file access
        self.github_api_token = None
        
    def set_github_token(self, token: str):
        """Set GitHub API token for faster file access"""
        self.github_api_token = token
        
    def add_repository(self, name: str, url: str, branch: str = "main", 
                      access_token: str = None, sync_interval: int = 300):
        """Add a git repository to monitor with enhanced validation"""
        if name in self.repos_config:
            raise ValueError(f"Repository {name} already exists")
            
        # Validate URL format
        if not (url.startswith('https://') or url.startswith('git@')):
            raise ValueError("Repository URL must start with https:// or git@")
            
        self.repos_config[name] = {
            "url": url,
            "branch": branch,
            "access_token": access_token,
            "sync_interval": sync_interval,
            "local_path": self.base_path / name,
            "created_at": datetime.now(),
            "sync_count": 0,
            "error_count": 0,
            "last_error": None,
            "files_processed": 0,
            "critical_files": 0,
            "sync_duration": 0
        }
        
        # Initialize sync lock and progress tracking
        self.sync_locks[name] = asyncio.Lock()
        self.sync_progress[name] = {"status": "idle", "progress": 0, "message": ""}
        
        # Set GitHub token if this is a GitHub repo
        if "github.com" in url and access_token:
            self.github_api_token = access_token
        
    async def clone_or_update_repo_with_timeout(self, repo_name: str, max_duration: int = 25) -> Tuple[bool, str, int]:
        """Clone/update repository with timeout handling for Render's 30-second limit"""
        if repo_name not in self.repos_config:
            return False, f"Repository {repo_name} not configured", 0
            
        # Use lock to prevent concurrent syncs
        async with self.sync_locks[repo_name]:
            try:
                return await asyncio.wait_for(
                    self._do_sync_with_progress(repo_name),
                    timeout=max_duration
                )
            except asyncio.TimeoutError:
                # Handle timeout gracefully
                self.sync_progress[repo_name] = {
                    "status": "timeout", 
                    "progress": 50, 
                    "message": "Sync timeout - will retry in background"
                }
                
                # Schedule background completion
                asyncio.create_task(self._complete_sync_in_background(repo_name))
                
                return False, f"Repository {repo_name} sync timeout - continuing in background", 0
    
    async def _complete_sync_in_background(self, repo_name: str):
        """Complete sync operation in background without timeout constraints"""
        print(f"🔄 Completing sync for {repo_name} in background...")
        
        try:
            success, message, file_count = await self._do_sync_with_progress(repo_name, background=True)
            
            self.sync_progress[repo_name] = {
                "status": "completed" if success else "failed",
                "progress": 100,
                "message": message
            }
            
            print(f"✅ Background sync completed for {repo_name}: {message}")
            
        except Exception as e:
            print(f"❌ Background sync failed for {repo_name}: {e}")
            self.sync_progress[repo_name] = {
                "status": "failed",
                "progress": 0,
                "message": str(e)
            }
    
    async def _do_sync_with_progress(self, repo_name: str, background: bool = False) -> Tuple[bool, str, int]:
        """Internal sync method with progress tracking"""
        config = self.repos_config[repo_name]
        local_path = config["local_path"]
        start_time = time.time()
        
        self.sync_progress[repo_name] = {"status": "syncing", "progress": 10, "message": "Starting sync..."}
        
        try:
            if local_path.exists():
                # Repository exists, pull latest changes
                self.sync_progress[repo_name]["message"] = "Pulling latest changes..."
                self.sync_progress[repo_name]["progress"] = 30
                
                repo = git.Repo(local_path)
                
                # Verify we're on the correct branch
                current_branch = repo.active_branch.name
                if current_branch != config["branch"]:
                    self.sync_progress[repo_name]["message"] = f"Switching to branch {config['branch']}..."
                    
                    origin = repo.remotes.origin
                    origin.fetch()
                    
                    if config["branch"] in [ref.name.split('/')[-1] for ref in origin.refs]:
                        repo.git.checkout(config["branch"])
                    else:
                        return False, f"Branch {config['branch']} not found", 0
                
                # Pull latest changes
                self.sync_progress[repo_name]["progress"] = 50
                origin = repo.remotes.origin
                pull_info = origin.pull(config["branch"])
                
                message = f"Updated repository: {repo_name}"
                
            else:
                # Clone repository
                self.sync_progress[repo_name]["message"] = "Cloning repository..."
                self.sync_progress[repo_name]["progress"] = 20
                
                auth_url = self._get_authenticated_url(config["url"], config.get("access_token"))
                
                # Use shallow clone for faster performance
                repo = git.Repo.clone_from(
                    auth_url, 
                    local_path, 
                    branch=config["branch"],
                    depth=1  # Shallow clone
                )
                
                message = f"Cloned repository: {repo_name}"
            
            # Process files with progress tracking
            self.sync_progress[repo_name]["message"] = "Processing files..."
            self.sync_progress[repo_name]["progress"] = 70
            
            relevant_files = await self._process_files_with_priority(repo_name, background)
            
            # Update statistics
            duration = time.time() - start_time
            self.last_sync[repo_name] = datetime.now()
            config["sync_count"] += 1
            config["last_error"] = None
            config["sync_duration"] = duration
            config["files_processed"] = len(relevant_files)
            config["critical_files"] = sum(1 for f in relevant_files if self._is_critical_file(f))
            
            self.sync_progress[repo_name] = {
                "status": "completed", 
                "progress": 100, 
                "message": f"Processed {len(relevant_files)} files"
            }
            
            final_message = f"{message} ({len(relevant_files)} files processed in {duration:.1f}s)"
            print(f"✅ {final_message}")
            
            return True, final_message, len(relevant_files)
            
        except git.exc.GitError as e:
            error_msg = f"Git error syncing repository {repo_name}: {str(e)}"
            print(f"❌ {error_msg}")
            config["error_count"] += 1
            config["last_error"] = error_msg
            self.sync_progress[repo_name] = {"status": "failed", "progress": 0, "message": error_msg}
            return False, error_msg, 0
            
        except Exception as e:
            error_msg = f"Error syncing repository {repo_name}: {str(e)}"
            print(f"❌ {error_msg}")
            config["error_count"] += 1
            config["last_error"] = error_msg
            self.sync_progress[repo_name] = {"status": "failed", "progress": 0, "message": error_msg}
            return False, error_msg, 0
    
    async def _process_files_with_priority(self, repo_name: str, background: bool = False) -> List[str]:
        """Process files with priority system and smart batching"""
        all_files = self._list_all_files(repo_name)
        
        # Categorize files by priority
        critical_files = [f for f in all_files if self._is_critical_file(f)]
        important_files = [f for f in all_files if self._is_important_file(f) and f not in critical_files]
        other_files = [f for f in all_files if f not in critical_files and f not in important_files]
        
        processed_files = []
        
        # Always process critical files (Swift, Obj-C)
        for file_path in critical_files[:50]:  # Limit for timeout protection
            if not background and len(processed_files) > 30:  # Limit for quick sync
                break
            processed_files.append(file_path)
        
        # Process important files if we have time/capacity
        if background or len(processed_files) < 20:
            for file_path in important_files[:30]:
                if not background and len(processed_files) > 40:
                    break
                processed_files.append(file_path)
        
        # Process other files only in background or if we have minimal files
        if background:
            for file_path in other_files[:20]:
                processed_files.append(file_path)
        
        print(f"📁 {repo_name}: Processed {len(processed_files)} files "
              f"({len(critical_files)} critical, {len(important_files)} important)")
        
        return processed_files
    
    def _is_critical_file(self, file_path: str) -> bool:
        """Check if file is critical (Swift, Objective-C)"""
        return any(file_path.endswith(ext) for ext in self.critical_extensions)
    
    def _is_important_file(self, file_path: str) -> bool:
        """Check if file is important (Python, JS, config files)"""
        return any(file_path.endswith(ext) for ext in self.important_extensions)
    
    def _get_authenticated_url(self, url: str, token: str) -> str:
        """Add authentication token to git URL"""
        if not token:
            return url
        
        if "github.com" in url:
            return url.replace("https://", f"https://{token}@")
        elif "gitlab.com" in url:
            return url.replace("https://", f"https://oauth2:{token}@")
        elif "bitbucket.org" in url:
            return url.replace("https://", f"https://{token}@")
        
        return url
    
    async def sync_all_repositories_batch(self, batch_size: int = 3, max_duration: int = 25) -> Dict[str, Tuple[bool, str, int]]:
        """Sync repositories in batches to handle timeout constraints"""
        if not self.repos_config:
            return {}
        
        repos_to_sync = [repo_name for repo_name in self.repos_config if self._should_sync(repo_name)]
        
        if not repos_to_sync:
            print("📝 No repositories need syncing")
            return {}
        
        print(f"🔄 Syncing {len(repos_to_sync)} repositories in batches of {batch_size}...")
        
        all_results = {}
        
        # Process repositories in batches
        for i in range(0, len(repos_to_sync), batch_size):
            batch = repos_to_sync[i:i + batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}: {batch}")
            
            # Sync batch concurrently with timeout
            batch_tasks = [self.clone_or_update_repo_with_timeout(repo_name, max_duration) 
                          for repo_name in batch]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for repo_name, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        all_results[repo_name] = (False, f"Exception: {str(result)}", 0)
                    else:
                        all_results[repo_name] = result
                
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                for repo_name in batch:
                    all_results[repo_name] = (False, f"Batch error: {str(e)}", 0)
        
        return all_results
    
    def _should_sync(self, repo_name: str) -> bool:
        """Check if repository should be synced based on interval"""
        if repo_name not in self.last_sync:
            return True
            
        config = self.repos_config[repo_name]
        time_since_sync = datetime.now() - self.last_sync[repo_name]
        
        # Force sync if there were previous errors
        if config.get("error_count", 0) > 0 and time_since_sync.total_seconds() > 60:
            return True
            
        return time_since_sync.total_seconds() > config["sync_interval"]
    
    def get_file_content(self, repo_name: str, file_path: str) -> Optional[str]:
        """Get content of a specific file with enhanced encoding handling"""
        if repo_name not in self.repos_config:
            return None
            
        local_path = self.repos_config[repo_name]["local_path"]
        full_path = local_path / file_path
        
        try:
            if not full_path.exists() or not full_path.is_file():
                return None
                
            # Check file size based on type
            file_size = full_path.stat().st_size
            max_size = self._get_max_file_size(file_path)
            
            if file_size > max_size:
                return f"File too large ({file_size} bytes, max {max_size}) - skipped for context"
            
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(full_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail
            return f"Unable to decode file {file_path} - binary or unsupported encoding"
                
        except Exception as e:
            print(f"❌ Error reading file {file_path} from {repo_name}: {str(e)}")
        
        return None
    
    def _get_max_file_size(self, file_path: str) -> int:
        """Get maximum file size based on file type"""
        if self._is_critical_file(file_path):
            return self.file_size_limits['critical']
        elif self._is_important_file(file_path):
            return self.file_size_limits['important']
        else:
            return self.file_size_limits['regular']
    
    def _list_all_files(self, repo_name: str) -> List[str]:
        """List all relevant files in repository with smart filtering"""
        if repo_name not in self.repos_config:
            return []
            
        local_path = self.repos_config[repo_name]["local_path"]
        if not local_path.exists():
            return []
        
        files = []
        
        try:
            for root, dirs, filenames in os.walk(local_path):
                # Remove excluded directories
                dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]
                
                for filename in filenames:
                    if self._should_exclude_file(filename):
                        continue
                        
                    file_path = Path(root) / filename
                    rel_path = file_path.relative_to(local_path)
                    files.append(str(rel_path))
        
        except Exception as e:
            print(f"❌ Error walking repository {repo_name}: {e}")
            return []
        
        return self._sort_files_by_priority(files)
    
    def list_files(self, repo_name: str, extensions: List[str] = None, 
                  exclude_dirs: List[str] = None) -> List[str]:
        """Public interface for listing files"""
        return self._list_all_files(repo_name)
    
    def _sort_files_by_priority(self, files: List[str]) -> List[str]:
        """Sort files by priority for processing"""
        def get_priority(file_path: str) -> int:
            if self._is_critical_file(file_path):
                return 0  # Highest priority
            elif self._is_important_file(file_path):
                return 1  # Medium priority
            elif file_path.endswith(('.json', '.plist', '.xml', '.yaml', '.yml')):
                return 2  # Config files
            elif file_path.endswith(('.md', '.txt')):
                return 3  # Documentation
            else:
                return 4  # Lowest priority
        
        return sorted(files, key=get_priority)
    
    def _should_exclude_dir(self, dirname: str) -> bool:
        """Check if directory should be excluded"""
        exclude_dirs = {
            '.git', 'node_modules', '__pycache__', '.vscode', '.idea',
            'build', 'Build', 'DerivedData', '.build', 'dist',
            'Pods', 'Carthage', '.bundle'
        }
        
        return dirname in exclude_dirs or any(
            fnmatch.fnmatch(dirname, pattern.split('/')[0]) 
            for pattern in self.exclude_patterns
        )
    
    def _should_exclude_file(self, filename: str) -> bool:
        """Check if file should be excluded"""
        # Direct filename exclusions
        if filename in {'.DS_Store', 'Package.resolved', 'Podfile.lock'}:
            return True
            
        # Pattern-based exclusions
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
                
        return False
    
    async def get_repository_files_github_api(self, repo_name: str, repo_url: str, branch: str = "main") -> List[Dict]:
        """Get repository files using GitHub API for faster access"""
        if not self.github_api_token:
            return []
            
        # Extract owner/repo from URL
        if "github.com" not in repo_url:
            return []
            
        parts = repo_url.replace("https://github.com/", "").replace(".git", "").split("/")
        if len(parts) != 2:
            return []
            
        owner, repo = parts
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        
        headers = {
            "Authorization": f"token {self.github_api_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {"path": item["path"], "type": item["type"], "size": item.get("size", 0)}
                            for item in data.get("tree", [])
                            if item["type"] == "blob" and not self._should_exclude_file(item["path"].split("/")[-1])
                        ]
        except Exception as e:
            print(f"❌ GitHub API error for {repo_name}: {e}")
        
        return []
    
    def get_sync_progress(self, repo_name: str) -> Dict:
        """Get current sync progress for a repository"""
        return self.sync_progress.get(repo_name, {"status": "unknown", "progress": 0, "message": ""})
    
    def get_repository_structure(self, repo_name: str) -> Dict:
        """Get complete repository structure with enhanced metadata"""
        if repo_name not in self.repos_config:
            return {}
            
        config = self.repos_config[repo_name]
        files = self._list_all_files(repo_name)
        
        # Calculate detailed statistics
        total_size = 0
        file_types = {}
        critical_files = 0
        important_files = 0
        
        for file_path in files[:100]:  # Limit for performance
            full_path = config["local_path"] / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                total_size += size
                
                ext = Path(file_path).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                
                if self._is_critical_file(file_path):
                    critical_files += 1
                elif self._is_important_file(file_path):
                    important_files += 1
        
        last_sync_time = self.last_sync.get(repo_name)
        progress = self.get_sync_progress(repo_name)
        
        structure = {
            "repository": repo_name,
            "url": config["url"],
            "branch": config["branch"],
            "local_path": str(config["local_path"]),
            "last_sync": last_sync_time.isoformat() if last_sync_time else "Never",
            "sync_count": config.get("sync_count", 0),
            "error_count": config.get("error_count", 0),
            "last_error": config.get("last_error"),
            "sync_duration": config.get("sync_duration", 0),
            "total_files": len(files),
            "critical_files": critical_files,
            "important_files": important_files,
            "total_size_kb": total_size // 1024,
            "file_types": file_types,
            "files": files[:50],  # Limit for UI
            "sync_progress": progress,
            "status": self._get_repo_health_status(repo_name),
            "performance_metrics": {
                "files_per_second": config.get("files_processed", 0) / max(config.get("sync_duration", 1), 1),
                "avg_sync_time": config.get("sync_duration", 0),
                "success_rate": (config.get("sync_count", 0) - config.get("error_count", 0)) / max(config.get("sync_count", 1), 1) * 100
            }
        }
        
        return structure
    
    def _get_repo_health_status(self, repo_name: str) -> str:
        """Get repository health status"""
        if repo_name not in self.repos_config:
            return "not_found"
            
        config = self.repos_config[repo_name]
        progress = self.get_sync_progress(repo_name)
        
        if progress["status"] == "syncing":
            return "syncing"
        elif progress["status"] == "timeout":
            return "timeout_recovery"
        elif config.get("error_count", 0) > 0:
            return "has_errors"
        elif not self.last_sync.get(repo_name):
            return "never_synced"
        else:
            time_since_sync = datetime.now() - self.last_sync[repo_name]
            if time_since_sync.total_seconds() > config["sync_interval"] * 2:
                return "sync_overdue"
            else:
                return "healthy"
    
    def get_sync_statistics(self) -> Dict[str, any]:
        """Get comprehensive sync statistics"""
        if not self.repos_config:
            return {
                "total_repos": 0,
                "healthy_repos": 0,
                "syncing_repos": 0,
                "repos_with_errors": 0,
                "total_files": 0,
                "critical_files": 0,
                "last_successful_sync": None,
                "avg_sync_time": 0
            }
        
        healthy_count = 0
        syncing_count = 0
        error_count = 0
        total_files = 0
        critical_files = 0
        total_sync_time = 0
        last_successful_sync = None
        
        for repo_name, config in self.repos_config.items():
            status = self._get_repo_health_status(repo_name)
            
            if status == "healthy":
                healthy_count += 1
            elif status == "syncing":
                syncing_count += 1
            elif "error" in status:
                error_count += 1
            
            files = self._list_all_files(repo_name)
            total_files += len(files)
            critical_files += sum(1 for f in files if self._is_critical_file(f))
            
            sync_duration = config.get("sync_duration", 0)
            total_sync_time += sync_duration
            
            repo_last_sync = self.last_sync.get(repo_name)
            if repo_last_sync:
                if not last_successful_sync or repo_last_sync > last_successful_sync:
                    last_successful_sync = repo_last_sync
        
        return {
            "total_repos": len(self.repos_config),
            "healthy_repos": healthy_count,
            "syncing_repos": syncing_count,
            "repos_with_errors": error_count,
            "total_files": total_files,
            "critical_files": critical_files,
            "last_successful_sync": last_successful_sync.isoformat() if last_successful_sync else None,
            "avg_sync_time": total_sync_time / max(len(self.repos_config), 1),
            "performance": {
                "total_sync_time": total_sync_time,
                "files_per_repo": total_files / max(len(self.repos_config), 1),
                "critical_file_ratio": critical_files / max(total_files, 1) * 100
            }
        }