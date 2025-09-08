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

class GitRepoManager:
    def __init__(self, base_path: str = "/tmp/repos"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.repos_config = {}
        self.last_sync = {}
        self.file_hashes = {}
        self.sync_locks = {}  # Prevent concurrent syncs
        
        # File filtering configuration
        self.important_extensions = {
            '.swift', '.m', '.h', '.mm', '.cpp', '.c',  # iOS/Native
            '.py', '.js', '.ts', '.json',  # Backend/Config
            '.plist', '.xml', '.yaml', '.yml',  # Config files
            '.md', '.txt',  # Documentation
        }
        
        self.exclude_patterns = {
            '*.xcworkspace', '*.xcodeproj', '.git', '.gitignore',
            'node_modules', '__pycache__', '*.pyc', '.DS_Store',
            'build', 'Build', 'DerivedData', '.build',
            '*.framework', '*.a', '*.so', '*.dylib',
            'Pods', 'Carthage', 'Package.resolved'
        }
        
        self.max_file_size = 100 * 1024  # 100KB max per file
        
    def add_repository(self, name: str, url: str, branch: str = "main", 
                      access_token: str = None, sync_interval: int = 300):
        """Add a git repository to monitor with validation"""
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
            "last_error": None
        }
        
        # Initialize sync lock
        self.sync_locks[name] = asyncio.Lock()
        
    async def clone_or_update_repo(self, repo_name: str) -> Tuple[bool, str, int]:
        """Clone repository if not exists, otherwise pull latest changes"""
        if repo_name not in self.repos_config:
            return False, f"Repository {repo_name} not configured", 0
            
        # Use lock to prevent concurrent syncs
        async with self.sync_locks[repo_name]:
            return await self._do_sync(repo_name)
    
    async def _do_sync(self, repo_name: str) -> Tuple[bool, str, int]:
        """Internal sync method"""
        config = self.repos_config[repo_name]
        local_path = config["local_path"]
        
        try:
            if local_path.exists():
                # Repository exists, pull latest changes
                repo = git.Repo(local_path)
                
                # Verify we're on the correct branch
                current_branch = repo.active_branch.name
                if current_branch != config["branch"]:
                    # Switch to correct branch
                    origin = repo.remotes.origin
                    origin.fetch()
                    
                    if config["branch"] in [ref.name.split('/')[-1] for ref in origin.refs]:
                        repo.git.checkout(config["branch"])
                    else:
                        return False, f"Branch {config['branch']} not found", 0
                
                # Pull latest changes
                origin = repo.remotes.origin
                pull_info = origin.pull(config["branch"])
                
                # Count changed files
                changed_files = len(self.get_changed_files(repo_name))
                
                message = f"Updated repository: {repo_name} ({changed_files} files changed)"
                print(message)
                
            else:
                # Clone repository
                auth_url = self._get_authenticated_url(config["url"], config.get("access_token"))
                
                # Clone with specific branch
                repo = git.Repo.clone_from(
                    auth_url, 
                    local_path, 
                    branch=config["branch"],
                    depth=1  # Shallow clone for faster performance
                )
                
                # Count all relevant files
                file_count = len(self.list_files(repo_name))
                message = f"Cloned repository: {repo_name} ({file_count} files)"
                print(message)
                
            # Update sync statistics
            self.last_sync[repo_name] = datetime.now()
            config["sync_count"] += 1
            config["last_error"] = None
            
            # Count relevant files
            relevant_files = len(self.list_files(repo_name))
            
            return True, message, relevant_files
            
        except git.exc.GitError as e:
            error_msg = f"Git error syncing repository {repo_name}: {str(e)}"
            print(error_msg)
            config["error_count"] += 1
            config["last_error"] = error_msg
            return False, error_msg, 0
            
        except Exception as e:
            error_msg = f"Error syncing repository {repo_name}: {str(e)}"
            print(error_msg)
            config["error_count"] += 1
            config["last_error"] = error_msg
            return False, error_msg, 0
    
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
    
    async def sync_all_repositories(self) -> Dict[str, Tuple[bool, str, int]]:
        """Sync all configured repositories concurrently"""
        if not self.repos_config:
            return {}
            
        sync_tasks = []
        repos_to_sync = []
        
        for repo_name in self.repos_config:
            if self._should_sync(repo_name):
                repos_to_sync.append(repo_name)
                sync_tasks.append(self.clone_or_update_repo(repo_name))
        
        if not sync_tasks:
            print("No repositories need syncing")
            return {}
        
        print(f"Syncing {len(sync_tasks)} repositories...")
        results = await asyncio.gather(*sync_tasks, return_exceptions=True)
        
        sync_results = {}
        for repo_name, result in zip(repos_to_sync, results):
            if isinstance(result, Exception):
                sync_results[repo_name] = (False, f"Exception: {str(result)}", 0)
            else:
                sync_results[repo_name] = result
        
        return sync_results
    
    def _should_sync(self, repo_name: str) -> bool:
        """Check if repository should be synced based on interval"""
        if repo_name not in self.last_sync:
            return True
            
        config = self.repos_config[repo_name]
        time_since_sync = datetime.now() - self.last_sync[repo_name]
        return time_since_sync.total_seconds() > config["sync_interval"]
    
    def get_file_content(self, repo_name: str, file_path: str) -> Optional[str]:
        """Get content of a specific file from repository with encoding handling"""
        if repo_name not in self.repos_config:
            return None
            
        local_path = self.repos_config[repo_name]["local_path"]
        full_path = local_path / file_path
        
        try:
            if full_path.exists() and full_path.is_file():
                # Check file size
                if full_path.stat().st_size > self.max_file_size:
                    return f"File too large ({full_path.stat().st_size} bytes) - skipped for context"
                
                # Try different encodings
                encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        with open(full_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            return content
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail, return error message
                return f"Unable to decode file {file_path} - binary or unsupported encoding"
                
        except Exception as e:
            print(f"Error reading file {file_path} from {repo_name}: {str(e)}")
        
        return None
    
    def list_files(self, repo_name: str, extensions: List[str] = None, 
                  exclude_dirs: List[str] = None) -> List[str]:
        """List all relevant files in repository with smart filtering"""
        if repo_name not in self.repos_config:
            return []
            
        local_path = self.repos_config[repo_name]["local_path"]
        if not local_path.exists():
            return []
        
        # Use default important extensions if none specified
        if extensions is None:
            extensions = list(self.important_extensions)
            
        exclude_dirs = exclude_dirs or ['.git', 'node_modules', '.vscode', '__pycache__']
        files = []
        
        for root, dirs, filenames in os.walk(local_path):
            # Remove excluded directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d, exclude_dirs)]
            
            for filename in filenames:
                # Skip files that match exclude patterns
                if self._should_exclude_file(filename):
                    continue
                    
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(local_path)
                
                # Check file extension
                if extensions:
                    if any(str(rel_path).endswith(ext) for ext in extensions):
                        files.append(str(rel_path))
                else:
                    files.append(str(rel_path))
        
        # Sort files by importance (Swift/Obj-C first, then others)
        def file_priority(file_path: str) -> int:
            if file_path.endswith(('.swift', '.m', '.h')):
                return 0  # Highest priority
            elif file_path.endswith(('.py', '.js', '.ts')):
                return 1  # Medium priority
            elif file_path.endswith(('.json', '.plist', '.xml')):
                return 2  # Lower priority
            else:
                return 3  # Lowest priority
        
        files.sort(key=file_priority)
        return files
    
    def _should_exclude_dir(self, dirname: str, exclude_dirs: List[str]) -> bool:
        """Check if directory should be excluded"""
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(dirname, pattern):
                return True
        return dirname in exclude_dirs
    
    def _should_exclude_file(self, filename: str) -> bool:
        """Check if file should be excluded"""
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False
    
    def get_repository_structure(self, repo_name: str) -> Dict:
        """Get complete repository structure with metadata"""
        if repo_name not in self.repos_config:
            return {}
            
        config = self.repos_config[repo_name]
        files = self.list_files(repo_name)
        
        # Calculate repository statistics
        total_size = 0
        file_types = {}
        
        for file_path in files:
            full_path = config["local_path"] / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                total_size += size
                
                ext = Path(file_path).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        last_sync_time = self.last_sync.get(repo_name)
        
        structure = {
            "repository": repo_name,
            "url": config["url"],
            "branch": config["branch"],
            "local_path": str(config["local_path"]),
            "last_sync": last_sync_time.isoformat() if last_sync_time else "Never",
            "sync_count": config.get("sync_count", 0),
            "error_count": config.get("error_count", 0),
            "last_error": config.get("last_error"),
            "total_files": len(files),
            "total_size_kb": total_size // 1024,
            "file_types": file_types,
            "files": files[:50],  # Limit to first 50 files for UI
            "status": "healthy" if config.get("error_count", 0) == 0 else "has_errors"
        }
        
        return structure
    
    def has_file_changed(self, repo_name: str, file_path: str) -> bool:
        """Check if file has changed since last check using file hash"""
        content = self.get_file_content(repo_name, file_path)
        if not content or isinstance(content, str) and content.startswith("File too large"):
            return False
            
        current_hash = hashlib.md5(content.encode()).hexdigest()
        key = f"{repo_name}:{file_path}"
        
        if key not in self.file_hashes:
            self.file_hashes[key] = current_hash
            return True
        
        if self.file_hashes[key] != current_hash:
            self.file_hashes[key] = current_hash
            return True
            
        return False
    
    def get_changed_files(self, repo_name: str) -> List[str]:
        """Get list of files that have changed since last check"""
        files = self.list_files(repo_name)
        changed = []
        
        for file_path in files:
            if self.has_file_changed(repo_name, file_path):
                changed.append(file_path)
                
        return changed
    
    def get_repository_health(self, repo_name: str) -> Dict[str, any]:
        """Get health status of a repository"""
        if repo_name not in self.repos_config:
            return {"status": "not_found", "message": "Repository not configured"}
        
        config = self.repos_config[repo_name]
        local_path = config["local_path"]
        
        health = {
            "status": "unknown",
            "message": "",
            "last_sync": self.last_sync.get(repo_name),
            "sync_count": config.get("sync_count", 0),
            "error_count": config.get("error_count", 0),
            "last_error": config.get("last_error"),
            "file_count": 0,
            "repo_exists": local_path.exists()
        }
        
        if not local_path.exists():
            health.update({
                "status": "not_cloned",
                "message": "Repository not cloned yet"
            })
        elif config.get("error_count", 0) > 0:
            health.update({
                "status": "has_errors", 
                "message": f"Last error: {config.get('last_error', 'Unknown error')}"
            })
        elif not self.last_sync.get(repo_name):
            health.update({
                "status": "never_synced",
                "message": "Repository never synced"
            })
        else:
            time_since_sync = datetime.now() - self.last_sync[repo_name]
            health["file_count"] = len(self.list_files(repo_name))
            
            if time_since_sync.total_seconds() > config["sync_interval"] * 2:
                health.update({
                    "status": "sync_overdue",
                    "message": f"Sync overdue by {time_since_sync}"
                })
            else:
                health.update({
                    "status": "healthy",
                    "message": "Repository is up to date"
                })
        
        return health
    
    def cleanup_old_repos(self, max_age_days: int = 7) -> List[str]:
        """Clean up repositories that haven't been synced in a while"""
        cleaned = []
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for repo_name, config in list(self.repos_config.items()):
            last_sync = self.last_sync.get(repo_name)
            
            if not last_sync or last_sync < cutoff_date:
                local_path = config["local_path"]
                
                if local_path.exists():
                    try:
                        shutil.rmtree(local_path)
                        print(f"Cleaned up old repository: {repo_name}")
                        cleaned.append(repo_name)
                    except Exception as e:
                        print(f"Error cleaning up repository {repo_name}: {e}")
        
        return cleaned
    
    def get_sync_statistics(self) -> Dict[str, any]:
        """Get overall sync statistics"""
        if not self.repos_config:
            return {
                "total_repos": 0,
                "healthy_repos": 0,
                "repos_with_errors": 0,
                "total_files": 0,
                "last_successful_sync": None
            }
        
        healthy_count = 0
        error_count = 0
        total_files = 0
        last_successful_sync = None
        
        for repo_name, config in self.repos_config.items():
            if config.get("error_count", 0) == 0:
                healthy_count += 1
            else:
                error_count += 1
                
            total_files += len(self.list_files(repo_name))
            
            repo_last_sync = self.last_sync.get(repo_name)
            if repo_last_sync:
                if not last_successful_sync or repo_last_sync > last_successful_sync:
                    last_successful_sync = repo_last_sync
        
        return {
            "total_repos": len(self.repos_config),
            "healthy_repos": healthy_count,
            "repos_with_errors": error_count,
            "total_files": total_files,
            "last_successful_sync": last_successful_sync.isoformat() if last_successful_sync else None
        }