import os
import git
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import hashlib

class GitRepoManager:
    def __init__(self, base_path: str = "/tmp/repos"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.repos_config = {}
        self.last_sync = {}
        self.file_hashes = {}
        
    def add_repository(self, name: str, url: str, branch: str = "main", 
                      access_token: str = None, sync_interval: int = 300):
        """Add a git repository to monitor"""
        self.repos_config[name] = {
            "url": url,
            "branch": branch,
            "access_token": access_token,
            "sync_interval": sync_interval,
            "local_path": self.base_path / name
        }
        
    async def clone_or_update_repo(self, repo_name: str) -> bool:
        """Clone repository if not exists, otherwise pull latest changes"""
        if repo_name not in self.repos_config:
            raise ValueError(f"Repository {repo_name} not configured")
            
        config = self.repos_config[repo_name]
        local_path = config["local_path"]
        
        try:
            if local_path.exists():
                # Repository exists, pull latest changes
                repo = git.Repo(local_path)
                origin = repo.remotes.origin
                origin.pull(config["branch"])
                print(f"Updated repository: {repo_name}")
            else:
                # Clone repository
                auth_url = self._get_authenticated_url(config["url"], config.get("access_token"))
                git.Repo.clone_from(auth_url, local_path, branch=config["branch"])
                print(f"Cloned repository: {repo_name}")
                
            self.last_sync[repo_name] = datetime.now()
            return True
            
        except Exception as e:
            print(f"Error syncing repository {repo_name}: {str(e)}")
            return False
    
    def _get_authenticated_url(self, url: str, token: str) -> str:
        """Add authentication token to git URL"""
        if not token:
            return url
        
        if "github.com" in url:
            return url.replace("https://", f"https://{token}@")
        elif "gitlab.com" in url:
            return url.replace("https://", f"https://oauth2:{token}@")
        
        return url
    
    async def sync_all_repositories(self):
        """Sync all configured repositories"""
        tasks = []
        for repo_name in self.repos_config:
            if self._should_sync(repo_name):
                tasks.append(self.clone_or_update_repo(repo_name))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _should_sync(self, repo_name: str) -> bool:
        """Check if repository should be synced based on interval"""
        if repo_name not in self.last_sync:
            return True
            
        config = self.repos_config[repo_name]
        time_since_sync = datetime.now() - self.last_sync[repo_name]
        return time_since_sync.total_seconds() > config["sync_interval"]
    
    def get_file_content(self, repo_name: str, file_path: str) -> Optional[str]:
        """Get content of a specific file from repository"""
        if repo_name not in self.repos_config:
            return None
            
        local_path = self.repos_config[repo_name]["local_path"]
        full_path = local_path / file_path
        
        try:
            if full_path.exists() and full_path.is_file():
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error reading file {file_path} from {repo_name}: {str(e)}")
        
        return None
    
    def list_files(self, repo_name: str, extensions: List[str] = None, 
                  exclude_dirs: List[str] = None) -> List[str]:
        """List all files in repository with optional filtering"""
        if repo_name not in self.repos_config:
            return []
            
        local_path = self.repos_config[repo_name]["local_path"]
        if not local_path.exists():
            return []
        
        exclude_dirs = exclude_dirs or ['.git', 'node_modules', '.vscode', '__pycache__']
        files = []
        
        for root, dirs, filenames in os.walk(local_path):
            # Remove excluded directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for filename in filenames:
                if extensions:
                    if any(filename.endswith(ext) for ext in extensions):
                        rel_path = os.path.relpath(os.path.join(root, filename), local_path)
                        files.append(rel_path)
                else:
                    rel_path = os.path.relpath(os.path.join(root, filename), local_path)
                    files.append(rel_path)
        
        return files
    
    def get_repository_structure(self, repo_name: str) -> Dict:
        """Get complete repository structure"""
        if repo_name not in self.repos_config:
            return {}
            
        files = self.list_files(repo_name)
        structure = {
            "repository": repo_name,
            "last_sync": self.last_sync.get(repo_name, "Never").isoformat() if isinstance(self.last_sync.get(repo_name), datetime) else "Never",
            "total_files": len(files),
            "files": files
        }
        
        return structure
    
    def has_file_changed(self, repo_name: str, file_path: str) -> bool:
        """Check if file has changed since last check"""
        content = self.get_file_content(repo_name, file_path)
        if not content:
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