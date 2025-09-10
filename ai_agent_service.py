import json
import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import google.generativeai as genai
from dataclasses import dataclass
import hashlib
import os

@dataclass
class FileContext:
    repo_name: str
    file_path: str
    content: str
    last_modified: datetime
    file_hash: str
    file_size: int

class AIAgentService:
    def __init__(self, gemini_api_key: str, deepseek_api_key: str = None):
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # DeepSeek configuration
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_base_url = "https://api.deepseek.com/chat/completions"
        
        # Context storage
        self.file_contexts: Dict[str, FileContext] = {}
        self.conversation_history: List[Dict] = []
        
        # Enhanced context management
        self.max_context_files = 300  # Increased for larger repositories
        self.context_refresh_interval = 180  # 3 minutes for more frequent updates
        self.last_context_refresh = datetime.now()
        
        # Code extraction patterns
        self.code_block_pattern = re.compile(r'```(?:swift|objc|objective-c|python|javascript)?\n(.*?)\n```', re.DOTALL)
        self.file_header_pattern = re.compile(r'(?:FileName?|File|PATH?):\s*([^\n]+)', re.IGNORECASE)
        
    async def update_file_context(self, repo_name: str, file_path: str, content: str):
        """Enhanced file context updating with better memory management"""
        import hashlib
        
        key = f"{repo_name}:{file_path}"
        file_hash = hashlib.md5(content.encode()).hexdigest()
        file_size = len(content)
        
        # Smart file size handling - allow larger files for important types
        max_size = 200000 if file_path.endswith(('.swift', '.m', '.h')) else 100000
        
        if file_size > max_size:
            # For large files, store a truncated version with key sections
            content = self._extract_key_sections(content, file_path)
            print(f"Truncated large file: {file_path} ({file_size} -> {len(content)} bytes)")
        
        # Update or add file context
        self.file_contexts[key] = FileContext(
            repo_name=repo_name,
            file_path=file_path,
            content=content,
            last_modified=datetime.now(),
            file_hash=file_hash,
            file_size=len(content)
        )
        
        # Manage context size
        await self._manage_context_size()
        
    def _extract_key_sections(self, content: str, file_path: str) -> str:
        """Extract key sections from large files to preserve important information"""
        lines = content.split('\n')
        
        if file_path.endswith('.swift'):
            # For Swift files, preserve imports, class/struct definitions, and function signatures
            key_lines = []
            in_important_section = False
            brace_count = 0
            
            for line in lines:
                stripped = line.strip()
                
                # Always include imports, class/struct definitions, function signatures
                if (stripped.startswith(('import ', 'class ', 'struct ', 'enum ', 'protocol ', 'extension ')) or
                    stripped.startswith(('func ', 'init(', 'deinit', '@')) or
                    'MARK:' in stripped or
                    stripped.startswith('//') and len(stripped) > 10):
                    
                    key_lines.append(line)
                    if '{' in line:
                        in_important_section = True
                        brace_count = line.count('{') - line.count('}')
                
                elif in_important_section:
                    key_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    
                    if brace_count <= 0:
                        in_important_section = False
                        key_lines.append("    // ... implementation details truncated ...")
                        key_lines.append("")
                
                # Limit total lines
                if len(key_lines) > 500:
                    break
            
            return '\n'.join(key_lines)
        
        else:
            # For other files, just take first and last portions
            if len(lines) > 200:
                return '\n'.join(lines[:100] + ["// ... middle section truncated ..."] + lines[-100:])
            return content
    
    async def _manage_context_size(self):
        """Enhanced context size management with intelligent prioritization"""
        if len(self.file_contexts) <= self.max_context_files:
            return
            
        # Calculate relevance scores for each file
        scored_contexts = []
        for key, context in self.file_contexts.items():
            score = self._calculate_file_relevance_score(context)
            scored_contexts.append((score, key, context))
        
        # Sort by relevance (highest first) and keep the most relevant files
        scored_contexts.sort(reverse=True)
        files_to_keep = self.max_context_files - 50  # Leave room for new files
        
        new_contexts = {}
        for i in range(min(files_to_keep, len(scored_contexts))):
            score, key, context = scored_contexts[i]
            new_contexts[key] = context
        
        removed_count = len(self.file_contexts) - len(new_contexts)
        self.file_contexts = new_contexts
        
        print(f"Context management: Removed {removed_count} files, kept {len(new_contexts)} most relevant files")
    
    def _calculate_file_relevance_score(self, context: FileContext) -> float:
        """Calculate relevance score for file context prioritization"""
        score = 0.0
        
        # File type importance
        if context.file_path.endswith(('.swift', '.m', '.h')):
            score += 10.0  # iOS files are most important
        elif context.file_path.endswith('.py'):
            score += 7.0
        elif context.file_path.endswith(('.js', '.ts')):
            score += 5.0
        elif context.file_path.endswith(('.json', '.plist')):
            score += 3.0
        
        # Recency bonus
        hours_old = (datetime.now() - context.last_modified).total_seconds() / 3600
        if hours_old < 1:
            score += 5.0
        elif hours_old < 24:
            score += 2.0
        
        # Content qualit...(truncated 17912 characters)...trip()
                code_block_index += 1
                current_filename = None  # Reset for next iteration
        
        # Handle remaining code blocks without explicit filenames
        while code_block_index < len(code_blocks):
            filename = f"solution_code_{code_block_index + 1}.swift"  # Default extension
            code_sections[filename] = code_blocks[code_block_index].strip()
            code_block_index += 1
        
        return code_sections
    
    def _combine_code_sections(self, deepseek_code: Dict[str, str], gemini_code: Dict[str, str]) -> Dict[str, str]:
        """Combine code sections from both models, prioritizing DeepSeek for implementation"""
        combined = {}
        
        # Start with DeepSeek code (it's specialized for coding)
        for filename, code in deepseek_code.items():
            if code and len(code.strip()) > 50:  # Only include substantial code
                combined[filename] = code
        
        # Add Gemini code for files not covered by DeepSeek
        for filename, code in gemini_code.items():
            if filename not in combined and code and len(code.strip()) > 50:
                combined[filename] = code
        
        return combined
    
    async def analyze_xcode_error(self, error_message: str, use_deepseek: str = "both") -> Dict[str, Any]:
        """Analyze XCode error with collaborative AI"""
        context = self.get_relevant_context(error_message)
        
        # Parse file name from error
        file_match = re.search(r'([A-Z][a-zA-Z0-9_]*\.swift)', error_message)
        target_file = file_match.group(1) if file_match else None
        
        # Get existing file content if found
        file_content = ""
        if target_file:
            # Assume first repo or search across
            for repo_name in repo_manager.get_repositories():
                if repo_manager.file_exists(repo_name, target_file):
                    file_content = repo_manager.get_file_content(repo_name, target_file)
                    break
        
        # System prompt for structured output, full code, existing files
        system_prompt = """
You are an expert Xcode coding assistant. Analyze the error and provide a fix.
Structure your response as:
**Analysis:**
[Your detailed analysis]

**Code Fix:**
```swift
// File: filename.swift
[Full updated code for the file, not snippet. Only for existing files.]