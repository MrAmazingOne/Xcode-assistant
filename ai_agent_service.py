import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import google.generativeai as genai
from dataclasses import dataclass

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
        self.deepseek_base_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Context storage
        self.file_contexts: Dict[str, FileContext] = {}
        self.conversation_history: List[Dict] = []
        
        # Context management
        self.max_context_files = 50  # Limit to prevent memory issues
        self.context_refresh_interval = 300  # 5 minutes
        self.last_context_refresh = datetime.now()
        
    async def update_file_context(self, repo_name: str, file_path: str, content: str):
        """Update file context for the AI agent with better management"""
        import hashlib
        
        key = f"{repo_name}:{file_path}"
        file_hash = hashlib.md5(content.encode()).hexdigest()
        file_size = len(content)
        
        # Skip very large files (>50KB) to prevent context overflow
        if file_size > 50000:
            print(f"Skipping large file: {file_path} ({file_size} bytes)")
            return
        
        # Update or add file context
        self.file_contexts[key] = FileContext(
            repo_name=repo_name,
            file_path=file_path,
            content=content,
            last_modified=datetime.now(),
            file_hash=file_hash,
            file_size=file_size
        )
        
        # Manage context size
        await self._manage_context_size()
        
    async def _manage_context_size(self):
        """Manage context size to prevent memory issues"""
        if len(self.file_contexts) > self.max_context_files:
            # Remove oldest files first
            sorted_contexts = sorted(
                self.file_contexts.items(),
                key=lambda x: x[1].last_modified
            )
            
            # Keep only the most recent files
            files_to_remove = len(self.file_contexts) - self.max_context_files
            for i in range(files_to_remove):
                key_to_remove = sorted_contexts[i][0]
                del self.file_contexts[key_to_remove]
                print(f"Removed old context: {key_to_remove}")
        
    def get_relevant_context(self, query: str, max_files: int = 15) -> str:
        """Get relevant file context based on the query with improved scoring"""
        relevant_files = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for key, context in self.file_contexts.items():
            relevance_score = 0
            
            # File path relevance
            file_path_lower = context.file_path.lower()
            if any(word in file_path_lower for word in query_words):
                relevance_score += 10
            
            # File extension relevance for XCode errors
            if any(ext in context.file_path for ext in ['.swift', '.m', '.h']):
                if any(keyword in query_lower for keyword in ['error', 'xcode', 'ios', 'swift', 'objective-c']):
                    relevance_score += 8
            
            # Programming language relevance
            if '.py' in context.file_path and any(keyword in query_lower for keyword in ['python', 'script']):
                relevance_score += 5
            
            # Content relevance with keyword matching
            content_lower = context.content.lower()
            matching_words = 0
            for word in query_words:
                if len(word) > 3:  # Only count meaningful words
                    word_count = content_lower.count(word)
                    matching_words += word_count
                    relevance_score += word_count * 0.5
            
            # Boost score for files with many keyword matches
            if matching_words > 5:
                relevance_score += 5
                
            # Recent files get slight boost
            hours_since_modified = (datetime.now() - context.last_modified).total_seconds() / 3600
            if hours_since_modified < 24:
                relevance_score += 2
                
            if relevance_score > 0:
                relevant_files.append((relevance_score, context))
        
        # Sort by relevance and limit results
        relevant_files.sort(key=lambda x: x[0], reverse=True)
        relevant_files = relevant_files[:max_files]
        
        # Format context with better structure
        if not relevant_files:
            return "No relevant files found in context."
        
        context_text = "CURRENT PROJECT CONTEXT:\n\n"
        
        total_context_size = 0
        for score, context in relevant_files:
            # Limit individual file content to prevent context overflow
            content_preview = context.content
            if len(content_preview) > 2000:
                content_preview = content_preview[:1500] + "\n\n... (file truncated for context) ..."
            
            total_context_size += len(content_preview)
            
            # Stop if context is getting too large (DeepSeek has ~32K token limit)
            if total_context_size > 20000:
                context_text += "\n... (additional files omitted due to context size limits) ...\n"
                break
                
            context_text += f"📁 {context.repo_name}/{context.file_path} (relevance: {score:.1f})\n"
            context_text += f"```{self._get_file_language(context.file_path)}\n{content_preview}\n```\n\n"
        
        return context_text
    
    def _get_file_language(self, file_path: str) -> str:
        """Get programming language for syntax highlighting"""
        if file_path.endswith('.swift'):
            return 'swift'
        elif file_path.endswith('.m') or file_path.endswith('.h'):
            return 'objective-c'
        elif file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.js'):
            return 'javascript'
        elif file_path.endswith('.json'):
            return 'json'
        elif file_path.endswith('.xml') or file_path.endswith('.plist'):
            return 'xml'
        else:
            return ''
    
    async def query_deepseek(self, prompt: str, system_prompt: str = None) -> str:
        """Query DeepSeek API with better error handling"""
        if not self.deepseek_api_key:
            return "DeepSeek API key not configured"
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "deepseek-coder",
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.1,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=120)  # Increase timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.deepseek_base_url,
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                        else:
                            error_text = await response.text()
                            print(f"DeepSeek API Error (attempt {attempt + 1}): {response.status} - {error_text}")
                            
                            if response.status == 429:  # Rate limit
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                return f"DeepSeek API Error: {response.status} - {error_text}"
            except asyncio.TimeoutError:
                print(f"DeepSeek API timeout (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return "DeepSeek API Error: Request timeout"
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"DeepSeek API Error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return f"DeepSeek API Error: {str(e)}"
                await asyncio.sleep(2 ** attempt)
        
        return "DeepSeek API Error: Max retries exceeded"
    
    async def query_gemini(self, prompt: str) -> str:
        """Query Gemini API with better error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.gemini_model.generate_content, prompt
                )
                return response.text
            except Exception as e:
                print(f"Gemini API Error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return f"Gemini API Error: {str(e)}"
                await asyncio.sleep(2 ** attempt)
    
    async def analyze_xcode_error(self, error_message: str, use_deepseek: bool = True) -> Dict[str, Any]:
        """Analyze XCode error and provide complete file solutions"""
        # Get relevant context
        context = self.get_relevant_context(error_message)
        
        system_prompt = """You are an expert iOS/Swift developer and XCode error resolver with deep knowledge of:
        - Swift programming language and iOS frameworks
        - Objective-C and bridging between Swift/Objective-C
        - XCode IDE, build systems, and common compilation errors
        - iOS app architecture patterns (MVC, MVVM, etc.)
        - Memory management, threading, and performance optimization

        When provided with an XCode error message and project file context:

        1. ANALYZE the error thoroughly, identifying:
           - The specific error type and root cause
           - Which files are affected
           - Dependencies and relationships between components

        2. PROVIDE COMPLETE FIXED FILES:
           - Output the entire corrected file content, not just snippets
           - Ensure all imports, class declarations, and methods are included
           - Maintain proper Swift/Objective-C syntax and conventions
           - Add helpful comments explaining the fixes

        3. EXPLAIN your solution:
           - What was wrong and why it caused the error
           - What changes were made and why
           - Any potential side effects or additional considerations

        Format your response as:
        ## ERROR ANALYSIS
        [Detailed analysis of the error]
        
        ## ROOT CAUSE
        [The underlying cause of the error]
        
        ## SOLUTION
        [Step-by-step explanation of the fix]
        
        ## COMPLETE FIXED FILES
        [Full file contents with all necessary fixes]
        
        ## ADDITIONAL RECOMMENDATIONS
        [Best practices, potential improvements, testing suggestions]
        """
        
        full_prompt = f"{context}\n\nXCODE ERROR MESSAGE:\n```\n{error_message}\n```\n\nPlease analyze this error and provide complete fixed files with detailed explanations."
        
        if use_deepseek:
            response = await self.query_deepseek(full_prompt, system_prompt)
        else:
            full_prompt_with_system = f"{system_prompt}\n\n{full_prompt}"
            response = await self.query_gemini(full_prompt_with_system)
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "xcode_error_analysis",
            "error": error_message,
            "response": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "context_files_count": len(self.file_contexts)
        })
        
        return {
            "analysis": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "context_files_used": len(self.file_contexts),
            "context_size_kb": sum(ctx.file_size for ctx in self.file_contexts.values()) // 1024,
            "timestamp": datetime.now().isoformat()
        }
    
    async def general_coding_query(self, query: str, use_deepseek: bool = True) -> Dict[str, Any]:
        """Handle general coding queries with comprehensive project context"""
        context = self.get_relevant_context(query)
        
        system_prompt = """You are an expert software development assistant with deep knowledge across multiple programming languages and frameworks, particularly:
        - iOS development (Swift, Objective-C, UIKit, SwiftUI)
        - Backend development (Python, Node.js, APIs)
        - Database design and optimization
        - Software architecture and design patterns
        - Testing, debugging, and performance optimization

        When answering coding questions:

        1. UNDERSTAND the context from the provided project files
        2. PROVIDE complete, working solutions rather than code snippets
        3. EXPLAIN your reasoning and approach
        4. INCLUDE best practices and potential improvements
        5. SUGGEST testing strategies where appropriate
        6. CONSIDER performance, security, and maintainability

        Always provide complete file contents when modifications are needed, ensuring:
        - All imports and dependencies are included
        - Proper error handling is implemented
        - Code follows language-specific conventions
        - Comments explain complex logic
        """
        
        full_prompt = f"{context}\n\nCODING QUESTION:\n{query}\n\nPlease provide a comprehensive answer with complete code solutions where applicable."
        
        if use_deepseek:
            response = await self.query_deepseek(full_prompt, system_prompt)
        else:
            full_prompt_with_system = f"{system_prompt}\n\n{full_prompt}"
            response = await self.query_gemini(full_prompt_with_system)
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "general_query",
            "query": query,
            "response": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "context_files_count": len(self.file_contexts)
        })
        
        return {
            "response": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "context_files_used": len(self.file_contexts),
            "context_size_kb": sum(ctx.file_size for ctx in self.file_contexts.values()) // 1024,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def clear_context(self):
        """Clear file context (useful for memory management)"""
        self.file_contexts.clear()
        print("File context cleared")
        
    def clear_old_conversations(self, max_conversations: int = 50):
        """Clear old conversations to prevent memory leaks"""
        if len(self.conversation_history) > max_conversations:
            removed_count = len(self.conversation_history) - max_conversations
            self.conversation_history = self.conversation_history[-max_conversations:]
            print(f"Cleared {removed_count} old conversations")
        
    def get_context_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current context"""
        if not self.file_contexts:
            return {
                "total_files": 0,
                "repositories": [],
                "file_types": [],
                "total_size_kb": 0,
                "last_update": None
            }
            
        repositories = list(set(ctx.repo_name for ctx in self.file_contexts.values()))
        file_types = list(set(ctx.file_path.split('.')[-1] for ctx in self.file_contexts.values() if '.' in ctx.file_path))
        total_size = sum(ctx.file_size for ctx in self.file_contexts.values())
        last_update = max(ctx.last_modified for ctx in self.file_contexts.values())
        
        return {
            "total_files": len(self.file_contexts),
            "repositories": repositories,
            "file_types": file_types,
            "total_size_kb": total_size // 1024,
            "last_update": last_update.isoformat(),
            "conversations_count": len(self.conversation_history),
            "context_health": "good" if len(self.file_contexts) < self.max_context_files else "at_limit"
        }
    
    async def refresh_context_if_needed(self):
        """Refresh context if enough time has passed"""
        time_since_refresh = (datetime.now() - self.last_context_refresh).total_seconds()
        
        if time_since_refresh > self.context_refresh_interval:
            # Clear old conversations
            self.clear_old_conversations()
            
            # Update last refresh time
            self.last_context_refresh = datetime.now()
            
            print(f"Context refreshed - Files: {len(self.file_contexts)}, Conversations: {len(self.conversation_history)}")
            
            return True
        return False