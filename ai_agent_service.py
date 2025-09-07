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
        
    async def update_file_context(self, repo_name: str, file_path: str, content: str):
        """Update file context for the AI agent"""
        key = f"{repo_name}:{file_path}"
        self.file_contexts[key] = FileContext(
            repo_name=repo_name,
            file_path=file_path,
            content=content,
            last_modified=datetime.now()
        )
        
    def get_relevant_context(self, query: str, max_files: int = 10) -> str:
        """Get relevant file context based on the query"""
        # Simple relevance scoring - can be improved with embeddings
        relevant_files = []
        query_lower = query.lower()
        
        for key, context in self.file_contexts.items():
            relevance_score = 0
            
            # Check if query mentions file name or path
            if context.file_path.lower() in query_lower:
                relevance_score += 10
            
            # Check for programming language relevance
            if any(ext in context.file_path for ext in ['.swift', '.m', '.h', '.py', '.js']):
                if any(keyword in query_lower for keyword in ['error', 'xcode', 'ios', 'swift']):
                    relevance_score += 5
            
            # Check content relevance (basic keyword matching)
            content_lower = context.content.lower()
            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    relevance_score += 1
                    
            if relevance_score > 0:
                relevant_files.append((relevance_score, context))
        
        # Sort by relevance and limit results
        relevant_files.sort(key=lambda x: x[0], reverse=True)
        relevant_files = relevant_files[:max_files]
        
        # Format context
        context_text = "CURRENT FILE CONTEXT:\n\n"
        for score, context in relevant_files:
            context_text += f"File: {context.repo_name}/{context.file_path}\n"
            context_text += f"Content:\n```\n{context.content}\n```\n\n"
        
        return context_text
    
    async def query_deepseek(self, prompt: str, system_prompt: str = None) -> str:
        """Query DeepSeek API"""
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
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.deepseek_base_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        return f"DeepSeek API Error: {response.status} - {error_text}"
        except Exception as e:
            return f"DeepSeek API Error: {str(e)}"
    
    async def query_gemini(self, prompt: str) -> str:
        """Query Gemini API"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.gemini_model.generate_content, prompt
            )
            return response.text
        except Exception as e:
            return f"Gemini API Error: {str(e)}"
    
    async def analyze_xcode_error(self, error_message: str, use_deepseek: bool = True) -> Dict[str, Any]:
        """Analyze XCode error and provide solutions"""
        # Get relevant context
        context = self.get_relevant_context(error_message)
        
        system_prompt = """You are an expert iOS/Swift developer and XCode error resolver. 
        When provided with an error message and file context:
        1. Analyze the error thoroughly
        2. Identify the root cause
        3. Provide a complete fixed version of any problematic files
        4. Explain the changes made
        5. Provide the complete file content, not just snippets
        
        Format your response as:
        ERROR ANALYSIS:
        [Your analysis]
        
        ROOT CAUSE:
        [The root cause]
        
        SOLUTION:
        [Your solution explanation]
        
        FIXED FILES:
        [Complete file contents for each fixed file, clearly labeled]
        """
        
        full_prompt = f"{context}\n\nXCODE ERROR:\n{error_message}\n\nPlease analyze this error and provide complete fixed files."
        
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
            "model_used": "deepseek" if use_deepseek else "gemini"
        })
        
        return {
            "analysis": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "context_files_used": len(self.file_contexts),
            "timestamp": datetime.now().isoformat()
        }
    
    async def general_coding_query(self, query: str, use_deepseek: bool = True) -> Dict[str, Any]:
        """Handle general coding queries with file context"""
        context = self.get_relevant_context(query)
        
        system_prompt = """You are an expert software developer assistant. 
        Use the provided file context to give accurate, contextual responses.
        When providing code fixes or implementations, always provide complete files rather than snippets.
        """
        
        full_prompt = f"{context}\n\nQUERY:\n{query}"
        
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
            "model_used": "deepseek" if use_deepseek else "gemini"
        })
        
        return {
            "response": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "context_files_used": len(self.file_contexts),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def clear_context(self):
        """Clear file context (useful for memory management)"""
        self.file_contexts.clear()
        
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context"""
        return {
            "total_files": len(self.file_contexts),
            "repositories": list(set(ctx.repo_name for ctx in self.file_contexts.values())),
            "file_types": list(set(ctx.file_path.split('.')[-1] for ctx in self.file_contexts.values() if '.' in ctx.file_path)),
            "last_update": max([ctx.last_modified for ctx in self.file_contexts.values()]).isoformat() if self.file_contexts else None
        }