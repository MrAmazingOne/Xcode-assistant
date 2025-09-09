import json
import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import google.generativeai as genai
from dataclasses import dataclass
import hashlib

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
        
        # Content quality (files with more content are likely more important)
        if context.file_size > 5000:
            score += 2.0
        elif context.file_size > 1000:
            score += 1.0
        
        # File name importance (files with "main", "app", "view", etc. are important)
        important_keywords = ['main', 'app', 'view', 'controller', 'model', 'service', 'manager', 'helper']
        file_name_lower = context.file_path.lower()
        for keyword in important_keywords:
            if keyword in file_name_lower:
                score += 1.0
        
        return score
    
    def get_relevant_context(self, query: str, max_files: int = 25) -> str:
        """Enhanced context retrieval with better relevance scoring"""
        if not self.file_contexts:
            return "No project files available in context."
        
        relevant_files = []
        query_lower = query.lower()
        query_words = set(word for word in query_lower.split() if len(word) > 2)
        
        # Extract specific file names from query
        mentioned_files = self._extract_file_names_from_query(query)
        
        for key, context in self.file_contexts.items():
            relevance_score = 0.0
            
            # Exact file name match gets highest priority
            if any(filename.lower() in context.file_path.lower() for filename in mentioned_files):
                relevance_score += 50.0
            
            # File path relevance
            file_path_lower = context.file_path.lower()
            path_matches = sum(1 for word in query_words if word in file_path_lower)
            relevance_score += path_matches * 5.0
            
            # File extension relevance for error analysis
            if any(ext in context.file_path for ext in ['.swift', '.m', '.h']):
                if any(keyword in query_lower for keyword in ['error', 'xcode', 'ios', 'swift', 'build']):
                    relevance_score += 15.0
            
            # Content relevance with weighted keyword matching
            content_lower = context.content.lower()
            content_matches = 0
            for word in query_words:
                if len(word) > 3:
                    word_count = content_lower.count(word)
                    content_matches += word_count
                    relevance_score += word_count * 1.0
            
            # Boost for files with many matches
            if content_matches > 10:
                relevance_score += 10.0
            elif content_matches > 5:
                relevance_score += 5.0
            
            # File type prioritization
            relevance_score += self._calculate_file_relevance_score(context) * 0.5
            
            if relevance_score > 0:
                relevant_files.append((relevance_score, context))
        
        # Sort by relevance and limit results
        relevant_files.sort(key=lambda x: x[0], reverse=True)
        relevant_files = relevant_files[:max_files]
        
        if not relevant_files:
            return "No relevant files found for this query."
        
        # Format context with improved structure
        context_text = f"PROJECT CONTEXT ({len(relevant_files)} most relevant files):\n\n"
        
        total_context_size = 0
        for score, context in relevant_files:
            content_preview = context.content
            
            # Smart content truncation
            if len(content_preview) > 3000:
                content_preview = content_preview[:2000] + "\n\n... (content truncated for context efficiency) ..."
            
            total_context_size += len(content_preview)
            
            # Context size limit for API constraints
            if total_context_size > 30000:
                context_text += "\n... (additional files omitted due to context size limits) ...\n"
                break
            
            context_text += f"📁 {context.repo_name}/{context.file_path} [Score: {score:.1f}]\n"
            context_text += f"```{self._get_file_language(context.file_path)}\n{content_preview}\n```\n\n"
        
        return context_text
    
    def _extract_file_names_from_query(self, query: str) -> List[str]:
        """Extract potential file names from the query"""
        # Look for file names with extensions
        file_pattern = re.compile(r'(\w+\.\w+)', re.IGNORECASE)
        matches = file_pattern.findall(query)
        
        # Also look for quoted file names or paths
        quoted_pattern = re.compile(r'["\']([^"\']*\.\w+)["\']', re.IGNORECASE)
        quoted_matches = quoted_pattern.findall(query)
        
        return list(set(matches + quoted_matches))
    
    def _get_file_language(self, file_path: str) -> str:
        """Get programming language for syntax highlighting"""
        ext_map = {
            '.swift': 'swift',
            '.m': 'objective-c',
            '.h': 'objective-c',
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.json': 'json',
            '.xml': 'xml',
            '.plist': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        
        ext = '.' + file_path.split('.')[-1].lower()
        return ext_map.get(ext, '')
    
    async def query_deepseek(self, prompt: str, system_prompt: str = None) -> str:
        """Enhanced DeepSeek API querying with better error handling and retry logic"""
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
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=180)  # 3 minutes timeout
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
                                delay = base_delay * (2 ** attempt) + (attempt * 5)
                                print(f"Rate limited, waiting {delay} seconds...")
                                await asyncio.sleep(delay)
                                continue
                            elif response.status >= 500:  # Server error, retry
                                delay = base_delay * (2 ** attempt)
                                await asyncio.sleep(delay)
                                continue
                            else:
                                return f"DeepSeek API Error: {response.status} - {error_text}"
                                
            except asyncio.TimeoutError:
                print(f"DeepSeek API timeout (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return "DeepSeek API Error: Request timeout after multiple attempts"
                await asyncio.sleep(base_delay * (2 ** attempt))
                
            except Exception as e:
                print(f"DeepSeek API Error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return f"DeepSeek API Error: {str(e)}"
                await asyncio.sleep(base_delay * (2 ** attempt))
        
        return "DeepSeek API Error: Max retries exceeded"
    
    async def query_gemini(self, prompt: str) -> str:
        """Enhanced Gemini API querying with better error handling"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.gemini_model.generate_content, prompt
                )
                return response.text
                
            except Exception as e:
                print(f"Gemini API Error (attempt {attempt + 1}): {str(e)}")
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + (attempt * 3)
                    print(f"Rate limited, waiting {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                    
                if attempt == max_retries - 1:
                    return f"Gemini API Error: {str(e)}"
                await asyncio.sleep(base_delay * (2 ** attempt))
        
        return "Gemini API Error: Max retries exceeded"
    
    async def collaborative_analysis(self, query: str, is_error_analysis: bool, error_message: str = None) -> Dict[str, Any]:
        """Enhanced collaborative analysis using both AI models with intelligent coordination"""
        print("Starting collaborative analysis...")
        
        # Determine which query to use
        analysis_query = error_message if is_error_analysis else query
        context = self.get_relevant_context(analysis_query, max_files=30)
        
        # Create specialized prompts for each model
        deepseek_system_prompt = self._get_deepseek_system_prompt(is_error_analysis)
        gemini_prompt = self._get_gemini_prompt(context, analysis_query, is_error_analysis)
        deepseek_prompt = self._get_deepseek_prompt(context, analysis_query, is_error_analysis)
        
        # Run both analyses concurrently
        print("Querying both AI models...")
        deepseek_task = self.query_deepseek(deepseek_prompt, deepseek_system_prompt)
        gemini_task = self.query_gemini(gemini_prompt)
        
        deepseek_result, gemini_result = await asyncio.gather(deepseek_task, gemini_task)
        
        # Extract code sections from responses
        deepseek_code = self._extract_code_sections(deepseek_result)
        gemini_code = self._extract_code_sections(gemini_result)
        
        # Combine and validate code sections
        combined_code = self._combine_code_sections(deepseek_code, gemini_code)
        
        # Create collaborative summary
        collaboration_prompt = self._create_collaboration_prompt(
            analysis_query, deepseek_result, gemini_result, is_error_analysis
        )
        
        print("Creating collaborative summary...")
        collaborative_summary = await self.query_deepseek(
            collaboration_prompt,
            "You are an expert technical analyst. Create a comprehensive, actionable summary based on the provided analyses."
        )
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "collaborative_analysis",
            "query": analysis_query,
            "is_error_analysis": is_error_analysis,
            "deepseek_analysis": deepseek_result,
            "gemini_analysis": gemini_result,
            "collaborative_analysis": collaborative_summary,
            "code_sections": combined_code,
            "context_files_count": len(self.file_contexts)
        })
        
        return {
            "collaborative_analysis": collaborative_summary,
            "deepseek_analysis": deepseek_result,
            "gemini_analysis": gemini_result,
            "code_sections": combined_code,
            "collaboration_notes": f"Analysis completed using {len(self.file_contexts)} project files",
            "context_files_used": len(self.file_contexts),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_deepseek_system_prompt(self, is_error_analysis: bool) -> str:
        """Get specialized system prompt for DeepSeek"""
        if is_error_analysis:
            return """You are DeepSeek Coder, an expert iOS/Swift developer specializing in XCode error resolution. Your role:

ANALYSIS FOCUS:
- Identify the exact error type and root cause
- Locate the problematic code sections
- Understand the technical context and dependencies

CODE SOLUTIONS:
- Provide COMPLETE, working file contents
- Include ALL necessary imports, class definitions, and methods
- Ensure proper Swift/Objective-C syntax and conventions
- Add explanatory comments for complex fixes

RESPONSE FORMAT:
## ERROR ANALYSIS
[Detailed technical analysis]

## ROOT CAUSE
[Specific cause identification]

## SOLUTION APPROACH
[Step-by-step fix strategy]

## COMPLETE FIXED FILES
[Full file contents with fixes]

Be thorough, accurate, and provide production-ready code."""
        else:
            return """You are DeepSeek Coder, an expert software developer. Focus on:

CODE SOLUTIONS:
- Provide complete, working implementations
- Include all necessary dependencies and imports
- Follow language-specific best practices
- Add clear documentation and comments

TECHNICAL ACCURACY:
- Ensure code compiles and runs correctly
- Handle edge cases and error conditions
- Consider performance and security implications
- Suggest testing approaches

Always provide complete file contents when code changes are needed."""
    
    def _get_gemini_prompt(self, context: str, query: str, is_error_analysis: bool) -> str:
        """Get specialized prompt for Gemini focusing on reasoning and validation"""
        prompt_type = "XCode Error Analysis" if is_error_analysis else "Code Review and Solution"
        
        return f"""As Gemini, you excel at reasoning, validation, and architectural thinking. Your task: {prompt_type}

PROJECT CONTEXT:
{context}

QUERY/ERROR:
{query}

YOUR ROLE - Focus on:
1. LOGICAL REASONING: Analyze the problem from multiple angles
2. VALIDATION: Verify that solutions will actually work
3. ARCHITECTURE: Consider system-wide implications
4. BEST PRACTICES: Suggest improvements and optimizations
5. TESTING: Recommend validation approaches

Provide your analysis with clear reasoning and architectural insights."""
    
    def _get_deepseek_prompt(self, context: str, query: str, is_error_analysis: bool) -> str:
        """Get specialized prompt for DeepSeek focusing on code implementation"""
        return f"""PROJECT CONTEXT:
{context}

{"XCODE ERROR TO FIX:" if is_error_analysis else "CODING TASK:"}
{query}

As DeepSeek Coder, provide:
1. Technical analysis of the issue
2. Complete, working code solutions
3. Full file contents (not snippets)
4. Detailed explanations of changes

Focus on delivering production-ready code that solves the problem completely."""
    
    def _create_collaboration_prompt(self, query: str, deepseek_result: str, gemini_result: str, is_error_analysis: bool) -> str:
        """Create prompt for collaborative summary"""
        task_type = "XCode error resolution" if is_error_analysis else "coding solution"
        
        return f"""COLLABORATIVE {task_type.upper()} ANALYSIS

ORIGINAL QUERY/ERROR:
{query}

DEEPSEEK ANALYSIS (Technical Implementation):
{deepseek_result}

GEMINI ANALYSIS (Reasoning & Validation):
{gemini_result}

Create a comprehensive, actionable summary that:
1. Combines the best insights from both analyses
2. Provides a clear, step-by-step solution
3. Explains WHY this approach is correct
4. Highlights any discrepancies between the analyses
5. Gives final recommendations

Focus on practical, implementable solutions."""
    
    def _extract_code_sections(self, response: str) -> Dict[str, str]:
        """Extract code sections from AI response with improved parsing"""
        code_sections = {}
        
        # Find all code blocks
        code_blocks = re.findall(self.code_block_pattern, response)
        
        if not code_blocks:
            return code_sections
        
        # Split response into sections to find file names
        sections = re.split(r'\n(?=#{1,3}\s)', response)
        
        current_filename = None
        code_block_index = 0
        
        for section in sections:
            # Look for file names in section headers or content
            file_matches = re.findall(r'(?:File|Path|Filename):\s*([^\n\r]+)', section, re.IGNORECASE)
            
            if file_matches:
                current_filename = file_matches[0].strip('` "\'')
            else:
                # Try to extract filename from common patterns
                swift_files = re.findall(r'(\w+\.swift)', section, re.IGNORECASE)
                objc_files = re.findall(r'(\w+\.[mh])', section, re.IGNORECASE)
                py_files = re.findall(r'(\w+\.py)', section, re.IGNORECASE)
                
                if swift_files:
                    current_filename = swift_files[0]
                elif objc_files:
                    current_filename = objc_files[0]
                elif py_files:
                    current_filename = py_files[0]
            
            # If we have a code block and filename, associate them
            if current_filename and code_block_index < len(code_blocks):
                code_sections[current_filename] = code_blocks[code_block_index].strip()
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
    
    async def analyze_xcode_error(self, error_message: str, use_deepseek: bool = True) -> Dict[str, Any]:
        """Analyze XCode error (single model version for backward compatibility)"""
        context = self.get_relevant_context(error_message)
        
        system_prompt = self._get_deepseek_system_prompt(True)
        prompt = self._get_deepseek_prompt(context, error_message, True)
        
        if use_deepseek:
            response = await self.query_deepseek(prompt, system_prompt)
        else:
            response = await self.query_gemini(f"{system_prompt}\n\n{prompt}")
        
        # Extract code sections
        code_sections = self._extract_code_sections(response)
        
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "xcode_error_analysis",
            "error": error_message,
            "response": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "code_sections": code_sections,
            "context_files_count": len(self.file_contexts)
        })
        
        return {
            "analysis": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "code_sections": code_sections,
            "context_files_used": len(self.file_contexts),
            "timestamp": datetime.now().isoformat()
        }
    
    async def general_coding_query(self, query: str, use_deepseek: bool = True) -> Dict[str, Any]:
        """Handle general coding queries (single model version for backward compatibility)"""
        context = self.get_relevant_context(query)
        
        system_prompt = self._get_deepseek_system_prompt(False)
        prompt = self._get_deepseek_prompt(context, query, False)
        
        if use_deepseek:
            response = await self.query_deepseek(prompt, system_prompt)
        else:
            response = await self.query_gemini(f"{system_prompt}\n\n{prompt}")
        
        # Extract code sections
        code_sections = self._extract_code_sections(response)
        
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "general_query",
            "query": query,
            "response": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "code_sections": code_sections,
            "context_files_count": len(self.file_contexts)
        })
        
        return {
            "response": response,
            "model_used": "deepseek" if use_deepseek else "gemini",
            "code_sections": code_sections,
            "context_files_used": len(self.file_contexts),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def clear_context(self):
        """Clear file context"""
        self.file_contexts.clear()
        print("File context cleared")
        
    def clear_old_conversations(self, max_conversations: int = 100):
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