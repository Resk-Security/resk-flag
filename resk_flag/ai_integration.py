"""
AI integration interface for connecting with language models.
Provides abstract interface for LLM communication and response processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
from .flags import Flag, FlagManager, FlagType
from .tree import SymbolicTree, TreeBuilder


@dataclass
class AIResponse:
    """Represents a response from an AI model."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AIRequest:
    """Represents a request to an AI model."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AIInterface(ABC):
    """Abstract interface for AI model communication."""
    
    @abstractmethod
    def send_request(self, request: AIRequest) -> AIResponse:
        """
        Send request to AI model and get response.
        
        Args:
            request: The request to send
            
        Returns:
            Response from the AI model
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name/identifier of the AI model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI model is available for requests."""
        pass


class MockAIInterface(AIInterface):
    """Mock AI interface for testing and development."""
    
    def __init__(self, model_name: str = "mock-ai"):
        """Initialize mock AI interface."""
        self.model_name = model_name
        self._available = True
        self._response_templates = self._default_templates()
    
    def _default_templates(self) -> Dict[str, str]:
        """Default response templates for different types of requests."""
        return {
            "reasoning": """
            <reasoning>
            Let me analyze this step by step:
            1. First, I need to understand the problem
            2. Then identify key components
            3. Finally, propose a solution
            </reasoning>
            
            <decision>
            Based on my analysis, I recommend proceeding with option A
            </decision>
            
            <conclusion>
            This approach should provide the best balance of efficiency and reliability
            </conclusion>
            """,
            "problem_solving": """
            <step>
            Identify the core issue: {problem}
            </step>
            
            <hypothesis>
            The root cause appears to be related to data validation
            </hypothesis>
            
            <validation>
            Testing this hypothesis by checking input constraints
            </validation>
            
            <conclusion>
            Implementing proper validation should resolve the issue
            </conclusion>
            """,
            "error_analysis": """
            <error>
            Critical issue detected in the logic flow
            </error>
            
            <reasoning>
            The error occurs because of invalid assumptions about input data
            </reasoning>
            
            <decision>
            Need to add comprehensive error handling
            </decision>
            """
        }
    
    def send_request(self, request: AIRequest) -> AIResponse:
        """Send mock request and return templated response."""
        if not self._available:
            raise RuntimeError("Mock AI interface is not available")
        
        # Select template based on prompt content
        template_key = "reasoning"  # default
        prompt_lower = request.prompt.lower()
        
        if "problem" in prompt_lower or "solve" in prompt_lower:
            template_key = "problem_solving"
        elif "error" in prompt_lower or "debug" in prompt_lower:
            template_key = "error_analysis"
        elif any(word in prompt_lower for word in ["how should", "approach", "what", "analyze"]):
            template_key = "reasoning"
        
        content = self._response_templates[template_key]
        
        # Replace placeholders if any
        if "{problem}" in content and "problem" in prompt_lower:
            content = content.replace("{problem}", "the identified issue")
        
        return AIResponse(
            content=content.strip(),
            model=self.model_name,
            tokens_used=len(content.split()),
            confidence=0.85,
            metadata={"template_used": template_key}
        )
    
    def get_model_name(self) -> str:
        """Get mock model name."""
        return self.model_name
    
    def is_available(self) -> bool:
        """Check if mock interface is available."""
        return self._available
    
    def set_available(self, available: bool) -> None:
        """Set availability status for testing."""
        self._available = available
    
    def add_template(self, key: str, template: str) -> None:
        """Add custom response template."""
        self._response_templates[key] = template


try:
    import openai
    
    class OpenAIInterface(AIInterface):
        """OpenAI API interface implementation."""
        
        def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
            """
            Initialize OpenAI interface.
            
            Args:
                api_key: OpenAI API key
                model: Model name to use
            """
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        
        def send_request(self, request: AIRequest) -> AIResponse:
            """Send request to OpenAI API."""
            messages = []
            
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            
            messages.append({"role": "user", "content": request.prompt})
            
            try:
                response = self.client.chat.completions.create(
                    model=request.model or self.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                return AIResponse(
                    content=response.choices[0].message.content,
                    model=response.model,
                    tokens_used=response.usage.total_tokens if response.usage else None,
                    metadata={
                        "finish_reason": response.choices[0].finish_reason,
                        "usage": response.usage.model_dump() if response.usage else None
                    }
                )
            except Exception as e:
                raise RuntimeError(f"OpenAI API error: {str(e)}")
        
        def get_model_name(self) -> str:
            """Get OpenAI model name."""
            return self.model
        
        def is_available(self) -> bool:
            """Check if OpenAI API is available."""
            try:
                # Simple test request
                self.client.models.list()
                return True
            except:
                return False

except ImportError:
    # OpenAI not available
    class OpenAIInterface(AIInterface):
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        def send_request(self, request: AIRequest) -> AIResponse:
            raise NotImplementedError
        
        def get_model_name(self) -> str:
            raise NotImplementedError
        
        def is_available(self) -> bool:
            return False


class AIProcessor:
    """High-level processor for AI-driven flag extraction and tree building."""
    
    def __init__(self, ai_interface: AIInterface):
        """
        Initialize AI processor.
        
        Args:
            ai_interface: AI interface implementation to use
        """
        self.ai_interface = ai_interface
        self.flag_manager = FlagManager()
        self.tree_builder = TreeBuilder()
        self._system_prompt = self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Default system prompt that instructs AI to use flags."""
        return """
You are an AI assistant that structures your responses using specific flags to indicate different types of reasoning steps. Please use the following XML-style flags in your responses:

- <reasoning>...</reasoning> for analytical thinking
- <decision>...</decision> for choices or determinations  
- <hypothesis>...</hypothesis> for proposed explanations
- <validation>...</validation> for testing or verification
- <step>...</step> for procedural steps
- <condition>...</condition> for conditional logic
- <conclusion>...</conclusion> for final results
- <error>...</error> for identifying problems

Structure your response to show your reasoning process clearly using these flags.
"""
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt."""
        self._system_prompt = prompt
    
    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query through AI and extract flags to build symbolic tree.
        
        Args:
            query: The query to send to the AI
            **kwargs: Additional arguments for AI request
            
        Returns:
            Dictionary containing response, flags, and tree
        """
        # Create AI request
        request = AIRequest(
            prompt=query,
            system_prompt=self._system_prompt,
            **kwargs
        )
        
        # Get AI response
        response = self.ai_interface.send_request(request)
        
        # Extract flags from response
        flags = self.flag_manager.add_flags_from_text(response.content)
        
        # Build symbolic tree from flags
        tree = self.tree_builder.build_from_flags(flags, f"Query: {query[:50]}...")
        
        return {
            "query": query,
            "response": response,
            "flags": flags,
            "tree": tree,
            "metadata": {
                "flag_count": len(flags),
                "tree_depth": tree.get_depth(),
                "tree_nodes": tree.get_node_count(),
                "model_used": response.model
            }
        }
    
    def process_conversation(self, messages: List[str]) -> List[Dict[str, Any]]:
        """
        Process a conversation with multiple messages.
        
        Args:
            messages: List of messages to process
            
        Returns:
            List of processing results for each message
        """
        results = []
        for message in messages:
            result = self.process_query(message)
            results.append(result)
        return results
    
    def get_all_flags(self) -> List[Flag]:
        """Get all flags collected during processing."""
        return self.flag_manager.get_flags()
    
    def get_flags_by_type(self, flag_type: FlagType) -> List[Flag]:
        """Get flags of specific type."""
        return self.flag_manager.get_flags(flag_type)
    
    def clear_history(self) -> None:
        """Clear all collected flags and processing history."""
        self.flag_manager.clear()
    
    def export_session_data(self) -> str:
        """Export all session data (flags) to JSON."""
        return self.flag_manager.export_json()
    
    def import_session_data(self, json_str: str) -> None:
        """Import session data from JSON."""
        self.flag_manager.import_json(json_str)


class BatchProcessor:
    """Processor for handling multiple queries in batch."""
    
    def __init__(self, ai_interface: AIInterface):
        """Initialize batch processor."""
        self.ai_processor = AIProcessor(ai_interface)
        self.results: List[Dict[str, Any]] = []
    
    def add_query(self, query: str, **kwargs) -> None:
        """Add query to batch for processing."""
        result = self.ai_processor.process_query(query, **kwargs)
        self.results.append(result)
    
    def process_file(self, file_path: str, **kwargs) -> None:
        """Process queries from a text file (one per line)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                query = line.strip()
                if query:
                    try:
                        result = self.ai_processor.process_query(query, **kwargs)
                        result["source"] = {"file": file_path, "line": line_num}
                        self.results.append(result)
                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all batch processing results."""
        return self.results.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for batch processing."""
        if not self.results:
            return {"message": "No results to summarize"}
        
        total_flags = sum(len(result["flags"]) for result in self.results)
        total_nodes = sum(result["metadata"]["tree_nodes"] for result in self.results)
        avg_depth = sum(result["metadata"]["tree_depth"] for result in self.results) / len(self.results)
        
        flag_types = {}
        for result in self.results:
            for flag in result["flags"]:
                flag_type = flag.type.value
                flag_types[flag_type] = flag_types.get(flag_type, 0) + 1
        
        return {
            "total_queries": len(self.results),
            "total_flags": total_flags,
            "total_nodes": total_nodes,
            "average_tree_depth": avg_depth,
            "flag_type_distribution": flag_types,
            "models_used": list(set(result["metadata"]["model_used"] for result in self.results))
        }
    
    def export_results(self, file_path: str) -> None:
        """Export all results to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": self.results,
                "summary": self.get_summary()
            }, f, indent=2, default=str)
    
    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()
        self.ai_processor.clear_history()