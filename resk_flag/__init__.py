"""
resk-flag: A Python library combining symbolic trees with AI generative models.

This library provides tools for extracting flags from AI model responses and
building symbolic trees for reasoning analysis, error prediction, and logging.
"""

__version__ = "0.1.0"
__author__ = "Resk Security"
__email__ = "contact@resk-security.com"

# Core imports
from .flags import (
    Flag,
    FlagType,
    FlagParser,
    FlagValidator,
    FlagManager
)

from .tree import (
    TreeNode,
    NodeType,
    SymbolicTree,
    TreeBuilder
)

from .ai_integration import (
    AIInterface,
    AIRequest,
    AIResponse,
    MockAIInterface,
    OpenAIInterface,
    AIProcessor,
    BatchProcessor
)

from .analysis import (
    ErrorType,
    ErrorPrediction,
    ReasoningPath,
    ReasoningAnalyzer,
    ReasoningLogger
)

# Convenience imports
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    
    # Flags module
    "Flag",
    "FlagType",
    "FlagParser",
    "FlagValidator", 
    "FlagManager",
    
    # Tree module
    "TreeNode",
    "NodeType",
    "SymbolicTree",
    "TreeBuilder",
    
    # AI integration module
    "AIInterface",
    "AIRequest",
    "AIResponse",
    "MockAIInterface",
    "OpenAIInterface",
    "AIProcessor",
    "BatchProcessor",
    
    # Analysis module
    "ErrorType",
    "ErrorPrediction",
    "ReasoningPath",
    "ReasoningAnalyzer",
    "ReasoningLogger",
    
    # High-level functions
    "create_processor",
    "analyze_text",
    "quick_analysis"
]


def create_processor(ai_interface=None, use_mock=True):
    """
    Create an AI processor with the specified interface.
    
    Args:
        ai_interface: Custom AI interface to use
        use_mock: If True and no interface provided, use MockAIInterface
        
    Returns:
        AIProcessor instance
    """
    if ai_interface is None and use_mock:
        ai_interface = MockAIInterface()
    elif ai_interface is None:
        raise ValueError("Must provide ai_interface or set use_mock=True")
    
    return AIProcessor(ai_interface)


def analyze_text(text, ai_interface=None, use_mock=True):
    """
    Quick analysis of text using the default pipeline.
    
    Args:
        text: Text to analyze
        ai_interface: Custom AI interface to use
        use_mock: If True and no interface provided, use MockAIInterface
        
    Returns:
        Analysis results dictionary
    """
    processor = create_processor(ai_interface, use_mock)
    result = processor.process_query(text)
    
    # Add analysis
    analyzer = ReasoningAnalyzer()
    analysis = analyzer.analyze_tree(result["tree"])
    
    return {
        "processing_result": result,
        "analysis": analysis
    }


def quick_analysis(query, **kwargs):
    """
    Perform quick analysis with minimal setup.
    
    Args:
        query: Query to analyze
        **kwargs: Additional arguments for AI request
        
    Returns:
        Simplified analysis results
    """
    result = analyze_text(query, **kwargs)
    
    # Extract key information
    flags = result["processing_result"]["flags"]
    tree = result["processing_result"]["tree"]
    errors = result["analysis"]["error_predictions"]
    quality = result["analysis"]["quality_scores"]
    
    return {
        "query": query,
        "flags_found": len(flags),
        "tree_nodes": tree.get_node_count(),
        "tree_depth": tree.get_depth(),
        "errors_detected": len(errors),
        "critical_errors": len([e for e in errors if e["severity"] == "critical"]),
        "quality_score": quality.get("overall", 0.0),
        "recommendations": result["analysis"]["recommendations"][:3]  # Top 3
    }


# Example usage and documentation
EXAMPLE_USAGE = """
# Basic usage
from resk_flag import create_processor, analyze_text

# Using mock AI (for testing)
processor = create_processor(use_mock=True)
result = processor.process_query("How should I solve this problem?")

# Quick analysis
summary = quick_analysis("What are the risks of this approach?")
print(f"Quality score: {summary['quality_score']:.2f}")

# Using with real AI (requires API key)
from resk_flag import OpenAIInterface
ai = OpenAIInterface(api_key="your-api-key")
processor = create_processor(ai_interface=ai)
result = processor.process_query("Analyze the market trends")

# Detailed analysis
full_analysis = analyze_text("Complex reasoning query", ai_interface=ai)
"""

# Package configuration
DEFAULT_CONFIG = {
    "flag_parser": {
        "min_content_length": 3,
        "max_content_length": 1000
    },
    "tree_builder": {
        "max_depth": 10,
        "max_children_per_node": 5
    },
    "analyzer": {
        "error_detection_enabled": True,
        "confidence_threshold": 0.5
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}


def get_config():
    """Get current package configuration."""
    return DEFAULT_CONFIG.copy()


def set_config(config_dict):
    """Set package configuration."""
    DEFAULT_CONFIG.update(config_dict)


# Package information
def get_info():
    """Get package information."""
    return {
        "name": "resk-flag",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "A Python library combining symbolic trees with AI generative models",
        "components": {
            "flags": "Flag extraction and management",
            "tree": "Symbolic tree construction and manipulation", 
            "ai_integration": "AI model integration and processing",
            "analysis": "Reasoning analysis and error prediction"
        },
        "dependencies": {
            "required": ["typing-extensions"],
            "optional": ["openai", "anthropic"]
        }
    }


# Initialize logging
import logging
logging.getLogger("resk_flag").setLevel(logging.INFO)