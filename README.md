# resk-flag

A Python library that combines symbolic trees with AI generative models using flags for reasoning analysis, error prediction, and logging.

## Overview

resk-flag extracts "flags" from AI model responses and uses them to construct symbolic trees that represent reasoning processes. This enables analysis of reasoning paths, error prediction, and structured logging of AI decision-making.

## Features

- **Flag Extraction**: Parse structured flags from AI responses in multiple formats (XML, JSON, Markdown)
- **Symbolic Trees**: Build tree structures representing reasoning flows and decision paths
- **AI Integration**: Connect with various AI models (OpenAI, Anthropic, or custom interfaces)
- **Error Prediction**: Detect potential issues in reasoning chains (logical inconsistencies, missing validation, etc.)
- **Analysis Tools**: Comprehensive analysis of reasoning quality and completeness
- **Flexible Interface**: Support for both mock testing and real AI model integration

## Installation

```bash
pip install resk-flag
```

For AI model integration, install optional dependencies:

```bash
# For OpenAI integration
pip install resk-flag[ai]

# For development
pip install resk-flag[dev]
```

## Quick Start

```python
from resk_flag import quick_analysis

# Simple analysis
result = quick_analysis("How should I approach learning machine learning?")
print(f"Quality score: {result['quality_score']:.2f}")
print(f"Flags found: {result['flags_found']}")
print("Recommendations:", result['recommendations'][:3])
```

## Basic Usage

### 1. Using Mock AI (for testing)

```python
from resk_flag import create_processor

# Create processor with mock AI
processor = create_processor(use_mock=True)

# Process a query
result = processor.process_query("What are the risks of this approach?")

# Access results
print(f"Flags extracted: {len(result['flags'])}")
print(f"Tree nodes: {result['tree'].get_node_count()}")
```

### 2. Using Real AI Models

```python
from resk_flag import OpenAIInterface, create_processor

# Setup OpenAI interface
ai = OpenAIInterface(api_key="your-api-key", model="gpt-3.5-turbo")
processor = create_processor(ai_interface=ai)

# Process query
result = processor.process_query("Analyze the market trends")
```

### 3. Comprehensive Analysis

```python
from resk_flag import analyze_text, ReasoningAnalyzer

# Full analysis with error detection
analysis = analyze_text("Should I invest in cryptocurrency?")

# Access analysis components
processing = analysis['processing_result']
analysis_data = analysis['analysis']

print(f"Quality scores: {analysis_data['quality_scores']}")
print(f"Errors detected: {len(analysis_data['error_predictions'])}")
print(f"Recommendations: {analysis_data['recommendations']}")
```

## Flag Formats

The library supports multiple flag formats in AI responses:

### XML Style
```xml
<reasoning>
Let me analyze this step by step...
</reasoning>

<decision>
I choose option A because...
</decision>

<conclusion>
The best approach is...
</conclusion>
```

### JSON Style
```
[FLAG:reasoning:analyzing the data step by step]
[FLAG:decision:choosing the optimal solution]
[FLAG:conclusion:final recommendation]
```

### Markdown Style
```markdown
```reasoning
This requires careful consideration of...
```

```validation
Testing confirms this approach works...
```
```

## Flag Types

- `reasoning` - Analytical thinking steps
- `decision` - Choices or determinations
- `hypothesis` - Proposed explanations
- `validation` - Testing or verification
- `step` - Procedural steps
- `condition` - Conditional logic
- `conclusion` - Final results
- `error` - Problem identification

## Tree Analysis

### Building Trees from Flags

```python
from resk_flag import TreeBuilder, FlagManager

# Extract flags from text
flag_manager = FlagManager()
flags = flag_manager.add_flags_from_text(response_text)

# Build symbolic tree
builder = TreeBuilder()
tree = builder.build_from_flags(flags, "Analysis Process")

# Analyze tree structure
print(f"Tree depth: {tree.get_depth()}")
print(f"Leaf nodes: {len(tree.get_leaves())}")
```

### Manual Tree Construction

```python
from resk_flag import SymbolicTree, NodeType

tree = SymbolicTree("Decision Process")

# Add nodes
tree.add_node("root", "analysis", NodeType.REASONING, 
              "Analyzing options", confidence=0.9)

tree.add_node("analysis", "decision", NodeType.DECISION,
              "Choose option A", confidence=0.8)

tree.add_node("decision", "conclusion", NodeType.CONCLUSION,
              "Implement solution", confidence=0.95)
```

## Error Detection

The library includes sophisticated error detection:

```python
from resk_flag import ReasoningAnalyzer

analyzer = ReasoningAnalyzer()
analysis = analyzer.analyze_tree(tree)

# Check for errors
errors = analysis['error_predictions']
for error in errors:
    print(f"Error: {error['error_type']}")
    print(f"Severity: {error['severity']}")
    print(f"Description: {error['description']}")
    print(f"Suggestions: {error['suggestions']}")
```

## Error Types Detected

- **Logical Inconsistency**: Contradictory reasoning steps
- **Missing Validation**: Hypotheses without verification
- **Circular Reasoning**: Self-referential logic loops
- **Insufficient Evidence**: Conclusions without adequate support
- **Contradictory Conclusions**: Conflicting final results
- **Incomplete Reasoning**: Too shallow analysis chains
- **Overconfident Conclusions**: High confidence with little evidence

## Batch Processing

Process multiple queries efficiently:

```python
from resk_flag import BatchProcessor, MockAIInterface

# Create batch processor
batch = BatchProcessor(MockAIInterface())

# Add multiple queries
batch.add_query("What are the risks?")
batch.add_query("How should I proceed?")
batch.add_query("What are the alternatives?")

# Get summary statistics
summary = batch.get_summary()
print(f"Total queries: {summary['total_queries']}")
print(f"Average tree depth: {summary['average_tree_depth']:.2f}")

# Export results
batch.export_results("analysis_results.json")
```

## Configuration

```python
from resk_flag import get_config, set_config

# Get current configuration
config = get_config()

# Modify configuration
set_config({
    "flag_parser": {
        "min_content_length": 5,
        "max_content_length": 500
    },
    "analyzer": {
        "confidence_threshold": 0.7
    }
})
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_example.py` - Complete walkthrough of all features
- `advanced_analysis.py` - Complex reasoning analysis
- `custom_ai_integration.py` - Implementing custom AI interfaces

## API Reference

### Core Classes

- **FlagManager**: Extract and manage flags from text
- **SymbolicTree**: Tree structure for reasoning representation
- **TreeBuilder**: Build trees from flag sequences
- **AIProcessor**: High-level AI processing with flag extraction
- **ReasoningAnalyzer**: Analyze reasoning quality and detect errors
- **MockAIInterface**: Mock AI for testing and development

### Key Functions

- `create_processor()` - Create AI processor with interface
- `analyze_text()` - Complete text analysis pipeline
- `quick_analysis()` - Simplified analysis for rapid insights

## Development

```bash
# Clone repository
git clone https://github.com/Resk-Security/resk-flag.git
cd resk-flag

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run tests with coverage
pytest --cov=resk_flag

# Format code
black resk_flag/
isort resk_flag/

# Type checking
mypy resk_flag/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [GitHub Wiki](https://github.com/Resk-Security/resk-flag/wiki)
- Issues: [GitHub Issues](https://github.com/Resk-Security/resk-flag/issues)
- Discussions: [GitHub Discussions](https://github.com/Resk-Security/resk-flag/discussions)

## Citation

If you use resk-flag in your research, please cite:

```bibtex
@software{resk_flag,
  title={resk-flag: Symbolic Trees with AI Generative Models},
  author={Resk Security},
  url={https://github.com/Resk-Security/resk-flag},
  year={2024}
}
```