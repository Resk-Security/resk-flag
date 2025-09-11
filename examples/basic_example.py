"""
Basic example demonstrating the resk-flag library functionality.
"""

from resk_flag import (
    create_processor, analyze_text, quick_analysis,
    FlagType, NodeType, MockAIInterface
)
import json


def main():
    """Main example function."""
    print("=== resk-flag Library Example ===\n")
    
    # Example 1: Quick analysis
    print("1. Quick Analysis Example:")
    print("-" * 30)
    
    query = "How should I approach learning machine learning? What are the risks?"
    summary = quick_analysis(query)
    
    print(f"Query: {summary['query']}")
    print(f"Flags found: {summary['flags_found']}")
    print(f"Tree nodes: {summary['tree_nodes']}")
    print(f"Tree depth: {summary['tree_depth']}")
    print(f"Quality score: {summary['quality_score']:.2f}")
    print(f"Errors detected: {summary['errors_detected']}")
    print("Top recommendations:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"  {i}. {rec}")
    print()
    
    # Example 2: Detailed processing
    print("2. Detailed Processing Example:")
    print("-" * 35)
    
    # Create processor with mock AI
    processor = create_processor(use_mock=True)
    
    # Process a query
    result = processor.process_query(
        "I need to analyze whether to invest in cryptocurrency. "
        "What factors should I consider and what are the potential risks?"
    )
    
    print(f"Response from AI model: {result['response'].model}")
    print(f"Flags extracted: {len(result['flags'])}")
    print("\nFlag types found:")
    flag_types = {}
    for flag in result['flags']:
        flag_type = flag.type.value
        flag_types[flag_type] = flag_types.get(flag_type, 0) + 1
    
    for flag_type, count in flag_types.items():
        print(f"  {flag_type}: {count}")
    
    # Show tree structure
    tree = result['tree']
    print(f"\nTree structure:")
    print(f"  Total nodes: {tree.get_node_count()}")
    print(f"  Max depth: {tree.get_depth()}")
    print(f"  Leaf nodes: {len(tree.get_leaves())}")
    
    # Show some flag contents
    print("\nSample flags:")
    for i, flag in enumerate(result['flags'][:3]):
        print(f"  {i+1}. [{flag.type.value}] {flag.content[:60]}...")
    print()
    
    # Example 3: Analysis with error detection
    print("3. Analysis with Error Detection:")
    print("-" * 37)
    
    # Analyze the reasoning
    analysis_result = analyze_text(
        "I think cryptocurrency is good because it's popular. "
        "Therefore, I should invest all my money in it."
    )
    
    analysis = analysis_result['analysis']
    
    print(f"Quality scores:")
    for metric, score in analysis['quality_scores'].items():
        print(f"  {metric}: {score:.2f}")
    
    print(f"\nError predictions: {len(analysis['error_predictions'])}")
    for i, error in enumerate(analysis['error_predictions'][:2]):
        print(f"  {i+1}. {error['error_type']} (severity: {error['severity']})")
        print(f"     {error['description']}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    print()
    
    # Example 4: Custom AI interface
    print("4. Custom AI Interface Example:")
    print("-" * 34)
    
    # Create custom mock with specific templates
    custom_ai = MockAIInterface("custom-model")
    custom_ai.add_template("analysis", """
    <reasoning>
    Let me break down this complex problem systematically.
    </reasoning>
    
    <step>
    First, I'll identify the key variables and constraints.
    </step>
    
    <step>
    Next, I'll evaluate different solution approaches.
    </step>
    
    <hypothesis>
    The optimal solution likely involves a combination of approaches.
    </hypothesis>
    
    <validation>
    I should test this hypothesis with some examples.
    </validation>
    
    <conclusion>
    Based on the analysis, I recommend a phased implementation.
    </conclusion>
    """)
    
    # Use custom processor
    custom_processor = create_processor(ai_interface=custom_ai, use_mock=False)
    custom_result = custom_processor.process_query("Analyze this complex situation")
    
    print(f"Custom AI model: {custom_result['response'].model}")
    print(f"Flags found: {len(custom_result['flags'])}")
    
    # Show the reasoning path
    paths = custom_result['tree'].get_paths_to_leaves()
    if paths:
        longest_path = max(paths, key=len)
        print(f"\nReasoning path (length {len(longest_path)}):")
        for i, node in enumerate(longest_path):
            indent = "  " * node.depth
            print(f"{indent}{i+1}. [{node.type.value}] {node.content[:50]}...")
    print()
    
    # Example 5: Flag extraction from custom text
    print("5. Direct Flag Extraction:")
    print("-" * 28)
    
    from resk_flag import FlagManager
    
    flag_manager = FlagManager()
    
    # Text with various flag formats
    text_with_flags = """
    Let me analyze this step by step:
    
    <reasoning>
    The problem requires careful consideration of multiple factors.
    We need to balance efficiency with accuracy.
    </reasoning>
    
    [FLAG:decision:I choose the hybrid approach for best results]
    
    ```validation
    Testing this approach with sample data shows promising results.
    The accuracy is 94% with good performance characteristics.
    ```
    
    <conclusion>
    The hybrid approach is the recommended solution.
    </conclusion>
    """
    
    extracted_flags = flag_manager.add_flags_from_text(text_with_flags)
    
    print(f"Extracted {len(extracted_flags)} flags from custom text:")
    for flag in extracted_flags:
        print(f"  [{flag.type.value}] {flag.content.strip()[:60]}...")
    
    # Export flags to JSON
    flags_json = flag_manager.export_json()
    print(f"\nFlags exported to JSON ({len(flags_json)} characters)")
    print()
    
    # Example 6: Tree manipulation
    print("6. Manual Tree Construction:")
    print("-" * 31)
    
    from resk_flag import SymbolicTree, NodeType
    
    # Create tree manually
    manual_tree = SymbolicTree("Investment Decision Process")
    
    # Add nodes step by step
    manual_tree.add_node("root", "analysis", NodeType.REASONING, 
                        "Analyzing investment options", confidence=0.9)
    
    manual_tree.add_node("analysis", "option1", NodeType.STEP,
                        "Evaluate stocks", confidence=0.8)
    
    manual_tree.add_node("analysis", "option2", NodeType.STEP,
                        "Evaluate bonds", confidence=0.8)
    
    manual_tree.add_node("option1", "stock_decision", NodeType.DECISION,
                        "Choose growth stocks", confidence=0.7)
    
    manual_tree.add_node("option2", "bond_decision", NodeType.DECISION,
                        "Choose government bonds", confidence=0.9)
    
    manual_tree.add_node("root", "final_conclusion", NodeType.CONCLUSION,
                        "Diversify portfolio with 60% stocks, 40% bonds", confidence=0.85)
    
    print(f"Manual tree created:")
    print(f"  Nodes: {manual_tree.get_node_count()}")
    print(f"  Depth: {manual_tree.get_depth()}")
    print(f"  Leaves: {len(manual_tree.get_leaves())}")
    
    # Show tree structure
    print(f"\nTree structure:")
    for node in manual_tree.traverse_dfs():
        indent = "  " * node.depth
        print(f"{indent}- [{node.type.value}] {node.content}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()