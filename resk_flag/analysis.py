"""
Analysis tools for symbolic trees and reasoning paths.
Provides error prediction, logging, and reasoning analysis capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import logging
from datetime import datetime
from .flags import Flag, FlagType
from .tree import SymbolicTree, TreeNode, NodeType


class ErrorType(Enum):
    """Types of errors that can be detected in reasoning."""
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    MISSING_VALIDATION = "missing_validation"
    CIRCULAR_REASONING = "circular_reasoning"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTORY_CONCLUSIONS = "contradictory_conclusions"
    INCOMPLETE_REASONING = "incomplete_reasoning"
    OVERCONFIDENT_CONCLUSION = "overconfident_conclusion"


@dataclass
class ErrorPrediction:
    """Represents a predicted error in reasoning."""
    error_type: ErrorType
    confidence: float
    description: str
    affected_nodes: List[str]
    severity: str  # "low", "medium", "high", "critical"
    suggestions: List[str]
    metadata: Dict[str, Any]


@dataclass
class ReasoningPath:
    """Represents a path through the reasoning tree."""
    nodes: List[TreeNode]
    confidence: float
    coherence_score: float
    completeness_score: float
    length: int
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        self.length = len(self.nodes)


class ReasoningAnalyzer:
    """Analyzer for reasoning paths and error detection."""
    
    def __init__(self):
        """Initialize reasoning analyzer."""
        self.error_patterns = self._init_error_patterns()
        self.logger = self._setup_logger()
    
    def _init_error_patterns(self) -> Dict[ErrorType, Dict[str, Any]]:
        """Initialize error detection patterns."""
        return {
            ErrorType.LOGICAL_INCONSISTENCY: {
                "node_types": [NodeType.DECISION, NodeType.CONCLUSION],
                "min_confidence": 0.3,
                "description": "Detected potential logical inconsistency"
            },
            ErrorType.MISSING_VALIDATION: {
                "required_after": [NodeType.HYPOTHESIS],
                "missing_type": NodeType.VALIDATION,
                "description": "Hypothesis without validation step"
            },
            ErrorType.CIRCULAR_REASONING: {
                "max_revisits": 2,
                "description": "Circular reasoning pattern detected"
            },
            ErrorType.INSUFFICIENT_EVIDENCE: {
                "min_reasoning_nodes": 2,
                "before_conclusion": True,
                "description": "Insufficient reasoning before conclusion"
            },
            ErrorType.CONTRADICTORY_CONCLUSIONS: {
                "node_type": NodeType.CONCLUSION,
                "similarity_threshold": 0.7,
                "description": "Contradictory conclusions detected"
            },
            ErrorType.INCOMPLETE_REASONING: {
                "min_depth": 3,
                "required_types": [NodeType.REASONING, NodeType.DECISION],
                "description": "Reasoning chain appears incomplete"
            },
            ErrorType.OVERCONFIDENT_CONCLUSION: {
                "confidence_threshold": 0.9,
                "min_evidence_nodes": 3,
                "description": "High confidence with insufficient evidence"
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for analysis events."""
        logger = logging.getLogger("resk_flag.analysis")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def analyze_tree(self, tree: SymbolicTree) -> Dict[str, Any]:
        """
        Comprehensive analysis of a symbolic tree.
        
        Args:
            tree: The symbolic tree to analyze
            
        Returns:
            Analysis results dictionary
        """
        self.logger.info(f"Starting analysis of tree with {tree.get_node_count()} nodes")
        
        # Get all reasoning paths
        paths = self.extract_reasoning_paths(tree)
        
        # Detect errors
        errors = self.predict_errors(tree)
        
        # Calculate tree metrics
        metrics = self.calculate_tree_metrics(tree)
        
        # Analyze reasoning quality
        quality_scores = self.analyze_reasoning_quality(tree, paths)
        
        analysis_result = {
            "tree_id": tree.root.id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "reasoning_paths": [self._path_to_dict(path) for path in paths],
            "error_predictions": [self._error_to_dict(error) for error in errors],
            "quality_scores": quality_scores,
            "recommendations": self.generate_recommendations(tree, errors, quality_scores)
        }
        
        self.logger.info(f"Analysis complete. Found {len(errors)} potential issues")
        return analysis_result
    
    def extract_reasoning_paths(self, tree: SymbolicTree) -> List[ReasoningPath]:
        """Extract all reasoning paths from root to leaves."""
        paths = []
        node_paths = tree.get_paths_to_leaves()
        
        for node_path in node_paths:
            # Calculate path metrics
            confidence = self._calculate_path_confidence(node_path)
            coherence = self._calculate_coherence_score(node_path)
            completeness = self._calculate_completeness_score(node_path)
            
            path = ReasoningPath(
                nodes=node_path,
                confidence=confidence,
                coherence_score=coherence,
                completeness_score=completeness,
                length=len(node_path),
                metadata={
                    "types_sequence": [node.type.value for node in node_path],
                    "has_validation": any(node.type == NodeType.VALIDATION for node in node_path),
                    "has_error": any(node.type == NodeType.ERROR for node in node_path)
                }
            )
            paths.append(path)
        
        return paths
    
    def predict_errors(self, tree: SymbolicTree) -> List[ErrorPrediction]:
        """Predict potential errors in the reasoning tree."""
        errors = []
        
        # Check for logical inconsistencies
        errors.extend(self._detect_logical_inconsistencies(tree))
        
        # Check for missing validations
        errors.extend(self._detect_missing_validations(tree))
        
        # Check for circular reasoning
        errors.extend(self._detect_circular_reasoning(tree))
        
        # Check for insufficient evidence
        errors.extend(self._detect_insufficient_evidence(tree))
        
        # Check for contradictory conclusions
        errors.extend(self._detect_contradictory_conclusions(tree))
        
        # Check for incomplete reasoning
        errors.extend(self._detect_incomplete_reasoning(tree))
        
        # Check for overconfident conclusions
        errors.extend(self._detect_overconfident_conclusions(tree))
        
        # Sort by severity and confidence
        errors.sort(key=lambda e: (self._severity_weight(e.severity), -e.confidence))
        
        return errors
    
    def calculate_tree_metrics(self, tree: SymbolicTree) -> Dict[str, Any]:
        """Calculate various metrics for the tree."""
        nodes = tree.get_all_nodes()
        
        # Basic metrics
        total_nodes = len(nodes)
        max_depth = tree.get_depth()
        leaf_count = len(tree.get_leaves())
        
        # Node type distribution
        type_counts = {}
        for node in nodes:
            node_type = node.type.value
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        # Confidence metrics
        confidences = [node.confidence for node in nodes if node.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        max_confidence = max(confidences) if confidences else 0
        
        # Tree balance (how evenly distributed children are)
        balance_score = self._calculate_tree_balance(tree)
        
        return {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "leaf_count": leaf_count,
            "avg_children_per_node": (total_nodes - 1) / max(1, total_nodes - leaf_count),
            "node_type_distribution": type_counts,
            "confidence_stats": {
                "average": avg_confidence,
                "minimum": min_confidence,
                "maximum": max_confidence,
                "count": len(confidences)
            },
            "balance_score": balance_score,
            "reasoning_density": type_counts.get("reasoning", 0) / total_nodes,
            "decision_points": type_counts.get("decision", 0),
            "error_nodes": type_counts.get("error", 0)
        }
    
    def analyze_reasoning_quality(self, tree: SymbolicTree, 
                                paths: List[ReasoningPath]) -> Dict[str, float]:
        """Analyze overall reasoning quality."""
        if not paths:
            return {"overall": 0.0}
        
        # Average path confidence
        avg_confidence = sum(path.confidence for path in paths) / len(paths)
        
        # Average coherence
        avg_coherence = sum(path.coherence_score for path in paths) / len(paths)
        
        # Average completeness
        avg_completeness = sum(path.completeness_score for path in paths) / len(paths)
        
        # Logical structure score
        structure_score = self._calculate_structure_score(tree)
        
        # Evidence sufficiency
        evidence_score = self._calculate_evidence_score(tree)
        
        # Overall quality (weighted average)
        overall = (
            avg_confidence * 0.2 +
            avg_coherence * 0.25 +
            avg_completeness * 0.2 +
            structure_score * 0.2 +
            evidence_score * 0.15
        )
        
        return {
            "overall": overall,
            "confidence": avg_confidence,
            "coherence": avg_coherence,
            "completeness": avg_completeness,
            "structure": structure_score,
            "evidence": evidence_score
        }
    
    def generate_recommendations(self, tree: SymbolicTree, 
                               errors: List[ErrorPrediction],
                               quality_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving reasoning."""
        recommendations = []
        
        # Error-based recommendations
        high_severity_errors = [e for e in errors if e.severity in ["high", "critical"]]
        if high_severity_errors:
            recommendations.append("Address critical reasoning errors before proceeding")
            for error in high_severity_errors[:3]:  # Top 3 critical errors
                recommendations.extend(error.suggestions)
        
        # Quality-based recommendations
        if quality_scores.get("coherence", 0) < 0.6:
            recommendations.append("Improve logical flow between reasoning steps")
        
        if quality_scores.get("completeness", 0) < 0.7:
            recommendations.append("Add more detailed reasoning steps")
        
        if quality_scores.get("evidence", 0) < 0.6:
            recommendations.append("Provide more supporting evidence for conclusions")
        
        # Structure-based recommendations
        validation_nodes = len(tree.get_nodes_by_type(NodeType.VALIDATION))
        decision_nodes = len(tree.get_nodes_by_type(NodeType.DECISION))
        
        if validation_nodes == 0 and decision_nodes > 0:
            recommendations.append("Add validation steps to verify key decisions")
        
        if tree.get_depth() < 3:
            recommendations.append("Develop more detailed reasoning chain")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    # Helper methods for error detection
    def _detect_logical_inconsistencies(self, tree: SymbolicTree) -> List[ErrorPrediction]:
        """Detect logical inconsistencies in reasoning."""
        errors = []
        # Simplified implementation - would need more sophisticated logic
        decision_nodes = tree.get_nodes_by_type(NodeType.DECISION)
        
        for node in decision_nodes:
            if node.confidence < 0.5:
                errors.append(ErrorPrediction(
                    error_type=ErrorType.LOGICAL_INCONSISTENCY,
                    confidence=0.7,
                    description=f"Low confidence decision: {node.content[:50]}...",
                    affected_nodes=[node.id],
                    severity="medium",
                    suggestions=["Review decision criteria", "Gather more evidence"],
                    metadata={"node_confidence": node.confidence}
                ))
        
        return errors
    
    def _detect_missing_validations(self, tree: SymbolicTree) -> List[ErrorPrediction]:
        """Detect hypotheses without validation."""
        errors = []
        hypothesis_nodes = tree.get_nodes_by_type(NodeType.HYPOTHESIS)
        
        for hyp_node in hypothesis_nodes:
            # Check if there's a validation node in the subtree
            descendants = hyp_node.get_descendants()
            has_validation = any(node.type == NodeType.VALIDATION for node in descendants)
            
            if not has_validation:
                errors.append(ErrorPrediction(
                    error_type=ErrorType.MISSING_VALIDATION,
                    confidence=0.8,
                    description=f"Hypothesis without validation: {hyp_node.content[:50]}...",
                    affected_nodes=[hyp_node.id],
                    severity="medium",
                    suggestions=["Add validation step", "Test hypothesis"],
                    metadata={"hypothesis_content": hyp_node.content}
                ))
        
        return errors
    
    def _detect_circular_reasoning(self, tree: SymbolicTree) -> List[ErrorPrediction]:
        """Detect circular reasoning patterns."""
        errors = []
        # Simplified implementation
        return errors
    
    def _detect_insufficient_evidence(self, tree: SymbolicTree) -> List[ErrorPrediction]:
        """Detect conclusions with insufficient evidence."""
        errors = []
        conclusion_nodes = tree.get_nodes_by_type(NodeType.CONCLUSION)
        
        for conc_node in conclusion_nodes:
            path_to_root = conc_node.get_path_to_root()
            reasoning_count = sum(1 for node in path_to_root 
                                if node.type == NodeType.REASONING)
            
            if reasoning_count < 2:
                errors.append(ErrorPrediction(
                    error_type=ErrorType.INSUFFICIENT_EVIDENCE,
                    confidence=0.75,
                    description=f"Conclusion with minimal reasoning: {conc_node.content[:50]}...",
                    affected_nodes=[conc_node.id],
                    severity="medium",
                    suggestions=["Add more reasoning steps", "Provide supporting evidence"],
                    metadata={"reasoning_count": reasoning_count}
                ))
        
        return errors
    
    def _detect_contradictory_conclusions(self, tree: SymbolicTree) -> List[ErrorPrediction]:
        """Detect contradictory conclusions."""
        errors = []
        # Would need semantic analysis for proper implementation
        return errors
    
    def _detect_incomplete_reasoning(self, tree: SymbolicTree) -> List[ErrorPrediction]:
        """Detect incomplete reasoning chains."""
        errors = []
        if tree.get_depth() < 3:
            errors.append(ErrorPrediction(
                error_type=ErrorType.INCOMPLETE_REASONING,
                confidence=0.6,
                description="Reasoning chain appears too shallow",
                affected_nodes=[tree.root.id],
                severity="low",
                suggestions=["Develop more detailed reasoning", "Add intermediate steps"],
                metadata={"tree_depth": tree.get_depth()}
            ))
        
        return errors
    
    def _detect_overconfident_conclusions(self, tree: SymbolicTree) -> List[ErrorPrediction]:
        """Detect overconfident conclusions."""
        errors = []
        conclusion_nodes = tree.get_nodes_by_type(NodeType.CONCLUSION)
        
        for node in conclusion_nodes:
            if node.confidence > 0.9:
                path_reasoning = sum(1 for n in node.get_path_to_root() 
                                  if n.type == NodeType.REASONING)
                if path_reasoning < 3:
                    errors.append(ErrorPrediction(
                        error_type=ErrorType.OVERCONFIDENT_CONCLUSION,
                        confidence=0.7,
                        description=f"High confidence with limited reasoning: {node.content[:50]}...",
                        affected_nodes=[node.id],
                        severity="medium",
                        suggestions=["Review confidence level", "Add more supporting reasoning"],
                        metadata={"confidence": node.confidence, "reasoning_count": path_reasoning}
                    ))
        
        return errors
    
    # Helper methods for calculations
    def _calculate_path_confidence(self, nodes: List[TreeNode]) -> float:
        """Calculate confidence for a reasoning path."""
        confidences = [node.confidence for node in nodes if node.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _calculate_coherence_score(self, nodes: List[TreeNode]) -> float:
        """Calculate coherence score for a reasoning path."""
        # Simplified implementation - would need semantic analysis
        return 0.7  # Placeholder
    
    def _calculate_completeness_score(self, nodes: List[TreeNode]) -> float:
        """Calculate completeness score for a reasoning path."""
        required_types = {NodeType.REASONING, NodeType.DECISION}
        present_types = {node.type for node in nodes}
        return len(required_types & present_types) / len(required_types)
    
    def _calculate_tree_balance(self, tree: SymbolicTree) -> float:
        """Calculate tree balance score."""
        # Simplified implementation
        return 0.5  # Placeholder
    
    def _calculate_structure_score(self, tree: SymbolicTree) -> float:
        """Calculate logical structure score."""
        # Check for good reasoning flow
        reasoning_nodes = len(tree.get_nodes_by_type(NodeType.REASONING))
        decision_nodes = len(tree.get_nodes_by_type(NodeType.DECISION))
        conclusion_nodes = len(tree.get_nodes_by_type(NodeType.CONCLUSION))
        
        total_nodes = tree.get_node_count()
        if total_nodes == 0:
            return 0.0
        
        # Good structure has balanced reasoning, decisions, and conclusions
        reasoning_ratio = reasoning_nodes / total_nodes
        decision_ratio = decision_nodes / total_nodes
        conclusion_ratio = conclusion_nodes / total_nodes
        
        # Ideal ratios (roughly)
        ideal_reasoning = 0.4
        ideal_decision = 0.3
        ideal_conclusion = 0.2
        
        score = 1.0 - (
            abs(reasoning_ratio - ideal_reasoning) +
            abs(decision_ratio - ideal_decision) +
            abs(conclusion_ratio - ideal_conclusion)
        ) / 3.0
        
        return max(0.0, score)
    
    def _calculate_evidence_score(self, tree: SymbolicTree) -> float:
        """Calculate evidence sufficiency score."""
        conclusion_nodes = tree.get_nodes_by_type(NodeType.CONCLUSION)
        if not conclusion_nodes:
            return 0.5  # No conclusions to evaluate
        
        evidence_scores = []
        for conc_node in conclusion_nodes:
            path = conc_node.get_path_to_root()
            evidence_count = sum(1 for node in path 
                               if node.type in [NodeType.REASONING, NodeType.VALIDATION])
            
            # Score based on evidence count (diminishing returns)
            score = min(1.0, evidence_count / 5.0)
            evidence_scores.append(score)
        
        return sum(evidence_scores) / len(evidence_scores)
    
    def _severity_weight(self, severity: str) -> int:
        """Get numeric weight for severity."""
        weights = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return weights.get(severity, 1)
    
    def _path_to_dict(self, path: ReasoningPath) -> Dict[str, Any]:
        """Convert reasoning path to dictionary."""
        return {
            "nodes": [node.id for node in path.nodes],
            "confidence": path.confidence,
            "coherence_score": path.coherence_score,
            "completeness_score": path.completeness_score,
            "length": path.length,
            "metadata": path.metadata
        }
    
    def _error_to_dict(self, error: ErrorPrediction) -> Dict[str, Any]:
        """Convert error prediction to dictionary."""
        return {
            "error_type": error.error_type.value,
            "confidence": error.confidence,
            "description": error.description,
            "affected_nodes": error.affected_nodes,
            "severity": error.severity,
            "suggestions": error.suggestions,
            "metadata": error.metadata
        }


class ReasoningLogger:
    """Logger for reasoning analysis events and results."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize reasoning logger.
        
        Args:
            log_file: Optional file path for logging
        """
        self.logger = logging.getLogger("resk_flag.reasoning")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file and not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_analysis_start(self, tree_id: str, node_count: int) -> None:
        """Log start of analysis."""
        self.logger.info(f"Starting analysis of tree {tree_id} with {node_count} nodes")
    
    def log_analysis_complete(self, tree_id: str, error_count: int, 
                            quality_score: float) -> None:
        """Log completion of analysis."""
        self.logger.info(
            f"Analysis complete for tree {tree_id}: "
            f"{error_count} errors, quality score {quality_score:.2f}"
        )
    
    def log_error_detected(self, error: ErrorPrediction) -> None:
        """Log detected error."""
        self.logger.warning(
            f"Error detected: {error.error_type.value} "
            f"(confidence: {error.confidence:.2f}, severity: {error.severity})"
        )
    
    def log_recommendation(self, recommendation: str) -> None:
        """Log recommendation."""
        self.logger.info(f"Recommendation: {recommendation}")
    
    def export_session_log(self, file_path: str) -> None:
        """Export session log to file."""
        # This would extract and save log entries
        pass