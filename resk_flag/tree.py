"""
Symbolic tree implementation for representing reasoning structures.
Trees are built from flags and allow analysis of reasoning paths,
error prediction, and logical flow analysis.
"""

from typing import Dict, List, Optional, Any, Set, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from .flags import Flag, FlagType


class NodeType(Enum):
    """Types of nodes in the symbolic tree."""
    ROOT = "root"
    REASONING = "reasoning"
    DECISION = "decision"
    CONDITION = "condition"
    CONCLUSION = "conclusion"
    ERROR = "error"
    HYPOTHESIS = "hypothesis"
    VALIDATION = "validation"
    STEP = "step"


@dataclass
class TreeNode:
    """Represents a node in the symbolic reasoning tree."""
    id: str
    type: NodeType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    children: List['TreeNode'] = field(default_factory=list)
    parent: Optional['TreeNode'] = None
    flag_id: Optional[str] = None
    depth: int = 0
    
    def add_child(self, child: 'TreeNode') -> None:
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def remove_child(self, child: 'TreeNode') -> bool:
        """Remove a child node."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            return True
        return False
    
    def get_path_to_root(self) -> List['TreeNode']:
        """Get path from this node to the root."""
        path = [self]
        current = self.parent
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_descendants(self) -> List['TreeNode']:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None and self.id == "root"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "flag_id": self.flag_id,
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional['TreeNode'] = None) -> 'TreeNode':
        """Create node from dictionary representation."""
        node = cls(
            id=data["id"],
            type=NodeType(data["type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0),
            flag_id=data.get("flag_id"),
            depth=data.get("depth", 0),
            parent=parent
        )
        
        # Recursively create children
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data, node)
            node.children.append(child)
        
        return node


class SymbolicTree:
    """A symbolic tree for representing reasoning structures."""
    
    def __init__(self, root_content: str = "Root"):
        """
        Initialize symbolic tree with root node.
        
        Args:
            root_content: Content for the root node
        """
        self.root = TreeNode(
            id="root",
            type=NodeType.ROOT,
            content=root_content,
            depth=0
        )
        self._node_map: Dict[str, TreeNode] = {"root": self.root}
        self._flag_to_node: Dict[str, TreeNode] = {}
    
    def add_node(self, parent_id: str, node_id: str, node_type: NodeType, 
                 content: str, flag_id: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None,
                 confidence: float = 1.0) -> Optional[TreeNode]:
        """
        Add a new node to the tree.
        
        Args:
            parent_id: ID of the parent node
            node_id: ID for the new node
            node_type: Type of the new node
            content: Content of the new node
            flag_id: Optional associated flag ID
            metadata: Optional metadata dictionary
            confidence: Confidence score for the node
            
        Returns:
            The created node, or None if parent not found
        """
        parent = self.get_node(parent_id)
        if not parent:
            return None
        
        if node_id in self._node_map:
            return None  # Node ID already exists
        
        node = TreeNode(
            id=node_id,
            type=node_type,
            content=content,
            metadata=metadata or {},
            confidence=confidence,
            flag_id=flag_id
        )
        
        parent.add_child(node)
        self._node_map[node_id] = node
        
        if flag_id:
            self._flag_to_node[flag_id] = node
        
        return node
    
    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Get node by ID."""
        return self._node_map.get(node_id)
    
    def get_node_by_flag(self, flag_id: str) -> Optional[TreeNode]:
        """Get node by associated flag ID."""
        return self._flag_to_node.get(flag_id)
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its descendants.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            True if node was removed, False if not found or is root
        """
        if node_id == "root":
            return False  # Cannot remove root
        
        node = self.get_node(node_id)
        if not node or not node.parent:
            return False
        
        # Remove from parent
        node.parent.remove_child(node)
        
        # Remove from maps (including descendants)
        nodes_to_remove = [node] + node.get_descendants()
        for n in nodes_to_remove:
            if n.id in self._node_map:
                del self._node_map[n.id]
            if n.flag_id and n.flag_id in self._flag_to_node:
                del self._flag_to_node[n.flag_id]
        
        return True
    
    def get_all_nodes(self) -> List[TreeNode]:
        """Get all nodes in the tree."""
        return list(self._node_map.values())
    
    def get_leaves(self) -> List[TreeNode]:
        """Get all leaf nodes."""
        return [node for node in self._node_map.values() if node.is_leaf()]
    
    def get_paths_to_leaves(self) -> List[List[TreeNode]]:
        """Get all paths from root to leaf nodes."""
        paths = []
        for leaf in self.get_leaves():
            paths.append(leaf.get_path_to_root())
        return paths
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[TreeNode]:
        """Get all nodes of a specific type."""
        return [node for node in self._node_map.values() if node.type == node_type]
    
    def get_depth(self) -> int:
        """Get maximum depth of the tree."""
        if not self._node_map:
            return 0
        return max(node.depth for node in self._node_map.values())
    
    def get_node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._node_map)
    
    def traverse_dfs(self, start_node: Optional[TreeNode] = None) -> Iterator[TreeNode]:
        """
        Depth-first traversal of the tree.
        
        Args:
            start_node: Node to start traversal from (default: root)
            
        Yields:
            Nodes in DFS order
        """
        if start_node is None:
            start_node = self.root
        
        yield start_node
        for child in start_node.children:
            yield from self.traverse_dfs(child)
    
    def traverse_bfs(self, start_node: Optional[TreeNode] = None) -> Iterator[TreeNode]:
        """
        Breadth-first traversal of the tree.
        
        Args:
            start_node: Node to start traversal from (default: root)
            
        Yields:
            Nodes in BFS order
        """
        if start_node is None:
            start_node = self.root
        
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)
    
    def find_path(self, start_id: str, end_id: str) -> Optional[List[TreeNode]]:
        """
        Find path between two nodes.
        
        Args:
            start_id: ID of start node
            end_id: ID of end node
            
        Returns:
            Path as list of nodes, or None if no path exists
        """
        start_node = self.get_node(start_id)
        end_node = self.get_node(end_id)
        
        if not start_node or not end_node:
            return None
        
        # Get paths to root for both nodes
        start_path = start_node.get_path_to_root()
        end_path = end_node.get_path_to_root()
        
        # Find common ancestor
        common_ancestor = None
        for i in range(min(len(start_path), len(end_path))):
            if start_path[i] == end_path[i]:
                common_ancestor = start_path[i]
            else:
                break
        
        if not common_ancestor:
            return None
        
        # Build path: start -> common ancestor -> end
        path_to_ancestor = []
        current = start_node
        while current != common_ancestor:
            path_to_ancestor.append(current)
            current = current.parent
        
        path_from_ancestor = []
        current = end_node
        while current != common_ancestor:
            path_from_ancestor.append(current)
            current = current.parent
        
        # Combine paths
        full_path = path_to_ancestor + [common_ancestor] + list(reversed(path_from_ancestor))
        return full_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Export tree to dictionary representation."""
        return {
            "root": self.root.to_dict(),
            "metadata": {
                "node_count": self.get_node_count(),
                "max_depth": self.get_depth(),
                "leaf_count": len(self.get_leaves())
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolicTree':
        """Create tree from dictionary representation."""
        tree = cls.__new__(cls)  # Create instance without calling __init__
        tree.root = TreeNode.from_dict(data["root"])
        tree._node_map = {}
        tree._flag_to_node = {}
        
        # Rebuild maps
        for node in tree.traverse_dfs():
            tree._node_map[node.id] = node
            if node.flag_id:
                tree._flag_to_node[node.flag_id] = node
        
        return tree
    
    def export_json(self) -> str:
        """Export tree to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def import_json(cls, json_str: str) -> 'SymbolicTree':
        """Import tree from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class TreeBuilder:
    """Builder for constructing symbolic trees from flags."""
    
    def __init__(self):
        """Initialize tree builder."""
        self._node_counter = 0
        
    def build_from_flags(self, flags: List[Flag], 
                        root_content: str = "Reasoning Process") -> SymbolicTree:
        """
        Build a symbolic tree from a list of flags.
        
        Args:
            flags: List of flags to build tree from
            root_content: Content for root node
            
        Returns:
            Constructed symbolic tree
        """
        tree = SymbolicTree(root_content)
        self._node_counter = 0
        
        # Sort flags by position to maintain order
        sorted_flags = sorted(flags, key=lambda f: f.position or 0)
        
        # Build tree structure
        current_parent = "root"
        depth_stack = ["root"]  # Track parent nodes at each depth
        
        for flag in sorted_flags:
            node_id = self._generate_node_id()
            node_type = self._flag_type_to_node_type(flag.type)
            
            # Determine parent based on flag type and current context
            parent_id = self._determine_parent(flag, depth_stack, tree)
            
            node = tree.add_node(
                parent_id=parent_id,
                node_id=node_id,
                node_type=node_type,
                content=flag.content,
                flag_id=flag.id,
                metadata=flag.metadata,
                confidence=flag.confidence
            )
            
            if node:
                # Update depth stack based on node type
                self._update_depth_stack(node, depth_stack, flag.type)
        
        return tree
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"
    
    def _flag_type_to_node_type(self, flag_type: FlagType) -> NodeType:
        """Convert flag type to node type."""
        mapping = {
            FlagType.REASONING: NodeType.REASONING,
            FlagType.DECISION: NodeType.DECISION,
            FlagType.ERROR: NodeType.ERROR,
            FlagType.CONCLUSION: NodeType.CONCLUSION,
            FlagType.HYPOTHESIS: NodeType.HYPOTHESIS,
            FlagType.VALIDATION: NodeType.VALIDATION,
            FlagType.STEP: NodeType.STEP,
            FlagType.CONDITION: NodeType.CONDITION
        }
        return mapping.get(flag_type, NodeType.REASONING)
    
    def _determine_parent(self, flag: Flag, depth_stack: List[str], 
                         tree: SymbolicTree) -> str:
        """Determine parent node for a new flag-based node."""
        # If flag has explicit parent_id, use it
        if flag.parent_id and tree.get_node(flag.parent_id):
            return flag.parent_id
        
        # Use heuristics based on flag type
        if flag.type in [FlagType.CONCLUSION, FlagType.VALIDATION]:
            # Conclusions and validations typically attach to the most recent reasoning
            for node_id in reversed(depth_stack):
                node = tree.get_node(node_id)
                if node and node.type in [NodeType.REASONING, NodeType.DECISION]:
                    return node_id
        
        # Default to the most recent parent in stack
        return depth_stack[-1] if depth_stack else "root"
    
    def _update_depth_stack(self, node: TreeNode, depth_stack: List[str], 
                           flag_type: FlagType) -> None:
        """Update the depth stack based on the new node."""
        # Add current node to stack
        depth_stack.append(node.id)
        
        # Trim stack based on node type to maintain logical structure
        if flag_type in [FlagType.CONCLUSION, FlagType.ERROR]:
            # Conclusions and errors typically end a reasoning branch
            # Keep only up to the parent
            if len(depth_stack) > 2:
                depth_stack = depth_stack[:-1]
        
        elif flag_type == FlagType.CONDITION:
            # Conditions might start new branches, maintain full stack
            pass
        
        # Limit stack depth to prevent overly deep nesting
        if len(depth_stack) > 5:
            depth_stack = depth_stack[-3:]  # Keep last 3 levels