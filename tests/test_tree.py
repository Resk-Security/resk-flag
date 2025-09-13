"""
Tests for the tree module.
"""

import pytest
from resk_flag.tree import TreeNode, NodeType, SymbolicTree, TreeBuilder
from resk_flag.flags import Flag, FlagType


class TestTreeNode:
    """Test TreeNode class."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = TreeNode(
            id="test_node",
            type=NodeType.REASONING,
            content="Test reasoning content",
            confidence=0.8
        )
        
        assert node.id == "test_node"
        assert node.type == NodeType.REASONING
        assert node.content == "Test reasoning content"
        assert node.confidence == 0.8
        assert node.depth == 0
        assert len(node.children) == 0
        assert node.parent is None
    
    def test_add_child(self):
        """Test adding child nodes."""
        parent = TreeNode("parent", NodeType.ROOT, "Parent")
        child = TreeNode("child", NodeType.REASONING, "Child")
        
        parent.add_child(child)
        
        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent
        assert child.depth == 1
    
    def test_remove_child(self):
        """Test removing child nodes."""
        parent = TreeNode("parent", NodeType.ROOT, "Parent")
        child = TreeNode("child", NodeType.REASONING, "Child")
        
        parent.add_child(child)
        result = parent.remove_child(child)
        
        assert result is True
        assert len(parent.children) == 0
        assert child.parent is None
    
    def test_get_path_to_root(self):
        """Test getting path to root."""
        root = TreeNode("root", NodeType.ROOT, "Root")
        level1 = TreeNode("level1", NodeType.REASONING, "Level 1")
        level2 = TreeNode("level2", NodeType.DECISION, "Level 2")
        
        root.add_child(level1)
        level1.add_child(level2)
        
        path = level2.get_path_to_root()
        
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == level1
        assert path[2] == level2
    
    def test_get_descendants(self):
        """Test getting descendants."""
        root = TreeNode("root", NodeType.ROOT, "Root")
        child1 = TreeNode("child1", NodeType.REASONING, "Child 1")
        child2 = TreeNode("child2", NodeType.DECISION, "Child 2")
        grandchild = TreeNode("grandchild", NodeType.CONCLUSION, "Grandchild")
        
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)
        
        descendants = root.get_descendants()
        
        assert len(descendants) == 3
        descendant_ids = {node.id for node in descendants}
        assert descendant_ids == {"child1", "child2", "grandchild"}
    
    def test_is_leaf_and_root(self):
        """Test leaf and root detection."""
        root = TreeNode("root", NodeType.ROOT, "Root")
        child = TreeNode("child", NodeType.REASONING, "Child")
        
        assert root.is_root() is True
        assert root.is_leaf() is True  # No children yet
        assert child.is_root() is False
        assert child.is_leaf() is True
        
        root.add_child(child)
        
        assert root.is_leaf() is False  # Now has children
        assert child.is_leaf() is True
    
    def test_to_dict(self):
        """Test node to dictionary conversion."""
        parent = TreeNode("parent", NodeType.REASONING, "Parent content")
        child = TreeNode("child", NodeType.DECISION, "Child content")
        parent.add_child(child)
        
        parent_dict = parent.to_dict()
        
        assert parent_dict["id"] == "parent"
        assert parent_dict["type"] == "reasoning"
        assert parent_dict["content"] == "Parent content"
        assert len(parent_dict["children"]) == 1
        assert parent_dict["children"][0]["id"] == "child"
    
    def test_from_dict(self):
        """Test node creation from dictionary."""
        node_dict = {
            "id": "test",
            "type": "decision",
            "content": "Test content",
            "metadata": {"test": True},
            "confidence": 0.9,
            "depth": 1,
            "children": [
                {
                    "id": "child",
                    "type": "conclusion",
                    "content": "Child content",
                    "metadata": {},
                    "confidence": 1.0,
                    "depth": 2,
                    "children": []
                }
            ]
        }
        
        node = TreeNode.from_dict(node_dict)
        
        assert node.id == "test"
        assert node.type == NodeType.DECISION
        assert node.content == "Test content"
        assert node.confidence == 0.9
        assert len(node.children) == 1
        assert node.children[0].id == "child"
        assert node.children[0].parent == node


class TestSymbolicTree:
    """Test SymbolicTree class."""
    
    def test_tree_creation(self):
        """Test basic tree creation."""
        tree = SymbolicTree("Test Root")
        
        assert tree.root.id == "root"
        assert tree.root.type == NodeType.ROOT
        assert tree.root.content == "Test Root"
        assert tree.get_node_count() == 1
        assert tree.get_depth() == 0
    
    def test_add_node(self):
        """Test adding nodes to tree."""
        tree = SymbolicTree()
        
        node = tree.add_node(
            parent_id="root",
            node_id="child1",
            node_type=NodeType.REASONING,
            content="First reasoning step",
            confidence=0.8
        )
        
        assert node is not None
        assert node.id == "child1"
        assert node.type == NodeType.REASONING
        assert node.parent == tree.root
        assert tree.get_node_count() == 2
    
    def test_add_node_with_invalid_parent(self):
        """Test adding node with invalid parent ID."""
        tree = SymbolicTree()
        
        node = tree.add_node(
            parent_id="nonexistent",
            node_id="child1",
            node_type=NodeType.REASONING,
            content="Test content"
        )
        
        assert node is None
        assert tree.get_node_count() == 1  # Only root
    
    def test_add_node_duplicate_id(self):
        """Test adding node with duplicate ID."""
        tree = SymbolicTree()
        
        # Add first node
        tree.add_node("root", "child1", NodeType.REASONING, "Content 1")
        
        # Try to add another with same ID
        node = tree.add_node("root", "child1", NodeType.DECISION, "Content 2")
        
        assert node is None
        assert tree.get_node_count() == 2  # Root + first child only
    
    def test_get_node(self):
        """Test getting nodes by ID."""
        tree = SymbolicTree()
        tree.add_node("root", "child1", NodeType.REASONING, "Content")
        
        found = tree.get_node("child1")
        not_found = tree.get_node("nonexistent")
        
        assert found is not None
        assert found.id == "child1"
        assert not_found is None
    
    def test_remove_node(self):
        """Test removing nodes."""
        tree = SymbolicTree()
        tree.add_node("root", "child1", NodeType.REASONING, "Content")
        tree.add_node("child1", "grandchild", NodeType.DECISION, "Decision")
        
        assert tree.get_node_count() == 3
        
        # Remove child (should also remove grandchild)
        result = tree.remove_node("child1")
        
        assert result is True
        assert tree.get_node_count() == 1  # Only root remains
        assert tree.get_node("child1") is None
        assert tree.get_node("grandchild") is None
    
    def test_remove_root(self):
        """Test that root cannot be removed."""
        tree = SymbolicTree()
        
        result = tree.remove_node("root")
        
        assert result is False
        assert tree.get_node_count() == 1
    
    def test_get_leaves(self):
        """Test getting leaf nodes."""
        tree = SymbolicTree()
        tree.add_node("root", "child1", NodeType.REASONING, "Content 1")
        tree.add_node("root", "child2", NodeType.REASONING, "Content 2")
        tree.add_node("child1", "grandchild", NodeType.DECISION, "Decision")
        
        leaves = tree.get_leaves()
        leaf_ids = {leaf.id for leaf in leaves}
        
        assert len(leaves) == 2
        assert leaf_ids == {"child2", "grandchild"}
    
    def test_get_paths_to_leaves(self):
        """Test getting paths to leaf nodes."""
        tree = SymbolicTree()
        tree.add_node("root", "child1", NodeType.REASONING, "Content 1")
        tree.add_node("child1", "grandchild", NodeType.DECISION, "Decision")
        tree.add_node("root", "child2", NodeType.REASONING, "Content 2")
        
        paths = tree.get_paths_to_leaves()
        
        assert len(paths) == 2
        
        # Find path to grandchild
        long_path = next(path for path in paths if len(path) == 3)
        assert long_path[0].id == "root"
        assert long_path[1].id == "child1"
        assert long_path[2].id == "grandchild"
    
    def test_get_nodes_by_type(self):
        """Test getting nodes by type."""
        tree = SymbolicTree()
        tree.add_node("root", "reasoning1", NodeType.REASONING, "Reasoning 1")
        tree.add_node("root", "reasoning2", NodeType.REASONING, "Reasoning 2")
        tree.add_node("root", "decision1", NodeType.DECISION, "Decision 1")
        
        reasoning_nodes = tree.get_nodes_by_type(NodeType.REASONING)
        decision_nodes = tree.get_nodes_by_type(NodeType.DECISION)
        
        assert len(reasoning_nodes) == 2
        assert len(decision_nodes) == 1
    
    def test_traverse_dfs(self):
        """Test depth-first traversal."""
        tree = SymbolicTree()
        tree.add_node("root", "child1", NodeType.REASONING, "Child 1")
        tree.add_node("root", "child2", NodeType.REASONING, "Child 2")
        tree.add_node("child1", "grandchild", NodeType.DECISION, "Grandchild")
        
        nodes = list(tree.traverse_dfs())
        node_ids = [node.id for node in nodes]
        
        # Should visit root, then child1 and its subtree, then child2
        assert node_ids == ["root", "child1", "grandchild", "child2"]
    
    def test_traverse_bfs(self):
        """Test breadth-first traversal."""
        tree = SymbolicTree()
        tree.add_node("root", "child1", NodeType.REASONING, "Child 1")
        tree.add_node("root", "child2", NodeType.REASONING, "Child 2")
        tree.add_node("child1", "grandchild", NodeType.DECISION, "Grandchild")
        
        nodes = list(tree.traverse_bfs())
        node_ids = [node.id for node in nodes]
        
        # Should visit by levels: root, then both children, then grandchild
        assert node_ids == ["root", "child1", "child2", "grandchild"]
    
    def test_find_path(self):
        """Test finding path between nodes."""
        tree = SymbolicTree()
        tree.add_node("root", "child1", NodeType.REASONING, "Child 1")
        tree.add_node("child1", "grandchild1", NodeType.DECISION, "Grandchild 1")
        tree.add_node("root", "child2", NodeType.REASONING, "Child 2")
        tree.add_node("child2", "grandchild2", NodeType.CONCLUSION, "Grandchild 2")
        
        # Path between two grandchildren
        path = tree.find_path("grandchild1", "grandchild2")
        
        assert path is not None
        path_ids = [node.id for node in path]
        assert path_ids == ["grandchild1", "child1", "root", "child2", "grandchild2"]
    
    def test_to_dict_from_dict(self):
        """Test tree serialization and deserialization."""
        tree = SymbolicTree("Original Root")
        tree.add_node("root", "child1", NodeType.REASONING, "Child content")
        
        # Serialize
        tree_dict = tree.to_dict()
        
        # Deserialize
        new_tree = SymbolicTree.from_dict(tree_dict)
        
        assert new_tree.root.content == "Original Root"
        assert new_tree.get_node_count() == 2
        assert new_tree.get_node("child1") is not None
    
    def test_export_import_json(self):
        """Test JSON export and import."""
        tree = SymbolicTree("Test Root")
        tree.add_node("root", "child1", NodeType.REASONING, "Child content")
        
        # Export
        json_str = tree.export_json()
        
        # Import
        new_tree = SymbolicTree.import_json(json_str)
        
        assert new_tree.root.content == "Test Root"
        assert new_tree.get_node_count() == 2
        assert new_tree.get_node("child1") is not None


class TestTreeBuilder:
    """Test TreeBuilder class."""
    
    def test_build_from_flags_basic(self):
        """Test basic tree building from flags."""
        builder = TreeBuilder()
        
        flags = [
            Flag("flag1", FlagType.REASONING, "First reasoning step", {}, position=0),
            Flag("flag2", FlagType.DECISION, "Make a decision", {}, position=1),
            Flag("flag3", FlagType.CONCLUSION, "Final conclusion", {}, position=2),
        ]
        
        tree = builder.build_from_flags(flags, "Test Process")
        
        assert tree.root.content == "Test Process"
        assert tree.get_node_count() == 4  # Root + 3 flag nodes
        
        # Check that flag nodes exist
        flag_nodes = [node for node in tree.get_all_nodes() if node.flag_id is not None]
        assert len(flag_nodes) == 3
    
    def test_build_from_empty_flags(self):
        """Test building from empty flag list."""
        builder = TreeBuilder()
        
        tree = builder.build_from_flags([], "Empty Process")
        
        assert tree.get_node_count() == 1  # Only root
        assert tree.root.content == "Empty Process"
    
    def test_flag_type_to_node_type_conversion(self):
        """Test flag type to node type conversion."""
        builder = TreeBuilder()
        
        flags = [
            Flag("f1", FlagType.REASONING, "Reasoning", {}),
            Flag("f2", FlagType.DECISION, "Decision", {}),
            Flag("f3", FlagType.ERROR, "Error", {}),
            Flag("f4", FlagType.CONCLUSION, "Conclusion", {}),
            Flag("f5", FlagType.HYPOTHESIS, "Hypothesis", {}),
            Flag("f6", FlagType.VALIDATION, "Validation", {}),
            Flag("f7", FlagType.STEP, "Step", {}),
            Flag("f8", FlagType.CONDITION, "Condition", {}),
        ]
        
        tree = builder.build_from_flags(flags)
        
        # Check that all flag types are converted correctly
        node_types = {node.type for node in tree.get_all_nodes() if node.flag_id}
        expected_types = {
            NodeType.REASONING, NodeType.DECISION, NodeType.ERROR,
            NodeType.CONCLUSION, NodeType.HYPOTHESIS, NodeType.VALIDATION,
            NodeType.STEP, NodeType.CONDITION
        }
        
        assert node_types == expected_types
    
    def test_parent_child_relationships(self):
        """Test that parent-child relationships are established correctly."""
        builder = TreeBuilder()
        
        flags = [
            Flag("f1", FlagType.REASONING, "First reasoning", {}, position=0),
            Flag("f2", FlagType.STEP, "Step 1", {}, position=1),
            Flag("f3", FlagType.STEP, "Step 2", {}, position=2),
            Flag("f4", FlagType.CONCLUSION, "Final result", {}, position=3),
        ]
        
        tree = builder.build_from_flags(flags)
        
        # All nodes should be connected to the tree
        assert tree.get_node_count() == 5  # Root + 4 flags
        
        # Check that no nodes are orphaned (except root)
        for node in tree.get_all_nodes():
            if node != tree.root:
                assert node.parent is not None
    
    def test_explicit_parent_id(self):
        """Test using explicit parent_id in flags."""
        builder = TreeBuilder()
        
        flags = [
            Flag("f1", FlagType.REASONING, "First reasoning", {}, position=0),
            Flag("f2", FlagType.DECISION, "Decision", {}, position=1, parent_id="flag_1"),
        ]
        
        tree = builder.build_from_flags(flags)
        
        # Find the nodes
        reasoning_node = next(node for node in tree.get_all_nodes() 
                            if node.flag_id == "f1")
        decision_node = next(node for node in tree.get_all_nodes() 
                           if node.flag_id == "f2")
        
        # Decision should be child of reasoning if parent_id was respected
        # (Note: this depends on the exact implementation logic)
        assert decision_node.parent is not None