"""
Tests for the flags module.
"""

import pytest
from resk_flag.flags import Flag, FlagType, FlagParser, FlagValidator, FlagManager


class TestFlag:
    """Test Flag class."""
    
    def test_flag_creation(self):
        """Test basic flag creation."""
        flag = Flag(
            id="test_1",
            type=FlagType.REASONING,
            content="This is a test reasoning step",
            metadata={"source": "test"}
        )
        
        assert flag.id == "test_1"
        assert flag.type == FlagType.REASONING
        assert flag.content == "This is a test reasoning step"
        assert flag.metadata["source"] == "test"
        assert flag.confidence == 1.0
    
    def test_flag_to_dict(self):
        """Test flag dictionary conversion."""
        flag = Flag(
            id="test_1",
            type=FlagType.DECISION,
            content="Test decision",
            metadata={"test": True}
        )
        
        flag_dict = flag.to_dict()
        
        assert flag_dict["id"] == "test_1"
        assert flag_dict["type"] == "decision"
        assert flag_dict["content"] == "Test decision"
        assert flag_dict["metadata"]["test"] is True
    
    def test_flag_from_dict(self):
        """Test flag creation from dictionary."""
        flag_dict = {
            "id": "test_1",
            "type": "reasoning",
            "content": "Test reasoning",
            "metadata": {"test": True},
            "confidence": 0.8
        }
        
        flag = Flag.from_dict(flag_dict)
        
        assert flag.id == "test_1"
        assert flag.type == FlagType.REASONING
        assert flag.content == "Test reasoning"
        assert flag.confidence == 0.8


class TestFlagParser:
    """Test FlagParser class."""
    
    def test_basic_parsing(self):
        """Test basic flag parsing."""
        parser = FlagParser()
        text = """
        <reasoning>
        First, let me analyze the problem.
        </reasoning>
        
        <decision>
        I will choose option A.
        </decision>
        """
        
        flags = parser.parse(text)
        
        assert len(flags) == 2
        assert flags[0].type == FlagType.REASONING
        assert "analyze the problem" in flags[0].content
        assert flags[1].type == FlagType.DECISION
        assert "option A" in flags[1].content
    
    def test_json_style_flags(self):
        """Test JSON-style flag parsing."""
        parser = FlagParser()
        text = "This is [FLAG:reasoning:analyzing the data] and [FLAG:conclusion:final result]"
        
        flags = parser.parse(text)
        
        assert len(flags) == 2
        assert flags[0].type == FlagType.REASONING
        assert flags[0].content == "analyzing the data"
        assert flags[1].type == FlagType.CONCLUSION
        assert flags[1].content == "final result"
    
    def test_markdown_style_flags(self):
        """Test markdown-style flag parsing."""
        parser = FlagParser()
        text = """
        ```reasoning
        This is my reasoning process
        ```
        
        ```conclusion
        This is my conclusion
        ```
        """
        
        flags = parser.parse(text)
        
        assert len(flags) == 2
        assert flags[0].type == FlagType.REASONING
        assert "reasoning process" in flags[0].content
        assert flags[1].type == FlagType.CONCLUSION
        assert "conclusion" in flags[1].content
    
    def test_custom_patterns(self):
        """Test custom pattern addition."""
        parser = FlagParser()
        parser.add_pattern("custom", r"<custom>(.*?)</custom>")
        
        text = "<custom>Custom content</custom>"
        flags = parser.parse(text)
        
        # Should not create a flag since 'custom' is not a valid FlagType
        assert len(flags) == 0
    
    def test_position_ordering(self):
        """Test that flags are ordered by position."""
        parser = FlagParser()
        text = "<conclusion>End result</conclusion> some text <reasoning>Start analysis</reasoning>"
        
        flags = parser.parse(text)
        
        assert len(flags) == 2
        # First flag should be conclusion (appears first in text)
        assert flags[0].type == FlagType.CONCLUSION
        assert flags[1].type == FlagType.REASONING


class TestFlagValidator:
    """Test FlagValidator class."""
    
    def test_valid_flag(self):
        """Test validation of valid flag."""
        validator = FlagValidator()
        flag = Flag(
            id="test_1",
            type=FlagType.REASONING,
            content="This is valid content",
            metadata={}
        )
        
        assert validator.validate(flag) is True
    
    def test_content_too_short(self):
        """Test validation fails for too short content."""
        validator = FlagValidator(min_content_length=10)
        flag = Flag(
            id="test_1",
            type=FlagType.REASONING,
            content="Hi",
            metadata={}
        )
        
        assert validator.validate(flag) is False
    
    def test_content_too_long(self):
        """Test validation fails for too long content."""
        validator = FlagValidator(max_content_length=10)
        flag = Flag(
            id="test_1",
            type=FlagType.REASONING,
            content="This content is way too long for the validator",
            metadata={}
        )
        
        assert validator.validate(flag) is False
    
    def test_empty_content(self):
        """Test validation fails for empty content."""
        validator = FlagValidator()
        flag = Flag(
            id="test_1",
            type=FlagType.REASONING,
            content="   ",  # whitespace only
            metadata={}
        )
        
        assert validator.validate(flag) is False
    
    def test_invalid_confidence(self):
        """Test validation fails for invalid confidence."""
        validator = FlagValidator()
        flag = Flag(
            id="test_1",
            type=FlagType.REASONING,
            content="Valid content",
            metadata={},
            confidence=1.5  # Invalid confidence > 1.0
        )
        
        assert validator.validate(flag) is False
    
    def test_filter_valid_flags(self):
        """Test filtering of valid flags."""
        validator = FlagValidator()
        flags = [
            Flag("1", FlagType.REASONING, "Valid content", {}),
            Flag("2", FlagType.REASONING, "V", {}),  # Too short
            Flag("3", FlagType.REASONING, "Another valid content", {}),
        ]
        
        valid_flags = validator.filter_valid_flags(flags)
        
        assert len(valid_flags) == 2
        assert valid_flags[0].id == "1"
        assert valid_flags[1].id == "3"


class TestFlagManager:
    """Test FlagManager class."""
    
    def test_add_flags_from_text(self):
        """Test adding flags from text."""
        manager = FlagManager()
        text = "<reasoning>Test reasoning</reasoning><decision>Test decision</decision>"
        
        flags = manager.add_flags_from_text(text)
        
        assert len(flags) == 2
        assert len(manager.get_flags()) == 2
    
    def test_add_valid_flag(self):
        """Test adding valid flag."""
        manager = FlagManager()
        flag = Flag("1", FlagType.REASONING, "Valid content", {})
        
        result = manager.add_flag(flag)
        
        assert result is True
        assert len(manager.get_flags()) == 1
    
    def test_add_invalid_flag(self):
        """Test adding invalid flag."""
        manager = FlagManager()
        flag = Flag("1", FlagType.REASONING, "V", {})  # Too short
        
        result = manager.add_flag(flag)
        
        assert result is False
        assert len(manager.get_flags()) == 0
    
    def test_get_flags_by_type(self):
        """Test getting flags by type."""
        manager = FlagManager()
        manager.add_flag(Flag("1", FlagType.REASONING, "Reasoning content", {}))
        manager.add_flag(Flag("2", FlagType.DECISION, "Decision content", {}))
        manager.add_flag(Flag("3", FlagType.REASONING, "More reasoning", {}))
        
        reasoning_flags = manager.get_flags(FlagType.REASONING)
        decision_flags = manager.get_flags(FlagType.DECISION)
        
        assert len(reasoning_flags) == 2
        assert len(decision_flags) == 1
    
    def test_get_flag_by_id(self):
        """Test getting flag by ID."""
        manager = FlagManager()
        flag = Flag("test_id", FlagType.REASONING, "Test content", {})
        manager.add_flag(flag)
        
        found_flag = manager.get_flag_by_id("test_id")
        not_found = manager.get_flag_by_id("nonexistent")
        
        assert found_flag is not None
        assert found_flag.id == "test_id"
        assert not_found is None
    
    def test_clear(self):
        """Test clearing all flags."""
        manager = FlagManager()
        manager.add_flag(Flag("1", FlagType.REASONING, "Content", {}))
        
        assert len(manager.get_flags()) == 1
        
        manager.clear()
        
        assert len(manager.get_flags()) == 0
    
    def test_export_import_json(self):
        """Test JSON export and import."""
        manager = FlagManager()
        flag1 = Flag("1", FlagType.REASONING, "First flag", {"test": True})
        flag2 = Flag("2", FlagType.DECISION, "Second flag", {"test": False})
        
        manager.add_flag(flag1)
        manager.add_flag(flag2)
        
        # Export
        json_str = manager.export_json()
        
        # Clear and import
        manager.clear()
        assert len(manager.get_flags()) == 0
        
        manager.import_json(json_str)
        
        # Verify import
        imported_flags = manager.get_flags()
        assert len(imported_flags) == 2
        
        flag_ids = {flag.id for flag in imported_flags}
        assert "1" in flag_ids
        assert "2" in flag_ids