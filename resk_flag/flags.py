"""
Flag system for parsing and managing flags from AI model responses.
Flags are special markers in AI responses that indicate reasoning steps,
decisions, or states that can be used to build symbolic trees.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import re
import json


class FlagType(Enum):
    """Types of flags that can be extracted from AI responses."""
    REASONING = "reasoning"
    DECISION = "decision" 
    ERROR = "error"
    CONCLUSION = "conclusion"
    HYPOTHESIS = "hypothesis"
    VALIDATION = "validation"
    STEP = "step"
    CONDITION = "condition"


@dataclass
class Flag:
    """Represents a flag extracted from an AI response."""
    id: str
    type: FlagType
    content: str
    metadata: Dict[str, Any]
    confidence: float = 1.0
    position: Optional[int] = None
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flag to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "position": self.position,
            "parent_id": self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Flag':
        """Create flag from dictionary representation."""
        return cls(
            id=data["id"],
            type=FlagType(data["type"]),
            content=data["content"],
            metadata=data["metadata"],
            confidence=data.get("confidence", 1.0),
            position=data.get("position"),
            parent_id=data.get("parent_id")
        )


class FlagParser:
    """Parser for extracting flags from AI model responses."""
    
    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        """
        Initialize flag parser with optional custom patterns.
        
        Args:
            custom_patterns: Dictionary mapping flag types to regex patterns
        """
        self.patterns = self._default_patterns()
        if custom_patterns:
            self.patterns.update(custom_patterns)
    
    def _default_patterns(self) -> Dict[str, str]:
        """Default regex patterns for flag detection."""
        return {
            "reasoning": r"<reasoning>(.*?)</reasoning>",
            "decision": r"<decision>(.*?)</decision>", 
            "error": r"<error>(.*?)</error>",
            "conclusion": r"<conclusion>(.*?)</conclusion>",
            "hypothesis": r"<hypothesis>(.*?)</hypothesis>",
            "validation": r"<validation>(.*?)</validation>",
            "step": r"<step>(.*?)</step>",
            "condition": r"<condition>(.*?)</condition>",
            # JSON-style flags
            "json_flag": r"\[FLAG:(\w+):([^\]]+)\]",
            # Markdown-style flags  
            "md_flag": r"```\s*(\w+)\s*\n(.*?)\n\s*```"
        }
    
    def parse(self, text: str) -> List[Flag]:
        """
        Parse flags from text response.
        
        Args:
            text: The text to parse for flags
            
        Returns:
            List of extracted flags
        """
        flags = []
        flag_id_counter = 0
        
        # Extract XML-style flags
        for flag_type_str, pattern in self.patterns.items():
            if flag_type_str in ["json_flag", "md_flag"]:
                continue
                
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    flag_type = FlagType(flag_type_str)
                    flag = Flag(
                        id=f"flag_{flag_id_counter}",
                        type=flag_type,
                        content=match.group(1).strip(),
                        metadata={"pattern": pattern, "match_groups": match.groups()},
                        position=match.start()
                    )
                    flags.append(flag)
                    flag_id_counter += 1
                except ValueError:
                    # Skip invalid flag types
                    continue
        
        # JSON-style flags
        json_matches = re.finditer(self.patterns["json_flag"], text)
        for match in json_matches:
            try:
                flag_type_str = match.group(1).lower()
                if flag_type_str in [ft.value for ft in FlagType]:
                    flag_type = FlagType(flag_type_str)
                    flag = Flag(
                        id=f"flag_{flag_id_counter}",
                        type=flag_type,
                        content=match.group(2).strip(),
                        metadata={"format": "json_style"},
                        position=match.start()
                    )
                    flags.append(flag)
                    flag_id_counter += 1
            except ValueError:
                continue
        
        # Extract markdown-style flags
        md_matches = re.finditer(self.patterns["md_flag"], text, re.DOTALL)
        for match in md_matches:
            try:
                flag_type_str = match.group(1).lower()
                if flag_type_str in [ft.value for ft in FlagType]:
                    flag_type = FlagType(flag_type_str)
                    flag = Flag(
                        id=f"flag_{flag_id_counter}",
                        type=flag_type,
                        content=match.group(2).strip(),
                        metadata={"format": "markdown_style"},
                        position=match.start()
                    )
                    flags.append(flag)
                    flag_id_counter += 1
            except ValueError:
                continue
        
        # Sort flags by position
        flags.sort(key=lambda f: f.position or 0)
        
        return flags
    
    def add_pattern(self, flag_type: str, pattern: str) -> None:
        """Add custom pattern for flag detection."""
        self.patterns[flag_type] = pattern
    
    def remove_pattern(self, flag_type: str) -> None:
        """Remove pattern for flag detection."""
        if flag_type in self.patterns:
            del self.patterns[flag_type]


class FlagValidator:
    """Validator for ensuring flag consistency and quality."""
    
    def __init__(self, min_content_length: int = 3, max_content_length: int = 1000):
        """
        Initialize validator with content length constraints.
        
        Args:
            min_content_length: Minimum allowed content length
            max_content_length: Maximum allowed content length
        """
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
    
    def validate(self, flag: Flag) -> bool:
        """
        Validate a flag for consistency and quality.
        
        Args:
            flag: The flag to validate
            
        Returns:
            True if flag is valid, False otherwise
        """
        # Check content length
        if len(flag.content) < self.min_content_length:
            return False
        if len(flag.content) > self.max_content_length:
            return False
        
        # Check for empty or whitespace-only content
        if not flag.content.strip():
            return False
        
        # Check confidence bounds
        if not 0.0 <= flag.confidence <= 1.0:
            return False
        
        return True
    
    def filter_valid_flags(self, flags: List[Flag]) -> List[Flag]:
        """Filter list to only include valid flags."""
        return [flag for flag in flags if self.validate(flag)]


class FlagManager:
    """Manager for handling flag collections and operations."""
    
    def __init__(self):
        """Initialize flag manager."""
        self.parser = FlagParser()
        self.validator = FlagValidator()
        self._flags: List[Flag] = []
    
    def add_flags_from_text(self, text: str) -> List[Flag]:
        """
        Extract and add flags from text.
        
        Args:
            text: Text to parse for flags
            
        Returns:
            List of newly added flags
        """
        new_flags = self.parser.parse(text)
        valid_flags = self.validator.filter_valid_flags(new_flags)
        self._flags.extend(valid_flags)
        return valid_flags
    
    def add_flag(self, flag: Flag) -> bool:
        """
        Add a single flag if valid.
        
        Args:
            flag: Flag to add
            
        Returns:
            True if flag was added, False if invalid
        """
        if self.validator.validate(flag):
            self._flags.append(flag)
            return True
        return False
    
    def get_flags(self, flag_type: Optional[FlagType] = None) -> List[Flag]:
        """
        Get flags, optionally filtered by type.
        
        Args:
            flag_type: Optional flag type to filter by
            
        Returns:
            List of flags matching criteria
        """
        if flag_type is None:
            return self._flags.copy()
        return [flag for flag in self._flags if flag.type == flag_type]
    
    def get_flag_by_id(self, flag_id: str) -> Optional[Flag]:
        """Get flag by ID."""
        for flag in self._flags:
            if flag.id == flag_id:
                return flag
        return None
    
    def clear(self) -> None:
        """Clear all flags."""
        self._flags.clear()
    
    def export_json(self) -> str:
        """Export flags to JSON string."""
        flag_dicts = [flag.to_dict() for flag in self._flags]
        return json.dumps(flag_dicts, indent=2)
    
    def import_json(self, json_str: str) -> None:
        """Import flags from JSON string."""
        flag_dicts = json.loads(json_str)
        for flag_dict in flag_dicts:
            flag = Flag.from_dict(flag_dict)
            if self.validator.validate(flag):
                self._flags.append(flag)