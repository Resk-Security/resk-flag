"""
Tests for the AI integration module.
"""

import pytest
from resk_flag.ai_integration import (
    AIRequest, AIResponse, MockAIInterface, AIProcessor, BatchProcessor
)
from resk_flag.flags import FlagType


class TestAIRequest:
    """Test AIRequest class."""
    
    def test_basic_request(self):
        """Test basic request creation."""
        request = AIRequest(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100
        )
        
        assert request.prompt == "Test prompt"
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert request.system_prompt is None
        assert request.metadata == {}
    
    def test_request_with_system_prompt(self):
        """Test request with system prompt."""
        request = AIRequest(
            prompt="User prompt",
            system_prompt="System instructions",
            metadata={"test": True}
        )
        
        assert request.system_prompt == "System instructions"
        assert request.metadata["test"] is True


class TestAIResponse:
    """Test AIResponse class."""
    
    def test_basic_response(self):
        """Test basic response creation."""
        response = AIResponse(
            content="Response content",
            model="test-model",
            tokens_used=50,
            confidence=0.8
        )
        
        assert response.content == "Response content"
        assert response.model == "test-model"
        assert response.tokens_used == 50
        assert response.confidence == 0.8
        assert response.metadata == {}


class TestMockAIInterface:
    """Test MockAIInterface class."""
    
    def test_mock_interface_creation(self):
        """Test mock interface creation."""
        interface = MockAIInterface("test-mock")
        
        assert interface.get_model_name() == "test-mock"
        assert interface.is_available() is True
    
    def test_send_request_reasoning(self):
        """Test sending reasoning request."""
        interface = MockAIInterface()
        request = AIRequest(prompt="What approach should I take?")
        
        response = interface.send_request(request)
        
        assert response.model == "mock-ai"
        assert "<reasoning>" in response.content
        assert "<decision>" in response.content
        assert "<conclusion>" in response.content
        assert response.tokens_used > 0
    
    def test_send_request_problem_solving(self):
        """Test sending problem-solving request."""
        interface = MockAIInterface()
        request = AIRequest(prompt="I need to solve this problem")
        
        response = interface.send_request(request)
        
        assert "<step>" in response.content
        assert "<hypothesis>" in response.content
        assert "<validation>" in response.content
        assert response.metadata["template_used"] == "problem_solving"
    
    def test_send_request_error_analysis(self):
        """Test sending error analysis request."""
        interface = MockAIInterface()
        request = AIRequest(prompt="Debug this error")
        
        response = interface.send_request(request)
        
        assert "<error>" in response.content
        assert "<reasoning>" in response.content
        assert response.metadata["template_used"] == "error_analysis"
    
    def test_availability_control(self):
        """Test availability control."""
        interface = MockAIInterface()
        
        assert interface.is_available() is True
        
        interface.set_available(False)
        assert interface.is_available() is False
        
        with pytest.raises(RuntimeError):
            interface.send_request(AIRequest("test"))
    
    def test_custom_template(self):
        """Test adding custom template."""
        interface = MockAIInterface()
        interface.add_template("custom", "<custom>Custom response</custom>")
        
        # This would require modifying the send_request logic to use custom templates
        # For now, just verify the template was added
        assert "custom" in interface._response_templates


class TestAIProcessor:
    """Test AIProcessor class."""
    
    def test_processor_creation(self):
        """Test processor creation."""
        interface = MockAIInterface()
        processor = AIProcessor(interface)
        
        assert processor.ai_interface == interface
        assert processor.flag_manager is not None
        assert processor.tree_builder is not None
    
    def test_process_query(self):
        """Test processing a query."""
        interface = MockAIInterface()
        processor = AIProcessor(interface)
        
        result = processor.process_query("How should I approach this problem?")
        
        assert "query" in result
        assert "response" in result
        assert "flags" in result
        assert "tree" in result
        assert "metadata" in result
        
        # Check that flags were extracted
        assert len(result["flags"]) > 0
        
        # Check that tree was built
        assert result["tree"].get_node_count() > 1
        
        # Check metadata
        metadata = result["metadata"]
        assert "flag_count" in metadata
        assert "tree_depth" in metadata
        assert "tree_nodes" in metadata
        assert "model_used" in metadata
    
    def test_process_conversation(self):
        """Test processing multiple messages."""
        interface = MockAIInterface()
        processor = AIProcessor(interface)
        
        messages = [
            "What is the problem?",
            "How should I solve it?",
            "What are the risks?"
        ]
        
        results = processor.process_conversation(messages)
        
        assert len(results) == 3
        for result in results:
            assert "query" in result
            assert "flags" in result
            assert "tree" in result
    
    def test_get_flags_functionality(self):
        """Test flag retrieval functionality."""
        interface = MockAIInterface()
        processor = AIProcessor(interface)
        
        # Process a query to generate flags
        processor.process_query("How should I solve this?")
        
        all_flags = processor.get_all_flags()
        reasoning_flags = processor.get_flags_by_type(FlagType.REASONING)
        
        assert len(all_flags) > 0
        assert len(reasoning_flags) >= 0
        
        # All reasoning flags should be in all flags
        for flag in reasoning_flags:
            assert flag in all_flags
    
    def test_clear_history(self):
        """Test clearing history."""
        interface = MockAIInterface()
        processor = AIProcessor(interface)
        
        # Process query and verify flags exist
        processor.process_query("Test query")
        assert len(processor.get_all_flags()) > 0
        
        # Clear and verify empty
        processor.clear_history()
        assert len(processor.get_all_flags()) == 0
    
    def test_export_import_session_data(self):
        """Test session data export/import."""
        interface = MockAIInterface()
        processor = AIProcessor(interface)
        
        # Process query to generate data
        processor.process_query("Test query")
        original_flag_count = len(processor.get_all_flags())
        
        # Export data
        session_data = processor.export_session_data()
        
        # Clear and import
        processor.clear_history()
        assert len(processor.get_all_flags()) == 0
        
        processor.import_session_data(session_data)
        
        # Verify data restored
        assert len(processor.get_all_flags()) == original_flag_count


class TestBatchProcessor:
    """Test BatchProcessor class."""
    
    def test_batch_processor_creation(self):
        """Test batch processor creation."""
        interface = MockAIInterface()
        batch_processor = BatchProcessor(interface)
        
        assert batch_processor.ai_processor is not None
        assert len(batch_processor.results) == 0
    
    def test_add_query(self):
        """Test adding queries to batch."""
        interface = MockAIInterface()
        batch_processor = BatchProcessor(interface)
        
        batch_processor.add_query("First query")
        batch_processor.add_query("Second query")
        
        results = batch_processor.get_results()
        assert len(results) == 2
        assert results[0]["query"] == "First query"
        assert results[1]["query"] == "Second query"
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        interface = MockAIInterface()
        batch_processor = BatchProcessor(interface)
        
        # Test empty summary
        empty_summary = batch_processor.get_summary()
        assert "message" in empty_summary
        
        # Add queries and test summary
        batch_processor.add_query("Query 1")
        batch_processor.add_query("Query 2")
        
        summary = batch_processor.get_summary()
        
        assert summary["total_queries"] == 2
        assert "total_flags" in summary
        assert "total_nodes" in summary
        assert "average_tree_depth" in summary
        assert "flag_type_distribution" in summary
        assert "models_used" in summary
    
    def test_clear(self):
        """Test clearing batch results."""
        interface = MockAIInterface()
        batch_processor = BatchProcessor(interface)
        
        batch_processor.add_query("Test query")
        assert len(batch_processor.get_results()) == 1
        
        batch_processor.clear()
        assert len(batch_processor.get_results()) == 0
    
    def test_export_results(self, tmp_path):
        """Test exporting results to file."""
        interface = MockAIInterface()
        batch_processor = BatchProcessor(interface)
        
        batch_processor.add_query("Test query")
        
        export_file = tmp_path / "results.json"
        batch_processor.export_results(str(export_file))
        
        assert export_file.exists()
        
        # Verify file contains valid JSON
        import json
        with open(export_file) as f:
            data = json.load(f)
        
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 1