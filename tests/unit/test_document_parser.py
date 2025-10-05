"""
Unit tests for Document Parser Agent.

This module tests the document parsing functionality for various file types
including text, PDF, DOCX, and HTML documents.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import os

from src.agents.document_parser import DocumentParserAgent
from src.models.document import Document, ProcessingStatus


class TestDocumentParserAgent:
    """Test cases for Document Parser Agent."""
    
    def test_init_with_config(self, test_config):
        """Test agent initialization with configuration."""
        config = test_config['document_processing']
        agent = DocumentParserAgent(config)
        
        assert agent.max_file_size_mb == config['max_file_size_mb']
        assert agent.supported_formats == config['supported_formats']
    
    def test_init_with_default_config(self):
        """Test agent initialization with default configuration."""
        agent = DocumentParserAgent({})
        
        assert agent.max_file_size_mb == 50  # Default value
        assert 'txt' in agent.supported_formats
        assert 'pdf' in agent.supported_formats
    
    def test_determine_document_type(self):
        """Test document type determination from file paths."""
        agent = DocumentParserAgent({})
        
        assert agent._determine_document_type("test.txt") == "text"
        assert agent._determine_document_type("test.pdf") == "pdf"
        assert agent._determine_document_type("test.docx") == "docx"
        assert agent._determine_document_type("test.html") == "html"
        assert agent._determine_document_type("test.unknown") == "unknown"
    
    def test_validate_file_exists(self, temp_dir):
        """Test file validation for existing files."""
        agent = DocumentParserAgent({})
        
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        result = agent.validate_file(str(test_file))
        
        assert result['valid'] is True
        assert 'error' not in result
        assert result['file_size'] > 0
        assert result['file_type'] == 'text'
    
    def test_validate_file_not_exists(self):
        """Test file validation for non-existent files."""
        agent = DocumentParserAgent({})
        
        result = agent.validate_file("non_existent_file.txt")
        
        assert result['valid'] is False
        assert 'does not exist' in result['error']
    
    def test_validate_file_too_large(self, temp_dir):
        """Test file validation for oversized files."""
        agent = DocumentParserAgent({'max_file_size_mb': 1})
        
        # Create a large test file (mock the size check)
        test_file = Path(temp_dir) / "large_test.txt"
        test_file.write_text("Test content")
        
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 2 * 1024 * 1024  # 2MB
            result = agent.validate_file(str(test_file))
        
        assert result['valid'] is False
        assert 'too large' in result['error']
    
    def test_validate_file_unsupported_format(self, temp_dir):
        """Test file validation for unsupported formats."""
        agent = DocumentParserAgent({'supported_formats': ['txt', 'pdf']})
        
        # Create test file with unsupported format
        test_file = Path(temp_dir) / "test.xyz"
        test_file.write_text("Test content")
        
        result = agent.validate_file(str(test_file))
        
        assert result['valid'] is False
        assert 'not supported' in result['error']
    
    def test_parse_text_document(self, temp_dir):
        """Test parsing of text documents."""
        agent = DocumentParserAgent({})
        
        # Create a text file
        test_content = "This is a test text document.\nWith multiple lines."
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        
        document = agent.parse_document(str(test_file))
        
        assert isinstance(document, Document)
        assert document.filename == "test.txt"
        assert document.content == test_content
        assert document.document_type == "text"
        assert document.processing_status == ProcessingStatus.PROCESSED
    
    @patch('src.agents.document_parser.BeautifulSoup')
    def test_parse_html_document(self, mock_soup, temp_dir):
        """Test parsing of HTML documents."""
        agent = DocumentParserAgent({})
        
        # Mock BeautifulSoup
        mock_soup_instance = Mock()
        mock_soup_instance.get_text.return_value = "Extracted text content"
        mock_soup.return_value = mock_soup_instance
        
        # Create HTML file
        html_content = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        test_file = Path(temp_dir) / "test.html"
        test_file.write_text(html_content, encoding='utf-8')
        
        document = agent.parse_document(str(test_file))
        
        assert document.content == "Extracted text content"
        assert document.document_type == "html"
        mock_soup.assert_called_once()
    
    @patch('src.agents.document_parser.pypdf.PdfReader')
    def test_parse_pdf_document(self, mock_pdf_reader, temp_dir):
        """Test parsing of PDF documents."""
        agent = DocumentParserAgent({})
        
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF text content"
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Create a mock PDF file
        test_file = Path(temp_dir) / "test.pdf"
        test_file.write_bytes(b"Mock PDF content")
        
        document = agent.parse_document(str(test_file))
        
        assert document.content == "PDF text content"
        assert document.document_type == "pdf"
        mock_pdf_reader.assert_called_once()
    
    @patch('src.agents.document_parser.docx.Document')
    def test_parse_docx_document(self, mock_docx, temp_dir):
        """Test parsing of DOCX documents."""
        agent = DocumentParserAgent({})
        
        # Mock DOCX document
        mock_paragraph = Mock()
        mock_paragraph.text = "DOCX paragraph content"
        mock_doc_instance = Mock()
        mock_doc_instance.paragraphs = [mock_paragraph]
        mock_docx.return_value = mock_doc_instance
        
        # Create a mock DOCX file
        test_file = Path(temp_dir) / "test.docx"
        test_file.write_bytes(b"Mock DOCX content")
        
        document = agent.parse_document(str(test_file))
        
        assert document.content == "DOCX paragraph content"
        assert document.document_type == "docx"
        mock_docx.assert_called_once()
    
    def test_parse_document_file_not_found(self):
        """Test parsing non-existent file."""
        agent = DocumentParserAgent({})
        
        document = agent.parse_document("non_existent.txt")
        
        assert document.processing_status == ProcessingStatus.FAILED
        assert "does not exist" in document.error_message
    
    def test_parse_document_invalid_format(self, temp_dir):
        """Test parsing unsupported file format."""
        agent = DocumentParserAgent({'supported_formats': ['txt']})
        
        # Create file with unsupported format
        test_file = Path(temp_dir) / "test.xyz"
        test_file.write_text("Test content")
        
        document = agent.parse_document(str(test_file))
        
        assert document.processing_status == ProcessingStatus.FAILED
        assert "not supported" in document.error_message
    
    @patch('src.agents.document_parser.pypdf.PdfReader')
    def test_parse_document_processing_error(self, mock_pdf_reader, temp_dir):
        """Test handling of processing errors."""
        agent = DocumentParserAgent({})
        
        # Make PDF reader raise an exception
        mock_pdf_reader.side_effect = Exception("PDF parsing error")
        
        # Create a mock PDF file
        test_file = Path(temp_dir) / "test.pdf"
        test_file.write_bytes(b"Mock PDF content")
        
        document = agent.parse_document(str(test_file))
        
        assert document.processing_status == ProcessingStatus.FAILED
        assert "PDF parsing error" in document.error_message
    
    def test_extract_metadata_text(self, temp_dir):
        """Test metadata extraction from text files."""
        agent = DocumentParserAgent({})
        
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        metadata = agent._extract_metadata(str(test_file), "text")
        
        assert 'file_size' in metadata
        assert 'created_date' in metadata
        assert 'modified_date' in metadata
        assert metadata['encoding'] is not None
    
    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        custom_formats = ['txt', 'pdf', 'html']
        agent = DocumentParserAgent({'supported_formats': custom_formats})
        
        formats = agent.get_supported_formats()
        
        assert formats == custom_formats
    
    def test_detect_encoding(self, temp_dir):
        """Test encoding detection."""
        agent = DocumentParserAgent({})
        
        # Create file with UTF-8 content
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test with émojis: 🤖", encoding='utf-8')
        
        encoding = agent._detect_encoding(str(test_file))
        
        # Should detect UTF-8 or similar
        assert encoding is not None
        assert 'utf' in encoding.lower() or 'ascii' in encoding.lower()
    
    def test_clean_text_content(self):
        """Test text cleaning functionality."""
        agent = DocumentParserAgent({})
        
        dirty_text = "  Text with\tmultiple   spaces\nand\n\nnewlines  "
        clean_text = agent._clean_text_content(dirty_text)
        
        assert clean_text == "Text with multiple spaces and newlines"
    
    def test_parse_document_empty_file(self, temp_dir):
        """Test parsing empty files."""
        agent = DocumentParserAgent({})
        
        # Create empty file
        test_file = Path(temp_dir) / "empty.txt"
        test_file.write_text("")
        
        document = agent.parse_document(str(test_file))
        
        assert document.processing_status == ProcessingStatus.PROCESSED
        assert document.content == ""
        assert document.file_size == 0
    
    def test_parse_document_with_special_characters(self, temp_dir):
        """Test parsing files with special characters."""
        agent = DocumentParserAgent({})
        
        # Create file with special characters
        special_content = "Special chars: àáâãäå, 中文, 🤖, ñoël"
        test_file = Path(temp_dir) / "special.txt"
        test_file.write_text(special_content, encoding='utf-8')
        
        document = agent.parse_document(str(test_file))
        
        assert document.processing_status == ProcessingStatus.PROCESSED
        assert document.content == special_content


@pytest.mark.integration
class TestDocumentParserIntegration:
    """Integration tests for Document Parser Agent."""
    
    def test_parse_real_documents(self, sample_documents):
        """Test parsing with real sample documents."""
        agent = DocumentParserAgent({})
        
        for doc_type, file_path in sample_documents.items():
            document = agent.parse_document(file_path)
            
            assert document.processing_status == ProcessingStatus.PROCESSED
            assert len(document.content) > 0
            assert document.document_type in ['text', 'html', 'json']
            assert document.filename is not None
    
    def test_batch_document_parsing(self, sample_documents):
        """Test parsing multiple documents in sequence."""
        agent = DocumentParserAgent({})
        parsed_docs = []
        
        for file_path in sample_documents.values():
            document = agent.parse_document(file_path)
            parsed_docs.append(document)
        
        assert len(parsed_docs) == len(sample_documents)
        assert all(doc.processing_status == ProcessingStatus.PROCESSED for doc in parsed_docs)


@pytest.mark.slow
class TestDocumentParserPerformance:
    """Performance tests for Document Parser Agent."""
    
    def test_parsing_performance(self, temp_dir, performance_timer):
        """Test document parsing performance."""
        agent = DocumentParserAgent({})
        
        # Create a moderately large text file
        large_content = "This is test content. " * 1000  # ~23KB
        test_file = Path(temp_dir) / "large_test.txt"
        test_file.write_text(large_content)
        
        performance_timer.start()
        document = agent.parse_document(str(test_file))
        performance_timer.stop()
        
        assert document.processing_status == ProcessingStatus.PROCESSED
        assert performance_timer.elapsed < 1.0  # Should parse in less than 1 second
    
    def test_memory_usage_large_file(self, temp_dir):
        """Test memory usage with large files."""
        agent = DocumentParserAgent({})
        
        # Create a large text file
        large_content = "This is test content for memory testing. " * 10000  # ~430KB
        test_file = Path(temp_dir) / "memory_test.txt"
        test_file.write_text(large_content)
        
        # Parse and verify
        document = agent.parse_document(str(test_file))
        
        assert document.processing_status == ProcessingStatus.PROCESSED
        assert len(document.content) == len(large_content)
        
        # Clean up - document should not hold reference to large content
        del document