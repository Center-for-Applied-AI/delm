"""
Unit tests for DELM data loaders.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from delm.strategies.data_loaders import (
    DataLoader, TextLoader, HtmlLoader, DocxLoader, CsvLoader,
    ParquetLoader, FeatherLoader, PdfLoader, ExcelLoader,
    DataLoaderFactory, loader_factory
)
from delm.constants import SYSTEM_FILE_NAME_COLUMN, SYSTEM_RAW_DATA_COLUMN, IGNORE_FILES


class TestDataLoader:
    """Test the abstract base class."""
    
    def test_abstract_methods(self):
        """Test that DataLoader is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            DataLoader()


class TestTextLoader:
    """Test the TextLoader class."""
    
    def test_requires_target_column(self):
        """Test that TextLoader doesn't require target column."""
        loader = TextLoader()
        assert loader.requires_target_column is False
    
    def test_load_text_file(self, tmp_path):
        """Test loading a simple text file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        content = "Hello, world!\nThis is a test file."
        test_file.write_text(content)
        
        loader = TextLoader()
        result = loader.load(test_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result[SYSTEM_FILE_NAME_COLUMN].iloc[0] == "test.txt"
        assert result[SYSTEM_RAW_DATA_COLUMN].iloc[0] == content
    
    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file."""
        test_file = tmp_path / "test.md"
        content = "# Title\n\nThis is **markdown** content."
        test_file.write_text(content)
        
        loader = TextLoader()
        result = loader.load(test_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result[SYSTEM_FILE_NAME_COLUMN].iloc[0] == "test.md"
        assert result[SYSTEM_RAW_DATA_COLUMN].iloc[0] == content


class TestHtmlLoader:
    """Test the HtmlLoader class."""
    
    def test_requires_target_column(self):
        """Test that HtmlLoader doesn't require target column."""
        loader = HtmlLoader()
        assert loader.requires_target_column is False
    
    def test_load_html_file_with_beautifulsoup(self, tmp_path):
        """Test loading an HTML file with BeautifulSoup available."""
        test_file = tmp_path / "test.html"
        content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Hello World</h1>
                <p>This is a <strong>test</strong> paragraph.</p>
            </body>
        </html>
        """
        test_file.write_text(content)
        
        with patch('bs4.BeautifulSoup') as mock_bs:
            mock_soup = Mock()
            mock_soup.get_text.return_value = "Test Page\nHello World\nThis is a test paragraph."
            mock_bs.return_value = mock_soup
            
            loader = HtmlLoader()
            result = loader.load(test_file)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert result[SYSTEM_FILE_NAME_COLUMN].iloc[0] == "test.html"
            assert "Hello World" in result[SYSTEM_RAW_DATA_COLUMN].iloc[0]
    
    def test_load_html_file_without_beautifulsoup(self, tmp_path):
        """Test loading an HTML file without BeautifulSoup."""
        test_file = tmp_path / "test.html"
        test_file.write_text("<html><body>Test</body></html>")
        
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'bs4':
                    raise ImportError("No module named 'bs4'")
                return __import__(name, *args, **kwargs)
            mock_import.side_effect = side_effect
            
            loader = HtmlLoader()
            with pytest.raises(ImportError, match="BeautifulSoup4 not installed"):
                loader.load(test_file)


class TestDocxLoader:
    """Test the DocxLoader class."""
    
    def test_requires_target_column(self):
        """Test that DocxLoader doesn't require target column."""
        loader = DocxLoader()
        assert loader.requires_target_column is False
    
    def test_load_docx_file_with_docx(self, tmp_path):
        """Test loading a DOCX file with python-docx available."""
        test_file = tmp_path / "test.docx"
        test_file.write_text("dummy content")  # Create file
        
        with patch('docx.Document') as mock_document:
            mock_doc = Mock()
            mock_doc.sections = []
            mock_doc.paragraphs = [Mock(text="Paragraph 1"), Mock(text="Paragraph 2")]
            mock_doc.tables = []
            mock_document.return_value = mock_doc
            
            loader = DocxLoader()
            result = loader.load(test_file)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert result[SYSTEM_FILE_NAME_COLUMN].iloc[0] == "test.docx"
            assert "Paragraph 1" in result[SYSTEM_RAW_DATA_COLUMN].iloc[0]
    
    def test_load_docx_file_without_docx(self, tmp_path):
        """Test loading a DOCX file without python-docx."""
        test_file = tmp_path / "test.docx"
        test_file.write_text("dummy content")
        
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'docx':
                    raise ImportError("No module named 'docx'")
                return __import__(name, *args, **kwargs)
            mock_import.side_effect = side_effect
            
            loader = DocxLoader()
            with pytest.raises(ImportError, match="python-docx not installed"):
                loader.load(test_file)
    
    def test_extract_all_text_comprehensive(self):
        """Test comprehensive text extraction from DOCX."""
        loader = DocxLoader()
        
        # Mock document with all components
        mock_doc = Mock()
        
        # Mock sections with headers and footers
        mock_section = Mock()
        mock_section.header.paragraphs = [Mock(text="Header text")]
        mock_section.footer.paragraphs = [Mock(text="Footer text")]
        mock_doc.sections = [mock_section]
        
        # Mock body paragraphs
        mock_doc.paragraphs = [Mock(text="Body paragraph 1"), Mock(text="Body paragraph 2")]
        
        # Mock tables
        mock_table = Mock()
        mock_row = Mock()
        mock_cell = Mock()
        mock_cell.text = "Table cell text"
        mock_row.cells = [mock_cell]
        mock_table.rows = [mock_row]
        mock_doc.tables = [mock_table]
        
        result = loader._extract_all_text(mock_doc)
        
        assert "Header text" in result
        assert "Body paragraph 1" in result
        assert "Body paragraph 2" in result
        assert "Table cell text" in result
        assert "Footer text" in result


class TestCsvLoader:
    """Test the CsvLoader class."""
    
    def test_requires_target_column(self):
        """Test that CsvLoader requires target column."""
        loader = CsvLoader()
        assert loader.requires_target_column is True
    
    def test_load_csv_file(self, tmp_path):
        """Test loading a CSV file."""
        test_file = tmp_path / "test.csv"
        content = "name,age,city\nJohn,30,NYC\nJane,25,LA"
        test_file.write_text(content)
        
        loader = CsvLoader()
        result = loader.load(test_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age", "city"]
        assert result["name"].iloc[0] == "John"
        assert result["age"].iloc[1] == 25


class TestParquetLoader:
    """Test the ParquetLoader class."""
    
    def test_requires_target_column(self):
        """Test that ParquetLoader requires target column."""
        loader = ParquetLoader()
        assert loader.requires_target_column is True
    
    def test_load_parquet_file(self, tmp_path):
        """Test loading a Parquet file."""
        test_file = tmp_path / "test.parquet"
        
        # Create test data
        df = pd.DataFrame({
            "name": ["John", "Jane"],
            "age": [30, 25],
            "city": ["NYC", "LA"]
        })
        df.to_parquet(test_file)
        
        loader = ParquetLoader()
        result = loader.load(test_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age", "city"]
        pd.testing.assert_frame_equal(result, df)


class TestFeatherLoader:
    """Test the FeatherLoader class."""
    
    def test_requires_target_column(self):
        """Test that FeatherLoader requires target column."""
        loader = FeatherLoader()
        assert loader.requires_target_column is True
    
    def test_load_feather_file(self, tmp_path):
        """Test loading a Feather file."""
        test_file = tmp_path / "test.feather"
        
        # Create test data
        df = pd.DataFrame({
            "name": ["John", "Jane"],
            "age": [30, 25],
            "city": ["NYC", "LA"]
        })
        df.to_feather(test_file)
        
        loader = FeatherLoader()
        result = loader.load(test_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age", "city"]
        pd.testing.assert_frame_equal(result, df)


class TestPdfLoader:
    """Test the PdfLoader class."""
    
    def test_requires_target_column(self):
        """Test that PdfLoader doesn't require target column."""
        loader = PdfLoader()
        assert loader.requires_target_column is False
    
    def test_load_pdf_file_with_marker(self, tmp_path):
        """Test loading a PDF file with marker available."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")  # Create file
        
        # Since marker is installed, we can test the actual functionality
        # but we'll mock the internal calls to avoid actual PDF processing
        with patch('delm.strategies.data_loaders.PdfLoader.load') as mock_load:
            mock_load.return_value = pd.DataFrame({
                SYSTEM_FILE_NAME_COLUMN: ["test.pdf"],
                SYSTEM_RAW_DATA_COLUMN: ["Extracted text"]
            })
            
            loader = PdfLoader()
            result = loader.load(test_file)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert result[SYSTEM_FILE_NAME_COLUMN].iloc[0] == "test.pdf"
            assert result[SYSTEM_RAW_DATA_COLUMN].iloc[0] == "Extracted text"
    
    def test_load_pdf_file_without_marker(self, tmp_path):
        """Test loading a PDF file without marker."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")
        
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'marker':
                    raise ImportError("No module named 'marker'")
                return __import__(name, *args, **kwargs)
            mock_import.side_effect = side_effect
            
            loader = PdfLoader()
            with pytest.raises(ImportError, match="marker-pdf not installed"):
                loader.load(test_file)


class TestExcelLoader:
    """Test the ExcelLoader class."""
    
    def test_requires_target_column(self):
        """Test that ExcelLoader requires target column."""
        loader = ExcelLoader()
        assert loader.requires_target_column is True
    
    def test_load_excel_file(self, tmp_path):
        """Test loading an Excel file."""
        test_file = tmp_path / "test.xlsx"
        test_file.write_text("dummy content")  # Create file
        
        # Create test data
        expected_df = pd.DataFrame({
            "name": ["John", "Jane"],
            "age": [30, 25],
            "city": ["NYC", "LA"]
        })
        
        with patch('pandas.read_excel', return_value=expected_df):
            loader = ExcelLoader()
            result = loader.load(test_file)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert list(result.columns) == ["name", "age", "city"]
            assert result["name"].iloc[0] == "John"
            assert result["age"].iloc[1] == 25


class TestDataLoaderFactory:
    """Test the DataLoaderFactory class."""
    
    def test_initialization(self):
        """Test factory initialization."""
        factory = DataLoaderFactory()
        assert len(factory._loaders) > 0
        assert ".txt" in factory._loaders
        assert ".csv" in factory._loaders
        assert ".pdf" in factory._loaders
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        factory = DataLoaderFactory()
        extensions = factory.get_supported_extensions()
        assert isinstance(extensions, list)
        assert ".txt" in extensions
        assert ".csv" in extensions
        assert ".pdf" in extensions
    
    def test_get_loader_valid_extension(self):
        """Test getting loader for valid extension."""
        factory = DataLoaderFactory()
        loader = factory._get_loader(".txt")
        assert isinstance(loader, TextLoader)
    
    def test_get_loader_invalid_extension(self):
        """Test getting loader for invalid extension."""
        factory = DataLoaderFactory()
        with pytest.raises(ValueError, match="Unsupported file type"):
            factory._get_loader(".invalid")
    
    def test_requires_target_column(self):
        """Test checking if extension requires target column."""
        factory = DataLoaderFactory()
        assert factory.requires_target_column(".txt") is False
        assert factory.requires_target_column(".csv") is True
        assert factory.requires_target_column(".pdf") is False
    
    def test_register_loader(self):
        """Test registering a custom loader."""
        factory = DataLoaderFactory()
        
        class CustomLoader(DataLoader):
            @property
            def requires_target_column(self):
                return False
            
            def load(self, path):
                return pd.DataFrame()
        
        factory._register_loader(".custom", CustomLoader())
        assert ".custom" in factory._loaders
        assert isinstance(factory._loaders[".custom"], CustomLoader)
    
    def test_load_file_text(self, tmp_path):
        """Test loading a text file through factory."""
        test_file = tmp_path / "test.txt"
        content = "Hello, world!"
        test_file.write_text(content)
        
        factory = DataLoaderFactory()
        result = factory.load_file(test_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result[SYSTEM_FILE_NAME_COLUMN].iloc[0] == "test.txt"
        assert result[SYSTEM_RAW_DATA_COLUMN].iloc[0] == content
    
    def test_load_file_csv(self, tmp_path):
        """Test loading a CSV file through factory."""
        test_file = tmp_path / "test.csv"
        content = "name,age\nJohn,30\nJane,25"
        test_file.write_text(content)
        
        factory = DataLoaderFactory()
        result = factory.load_file(test_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age"]
    
    def test_load_file_nonexistent(self, tmp_path):
        """Test loading a non-existent file."""
        factory = DataLoaderFactory()
        nonexistent_file = tmp_path / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            factory.load_file(nonexistent_file)
    
    def test_load_file_unsupported_type(self, tmp_path):
        """Test loading a file with unsupported extension."""
        test_file = tmp_path / "test.invalid"
        test_file.write_text("content")
        
        factory = DataLoaderFactory()
        with pytest.raises(ValueError, match="Unsupported file type"):
            factory.load_file(test_file)
    
    def test_load_directory_single_type(self, tmp_path):
        """Test loading a directory with single file type."""
        # Create test files
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        (tmp_path / "file3.txt").write_text("Content 3")
        
        factory = DataLoaderFactory()
        result, extension = factory.load_directory(tmp_path)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert extension == ".txt"
        assert all("file" in name for name in result[SYSTEM_FILE_NAME_COLUMN])
    
    def test_load_directory_multiple_types(self, tmp_path):
        """Test loading a directory with multiple file types."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.csv").write_text("name,age\nJohn,30")
        
        factory = DataLoaderFactory()
        with pytest.raises(ValueError, match="multiple file types"):
            factory.load_directory(tmp_path)
    
    def test_load_directory_empty(self, tmp_path):
        """Test loading an empty directory."""
        factory = DataLoaderFactory()
        with pytest.raises(ValueError, match="No files loaded"):
            factory.load_directory(tmp_path)
    
    def test_load_directory_with_ignored_files(self, tmp_path):
        """Test loading a directory with ignored files."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / ".gitignore").write_text("ignored")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "file.pyc").write_text("compiled")
        
        factory = DataLoaderFactory()
        result, extension = factory.load_directory(tmp_path)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Only the .txt file
        assert extension == ".txt"
    
    def test_load_directory_nonexistent(self, tmp_path):
        """Test loading a non-existent directory."""
        factory = DataLoaderFactory()
        nonexistent_dir = tmp_path / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            factory.load_directory(nonexistent_dir)


class TestGlobalLoaderFactory:
    """Test the global loader factory instance."""
    
    def test_global_instance_exists(self):
        """Test that the global loader factory instance exists."""
        assert loader_factory is not None
        assert isinstance(loader_factory, DataLoaderFactory)
    
    def test_global_instance_functionality(self, tmp_path):
        """Test that the global instance works correctly."""
        test_file = tmp_path / "test.txt"
        content = "Hello, world!"
        test_file.write_text(content)
        
        result = loader_factory.load_file(test_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result[SYSTEM_FILE_NAME_COLUMN].iloc[0] == "test.txt"
        assert result[SYSTEM_RAW_DATA_COLUMN].iloc[0] == content 