"""
DELM Data Loaders
================
Factory pattern for loading different file formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Callable
import pandas as pd

from delm.exceptions import DataError, FileError, DependencyError
from delm.constants import SYSTEM_FILE_NAME_COLUMN, SYSTEM_RAW_DATA_COLUMN

# Optional dependencies
try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    import docx  # python‑docx
except ImportError:  # pragma: no cover
    docx = None  # type: ignore

# try:
#     import marker  # OCR & PDF
# except ImportError:  # pragma: no cover
#     marker = None  # type: ignore


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @property
    @abstractmethod
    def requires_target_column(self) -> bool:
        """Whether this loader requires a target column specification."""
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: Path) -> pd.DataFrame:
        """Load data from file and return as string or DataFrame."""
        raise NotImplementedError


class TextLoader(DataLoader):
    """Load plain text files."""
    
    @property
    def requires_target_column(self) -> bool:
        return False
    
    def load(self, path: Path) -> pd.DataFrame:
        return pd.DataFrame({SYSTEM_FILE_NAME_COLUMN: [path.name], SYSTEM_RAW_DATA_COLUMN: [path.read_text(encoding="utf-8", errors="replace")]})


class HtmlLoader(DataLoader):
    """Load HTML/Markdown files."""
    
    @property
    def requires_target_column(self) -> bool:
        return False
    
    def load(self, path: Path) -> pd.DataFrame:
        if BeautifulSoup is None:
            raise DependencyError(
                "BeautifulSoup4 not installed but required for .html/.md loading",
                {"file_path": str(path), "file_type": "html/markdown"}
            )
        try:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
            return pd.DataFrame({SYSTEM_FILE_NAME_COLUMN: [path.name], SYSTEM_RAW_DATA_COLUMN: [soup.get_text("\n")]})
        except FileNotFoundError as e:
            raise FileError(f"HTML/Markdown file not found: {path}", {"file_path": str(path)}) from e
        except Exception as e:
            raise DataError(f"Failed to load HTML/Markdown file: {path}", {"file_path": str(path)}) from e


class DocxLoader(DataLoader):
    """Load Word documents."""
    
    @property
    def requires_target_column(self) -> bool:
        return False
    
    def load(self, path: Path) -> pd.DataFrame:
        if docx is None:
            raise DependencyError(
                "python-docx not installed but required for .docx loading",
                {"file_path": str(path), "file_type": "docx"}
            )
        try:
            text = self._extract_all_text(docx.Document(str(path)))
            return pd.DataFrame({SYSTEM_FILE_NAME_COLUMN: [path.name], SYSTEM_RAW_DATA_COLUMN: [text]})
        except FileNotFoundError as e:
            raise FileError(f"Word document not found: {path}", {"file_path": str(path)}) from e
        except Exception as e:
            raise DataError(f"Failed to load Word document: {path}", {"file_path": str(path)}) from e
    
    def _extract_all_text(self, doc) -> str:
        text_parts = []

        # 1. Headers (for each section)
        for section in doc.sections:
            for p in section.header.paragraphs:
                if p.text.strip():
                    text_parts.append(p.text)

        # 2. Main body paragraphs (includes titles/headings)
        for p in doc.paragraphs:
            if p.text.strip():
                text_parts.append(p.text)

        # 3. Tables (in order of appearance)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        text_parts.append(cell_text)

        # 4. Footers (for each section)
        for section in doc.sections:
            for p in section.footer.paragraphs:
                if p.text.strip():
                    text_parts.append(p.text)

        return "\n".join(text_parts)


class CsvLoader(DataLoader):
    """Load CSV files."""
    
    @property
    def requires_target_column(self) -> bool:
        return True
    
    def load(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except FileNotFoundError as e:
            raise FileError(f"CSV file not found: {path}", {"file_path": str(path)}) from e
        except Exception as e:
            raise DataError(f"Failed to load CSV file: {path}", {"file_path": str(path)}) from e


# class PdfLoader(DataLoader):
#     """Load PDF files using OCR."""
    
#     def load(self, path: Path) -> str:
#         pass


class DataLoaderFactory:
    """Factory for creating data loaders based on file extension."""
    
    def __init__(self):
        self._loaders: Dict[str, DataLoader] = {
            ".txt": TextLoader(),
            ".md": HtmlLoader(),
            ".html": HtmlLoader(),
            ".htm": HtmlLoader(),
            ".docx": DocxLoader(),
            # TODO: add pdf loader
            # ".pdf": PdfLoader(),
            ".csv": CsvLoader(),
        }
    
    def _get_loader(self, extension: str) -> DataLoader:
        """Get the appropriate loader for a file extension."""
        loader = self._loaders.get(extension.lower())
        if loader is None:
            supported = ", ".join(self.get_supported_extensions())
            raise DataError(
                f"Unsupported file type: {extension}",
                {
                    "file_extension": extension,
                    "supported_extensions": self.get_supported_extensions(),
                    "suggestion": f"Supported formats: {supported}"
                }
            )
        return loader

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions."""
        return list(self._loaders.keys())

    def requires_target_column(self, extension: str) -> bool:
        """Check if a file extension requires a target column specification."""
        loader = self._loaders.get(extension.lower())
        if loader is None:
            supported = ", ".join(self.get_supported_extensions())
            raise DataError(
                f"Unsupported file type: {extension}",
                {
                    "file_extension": extension,
                    "supported_extensions": self.get_supported_extensions(),
                    "suggestion": f"Supported formats: {supported}"
                }
            )
        return loader.requires_target_column

    def _register_loader(self, extension: str, loader: DataLoader) -> None:
        """Register a new loader for a file extension."""
        self._loaders[extension.lower()] = loader
    
    def load_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a file using the appropriate loader."""
        path = Path(file_path)
        if not path.exists():
            raise FileError(f"File does not exist: {path}", {"file_path": str(path)})
        
        loader = self._get_loader(path.suffix)
        return loader.load(path)
    
    def load_directory(self, directory_path: Union[str, Path]) -> tuple[pd.DataFrame, str]:
        """Load a directory of files using the appropriate loader.
        
        Returns:
            tuple[pd.DataFrame, str]: A tuple containing the loaded dataframe and the extension of the loaded files.
        Raises:
            FileError: If the directory does not exist.
            FileError: If the directory contains multiple file types.
        """
        extensions = set()
        path = Path(directory_path)
        if not path.exists():
            raise FileError(f"Directory does not exist: {path}", {"directory_path": str(path)})
        
        # Load all files into a dataframe. The record_id should be the file name.
        data = pd.DataFrame()
        for file in path.glob("**/*"):
            if file.is_file():
                data = pd.concat([data, self.load_file(file)], ignore_index=True)
                extensions.add(file.suffix)

        if len(extensions) != 1:
            raise FileError(
                f"Directory contains multiple file types: {path}",
                {"directory_path": str(path), "extensions found": list(extensions)}
            )
        return data, list(extensions)[0]

    def get_loaded_extension(self) -> str:
        """Get the extension of the loaded files."""
        if len(self._loaded_extensions) != 1:
            raise FileError(
                f"Directory contains multiple file types, or no files were loaded",
                {"extensions found": list(self._loaded_extensions)}
            )
        return list(self._loaded_extensions)[0]

# Global factory instance
loader_factory = DataLoaderFactory() 