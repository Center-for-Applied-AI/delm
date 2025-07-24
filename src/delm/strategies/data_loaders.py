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
    def load(self, path: Path) -> Union[str, pd.DataFrame]:
        """Load data from file and return as string or DataFrame."""
        raise NotImplementedError


class TextLoader(DataLoader):
    """Load plain text files."""
    
    @property
    def requires_target_column(self) -> bool:
        return False
    
    def load(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class HtmlLoader(DataLoader):
    """Load HTML/Markdown files."""
    
    @property
    def requires_target_column(self) -> bool:
        return False
    
    def load(self, path: Path) -> str:
        if BeautifulSoup is None:
            raise DependencyError(
                "BeautifulSoup4 not installed but required for .html/.md loading",
                {"file_path": str(path), "file_type": "html/markdown"}
            )
        try:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
            return soup.get_text("\n")
        except FileNotFoundError as e:
            raise FileError(f"HTML/Markdown file not found: {path}", {"file_path": str(path)}) from e
        except Exception as e:
            raise DataError(f"Failed to load HTML/Markdown file: {path}", {"file_path": str(path)}) from e


class DocxLoader(DataLoader):
    """Load Word documents."""
    
    @property
    def requires_target_column(self) -> bool:
        return False
    
    def load(self, path: Path) -> str:
        if docx is None:
            raise DependencyError(
                "python-docx not installed but required for .docx loading",
                {"file_path": str(path), "file_type": "docx"}
            )
        try:
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except FileNotFoundError as e:
            raise FileError(f"Word document not found: {path}", {"file_path": str(path)}) from e
        except Exception as e:
            raise DataError(f"Failed to load Word document: {path}", {"file_path": str(path)}) from e


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
    
    def load_file(self, file_path: Union[str, Path]) -> Union[str, pd.DataFrame]:
        """Load a file using the appropriate loader."""
        path = Path(file_path)
        if not path.exists():
            raise FileError(f"File does not exist: {path}", {"file_path": str(path)})
        
        loader = self._get_loader(path.suffix)
        return loader.load(path)

# Global factory instance
loader_factory = DataLoaderFactory() 