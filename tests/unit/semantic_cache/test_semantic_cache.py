"""
Unit tests for DELM semantic cache.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import threading
import time

from delm.utils.semantic_cache import (
    SemanticCache, FilesystemJSONCache, SQLiteWALCache, LMDBCache,
    SemanticCacheFactory, make_semantic_key, make_cache_key, _canonical_json
)


class TestUtilities:
    """Test utility functions."""
    
    def test_canonical_json(self):
        """Test canonical JSON generation."""
        data = {"b": 2, "a": 1, "c": 3}
        result = _canonical_json(data)
        
        # Should be sorted and compact
        assert result == '{"a":1,"b":2,"c":3}'
    
    def test_make_semantic_key(self):
        """Test semantic key generation."""
        material = {
            "prompt": "test prompt",
            "model": "gpt-4",
            "temperature": 0.1
        }
        
        key1 = make_semantic_key(material)
        key2 = make_semantic_key(material)
        
        # Should be deterministic
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex digest
        assert isinstance(key1, str)
    
    def test_make_cache_key(self):
        """Test cache key generation."""
        key = make_cache_key(
            prompt_text="test prompt",
            system_prompt="system prompt",
            model_name="gpt-4",
            temperature=0.1
        )
        
        assert len(key) == 64
        assert isinstance(key, str)


class TestSemanticCache:
    """Test the abstract base class."""
    
    def test_abstract_methods(self):
        """Test that SemanticCache is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            SemanticCache()


class TestFilesystemJSONCache:
    """Test the FilesystemJSONCache class."""
    
    def test_initialization(self, tmp_path):
        """Test FilesystemJSONCache initialization."""
        cache = FilesystemJSONCache(tmp_path)
        
        assert cache.root == tmp_path.resolve()
        assert cache.root.exists()
        assert cache._hits == 0
        assert cache._miss == 0
    
    def test_initialization_with_zstd(self, tmp_path):
        """Test initialization with zstd available."""
        with patch('delm.utils.semantic_cache.zstd') as mock_zstd:
            mock_compressor = Mock()
            mock_decompressor = Mock()
            mock_zstd.ZstdCompressor.return_value = mock_compressor
            mock_zstd.ZstdDecompressor.return_value = mock_decompressor
            
            cache = FilesystemJSONCache(tmp_path)
            
            assert cache._compress == mock_compressor.compress
            assert cache._decompress == mock_decompressor.decompress
    
    def test_initialization_without_zstd(self, tmp_path):
        """Test initialization without zstd."""
        with patch('delm.utils.semantic_cache.zstd', None):
            cache = FilesystemJSONCache(tmp_path)
            
            # Should use identity functions
            test_data = b"test data"
            assert cache._compress(test_data) == test_data
            assert cache._decompress(test_data) == test_data
    
    def test_path_generation(self, tmp_path):
        """Test path generation for cache entries."""
        cache = FilesystemJSONCache(tmp_path)
        key = "a" * 64  # 64-char hex key
        
        path = cache._path(key)
        expected_path = tmp_path / "aa" / "aa" / f"{key}.zst"
        
        assert path == expected_path
    
    def test_get_miss(self, tmp_path):
        """Test cache get with miss."""
        cache = FilesystemJSONCache(tmp_path)
        key = "a" * 64
        
        result = cache.get(key)
        
        assert result is None
        assert cache._miss == 1
        assert cache._hits == 0
    
    def test_get_hit(self, tmp_path):
        """Test cache get with hit."""
        cache = FilesystemJSONCache(tmp_path)
        key = "a" * 64
        value = b"test value"
        
        # Set a value first
        cache.set(key, value)
        
        # Get it back
        result = cache.get(key)
        
        assert result == value
        assert cache._hits == 1
        assert cache._miss == 0
    
    def test_set_with_metadata(self, tmp_path):
        """Test setting cache entry with metadata."""
        cache = FilesystemJSONCache(tmp_path)
        key = "a" * 64
        value = b"test value"
        meta = {"timestamp": 1234567890, "size": len(value)}
        
        cache.set(key, value, meta)
        
        # Check that the file exists
        path = cache._path(key)
        assert path.exists()
        
        # Check that metadata file exists
        meta_path = path.with_suffix(".meta.json")
        assert meta_path.exists()
        
        # Verify metadata content
        meta_content = json.loads(meta_path.read_text())
        assert meta_content == meta
    
    def test_stats(self, tmp_path):
        """Test cache statistics."""
        cache = FilesystemJSONCache(tmp_path)
        
        # Add some entries
        cache.set("a" * 64, b"value1")
        cache.set("b" * 64, b"value2")
        
        # Get one to create a hit
        cache.get("a" * 64)
        
        # Try to get non-existent key
        cache.get("c" * 64)
        
        stats = cache.stats()
        
        assert stats["backend"] == "filesystem"
        assert stats["entries"] == 2
        assert stats["hit"] == 1
        assert stats["miss"] == 1
        assert stats["bytes"] > 0
    
    def test_prune(self, tmp_path):
        """Test cache pruning."""
        cache = FilesystemJSONCache(tmp_path)
        
        # Add some entries
        cache.set("a" * 64, b"value1")
        cache.set("b" * 64, b"value2")
        cache.set("c" * 64, b"value3")
        
        # Get initial stats
        initial_stats = cache.stats()
        initial_size = initial_stats["bytes"]
        
        # Prune to a very small size
        cache.prune(max_size_bytes=1)
        
        # Check that entries were removed
        final_stats = cache.stats()
        assert final_stats["entries"] < initial_stats["entries"]
        assert final_stats["bytes"] <= 1


class TestSQLiteWALCache:
    """Test the SQLiteWALCache class."""
    
    def test_initialization(self, tmp_path):
        """Test SQLiteWALCache initialization."""
        db_path = tmp_path / "test.db"
        cache = SQLiteWALCache(db_path)
        
        assert cache.path == db_path.resolve()
        assert db_path.exists()
        assert cache._hits == 0
        assert cache._miss == 0
    
    def test_initialization_with_zstd(self, tmp_path):
        """Test initialization with zstd available."""
        with patch('delm.utils.semantic_cache.zstd') as mock_zstd:
            mock_compressor = Mock()
            mock_decompressor = Mock()
            mock_zstd.ZstdCompressor.return_value = mock_compressor
            mock_zstd.ZstdDecompressor.return_value = mock_decompressor
            
            db_path = tmp_path / "test.db"
            cache = SQLiteWALCache(db_path)
            
            # Check that zstd objects are available
            compressor, decompressor = cache._get_zstd_objects()
            assert compressor == mock_compressor
            assert decompressor == mock_decompressor
    
    def test_initialization_without_zstd(self, tmp_path):
        """Test initialization without zstd."""
        with patch('delm.utils.semantic_cache.zstd', None):
            db_path = tmp_path / "test.db"
            cache = SQLiteWALCache(db_path)
            
            # Check that zstd objects are None
            compressor, decompressor = cache._get_zstd_objects()
            assert compressor is None
            assert decompressor is None
    
    def test_get_miss(self, tmp_path):
        """Test cache get with miss."""
        db_path = tmp_path / "test.db"
        cache = SQLiteWALCache(db_path)
        key = "a" * 64
        
        result = cache.get(key)
        
        assert result is None
        assert cache._miss == 1
        assert cache._hits == 0
    
    def test_get_hit(self, tmp_path):
        """Test cache get with hit."""
        db_path = tmp_path / "test.db"
        cache = SQLiteWALCache(db_path)
        key = "a" * 64
        value = b"test value"
        
        # Set a value first
        cache.set(key, value)
        
        # Get it back
        result = cache.get(key)
        
        assert result == value
        assert cache._hits == 1
        assert cache._miss == 0
    
    def test_get_hit_with_compression(self, tmp_path):
        """Test cache get with compressed data."""
        with patch('delm.utils.semantic_cache.zstd') as mock_zstd:
            mock_compressor = Mock()
            mock_decompressor = Mock()
            mock_zstd.ZstdCompressor.return_value = mock_compressor
            mock_zstd.ZstdDecompressor.return_value = mock_decompressor
            
            # Mock compression/decompression
            original_data = b"test value"
            compressed_data = b"compressed"
            mock_compressor.compress.return_value = compressed_data
            mock_decompressor.decompress.return_value = original_data
            
            db_path = tmp_path / "test.db"
            cache = SQLiteWALCache(db_path)
            key = "a" * 64
            
            # Set compressed value
            cache.set(key, original_data)
            
            # Get it back (should decompress)
            result = cache.get(key)
            
            assert result == original_data
            mock_decompressor.decompress.assert_called_once_with(compressed_data)
    
    def test_set_with_metadata(self, tmp_path):
        """Test setting cache entry with metadata."""
        db_path = tmp_path / "test.db"
        cache = SQLiteWALCache(db_path)
        key = "a" * 64
        value = b"test value"
        meta = {"timestamp": 1234567890, "size": len(value)}
        
        cache.set(key, value, meta)
        
                # Verify the entry exists in the database
        db = cache._get_db()
        row = db.execute("SELECT v, meta FROM cache WHERE k=?", (key,)).fetchone()
    
        assert row is not None
        # The value might be compressed, so we need to decompress it for comparison
        if cache._zstd_available:
            import zstandard as zstd
            decompressed_value = zstd.decompress(row[0])
            assert decompressed_value == value
        else:
            assert row[0] == value  # value
        assert json.loads(row[1]) == meta  # metadata
    
    def test_stats(self, tmp_path):
        """Test cache statistics."""
        db_path = tmp_path / "test.db"
        cache = SQLiteWALCache(db_path)
        
        # Add some entries
        cache.set("a" * 64, b"value1")
        cache.set("b" * 64, b"value2")
        
        # Get one to create a hit
        cache.get("a" * 64)
        
        # Try to get non-existent key
        cache.get("c" * 64)
        
        stats = cache.stats()
        
        assert stats["backend"] == "sqlite"
        assert stats["entries"] == 2
        assert stats["hit"] == 1
        assert stats["miss"] == 1
        assert stats["bytes"] > 0
        assert stats["file"] == str(db_path)
    
    def test_prune(self, tmp_path):
        """Test cache pruning."""
        db_path = tmp_path / "test.db"
        cache = SQLiteWALCache(db_path)
        
        # Add some entries
        cache.set("a" * 64, b"value1")
        cache.set("b" * 64, b"value2")
        cache.set("c" * 64, b"value3")
        
        # Get initial stats
        initial_stats = cache.stats()
        
        # Prune to a very small size
        cache.prune(max_size_bytes=1)
        
        # Check that entries were removed
        final_stats = cache.stats()
        assert final_stats["entries"] < initial_stats["entries"]
    
    def test_close(self, tmp_path):
        """Test closing the cache."""
        db_path = tmp_path / "test.db"
        cache = SQLiteWALCache(db_path)
        
        # Create a connection
        db = cache._get_db()
        assert db is not None
        
        # Close the cache
        cache.close()
        
        # Verify connection is closed
        assert not hasattr(cache._local, 'db')
    
    def test_thread_safety(self, tmp_path):
        """Test thread safety of SQLiteWALCache."""
        db_path = tmp_path / "test.db"
        cache = SQLiteWALCache(db_path)
        
        def worker(thread_id):
            key = f"key_{thread_id}".ljust(64, '0')
            value = f"value_{thread_id}".encode()
            
            # Set and get in this thread
            cache.set(key, value)
            result = cache.get(key)
            assert result == value
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all entries are present
        stats = cache.stats()
        assert stats["entries"] == 5


class TestLMDBCache:
    """Test the LMDBCache class."""
    
    def test_initialization_with_lmdb(self, tmp_path):
        """Test LMDBCache initialization with lmdb available."""
        with patch('delm.utils.semantic_cache.lmdb') as mock_lmdb:
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            db_path = tmp_path / "test.lmdb"
            cache = LMDBCache(db_path)
            
            assert cache.path == db_path.resolve()
            assert cache.env == mock_env
            assert cache._hits == 0
            assert cache._miss == 0
    
    def test_initialization_without_lmdb(self, tmp_path):
        """Test LMDBCache initialization without lmdb."""
        with patch('delm.utils.semantic_cache.lmdb', None):
            db_path = tmp_path / "test.lmdb"
            
            with pytest.raises(ImportError, match="lmdb package not installed"):
                LMDBCache(db_path)
    
    def test_get_miss(self, tmp_path):
        """Test cache get with miss."""
        with patch('delm.utils.semantic_cache.lmdb') as mock_lmdb:
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            # Mock transaction that returns None
            mock_txn = Mock()
            mock_txn.get.return_value = None
            mock_begin = Mock()
            mock_begin.__enter__ = Mock(return_value=mock_txn)
            mock_begin.__exit__ = Mock(return_value=None)
            mock_env.begin.return_value = mock_begin
            
            db_path = tmp_path / "test.lmdb"
            cache = LMDBCache(db_path)
            key = "a" * 64
            
            result = cache.get(key)
            
            assert result is None
            assert cache._miss == 1
            assert cache._hits == 0
    
    def test_get_hit(self, tmp_path):
        """Test cache get with hit."""
        with patch('delm.utils.semantic_cache.lmdb') as mock_lmdb, \
             patch('delm.utils.semantic_cache.zstd') as mock_zstd:
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            # Mock decompressor
            mock_decompressor = Mock()
            mock_zstd.ZstdDecompressor.return_value = mock_decompressor
            mock_decompressor.decompress.return_value = b"test value"
            
            # Mock transaction that returns a value
            mock_txn = Mock()
            mock_txn.get.return_value = b"compressed_value"
            mock_begin = Mock()
            mock_begin.__enter__ = Mock(return_value=mock_txn)
            mock_begin.__exit__ = Mock(return_value=None)
            mock_env.begin.return_value = mock_begin
            
            db_path = tmp_path / "test.lmdb"
            cache = LMDBCache(db_path)
            key = "a" * 64
            
            result = cache.get(key)
            
            assert result == b"test value"
            assert cache._hits == 1
            assert cache._miss == 0
    
    def test_get_hit_with_compression(self, tmp_path):
        """Test cache get with compressed data."""
        with patch('delm.utils.semantic_cache.lmdb') as mock_lmdb, \
             patch('delm.utils.semantic_cache.zstd') as mock_zstd:
            
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            mock_compressor = Mock()
            mock_decompressor = Mock()
            mock_zstd.ZstdCompressor.return_value = mock_compressor
            mock_zstd.ZstdDecompressor.return_value = mock_decompressor
            
            # Mock compression/decompression
            original_data = b"test value"
            compressed_data = b"compressed"
            mock_compressor.compress.return_value = compressed_data
            mock_decompressor.decompress.return_value = original_data
            
            # Mock transaction
            mock_txn = Mock()
            mock_txn.get.return_value = compressed_data
            mock_begin = Mock()
            mock_begin.__enter__ = Mock(return_value=mock_txn)
            mock_begin.__exit__ = Mock(return_value=None)
            mock_env.begin.return_value = mock_begin
            
            db_path = tmp_path / "test.lmdb"
            cache = LMDBCache(db_path)
            key = "a" * 64
            
            result = cache.get(key)
            
            assert result == original_data
            mock_decompressor.decompress.assert_called_once_with(compressed_data)
    
    def test_set(self, tmp_path):
        """Test setting cache entry."""
        with patch('delm.utils.semantic_cache.lmdb') as mock_lmdb, \
             patch('delm.utils.semantic_cache.zstd') as mock_zstd:
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            # Mock compressor
            mock_compressor = Mock()
            mock_zstd.ZstdCompressor.return_value = mock_compressor
            mock_compressor.compress.return_value = b"compressed_value"
            
            # Mock transaction
            mock_txn = Mock()
            mock_begin = Mock()
            mock_begin.__enter__ = Mock(return_value=mock_txn)
            mock_begin.__exit__ = Mock(return_value=None)
            mock_env.begin.return_value = mock_begin
            
            db_path = tmp_path / "test.lmdb"
            cache = LMDBCache(db_path)
            key = "a" * 64
            value = b"test value"
            
            cache.set(key, value)
            
            # Verify the transaction was used correctly
            mock_txn.put.assert_called_once_with(
                key.encode("utf‑8"), b"compressed_value", overwrite=True
            )
    
    def test_set_with_compression(self, tmp_path):
        """Test setting cache entry with compression."""
        with patch('delm.utils.semantic_cache.lmdb') as mock_lmdb, \
             patch('delm.utils.semantic_cache.zstd') as mock_zstd:
            
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            mock_compressor = Mock()
            mock_zstd.ZstdCompressor.return_value = mock_compressor
            
            # Mock compression
            original_data = b"test value"
            compressed_data = b"compressed"
            mock_compressor.compress.return_value = compressed_data
            
            # Mock transaction
            mock_txn = Mock()
            mock_begin = Mock()
            mock_begin.__enter__ = Mock(return_value=mock_txn)
            mock_begin.__exit__ = Mock(return_value=None)
            mock_env.begin.return_value = mock_begin
            
            db_path = tmp_path / "test.lmdb"
            cache = LMDBCache(db_path)
            key = "a" * 64
            
            cache.set(key, original_data)
            
            # Verify compression was used
            mock_compressor.compress.assert_called_once_with(original_data)
            mock_txn.put.assert_called_once_with(
                key.encode("utf‑8"), compressed_data, overwrite=True
            )
    
    def test_stats(self, tmp_path):
        """Test cache statistics."""
        with patch('delm.utils.semantic_cache.lmdb') as mock_lmdb:
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            # Mock environment stats
            mock_env.stat.return_value = {"entries": 5}
            mock_env.info.return_value = {"map_size": 1024 * 1024}
            
            db_path = tmp_path / "test.lmdb"
            cache = LMDBCache(db_path)
            
            # Simulate some hits and misses
            cache._hits = 3
            cache._miss = 2
            
            stats = cache.stats()
            
            assert stats["backend"] == "lmdb"
            assert stats["entries"] == 5
            assert stats["map_size"] == 1024 * 1024
            assert stats["hit"] == 3
            assert stats["miss"] == 2
            assert stats["file"] == str(db_path)
    
    def test_prune(self, tmp_path):
        """Test cache pruning (should be no-op for LMDB)."""
        with patch('delm.utils.semantic_cache.lmdb') as mock_lmdb:
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            db_path = tmp_path / "test.lmdb"
            cache = LMDBCache(db_path)
            
            # Prune should not raise an exception
            cache.prune(max_size_bytes=1000)


class TestSemanticCacheFactory:
    """Test the SemanticCacheFactory class."""
    
    def test_from_config_none(self):
        """Test factory with None config."""
        cache = SemanticCacheFactory.from_config(None)
        assert isinstance(cache, SQLiteWALCache)
    
    def test_from_config_dict(self):
        """Test factory with dict config."""
        config = {"backend": "filesystem", "path": "/tmp/test_cache"}
        
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value = Path("/tmp/test_cache")
            cache = SemanticCacheFactory.from_config(config)
            
            assert isinstance(cache, FilesystemJSONCache)
    
    def test_from_config_dataclass(self):
        """Test factory with dataclass config."""
        from dataclasses import dataclass
        
        @dataclass
        class CacheConfig:
            backend: str = "sqlite"
            path: str = "/tmp/test_cache"
        
        config = CacheConfig()
        
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value = Path("/tmp/test_cache")
            cache = SemanticCacheFactory.from_config(config)
            
            assert isinstance(cache, SQLiteWALCache)
    
    def test_from_config_filesystem(self):
        """Test factory with filesystem backend."""
        config = {"backend": "filesystem", "path": "/tmp/test_cache"}
        
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value = Path("/tmp/test_cache")
            cache = SemanticCacheFactory.from_config(config)
            
            assert isinstance(cache, FilesystemJSONCache)
    
    def test_from_config_sqlite(self):
        """Test factory with sqlite backend."""
        config = {"backend": "sqlite", "path": "/tmp/test_cache", "synchronous": "FULL"}
        
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value = Path("/tmp/test_cache")
            cache = SemanticCacheFactory.from_config(config)
            
            assert isinstance(cache, SQLiteWALCache)
    
    def test_from_config_lmdb(self):
        """Test factory with lmdb backend."""
        config = {"backend": "lmdb", "path": "/tmp/test_cache", "map_size_mb": 2048}
        
        with patch('pathlib.Path') as mock_path, \
             patch('delm.utils.semantic_cache.lmdb') as mock_lmdb:
            
            mock_path.return_value = Path("/tmp/test_cache")
            mock_env = Mock()
            mock_lmdb.open.return_value = mock_env
            
            cache = SemanticCacheFactory.from_config(config)
            
            assert isinstance(cache, LMDBCache)
    
    def test_from_config_unknown_backend(self):
        """Test factory with unknown backend."""
        config = {"backend": "unknown", "path": "/tmp/test_cache"}
        
        with pytest.raises(ValueError, match="Unknown cache backend"):
            SemanticCacheFactory.from_config(config)
    
    def test_from_config_unknown_type(self):
        """Test factory with unknown config type."""
        config = "invalid"
        
        with pytest.raises(ValueError, match="Unknown cache config type"):
            SemanticCacheFactory.from_config(config) 