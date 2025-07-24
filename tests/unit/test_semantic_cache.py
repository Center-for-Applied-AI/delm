import os
import shutil
import tempfile
import json
from pathlib import Path
from dataclasses import asdict

from delm.utils import semantic_cache

lmdb_available = semantic_cache.lmdb is not None

def temp_dir():
    d = tempfile.mkdtemp()
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d)

def sample_key_and_value():
    key = semantic_cache.make_cache_key(
        prompt_text="Extract foo.",
        system_prompt="System.",
        model_name="gpt-4",
        temperature=0.7,
    )
    value = json.dumps({"result": 42}).encode("utf-8")
    meta = {"meta": "info"}
    return key, value, meta

def test_make_cache_key_deterministic():
    k1 = semantic_cache.make_cache_key(
        prompt_text="A", system_prompt="B", model_name="C", temperature=1.0
    )
    k2 = semantic_cache.make_cache_key(
        prompt_text="A", system_prompt="B", model_name="C", temperature=1.0
    )
    assert k1 == k2, "Cache key should be deterministic for same input"
    k3 = semantic_cache.make_cache_key(
        prompt_text="A", system_prompt="B", model_name="C", temperature=2.0
    )
    assert k1 != k3, "Cache key should differ for different input"

def test_filesystem_cache_basic():
    for d in temp_dir():
        key, value, meta = sample_key_and_value()
        cache = semantic_cache.FilesystemJSONCache(d)
        assert cache.get(key) is None
        cache.set(key, value, meta)
        out = cache.get(key)
        assert out == value or out == value
        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["hit"] == 1
        assert stats["miss"] == 1
        cache.prune(max_size_bytes=1000000)
        assert cache.get(key) == value
        cache.prune(max_size_bytes=0)
        assert cache.get(key) is None

def test_filesystem_cache_edge_cases():
    for d in temp_dir():
        cache = semantic_cache.FilesystemJSONCache(d)
        assert cache.get("deadbeef" * 8) is None
        cache.prune(max_size_bytes=100)
        stats = cache.stats()
        assert stats["entries"] == 0

def test_sqlite_cache_basic():
    for d in temp_dir():
        key, value, meta = sample_key_and_value()
        db_path = d / "test.db"
        cache = semantic_cache.SQLiteWALCache(db_path)
        assert cache.get(key) is None
        cache.set(key, value, meta)
        out = cache.get(key)
        assert out == value or out == value
        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["hit"] == 1
        assert stats["miss"] == 1
        cache.prune(max_size_bytes=1000000)
        assert cache.get(key) == value
        cache.prune(max_size_bytes=0)
        assert cache.get(key) is None

def test_sqlite_cache_edge_cases():
    for d in temp_dir():
        db_path = d / "test.db"
        cache = semantic_cache.SQLiteWALCache(db_path)
        assert cache.get("deadbeef" * 8) is None
        cache.prune(max_size_bytes=100)
        stats = cache.stats()
        assert stats["entries"] == 0

def test_factory_from_config_dict():
    for d in temp_dir():
        cfg = {"backend": "filesystem", "path": str(d)}
        cache = semantic_cache.SemanticCacheFactory.from_config(cfg)
        assert isinstance(cache, semantic_cache.FilesystemJSONCache)
        cfg = {"backend": "sqlite", "path": str(d)}
        cache = semantic_cache.SemanticCacheFactory.from_config(cfg)
        assert isinstance(cache, semantic_cache.SQLiteWALCache)
        if lmdb_available:
            cfg = {"backend": "lmdb", "path": str(d)}
            cache = semantic_cache.SemanticCacheFactory.from_config(cfg)
            assert isinstance(cache, semantic_cache.LMDBCache)

def test_factory_from_config_dataclass():
    from delm.config import SemanticCacheConfig
    for d in temp_dir():
        cfg = SemanticCacheConfig(backend="filesystem", path=str(d))
        cache = semantic_cache.SemanticCacheFactory.from_config(cfg)
        assert isinstance(cache, semantic_cache.FilesystemJSONCache)
        cfg = SemanticCacheConfig(backend="sqlite", path=str(d))
        cache = semantic_cache.SemanticCacheFactory.from_config(cfg)
        assert isinstance(cache, semantic_cache.SQLiteWALCache)
        if lmdb_available:
            cfg = SemanticCacheConfig(backend="lmdb", path=str(d))
            cache = semantic_cache.SemanticCacheFactory.from_config(cfg)
            assert isinstance(cache, semantic_cache.LMDBCache)

def test_lmdb_cache_basic():
    if not lmdb_available:
        print("LMDB not available, skipping test_lmdb_cache_basic.")
        return
    for d in temp_dir():
        key, value, meta = sample_key_and_value()
        cache = semantic_cache.LMDBCache(d / "test.lmdb", map_size_mb=10)
        assert cache.get(key) is None
        cache.set(key, value, meta)
        out = cache.get(key)
        assert out == value or out == value
        stats = cache.stats()
        assert stats["entries"] >= 1
        assert stats["hit"] == 1
        assert stats["miss"] == 1
        cache.prune(max_size_bytes=0)
        assert cache.get(key) == value

def test_canonical_json_and_semantic_key():
    obj1 = {"a": 1, "b": 2}
    obj2 = {"b": 2, "a": 1}
    s1 = semantic_cache._canonical_json(obj1)
    s2 = semantic_cache._canonical_json(obj2)
    assert s1 == s2
    k1 = semantic_cache.make_semantic_key(obj1)
    k2 = semantic_cache.make_semantic_key(obj2)
    assert k1 == k2

def test_factory_invalid_backend():
    for d in temp_dir():
        cfg = {"backend": "notarealbackend", "path": str(d)}
        try:
            semantic_cache.SemanticCacheFactory.from_config(cfg)
            assert False, "Should have raised ValueError for invalid backend"
        except ValueError:
            pass

def test_factory_invalid_type():
    try:
        semantic_cache.SemanticCacheFactory.from_config(12345)
        assert False, "Should have raised ValueError for invalid type"
    except ValueError:
        pass

def main():
    print("Running semantic_cache tests...")
    test_make_cache_key_deterministic()
    print("test_make_cache_key_deterministic passed")
    test_filesystem_cache_basic()
    print("test_filesystem_cache_basic passed")
    test_filesystem_cache_edge_cases()
    print("test_filesystem_cache_edge_cases passed")
    test_sqlite_cache_basic()
    print("test_sqlite_cache_basic passed")
    test_sqlite_cache_edge_cases()
    print("test_sqlite_cache_edge_cases passed")
    test_factory_from_config_dict()
    print("test_factory_from_config_dict passed")
    test_factory_from_config_dataclass()
    print("test_factory_from_config_dataclass passed")
    test_canonical_json_and_semantic_key()
    print("test_canonical_json_and_semantic_key passed")
    test_factory_invalid_backend()
    print("test_factory_invalid_backend passed")
    test_factory_invalid_type()
    print("test_factory_invalid_type passed")
    test_lmdb_cache_basic()
    print("test_lmdb_cache_basic passed (if available)")
    print("All tests passed.")

if __name__ == "__main__":
    main() 