r"""
DELM Semantic Cache
===================
A configurable, persistent, exact‑match cache for Instructor calls.

* Users choose the backend (`sqlite` | `lmdb` | `filesystem`) via
  `config.cache`.
* Keys are SHA‑256 hashes of **canonical JSON** containing:
  - rendered prompt/chunk text
  - model provider + name + generation params
  - extraction schema hash
  - prompt template hash
  - major DELM version
* Values are **zstd‑compressed** JSON bytes of the Instructor response
  plus a small metadata JSON envelope.

Back‑ends:
----------
* **SQLiteWALCache**  (default, std‑lib only)
* **LMDBCache**       (fastest, optional `lmdb` wheel)
* **FilesystemJSONCache** (zero deps, debug‑friendly)

The cache instance is created by `CacheFactory.from_config()` and passed to
`ExtractionManager`, which calls `cache.get(key)` before hitting the API and
`cache.set(key, response_bytes, meta)` afterwards.

This single file keeps import overhead minimal and avoids circular refs. If
size grows, split into a sub‑package (`delm.cache.*`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
import json
import hashlib
import time
import sqlite3
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union, Optional, Mapping

# Module-level logger
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional deps                                                                #
# --------------------------------------------------------------------------- #
try:
    import lmdb  # type: ignore
except ImportError:  # pragma: no cover
    lmdb = None

try:
    import zstandard as zstd  # type: ignore
except ImportError:  # pragma: no cover
    zstd = None  # fallback to no compression later

_ZSTD_LEVEL = 3  # good balance of speed / ratio

# --------------------------------------------------------------------------- #
# Utility helpers                                                              #
# --------------------------------------------------------------------------- #


def _canonical_json(obj: Any) -> str:
    """Return JSON string with sorted keys & no whitespace (deterministic)."""
    result = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    log.debug("Created canonical JSON (length: %d)", len(result))
    return result


def make_semantic_key(material: Mapping[str, Any]) -> str:
    """Hash canonical JSON material to a 64‑char hex string."""
    log.debug("Creating semantic key from material with %d keys", len(material))
    digest = hashlib.sha256(_canonical_json(material).encode("utf‑8")).hexdigest()
    log.debug("Generated semantic key: %s", digest[:16] + "...")
    return digest


def make_cache_key(
    *,
    prompt_text: str,
    system_prompt: str,
    model_name: str,
    temperature: float,
) -> str:
    """
    Build a deterministic cache key that depends **only** on:
      • rendered user prompt text  (includes chunk & any template vars)
      • system prompt text
      • model_name  (e.g. 'gpt‑4o-mini')
      • temperature
    """
    log.debug(
        "Creating cache key: model=%s, temperature=%s, prompt_length=%d, system_length=%d",
        model_name,
        temperature,
        len(prompt_text),
        len(system_prompt),
    )
    material = {
        "prompt": prompt_text,
        "system": system_prompt,
        "model": model_name,
        "temperature": temperature,
    }
    return make_semantic_key(material)


# --------------------------------------------------------------------------- #
# Abstract interface                                                           #
# --------------------------------------------------------------------------- #
class SemanticCache(ABC):
    """Minimal interface all cache back‑ends must implement."""

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Return raw (compressed) bytes or None if missing."""

    @abstractmethod
    def set(
        self, key: str, value: bytes, meta: Mapping[str, Any] | None = None
    ) -> None:  # noqa: E501
        """Insert `value` for `key` (no return). Must be *durable* when the method returns."""

    @abstractmethod
    def stats(self) -> Mapping[str, Any]:
        """Return diagnostic info (rows, size_bytes, hit_rate, etc.)."""

    @abstractmethod
    def prune(self, *, max_size_bytes: int) -> None:
        """Delete oldest entries until on‑disk size ≤ *max_size_bytes*."""


# --------------------------------------------------------------------------- #
# Filesystem JSON back‑end (debug / tiny workloads)                             #
# --------------------------------------------------------------------------- #
class FilesystemJSONCache(SemanticCache):
    """Stores each entry in `<root>/<first4>/<key>.json.zst`.

    Pros: zero deps, inspectable. Cons: many inodes, slower for 50k+ rows.
    """

    def __init__(self, root: Path):
        log.debug("Initializing FilesystemJSONCache with root: %s", root)
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        log.debug("Cache root directory created/verified: %s", self.root)

        if zstd is None:
            log.debug("zstd not available, using no compression")
            self._compress = lambda b: b
            self._decompress = lambda b: b
        else:
            log.debug("zstd available, using compression level %d", _ZSTD_LEVEL)
            self._compress = zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress
            self._decompress = zstd.ZstdDecompressor().decompress

        self._hits = 0
        self._miss = 0
        log.debug("FilesystemJSONCache initialized successfully")

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.zst"

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        log.debug("FilesystemJSONCache get: key=%s, path=%s", key[:16] + "...", p)
        if p.exists():
            self._hits += 1
            data = self._decompress(p.read_bytes())
            log.debug(
                "FilesystemJSONCache hit: key=%s, data_size=%d bytes",
                key[:16] + "...",
                len(data),
            )
            return data
        self._miss += 1
        log.debug("FilesystemJSONCache miss: key=%s", key[:16] + "...")
        return None

    def set(
        self, key: str, value: bytes, meta: Mapping[str, Any] | None = None
    ) -> None:
        p = self._path(key)
        log.debug(
            "FilesystemJSONCache set: key=%s, path=%s, value_size=%d bytes",
            key[:16] + "...",
            p,
            len(value),
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        compressed = self._compress(value)
        p.write_bytes(compressed)
        log.debug(
            "FilesystemJSONCache stored: key=%s, compressed_size=%d bytes",
            key[:16] + "...",
            len(compressed),
        )
        # meta goes in a sidecar .meta for transparency
        if meta:
            meta_path = p.with_suffix(".meta.json")
            meta_path.write_text(_canonical_json(meta))
            log.debug(
                "FilesystemJSONCache stored metadata: key=%s, meta_path=%s",
                key[:16] + "...",
                meta_path,
            )

    def stats(self):
        log.debug("FilesystemJSONCache calculating stats")
        total = sum(1 for _ in self.root.rglob("*.zst"))
        size = sum(p.stat().st_size for p in self.root.rglob("*.zst"))
        stats = {
            "backend": "filesystem",
            "entries": total,
            "bytes": size,
            "hit": self._hits,
            "miss": self._miss,
        }
        log.debug("FilesystemJSONCache stats: %s", stats)
        return stats

    def prune(self, *, max_size_bytes: int):
        log.debug("FilesystemJSONCache pruning to max_size_bytes=%d", max_size_bytes)
        files = sorted(self.root.rglob("*.zst"), key=lambda p: p.stat().st_mtime)
        size = sum(p.stat().st_size for p in files)
        log.debug(
            "FilesystemJSONCache current size: %d bytes, %d files", size, len(files)
        )

        deleted_count = 0
        for p in files:
            if size <= max_size_bytes:
                break
            file_size = p.stat().st_size
            size -= file_size
            p.unlink(missing_ok=True)
            meta = p.with_suffix(".meta.json")
            meta.unlink(missing_ok=True)
            deleted_count += 1
            log.debug("FilesystemJSONCache deleted: %s (%d bytes)", p.name, file_size)

        log.debug(
            "FilesystemJSONCache pruning completed: deleted %d files, final size %d bytes",
            deleted_count,
            size,
        )


# --------------------------------------------------------------------------- #
# SQLite back‑end (default)                                                    #
# --------------------------------------------------------------------------- #
class SQLiteWALCache(SemanticCache):

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS cache (
        k   TEXT PRIMARY KEY,
        v   BLOB NOT NULL,
        ts  INTEGER DEFAULT (strftime('%s','now')),
        meta JSON
    );
    """

    def __init__(self, path: Path, synchronous: str = "NORMAL"):
        log.debug(
            "Initializing SQLiteWALCache with path: %s, synchronous: %s",
            path,
            synchronous,
        )
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log.debug(
            "SQLiteWALCache database directory created/verified: %s", self.path.parent
        )

        # Store configuration for per-thread connections
        self._synchronous = synchronous
        self._local = (
            threading.local()
        )  # Thread-local storage for connections and zstd objects

        # Initialize database schema with a temporary connection
        temp_db = sqlite3.connect(self.path, check_same_thread=False, timeout=120)
        temp_db.execute("PRAGMA journal_mode=WAL;")
        temp_db.execute(f"PRAGMA synchronous={synchronous};")
        temp_db.execute(self._CREATE_SQL)
        temp_db.commit()
        temp_db.close()
        log.debug("SQLiteWALCache database initialized: %s", self.path)

        # Store zstd availability for thread-local initialization
        self._zstd_available = zstd is not None
        if self._zstd_available:
            log.debug("zstd available, will use compression level %d", _ZSTD_LEVEL)
        else:
            log.debug("zstd not available, using no compression")

        self._lock = threading.Lock()  # protect writes; many readers ok
        self._hits = 0
        self._miss = 0
        log.debug("SQLiteWALCache initialized successfully")

    def _get_db(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, "db"):
            self._local.db = sqlite3.connect(
                self.path, check_same_thread=False, timeout=120
            )
            self._local.db.execute("PRAGMA journal_mode=WAL;")
            self._local.db.execute(f"PRAGMA synchronous={self._synchronous};")
        return self._local.db

    def _get_zstd_objects(self):
        """Get thread-local zstd compressor and decompressor."""
        if not hasattr(self._local, "zstd_compressor"):
            if self._zstd_available:
                self._local.zstd_compressor = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
                self._local.zstd_decompressor = zstd.ZstdDecompressor()
            else:
                self._local.zstd_compressor = None
                self._local.zstd_decompressor = None
        return self._local.zstd_compressor, self._local.zstd_decompressor

    def get(self, key: str) -> Optional[bytes]:
        log.debug("SQLiteWALCache get: key=%s", key[:16] + "...")
        db = self._get_db()
        row = db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        if row:
            self._hits += 1
            data = row[0]
            _, decompressor = self._get_zstd_objects()
            if decompressor:
                try:
                    decompressed = decompressor.decompress(data)
                    log.debug(
                        "SQLiteWALCache hit: key=%s, compressed_size=%d, decompressed_size=%d",
                        key[:16] + "...",
                        len(data),
                        len(decompressed),
                    )
                    return decompressed
                except Exception as e:
                    log.error(
                        "Cache error decompression error: %s, performing new extraction",
                        e,
                    )
                    return None
            else:
                log.debug(
                    "SQLiteWALCache hit: key=%s, data_size=%d",
                    key[:16] + "...",
                    len(data),
                )
                return data
        self._miss += 1
        log.debug("SQLiteWALCache miss: key=%s", key[:16] + "...")
        return None

    def set(
        self, key: str, value: bytes, meta: Mapping[str, Any] | None = None
    ) -> None:
        log.debug(
            "SQLiteWALCache set: key=%s, value_size=%d bytes",
            key[:16] + "...",
            len(value),
        )
        compressor, _ = self._get_zstd_objects()
        payload = compressor.compress(value) if compressor else value
        meta_json = _canonical_json(meta) if meta else None
        log.debug(
            "SQLiteWALCache compressed: key=%s, original_size=%d, compressed_size=%d",
            key[:16] + "...",
            len(value),
            len(payload),
        )

        with self._lock:
            db = self._get_db()
            db.execute(
                "INSERT OR REPLACE INTO cache (k, v, meta) VALUES (?, ?, ?)",
                (key, payload, meta_json),
            )
            db.commit()
        log.debug("SQLiteWALCache stored: key=%s", key[:16] + "...")

    def stats(self):
        log.debug("SQLiteWALCache calculating stats")
        db = self._get_db()
        rows = db.execute(
            "SELECT COUNT(*), IFNULL(SUM(LENGTH(v)),0) FROM cache"
        ).fetchone()
        stats = {
            "backend": "sqlite",
            "entries": rows[0],
            "bytes": rows[1],
            "hit": self._hits,
            "miss": self._miss,
            "file": str(self.path),
        }
        log.debug("SQLiteWALCache stats: %s", stats)
        return stats

    def prune(self, *, max_size_bytes: int):
        log.debug("SQLiteWALCache pruning to max_size_bytes=%d", max_size_bytes)
        db = self._get_db()
        cur = db.execute("SELECT IFNULL(SUM(LENGTH(v)),0) FROM cache")
        size = cur.fetchone()[0]
        log.debug("SQLiteWALCache current size: %d bytes", size)

        if size <= max_size_bytes:
            log.debug("SQLiteWALCache no pruning needed")
            return

        # delete oldest first
        deleted_batches = 0
        with self._lock:
            while size > max_size_bytes:
                db.execute(
                    "DELETE FROM cache WHERE k IN (SELECT k FROM cache ORDER BY ts ASC LIMIT 1000)"
                )
                db.commit()
                deleted_batches += 1
                size = db.execute(
                    "SELECT IFNULL(SUM(LENGTH(v)),0) FROM cache"
                ).fetchone()[0]
                log.debug(
                    "SQLiteWALCache pruning batch %d: size now %d bytes",
                    deleted_batches,
                    size,
                )

        log.debug(
            "SQLiteWALCache pruning completed: deleted %d batches, final size %d bytes",
            deleted_batches,
            size,
        )

    def close(self):
        """Close all thread-local database connections and clean up zstd objects."""
        if hasattr(self._local, "db"):
            self._local.db.close()
            delattr(self._local, "db")
        # Clean up zstd objects (they don't need explicit cleanup, but we can clear them)
        if hasattr(self._local, "zstd_compressor"):
            delattr(self._local, "zstd_compressor")
        if hasattr(self._local, "zstd_decompressor"):
            delattr(self._local, "zstd_decompressor")


# --------------------------------------------------------------------------- #
# LMDB back‑end (fast path)                                                    #
# --------------------------------------------------------------------------- #
class LMDBCache(SemanticCache):
    def __init__(self, path: Path, map_size_mb: int = 1024):
        log.debug(
            "Initializing LMDBCache with path: %s, map_size_mb: %d", path, map_size_mb
        )
        if lmdb is None:
            log.error("lmdb package not installed")
            raise ImportError("lmdb package not installed. `pip install lmdb`.")

        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log.debug("LMDBCache directory created/verified: %s", self.path.parent)

        self.env = lmdb.open(
            str(self.path),
            map_size=map_size_mb * 1024 * 1024,
            lock=True,
            writemap=True,
            max_dbs=1,
        )
        log.debug("LMDBCache environment opened: %s", self.path)

        if zstd is None:
            log.debug("zstd not available, using no compression")
            self._c = None
            self._d = None
        else:
            log.debug("zstd available, using compression level %d", _ZSTD_LEVEL)
            self._c = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
            self._d = zstd.ZstdDecompressor()

        self._hits = 0
        self._miss = 0
        log.debug("LMDBCache initialized successfully")

    def get(self, key: str) -> Optional[bytes]:
        log.debug("LMDBCache get: key=%s", key[:16] + "...")
        with self.env.begin(buffers=True) as txn:
            val = txn.get(key.encode("utf‑8"))
            if val is None:
                self._miss += 1
                log.debug("LMDBCache miss: key=%s", key[:16] + "...")
                return None
            self._hits += 1
            if self._d:
                decompressed = self._d.decompress(val)
                log.debug(
                    "LMDBCache hit: key=%s, compressed_size=%d, decompressed_size=%d",
                    key[:16] + "...",
                    len(val),
                    len(decompressed),
                )
                return decompressed
            else:
                log.debug(
                    "LMDBCache hit: key=%s, data_size=%d", key[:16] + "...", len(val)
                )
                return bytes(val)

    def set(self, key: str, value: bytes, meta: Mapping[str, Any] | None = None):
        log.debug(
            "LMDBCache set: key=%s, value_size=%d bytes", key[:16] + "...", len(value)
        )
        payload = self._c.compress(value) if self._c else value
        log.debug(
            "LMDBCache compressed: key=%s, original_size=%d, compressed_size=%d",
            key[:16] + "...",
            len(value),
            len(payload),
        )

        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf‑8"), payload, overwrite=True)
        log.debug("LMDBCache stored: key=%s", key[:16] + "...")

    def stats(self):
        log.debug("LMDBCache calculating stats")
        stat = self.env.stat()
        stats = {
            "backend": "lmdb",
            "entries": stat["entries"],
            "map_size": self.env.info()["map_size"],
            "hit": self._hits,
            "miss": self._miss,
            "file": str(self.path),
        }
        log.debug("LMDBCache stats: %s", stats)
        return stats

    def prune(self, *, max_size_bytes: int):
        log.debug("LMDBCache prune called (not implemented)")
        # LMDB doesn't auto‑prune; we simply skip (user can drop & recreate).
        pass


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
class SemanticCacheFactory:
    """Create a cache instance from a config mapping (dict or attr‑access)."""

    @staticmethod
    def from_config(cfg) -> SemanticCache:
        log.debug("Creating SemanticCache from config: %s", cfg)

        if cfg is None:
            cfg_dict = {}
            log.debug("Config is None, using defaults")
        elif is_dataclass(cfg) and not isinstance(cfg, type):
            cfg_dict = asdict(cfg)
            log.debug("Config is dataclass, converted to dict")
        elif isinstance(cfg, dict):
            cfg_dict = cfg
            log.debug("Config is dict")
        else:
            log.error("Unknown cache config type: %s", type(cfg))
            raise ValueError(f"Unknown cache config type: {type(cfg)}")

        backend = cfg_dict.get("backend", "sqlite").lower()
        path = Path(cfg_dict.get("path", ".delm_cache"))
        log.debug("Cache config: backend=%s, path=%s", backend, path)

        if backend == "filesystem":
            log.debug("Creating FilesystemJSONCache")
            return FilesystemJSONCache(path)
        if backend == "sqlite":
            synchronous = cfg_dict.get("synchronous", "NORMAL").upper()
            log.debug("Creating SQLiteWALCache with synchronous=%s", synchronous)
            return SQLiteWALCache(path / "semantic.db", synchronous=synchronous)
        if backend == "lmdb":
            map_size_mb = cfg_dict.get("map_size_mb", 1024)
            log.debug("Creating LMDBCache with map_size_mb=%d", map_size_mb)
            return LMDBCache(path / "semantic.lmdb", map_size_mb=map_size_mb)

        log.error("Unknown cache backend: %s", backend)
        raise ValueError(f"Unknown cache backend: {backend}")


# --------------------------------------------------------------------------- #
# Convenience CLI hooks (optional)                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, shutil, textwrap

    ap = argparse.ArgumentParser(description="Inspect or prune DELM semantic cache")
    ap.add_argument("cache_dir", type=Path, help="Path to cache directory")
    ap.add_argument(
        "--backend", default="sqlite", choices=["sqlite", "lmdb", "filesystem"]
    )
    ap.add_argument("--stats", action="store_true", help="Show stats and exit")
    ap.add_argument(
        "--prune", type=int, metavar="MEGABYTES", help="Prune to <= this many MB"
    )
    ns = ap.parse_args()

    cache = SemanticCacheFactory.from_config(
        {"backend": ns.backend, "path": ns.cache_dir}
    )
    if ns.stats:
        print(json.dumps(cache.stats(), indent=2))
    if ns.prune is not None:
        cache.prune(max_size_bytes=ns.prune * 1024 * 1024)
        print("Pruned.")
