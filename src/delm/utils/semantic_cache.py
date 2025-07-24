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

from dataclasses import asdict, is_dataclass
import json
import hashlib
import time
import sqlite3
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional

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
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_semantic_key(material: Mapping[str, Any]) -> str:
    """Hash canonical JSON material to a 64‑char hex string."""
    digest = hashlib.sha256(_canonical_json(material).encode("utf‑8")).hexdigest()
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
    material = {
        "prompt"   : prompt_text,
        "system"   : system_prompt,
        "model"    : model_name,
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
    def set(self, key: str, value: bytes, meta: Mapping[str, Any] | None = None) -> None:  # noqa: E501
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
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._compress = (lambda b: b) if zstd is None else zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress  # noqa: E501
        self._decompress = (lambda b: b) if zstd is None else zstd.ZstdDecompressor().decompress
        self._hits = 0
        self._miss = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.zst"

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        if p.exists():
            self._hits += 1
            return self._decompress(p.read_bytes())
        self._miss += 1
        return None

    def set(self, key: str, value: bytes, meta: Mapping[str, Any] | None = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self._compress(value))
        # meta goes in a sidecar .meta for transparency
        if meta:
            p.with_suffix(".meta.json").write_text(_canonical_json(meta))

    def stats(self):
        total = sum(1 for _ in self.root.rglob("*.zst"))
        size = sum(p.stat().st_size for p in self.root.rglob("*.zst"))
        return {"backend": "filesystem", "entries": total, "bytes": size, "hit": self._hits, "miss": self._miss}

    def prune(self, *, max_size_bytes: int):
        files = sorted(self.root.rglob("*.zst"), key=lambda p: p.stat().st_mtime)
        size = sum(p.stat().st_size for p in files)
        for p in files:
            if size <= max_size_bytes:
                break
            size -= p.stat().st_size
            p.unlink(missing_ok=True)
            meta = p.with_suffix(".meta.json")
            meta.unlink(missing_ok=True)

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
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute(f"PRAGMA synchronous={synchronous};")
        self.db.execute(self._CREATE_SQL)
        self.db.commit()
        self._c = zstd.ZstdCompressor(level=_ZSTD_LEVEL) if zstd else None
        self._d = zstd.ZstdDecompressor() if zstd else None
        self._lock = threading.Lock()  # protect writes; many readers ok
        self._hits = 0
        self._miss = 0

    def get(self, key: str) -> Optional[bytes]:
        row = self.db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        if row:
            self._hits += 1
            data = row[0]
            return self._d.decompress(data) if self._d else data
        self._miss += 1
        return None

    def set(self, key: str, value: bytes, meta: Mapping[str, Any] | None = None) -> None:
        payload = self._c.compress(value) if self._c else value
        meta_json = _canonical_json(meta) if meta else None
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO cache (k, v, meta) VALUES (?, ?, ?)",
                (key, payload, meta_json),
            )
            self.db.commit()

    def stats(self):
        rows = self.db.execute("SELECT COUNT(*), IFNULL(SUM(LENGTH(v)),0) FROM cache").fetchone()
        return {
            "backend": "sqlite",
            "entries": rows[0],
            "bytes": rows[1],
            "hit": self._hits,
            "miss": self._miss,
            "file": str(self.path),
        }

    def prune(self, *, max_size_bytes: int):
        cur = self.db.execute("SELECT IFNULL(SUM(LENGTH(v)),0) FROM cache")
        size = cur.fetchone()[0]
        if size <= max_size_bytes:
            return
        # delete oldest first
        with self._lock:
            while size > max_size_bytes:
                self.db.execute("DELETE FROM cache WHERE k IN (SELECT k FROM cache ORDER BY ts ASC LIMIT 1000)")
                self.db.commit()
                size = self.db.execute("SELECT IFNULL(SUM(LENGTH(v)),0) FROM cache").fetchone()[0]

# --------------------------------------------------------------------------- #
# LMDB back‑end (fast path)                                                    #
# --------------------------------------------------------------------------- #
class LMDBCache(SemanticCache):
    def __init__(self, path: Path, map_size_mb: int = 1024):
        if lmdb is None:
            raise RuntimeError("lmdb package not installed. `pip install lmdb`.")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            map_size=map_size_mb * 1024 * 1024,
            lock=True,
            writemap=True,
            max_dbs=1,
        )
        self._c = zstd.ZstdCompressor(level=_ZSTD_LEVEL) if zstd else None
        self._d = zstd.ZstdDecompressor() if zstd else None
        self._hits = 0
        self._miss = 0

    def get(self, key: str) -> Optional[bytes]:
        with self.env.begin(buffers=True) as txn:
            val = txn.get(key.encode("utf‑8"))
            if val is None:
                self._miss += 1
                return None
            self._hits += 1
            return self._d.decompress(val) if self._d else bytes(val)

    def set(self, key: str, value: bytes, meta: Mapping[str, Any] | None = None):
        payload = self._c.compress(value) if self._c else value
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf‑8"), payload, overwrite=True)

    def stats(self):
        stat = self.env.stat()
        return {
            "backend": "lmdb",
            "entries": stat["entries"],
            "map_size": self.env.info()["map_size"],
            "hit": self._hits,
            "miss": self._miss,
            "file": str(self.path),
        }

    def prune(self, *, max_size_bytes: int):
        # LMDB doesn't auto‑prune; we simply skip (user can drop & recreate).
        pass

# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
class SemanticCacheFactory:
    """Create a cache instance from a config mapping (dict or attr‑access)."""

    @staticmethod
    def from_config(cfg) -> SemanticCache:
        if cfg is None:
            cfg_dict = {}
        elif is_dataclass(cfg) and not isinstance(cfg, type):
            cfg_dict = asdict(cfg) 
        elif isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            raise ValueError(f"Unknown cache config type: {type(cfg)}")
        backend = cfg_dict.get("backend", "sqlite").lower()
        path = Path(cfg_dict.get("path", ".delm_cache"))
        if backend == "filesystem":
            return FilesystemJSONCache(path)
        if backend == "sqlite":
            synchronous = cfg_dict.get("synchronous", "NORMAL").upper()
            return SQLiteWALCache(path / "semantic.db", synchronous=synchronous)
        if backend == "lmdb":
            return LMDBCache(path / "semantic.lmdb", map_size_mb=cfg_dict.get("map_size_mb", 1024))
        raise ValueError(f"Unknown cache backend: {backend}")


# --------------------------------------------------------------------------- #
# Convenience CLI hooks (optional)                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, shutil, textwrap

    ap = argparse.ArgumentParser(description="Inspect or prune DELM semantic cache")
    ap.add_argument("cache_dir", type=Path, help="Path to cache directory")
    ap.add_argument("--backend", default="sqlite", choices=["sqlite", "lmdb", "filesystem"])
    ap.add_argument("--stats", action="store_true", help="Show stats and exit")
    ap.add_argument("--prune", type=int, metavar="MEGABYTES", help="Prune to <= this many MB")
    ns = ap.parse_args()

    cache = SemanticCacheFactory.from_config({"backend": ns.backend, "path": ns.cache_dir})
    if ns.stats:
        print(json.dumps(cache.stats(), indent=2))
    if ns.prune is not None:
        cache.prune(max_size_bytes=ns.prune * 1024 * 1024)
        print("Pruned.")
