# src/pyfundlib/utils/cache.py
from __future__ import annotations

import hashlib
import logging
import pickle
import zlib
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


def stable_hash(obj: Any) -> str:
    """Deterministic SHA256 hash for any picklable object"""
    pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(pickled)
    return hashlib.sha256(compressed).hexdigest()


class CachedFunction:
    """
    Elite-level disk cache with:
    - Custom key via key_lambda
    - Dynamic TTL via expire_seconds (int or lambda)
    - Compression + size limits
    - Cache clearing
    """

    def __init__(
        self,
        dir_name: str,
        *,
        key_lambda: Callable[..., str] | None = None,
        expire_seconds: int | Callable[..., int] | None = None,
        expire_days: int | None = 30,
        max_size_mb: int | None = 5000,
        compress: bool = True,
    ):
        self.cache_dir = Path("./cache") / dir_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.key_lambda = key_lambda
        self.expire_seconds = expire_seconds
        self.expire_days = expire_days
        self.max_size_mb = max_size_mb
        self.compress = compress

        if max_size_mb:
            self._enforce_size_limit()

    def _get_ttl(self, **kwargs) -> int:
        """Resolve TTL in seconds"""
        if callable(self.expire_seconds):
            return self.expire_seconds(**kwargs)
        if self.expire_seconds is not None:
            return self.expire_seconds
        if self.expire_days is not None:
            return self.expire_days * 24 * 60 * 60
        return 30 * 24 * 60 * 60  # 30 days default

    def _get_cache_path(self, key: str) -> Path:
        suffix = ".pklz" if self.compress else ".pkl"
        return self.cache_dir / f"{key}{suffix}"

    def _enforce_size_limit(self):
        files = sorted(
            self.cache_dir.iterdir(),
            key=lambda p: p.stat().st_mtime
        )
        total_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)

        while total_mb > (self.max_size_mb or 0) and len(files) > 10:
            oldest = files.pop(0)
            size_mb = oldest.stat().st_size / (1024 ** 2)
            oldest.unlink(missing_ok=True)
            total_mb -= size_mb
            logger.debug(f"Cache pruned: {oldest.name} ({size_mb:.1f}MB)")

    def _is_expired(self, filepath: Path, **kwargs) -> bool:
        ttl = self._get_ttl(**kwargs)
        if ttl <= 0:
            return True  # 0 or negative = never cache
        age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        return age.total_seconds() > ttl

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if self.key_lambda:
                cache_key = self.key_lambda(*args, **kwargs)
            else:
                key_obj = {"func": func.__name__, "args": args, "kwargs": kwargs}
                cache_key = stable_hash(key_obj)

            cache_file = self._get_cache_path(cache_key)

            # Cache hit?
            if cache_file.exists() and not self._is_expired(cache_file, **kwargs):
                try:
                    with open(cache_file, "rb") as f:
                        data = f.read()
                        if self.compress:
                            data = zlib.decompress(data)
                        result = pickle.loads(data)
                    logger.debug(f"Cache HIT → {cache_file.name}")
                    return result
                except Exception as e:
                    logger.warning(f"Cache corrupted, refetching: {e}")

            # Cache miss
            logger.debug(f"Cache MISS → computing {func.__name__}")
            result = func(*args, **kwargs)

            # Save
            try:
                data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                if self.compress:
                    data = zlib.compress(data)
                with open(cache_file, "wb") as f:
                    f.write(data)
                if self.max_size_mb:
                    self._enforce_size_limit()
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")

            return result

        # Attach utilities
        wrapper_any: Any = wrapper
        wrapper_any.cache_dir = self.cache_dir
        wrapper_any.clear = lambda: [f.unlink(missing_ok=True) for f in self.cache_dir.glob("*")]
        wrapper_any.size_mb = lambda: sum(f.stat().st_size for f in self.cache_dir.iterdir()) / (1024**2)

        return wrapper


# Convenience factory — now supports key_lambda and expire_seconds!
def cached_function(
    dir_name: str,
    *,
    key_lambda: Callable[..., str] | None = None,
    expire_seconds: int | Callable[..., int] | None = None,
    expire_days: int | None = None,
    max_size_mb: int | None = 5000,
    compress: bool = True,
):
    return CachedFunction(
        dir_name=dir_name,
        key_lambda=key_lambda,
        expire_seconds=expire_seconds,
        expire_days=expire_days,
        max_size_mb=max_size_mb,
        compress=compress,
    )