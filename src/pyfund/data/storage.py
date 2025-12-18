# src/pyfund/data/storage.py
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_BASE_PATH = Path("./cache/parquet")


class DataStorage:
    """
    High-performance, versioned Parquet storage for financial time-series data.

    - Writes Parquet with schema metadata (encoded as bytes) and atomic replace.
    - Saves DatetimeIndex under the 'Date' index name (if present).
    - Simple load with optional column/time filtering.
    - Stable DataFrame content hash for cache invalidation.
    """

    def __init__(
        self,
        base_path: str | Path = DEFAULT_BASE_PATH,
        partition_by: Optional[str] = "ticker",  # None, "ticker", or "year/month" (not fully implemented)
        compression: str = "zstd",
    ) -> None:
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.partition_by = partition_by
        self.compression = compression

    def _get_path(self, name: str, ticker: Optional[str] = None) -> Path:
        """Resolve final storage path with optional partitioning by ticker."""
        if self.partition_by == "ticker" and ticker:
            path = self.base_path / ticker
        elif self.partition_by == "year/month" and ticker:
            # simple partition: keep under ticker; user can extend to year/month subfolders
            path = self.base_path / ticker
        else:
            path = self.base_path

        path.mkdir(parents=True, exist_ok=True)
        return path / f"{name}.parquet"

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        ticker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Save DataFrame to Parquet with rich metadata and atomic write.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save (prefer DatetimeIndex)
        name : str
            Logical file name (without extension)
        ticker : Optional[str]
            Optional ticker partition folder
        metadata : Optional[dict]
            Additional metadata to attach
        overwrite : bool
            If False and file exists -> raise FileExistsError
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            raise ValueError("Cannot save empty DataFrame")

        # Ensure DatetimeIndex name for portability
        df_to_save = df.copy()
        if isinstance(df_to_save.index, pd.DatetimeIndex):
            df_to_save.index.name = df_to_save.index.name or "Date"

        # Build metadata
        default_metadata: Dict[str, Any] = {
            "pyfund_version": "0.1.0",
            "saved_at": datetime.utcnow().isoformat(),
            "ticker": ticker or "unknown",
            "rows": int(len(df_to_save)),
            "columns": list(df_to_save.columns),
            "start_date": str(df_to_save.index.min()) if isinstance(df_to_save.index, pd.DatetimeIndex) else None,
            "end_date": str(df_to_save.index.max()) if isinstance(df_to_save.index, pd.DatetimeIndex) else None,
            "data_hash": self._df_hash(df_to_save),
        }

        if metadata:
            # merge without losing types (we'll stringify later)
            default_metadata.update(metadata)

        path = self._get_path(name, ticker=ticker)

        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists and overwrite=False")

        # Convert metadata keys/values to bytes mapping for pyarrow
        metadata_bytes = {str(k).encode(): str(v).encode() for k, v in default_metadata.items()}

        # Create Arrow table
        table = pa.Table.from_pandas(df_to_save, preserve_index=True)

        # Merge with existing schema metadata if present (prefer our metadata values)
        existing_meta = table.schema.metadata or {}
        merged_meta = dict(existing_meta)  # copy (keys and values are bytes)
        merged_meta.update(metadata_bytes)

        table = table.replace_schema_metadata(merged_meta)

        # Atomic write: write to temp file then move
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet", prefix=f".{name}-", dir=str(self.base_path))
        Path(tmp_path).unlink(missing_ok=True)  # ensure clean
        try:
            # Use pyarrow writer to write table
            pq.write_table(
                table,
                tmp_path,
                compression=self.compression,
                use_dictionary=True,
                write_statistics=True,
            )
            # ensure destination directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            Path(tmp_path).replace(path)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                # tmp may already be moved; ignore
                pass

        return path

    def load(
        self,
        name: str,
        ticker: Optional[str] = None,
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a DataFrame saved by `save`. Optional column and time filtering.

        Notes:
        - For date filtering we read the file and filter in pandas. This is safe and
          keeps implementation simple and cross-platform.
        """
        path = self._get_path(name, ticker=ticker)
        if not path.exists():
            raise FileNotFoundError(f"No data found at {path}")

        # Read the parquet file (pandas will use pyarrow engine if available)
        df = pd.read_parquet(path, columns=columns)

        # If the index was preserved as a column named 'Date', restore it
        if "Date" in df.columns:
            try:
                df = df.set_index("Date")
                df.index = pd.to_datetime(df.index)
            except Exception:
                # leave as-is if conversion fails
                pass
        # If index is not datetime and user requested date filtering, try converting index
        if (start_date or end_date) and not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                # If conversion fails, attempt to use 'Date' column if present
                if "Date" in df.columns:
                    df = df.set_index("Date")
                    df.index = pd.to_datetime(df.index)

        # Apply date filters if requested
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df.sort_index()

    def exists(self, name: str, ticker: Optional[str] = None) -> bool:
        """Return True if dataset file exists."""
        return self._get_path(name, ticker=ticker).exists()

    def list_datasets(self) -> List[str]:
        """Return list of dataset file stems in base_path (recursively)."""
        files = list(self.base_path.rglob("*.parquet"))
        stems = sorted({p.stem for p in files})
        return stems

    def delete(self, name: str, ticker: Optional[str] = None) -> bool:
        """Delete a dataset file if it exists. Returns True if removed."""
        path = self._get_path(name, ticker=ticker)
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        return False

    @staticmethod
    def _df_hash(df: pd.DataFrame) -> str:
        """
        Stable content hash for a DataFrame.

        Converts hash_pandas_object result to a numpy array to guarantee .tobytes().
        """
        # Hash values include index if requested
        hashed = pd.util.hash_pandas_object(df, index=True)

        # Ensure numpy ndarray backing so .tobytes() exists (silences type-checkers)
        arr = hashed.to_numpy(dtype="uint64", copy=False)

        # Use md5 over the raw bytes (fast and stable)
        return hashlib.md5(arr.tobytes()).hexdigest()

    # Context manager support
    def __enter__(self) -> "DataStorage":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Nothing to clean up for now (placeholder for future resources)
        return None
