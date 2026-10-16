"""Load DataCo files without coupling them to the production dataset pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
DATACO_DIR = BASE_DIR / "data" / "raw" / "dataco"

EXPECTED_FILES = (
    "DataCoSupplyChainDataset.csv",
    "DescriptionDataCoSupplyChain.csv",
    "tokenized_access_logs.csv",
)


def find_file(file_name: str, directory: Path = DATACO_DIR) -> Optional[Path]:
    """Find a DataCo file case-insensitively, if it exists."""
    exact = directory / file_name
    if exact.exists():
        return exact
    if not directory.exists():
        return None
    matches = [path for path in directory.iterdir() if path.name.lower() == file_name.lower()]
    return sorted(matches)[0] if matches else None


def discover_files(directory: Path = DATACO_DIR) -> Dict[str, Optional[Path]]:
    """Return paths for expected DataCo files, preserving source isolation."""
    return {name: find_file(name, directory) for name in EXPECTED_FILES}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a DataCo CSV with common encoding fallbacks."""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("dataco", b"", 0, 1, "; ".join(errors))


def load_main_dataset(directory: Path = DATACO_DIR) -> tuple[pd.DataFrame, Optional[Path]]:
    """Load the operational DataCo table, or return an empty frame if unavailable."""
    path = find_file("DataCoSupplyChainDataset.csv", directory)
    if path is None:
        LOGGER.warning("DataCo operational dataset is not available under %s", directory)
        return pd.DataFrame(), None
    return read_csv(path), path
