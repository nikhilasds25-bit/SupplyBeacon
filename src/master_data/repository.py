from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.dynamic_risk.current_state import BASE_DIR


def db_path() -> Path:
    return Path(os.getenv("SUPPLYBEACON_MASTER_DB_PATH", str(BASE_DIR / "data" / "master" / "supplybeacon_master.db")))


def generated_dir() -> Path:
    return Path(os.getenv("SUPPLYBEACON_MASTER_GENERATED_DIR", str(BASE_DIR / "data" / "master" / "generated")))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    initialize(connection)
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript("""
    CREATE TABLE IF NOT EXISTS products (
      id TEXT PRIMARY KEY, external_id TEXT, name TEXT NOT NULL, sku TEXT, brand TEXT, category TEXT,
      status TEXT NOT NULL, data_origin TEXT NOT NULL, entry_method TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS suppliers (
      id TEXT PRIMARY KEY, external_id TEXT, name TEXT NOT NULL, supplier_code TEXT, location_id TEXT, status TEXT NOT NULL,
      data_origin TEXT NOT NULL, entry_method TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS warehouses (
      id TEXT PRIMARY KEY, external_id TEXT, name TEXT NOT NULL, warehouse_code TEXT, location_id TEXT, capacity REAL,
      capacity_unit TEXT, status TEXT NOT NULL, data_origin TEXT NOT NULL, entry_method TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS locations (
      id TEXT PRIMARY KEY, external_id TEXT, name TEXT NOT NULL, city TEXT, region TEXT, country TEXT, status TEXT NOT NULL,
      data_origin TEXT NOT NULL, entry_method TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS relationships (
      id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_type TEXT NOT NULL, relationship_type TEXT NOT NULL,
      target_id TEXT NOT NULL, target_type TEXT NOT NULL, status TEXT NOT NULL, data_origin TEXT NOT NULL, entry_method TEXT NOT NULL,
      created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS change_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
      action TEXT NOT NULL, changed_fields TEXT, entry_method TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS import_batches (
      id TEXT PRIMARY KEY, import_type TEXT NOT NULL, status TEXT NOT NULL, total_rows INTEGER NOT NULL,
      valid_rows INTEGER NOT NULL, warning_rows INTEGER NOT NULL, invalid_rows INTEGER NOT NULL, duplicate_warnings INTEGER NOT NULL,
      payload TEXT NOT NULL, created_at TEXT NOT NULL, confirmed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS master_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    defaults = {"published_version": "0", "published_at": "", "last_published_change_id": "0"}
    for key, value in defaults.items():
        connection.execute("INSERT OR IGNORE INTO master_meta(key,value) VALUES(?,?)", (key, value))
    connection.commit()


def table_for(entity_type: str) -> str:
    table = {"Product": "products", "Supplier": "suppliers", "Warehouse": "warehouses", "Location": "locations"}.get(entity_type)
    if not table:
        raise ValueError("Unsupported entity type")
    return table


def row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def all_rows(connection: sqlite3.Connection, entity_type: str) -> list[dict[str, Any]]:
    return [row_dict(row) for row in connection.execute(f"SELECT * FROM {table_for(entity_type)} ORDER BY name")]


def get_meta(connection: sqlite3.Connection, key: str, default: str = "") -> str:
    row = connection.execute("SELECT value FROM master_meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute("INSERT OR REPLACE INTO master_meta(key,value) VALUES(?,?)", (key, value))


def audit(connection: sqlite3.Connection, entity_type: str, entity_id: str, action: str, fields: list[str], entry_method: str) -> None:
    connection.execute(
        "INSERT INTO change_log(timestamp,entity_type,entity_id,action,changed_fields,entry_method) VALUES(?,?,?,?,?,?)",
        (now(), entity_type, entity_id, action, json.dumps(fields), entry_method),
    )


def new_id(prefix: str) -> str:
    return f"{prefix}:COMPANY:{uuid4()}"
