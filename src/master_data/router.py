from __future__ import annotations

import csv
import io
import json
import os
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .graph_overlay import build_overlay
from .repository import all_rows, audit, connect, db_path, get_meta, new_id, now, set_meta
from .validators import ALLOWED, brand_id, brand_registry, duplicate_candidates, validate_entity, validate_network

router = APIRouter(prefix="/master", tags=["master-data"])


class EntityRequest(BaseModel):
    external_id: str | None = None
    name: str | None = None
    sku: str | None = None
    brand: str | None = None
    category: str | None = None
    supplier_code: str | None = None
    warehouse_code: str | None = None
    location_id: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    capacity: float | None = None
    capacity_unit: str | None = None
    status: str = "ACTIVE"
    continue_on_duplicate: bool = False


class RelationshipRequest(BaseModel):
    source_id: str
    source_type: str
    relationship_type: str
    target_id: str
    target_type: str
    status: str = "ACTIVE"


def _write_enabled() -> bool:
    return os.getenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "false").lower() == "true"


def _require_write() -> None:
    if not _write_enabled():
        raise HTTPException(status_code=403, detail="Master-data writes are disabled. Enable SUPPLYBEACON_MASTER_WRITE_ENABLED for local development only.")


def _entity_payload(entity_type: str, request: EntityRequest, entry_method: str = "FORM") -> dict[str, Any]:
    fields = {"Product": ["external_id", "name", "sku", "brand", "category", "status"], "Supplier": ["external_id", "name", "supplier_code", "location_id", "status"], "Warehouse": ["external_id", "name", "warehouse_code", "location_id", "capacity", "capacity_unit", "status"], "Location": ["external_id", "name", "city", "region", "country", "status"]}[entity_type]
    values = request.model_dump()
    timestamp = now()
    values.update({"id": new_id(entity_type.upper()), "data_origin": "USER_ENTERED", "entry_method": entry_method, "created_at": timestamp, "updated_at": timestamp})
    return {key: values.get(key) for key in ["id", *fields, "data_origin", "entry_method", "created_at", "updated_at"]}


def _duplicate_reasons(entity_type: str, record: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    reasons = []
    if record.get("external_id") and record.get("external_id") == candidate.get("external_id"):
        reasons.append("matching external ID")
    if entity_type == "Product" and record.get("sku") and record.get("sku") == candidate.get("sku"):
        reasons.append("matching SKU")
    if record.get("name") and str(record["name"]).strip().lower() == str(candidate.get("name") or "").strip().lower():
        reasons.append("matching normalized name")
    if entity_type == "Location" and all(str(record.get(key) or "").strip().lower() == str(candidate.get(key) or "").strip().lower() for key in ("city", "region", "country")):
        reasons.append("matching city, region and country")
    return reasons


def _entity_type_from_path(path: str) -> str:
    return path.rstrip("/").split("/")[-1].rstrip("s").capitalize()


def _sync_warehouse_location(connection: Any, warehouse_id: str, location_id: str | None, entry_method: str) -> None:
    if location_id and not connection.execute("SELECT 1 FROM locations WHERE id=? AND status='ACTIVE'", (location_id,)).fetchone():
        raise HTTPException(status_code=422, detail="Warehouse location is unknown or inactive")
    previous = connection.execute(
        "SELECT id, target_id FROM relationships WHERE source_id=? AND source_type='Warehouse' AND relationship_type='LOCATED_IN' AND target_type='Location' AND status='ACTIVE'",
        (warehouse_id,),
    ).fetchall()
    if previous:
        connection.execute(
            "UPDATE relationships SET status='INACTIVE', updated_at=? WHERE source_id=? AND source_type='Warehouse' AND relationship_type='LOCATED_IN' AND target_type='Location' AND status='ACTIVE'",
            (now(), warehouse_id),
        )
        for row in previous:
            audit(connection, "Relationship", row["id"], "RELATIONSHIP_DEACTIVATE", ["status"], entry_method)
    if location_id:
        relationship_id = new_id("RELATIONSHIP")
        timestamp = now()
        values = {
            "id": relationship_id,
            "source_id": warehouse_id,
            "source_type": "Warehouse",
            "relationship_type": "LOCATED_IN",
            "target_id": location_id,
            "target_type": "Location",
            "status": "ACTIVE",
            "data_origin": "USER_ENTERED",
            "entry_method": entry_method,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        columns = list(values)
        connection.execute(f"INSERT INTO relationships({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", [values[column] for column in columns])
        audit(connection, "Relationship", relationship_id, "RELATIONSHIP_CREATE", columns, entry_method)


@router.get("/status")
def status() -> dict[str, Any]:
    connection = connect()
    try:
        validation = validate_network(connection)
        published = get_meta(connection, "published_version", "0")
        changes = int(connection.execute("SELECT COUNT(*) FROM change_log WHERE id > ?", (get_meta(connection, "last_published_change_id", "0"),)).fetchone()[0])
        from .repository import generated_dir
        counts = {table: connection.execute(f"SELECT COUNT(*) FROM {table} WHERE status='ACTIVE'").fetchone()[0] for table in ("products", "suppliers", "warehouses", "locations")}
        last_modified = connection.execute("SELECT MAX(updated_at) FROM (SELECT updated_at FROM products UNION ALL SELECT updated_at FROM suppliers UNION ALL SELECT updated_at FROM warehouses UNION ALL SELECT updated_at FROM locations UNION ALL SELECT updated_at FROM relationships)").fetchone()[0]
        return {"write_enabled": _write_enabled(), "database_initialized": db_path().exists(), "draft_change_count": changes, "published_version": int(published), "published_at": get_meta(connection, "published_at"), "last_modified": last_modified, "overlay_available": (generated_dir() / "company_overlay.graphml").exists(), "validation_error_count": len(validation["errors"]), "validation_warning_count": len(validation["warnings"]), "entity_counts": {**counts, "relationships": connection.execute("SELECT COUNT(*) FROM relationships WHERE status='ACTIVE'").fetchone()[0]}}
    finally:
        connection.close()


@router.get("/products")
@router.get("/suppliers")
@router.get("/warehouses")
@router.get("/locations")
def list_entities(http_request: Request) -> list[dict[str, Any]]:
    entity_type = _entity_type_from_path(http_request.url.path)
    try:
        connection = connect()
        try:
            return all_rows(connection, entity_type.rstrip("s").capitalize())
        finally:
            connection.close()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/brands")
def list_brands() -> list[dict[str, Any]]:
    connection = connect()
    try:
        return list(brand_registry(connection).values())
    finally:
        connection.close()


@router.post("/products")
@router.post("/suppliers")
@router.post("/warehouses")
@router.post("/locations")
def create_entity(http_request: Request, request: EntityRequest) -> dict[str, Any]:
    _require_write()
    entity_type = _entity_type_from_path(http_request.url.path)
    canonical = entity_type.rstrip("s").capitalize()
    values = _entity_payload(canonical, request)
    errors = validate_entity(canonical, values)
    connection = connect()
    try:
        existing = all_rows(connection, canonical)
        duplicates = duplicate_candidates(existing, canonical, values)
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        if duplicates and not request.continue_on_duplicate:
            return {"status": "POSSIBLE DUPLICATE", "duplicate_candidates": duplicates, "duplicate_reasons": [_duplicate_reasons(canonical, values, candidate) for candidate in duplicates]}
        table = {"Product": "products", "Supplier": "suppliers", "Warehouse": "warehouses", "Location": "locations"}[canonical]
        columns = list(values)
        connection.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", [values[column] for column in columns])
        if canonical == "Warehouse":
            _sync_warehouse_location(connection, values["id"], values.get("location_id"), "FORM")
        audit(connection, canonical, values["id"], "DUPLICATE_OVERRIDE" if duplicates else "CREATE", columns, "FORM")
        connection.commit()
        return values
    finally:
        connection.close()


@router.put("/products/{entity_id}")
@router.put("/suppliers/{entity_id}")
@router.put("/warehouses/{entity_id}")
@router.put("/locations/{entity_id}")
def update_entity(http_request: Request, entity_id: str, request: EntityRequest) -> dict[str, Any]:
    _require_write()
    entity_type = _entity_type_from_path(http_request.url.path.split(f"/{entity_id}")[0])
    canonical = entity_type.rstrip("s").capitalize()
    connection = connect()
    try:
        table = {"Product": "products", "Supplier": "suppliers", "Warehouse": "warehouses", "Location": "locations"}.get(canonical)
        if not table or not connection.execute(f"SELECT 1 FROM {table} WHERE id=?", (entity_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Master entity not found")
        existing = dict(connection.execute(f"SELECT * FROM {table} WHERE id=?", (entity_id,)).fetchone())
        incoming = request.model_dump(exclude_unset=True)
        incoming.pop("continue_on_duplicate", None)
        values = {**existing, **incoming}
        values["id"] = entity_id
        values["updated_at"] = now()
        errors = validate_entity(canonical, values)
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        fields = [field for field in incoming if field in existing and field not in {"id", "created_at", "data_origin", "entry_method"}]
        fields = list(dict.fromkeys(fields + ["updated_at"]))
        connection.execute(f"UPDATE {table} SET {','.join(f'{field}=?' for field in fields)} WHERE id=?", [values[field] for field in fields] + [entity_id])
        if canonical == "Warehouse" and "location_id" in incoming:
            _sync_warehouse_location(connection, entity_id, values.get("location_id"), "FORM")
        audit(connection, canonical, entity_id, "UPDATE", fields, "FORM")
        connection.commit()
        return dict(connection.execute(f"SELECT * FROM {table} WHERE id=?", (entity_id,)).fetchone())
    finally:
        connection.close()


@router.post("/products/{entity_id}/deactivate")
@router.post("/suppliers/{entity_id}/deactivate")
@router.post("/warehouses/{entity_id}/deactivate")
@router.post("/locations/{entity_id}/deactivate")
def deactivate_entity(http_request: Request, entity_id: str) -> dict[str, str]:
    _require_write()
    entity_type = _entity_type_from_path(http_request.url.path.split(f"/{entity_id}")[0])
    canonical = entity_type.rstrip("s").capitalize()
    connection = connect()
    try:
        table = {"Product": "products", "Supplier": "suppliers", "Warehouse": "warehouses", "Location": "locations"}.get(canonical)
        result = connection.execute(f"UPDATE {table} SET status='INACTIVE', updated_at=? WHERE id=?", (now(), entity_id))
        if result.rowcount != 1:
            raise HTTPException(status_code=404, detail="Master entity not found")
        audit(connection, canonical, entity_id, "DEACTIVATE", ["status"], "FORM")
        connection.commit()
        return {"status": "INACTIVE", "id": entity_id}
    finally:
        connection.close()


@router.get("/relationships")
@router.get("/relationships/list")
def relationships() -> list[dict[str, Any]]:
    connection = connect()
    try:
        return [dict(row) for row in connection.execute("SELECT * FROM relationships ORDER BY created_at DESC")]
    finally:
        connection.close()


@router.post("/relationships/{relationship_id}/deactivate")
def deactivate_relationship(relationship_id: str) -> dict[str, str]:
    _require_write()
    connection = connect()
    try:
        result = connection.execute("UPDATE relationships SET status='INACTIVE', updated_at=? WHERE id=? AND status='ACTIVE'", (now(), relationship_id))
        if result.rowcount != 1:
            raise HTTPException(status_code=404, detail="Active relationship not found")
        audit(connection, "Relationship", relationship_id, "RELATIONSHIP_DEACTIVATE", ["status"], "FORM")
        connection.commit()
        return {"status": "INACTIVE", "id": relationship_id}
    finally:
        connection.close()


@router.post("/relationships")
@router.post("/relationships/create")
def create_relationship(request: RelationshipRequest) -> dict[str, Any]:
    _require_write()
    if (request.source_type, request.relationship_type, request.target_type) not in ALLOWED:
        raise HTTPException(status_code=422, detail="Invalid relationship type or endpoint types")
    connection = connect()
    try:
        if request.target_type != "Brand" and not connection.execute(f"SELECT 1 FROM {'products' if request.target_type == 'Product' else 'warehouses' if request.target_type == 'Warehouse' else 'locations' if request.target_type == 'Location' else 'suppliers'} WHERE id=? AND status='ACTIVE'", (request.target_id,)).fetchone():
            raise HTTPException(status_code=422, detail="Target endpoint is unknown or inactive")
        source_table = {"Product": "products", "Supplier": "suppliers", "Warehouse": "warehouses"}[request.source_type]
        if not connection.execute(f"SELECT 1 FROM {source_table} WHERE id=? AND status='ACTIVE'", (request.source_id,)).fetchone():
            raise HTTPException(status_code=422, detail="Source endpoint is unknown or inactive")
        if request.target_type == "Brand":
            registry = brand_registry(connection)
            if brand_id(request.target_id.removeprefix("BRAND:")) not in registry:
                raise HTTPException(status_code=422, detail="Target brand is not available from active product master data")
        if connection.execute(
            "SELECT 1 FROM relationships WHERE source_id=? AND source_type=? AND relationship_type=? AND target_id=? AND target_type=? AND status='ACTIVE'",
            (request.source_id, request.source_type, request.relationship_type, request.target_id, request.target_type),
        ).fetchone():
            raise HTTPException(status_code=409, detail="An active relationship with the same endpoints already exists")
        if request.relationship_type == "LOCATED_IN":
            _sync_warehouse_location(connection, request.source_id, request.target_id, "FORM")
            connection.execute("UPDATE warehouses SET location_id=?, updated_at=? WHERE id=?", (request.target_id, now(), request.source_id))
            connection.commit()
            return dict(connection.execute(
                "SELECT * FROM relationships WHERE source_id=? AND relationship_type='LOCATED_IN' AND target_id=? AND status='ACTIVE' ORDER BY created_at DESC LIMIT 1",
                (request.source_id, request.target_id),
            ).fetchone())
        values = {"id": new_id("RELATIONSHIP"), **request.model_dump(), "data_origin": "USER_ENTERED", "entry_method": "FORM", "created_at": now(), "updated_at": now()}
        columns = list(values)
        connection.execute(f"INSERT INTO relationships({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", [values[key] for key in columns])
        audit(connection, "Relationship", values["id"], "RELATIONSHIP_CREATE", columns, "FORM")
        connection.commit()
        return values
    finally:
        connection.close()


@router.get("/change-log")
@router.get("/change-log/list")
def change_log() -> list[dict[str, Any]]:
    connection = connect()
    try:
        return [dict(row) for row in connection.execute("SELECT * FROM change_log ORDER BY id DESC")]
    finally:
        connection.close()


@router.post("/validate")
def validate() -> dict[str, Any]:
    connection = connect()
    try:
        return validate_network(connection)
    finally:
        connection.close()


@router.get("/preview")
def preview() -> dict[str, Any]:
    connection = connect()
    try:
        validation = validate_network(connection)
        current_version = int(get_meta(connection, "published_version", "0"))
        nodes = {entity: connection.execute(f"SELECT COUNT(*) FROM {table} WHERE status='ACTIVE'").fetchone()[0] for entity, table in {"products": "products", "suppliers": "suppliers", "warehouses": "warehouses", "locations": "locations"}.items()}
        registry = brand_registry(connection)
        company_brands = {brand_id(row["brand"]) for row in connection.execute("SELECT brand FROM products WHERE status='ACTIVE' AND brand IS NOT NULL")}
        reused = sorted(key for key in company_brands if registry.get(key, {}).get("source") == "BASE")
        new_brands = sorted(company_brands - set(reused))
        company_entity_nodes = sum(nodes.values())
        edges = connection.execute("SELECT COUNT(*) FROM relationships WHERE status='ACTIVE'").fetchone()[0]
        derived_edges = connection.execute("SELECT COUNT(*) FROM products WHERE status='ACTIVE' AND brand IS NOT NULL").fetchone()[0]
        draft_changes = int(connection.execute("SELECT COUNT(*) FROM change_log WHERE id > ?", (get_meta(connection, "last_published_change_id", "0"),)).fetchone()[0])
        return {"validation": validation, "nodes": nodes, "relationships": connection.execute("SELECT COUNT(*) FROM relationships WHERE status='ACTIVE'").fetchone()[0], "company_entity_nodes": company_entity_nodes, "new_company_brand_nodes": len(new_brands), "reused_base_brand_references": len(reused), "company_relationship_edges": edges, "derived_PRODUCES_edges": derived_edges, "new_runtime_node_count": company_entity_nodes + len(new_brands), "overlay_graph_node_count": company_entity_nodes + len(company_brands), "nodes_to_publish": company_entity_nodes + len(new_brands), "edges_to_publish": edges + derived_edges, "draft_changes": draft_changes, "current_version": current_version, "expected_next_version": current_version + 1, "last_published": get_meta(connection, "published_at")}
    finally:
        connection.close()


@router.post("/publish")
def publish() -> dict[str, Any]:
    _require_write()
    connection = connect()
    try:
        validation = validate_network(connection)
        if validation["errors"]:
            raise HTTPException(status_code=422, detail=validation)
        manifest = build_overlay(connection, validation)
        set_meta(connection, "published_version", str(manifest["version"]))
        set_meta(connection, "published_at", manifest["published_at"])
        last_change = connection.execute("SELECT COALESCE(MAX(id),0) FROM change_log").fetchone()[0]
        set_meta(connection, "last_published_change_id", str(last_change))
        audit(connection, "Network", f"VERSION:{manifest['version']}", "PUBLISH", [], "FORM")
        connection.commit()
        from .repository import generated_dir
        (generated_dir() / "publish_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        connection.close()


@router.post("/import/preview")
async def import_preview(import_type: str, file: UploadFile = File(...)) -> dict[str, Any]:
    _require_write()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV uploads are supported")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV file exceeds the 10 MB limit")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if import_type not in {"Products", "Suppliers", "Warehouses", "Locations", "Relationships"}:
        raise HTTPException(status_code=400, detail="Unsupported import type")
    connection = connect()
    try:
        issues = []
        valid = []
        for index, row in enumerate(rows, 2):
            errors = [] if import_type == "Relationships" else validate_entity(import_type.rstrip("s").capitalize(), row)
            warnings = []
            if import_type == "Relationships":
                key = (row.get("source_type"), row.get("relationship_type"), row.get("target_type"))
                if key not in ALLOWED:
                    errors.append("invalid relationship type")
                table_by_type = {"Product": "products", "Supplier": "suppliers", "Warehouse": "warehouses", "Location": "locations"}
                source_table = table_by_type.get(row.get("source_type"))
                target_table = table_by_type.get(row.get("target_type"))
                if source_table and not connection.execute(f"SELECT 1 FROM {source_table} WHERE id=? AND status='ACTIVE'", (row.get("source_id"),)).fetchone():
                    errors.append("source endpoint is unknown or inactive")
                if target_table and not connection.execute(f"SELECT 1 FROM {target_table} WHERE id=? AND status='ACTIVE'", (row.get("target_id"),)).fetchone():
                    errors.append("target endpoint is unknown or inactive")
            else:
                canonical = import_type.rstrip("s").capitalize()
                candidates = duplicate_candidates(all_rows(connection, canonical), canonical, row)
                if candidates:
                    warnings.append("possible duplicate: " + ", ".join(_duplicate_reasons(canonical, row, candidate)[0] if _duplicate_reasons(canonical, row, candidate) else "matching fields" for candidate in candidates))
            if errors:
                issues.append({"row": index, "status": "INVALID", "messages": errors})
            else:
                valid.append(row)
                issues.append({"row": index, "status": "WARNING" if warnings else "VALID", "messages": warnings})
        batch_id = new_id("IMPORT")
        warning_count = sum(1 for item in issues if item["status"] == "WARNING")
        invalid_count = sum(1 for item in issues if item["status"] == "INVALID")
        strict_valid_count = len(valid) - warning_count
        connection.execute("INSERT INTO import_batches VALUES(?,?,?,?,?,?,?,?,?,?,?)", (batch_id, import_type, "PREVIEW", len(rows), strict_valid_count, warning_count, invalid_count, warning_count, json.dumps(valid), now(), None))
        connection.commit()
        return {"batch_id": batch_id, "import_type": import_type, "total_rows": len(rows), "valid_rows": strict_valid_count, "warning_rows": warning_count, "invalid_rows": invalid_count, "duplicate_warnings": warning_count, "rows": issues}
    finally:
        connection.close()


@router.post("/import/{batch_id}/confirm")
def confirm_import(batch_id: str) -> dict[str, Any]:
    _require_write()
    connection = connect()
    try:
        batch = connection.execute("SELECT * FROM import_batches WHERE id=? AND status='PREVIEW'", (batch_id,)).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail="Import batch not found or already confirmed")
        payload = json.loads(batch["payload"])
        import_type = batch["import_type"].rstrip("s")
        for row in payload:
            if import_type == "Relationships":
                relationship = RelationshipRequest(**row)
                if (relationship.source_type, relationship.relationship_type, relationship.target_type) not in ALLOWED:
                    raise ValueError("invalid relationship type")
                values = {"id": new_id("RELATIONSHIP"), **relationship.model_dump(), "data_origin": "USER_ENTERED", "entry_method": "CSV_IMPORT", "created_at": now(), "updated_at": now()}
                columns = list(values)
                connection.execute(f"INSERT INTO relationships({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", [values[column] for column in columns])
                audit(connection, "Relationship", values["id"], "IMPORT_CONFIRM", columns, "CSV_IMPORT")
                continue
            request = EntityRequest(**row)
            values = _entity_payload(import_type, request, "CSV_IMPORT")
            table = {"Product": "products", "Supplier": "suppliers", "Warehouse": "warehouses", "Location": "locations"}[import_type]
            columns = list(values)
            connection.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", [values[column] for column in columns])
            audit(connection, import_type, values["id"], "IMPORT_CONFIRM", columns, "CSV_IMPORT")
        connection.execute("UPDATE import_batches SET status='CONFIRMED', confirmed_at=? WHERE id=?", (now(), batch_id))
        connection.commit()
        return {"batch_id": batch_id, "status": "CONFIRMED", "imported": len(payload), "draft": True}
    except HTTPException:
        connection.rollback()
        raise
    except Exception as exc:
        connection.rollback()
        raise HTTPException(status_code=422, detail=f"Import rolled back: {exc}") from exc
    finally:
        connection.close()
