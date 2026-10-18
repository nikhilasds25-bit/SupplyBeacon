"""Fail closed if the reviewed deployment seed is changed or gains extra files.

Uses only the standard library; never prints matching secret values. Pattern
checks supplement manual review and are not a general PII/security guarantee.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CURRENT_FILES = {
    "current_weather.json", "current_events.json", "current_business_impacts.csv",
    "current_supply_chain_graph.graphml", "current_risk_scores.csv", "current_run_metadata.json",
}
METRIC_FILES = {"baseline_metrics.csv", "run_summary.json", "feature_audit.json"}
PATTERNS = {
    "credential assignment": r"(?i)(?:api[_-]?key|access[_-]?token|authorization|password|secret)\s*[=:]\s*[\"']?[A-Za-z0-9_/-]{10,}",
    "credential URL parameter": r"(?i)[?&](?:api[_-]?key|appid|token|key)=",
    "private key": r"-----BEGIN .*PRIVATE KEY-----",
    "email address": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    "developer path": r"(?i)(?:[A-Z]:[\\/](?:Users|Projects)|/Users/|/home/)",
}


def audit(root: Path = ROOT) -> dict[str, int]:
    manifest = json.loads((root / "deploy/seed-manifest.json").read_text(encoding="utf-8"))
    expected = {"deploy/seed/current/" + name for name in CURRENT_FILES}
    expected |= {"deploy/seed/analysis/dataco/modeling/" + name for name in METRIC_FILES}
    actual = {p.relative_to(root).as_posix() for p in (root / "deploy/seed").rglob("*") if p.is_file()}
    if actual != expected or set(manifest["files"]) != expected:
        raise ValueError("Deployment seed file allowlist mismatch")
    total = 0
    for relative in sorted(expected):
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"Symlink is not an approved seed: {relative}")
        raw = path.read_bytes()
        record = manifest["files"][relative]
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Seed checksum mismatch: {relative}")
        text = raw.decode("utf-8-sig")
        for label, pattern in PATTERNS.items():
            if re.search(pattern, text):
                raise ValueError(f"Seed rejected ({label}): {relative}")
        if path.suffix == ".json":
            json.loads(text)
        elif path.suffix == ".graphml":
            ET.fromstring(text)
        else:
            rows = csv.DictReader(io.StringIO(text))
            if not rows.fieldnames:
                raise ValueError(f"Missing CSV header: {relative}")
            for row in rows:
                if None in row:
                    raise ValueError(f"Unexpected CSV fields: {relative}")
        total += len(raw)
    return {"files": len(expected), "bytes": total}


if __name__ == "__main__":
    print("Deployment seed audit passed:", audit())
