"""Public read-only container entry point; no collection or refresh at startup."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def configuration() -> int:
    for name in ("SUPPLYBEACON_MASTER_WRITE_ENABLED", "SUPPLYBEACON_LLM_ENABLED"):
        if os.environ.get(name, "false").strip().lower() != "false":
            raise RuntimeError(f"{name} must be false for the public deployment")
        os.environ[name] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.setdefault("SUPPLYBEACON_ALLOWED_ORIGINS", "https://supply-beacon.vercel.app")
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError:
        raise RuntimeError("PORT must be an integer") from None
    if not 1 <= port <= 65535:
        raise RuntimeError("PORT must be between 1 and 65535")

    if os.environ.get("RAILWAY_ENVIRONMENT_ID"):
        mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
        if not mount:
            raise RuntimeError("Attach a Railway volume at /app/data/persistent before starting")
        root = Path(mount).resolve()
        for name in ("SUPPLYBEACON_CURRENT_DATA_DIR", "SUPPLYBEACON_MASTER_DB_PATH", "SUPPLYBEACON_MASTER_GENERATED_DIR"):
            value = os.environ.get(name)
            if not value or not Path(value).is_absolute():
                raise RuntimeError(f"{name} must be an absolute persistent-volume path")
            path = Path(value).resolve()
            if not path.is_relative_to(root) or path == root:
                raise RuntimeError(f"{name} must be inside the Railway volume")
    return port


def main() -> None:
    port = configuration()
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "src.api.app:app",
                             "--host", "0.0.0.0", "--port", str(port), "--workers", "1"])


if __name__ == "__main__":
    main()
