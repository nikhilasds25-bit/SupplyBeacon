# SupplyBeacon Railway preparation

These are reviewable local changes, not a deployed service. Nothing in this
preparation creates Railway resources, connects GitHub, pushes commits, or
enables auto-deployment. Review and publish the files manually first.

## Runtime and artifact boundary

The repository-root Dockerfile uses the official Python 3.13 Debian Bookworm
image pinned by manifest digest. `requirements-api.lock` pins the complete
50-package Linux amd64 / CPython 3.13 dependency closure with wheel hashes.
`requirements-api.txt` contains the same versions for local installation,
plus a Windows-only colorama requirement. `requirements-test.txt` adds pytest
and explicitly declares the `httpx` client used by FastAPI tests. `httpx2` is
also a legitimate transitive SDK dependency in this pinned environment; it
does not substitute for the test client's `httpx` requirement.

`python-multipart==0.0.32` is now declared. The optional research dependencies
in `requirements.txt` remain available but are not installed in the API image:
no PyTorch, PyTorch Geometric, scikit-learn, matplotlib, or Streamlit is needed
to serve the retained graph/evidence routes. No model training or inference
from saved joblib files runs on startup.

The default-deny `.dockerignore` and explicit COPY inputs exclude populated
environment files, backups, raw DataCo/customer inputs, mutable current state,
SQLite files, generated company overlays, models, frontend assets/dependencies,
and caches. Required base graph, node/edge tables, embeddings, reference CSVs,
and baseline risk CSV are included. The curated warehouse CSV is not needed
for the packaged read-only snapshot; rebuilding the base graph is outside this
image's supported workflow.

The reviewed seed is exactly six `deploy/seed/current/` files, plus three small
aggregate historical DataCo evaluation files under
`deploy/seed/analysis/dataco/modeling/`. The seed totals **5,080,080 bytes**.
`seed-manifest.json` records exact sizes and SHA-256 checksums. `.gitattributes`
preserves seed bytes across checkouts. Run `python deploy/audit_seed.py` before
publication; the Docker build runs it too. It validates file membership,
checksums, parseability, and patterns for credential assignments, key-bearing
URLs, private keys, email addresses, and developer paths without printing
matched values. This targeted audit supplements manual review; it is not a
general proof that arbitrary future data is sanitized. Re-audit intentional
seed updates and update the manifest only after review.

The existing seed was retained unchanged. It contains October 2026 historical
weather/news evidence, rule-derived scores, and modeled/simulated supply
relationships. It is not live inventory or a new provider fetch. Expired
events are evaluated using the existing runtime temporal rules. The DataCo
CSV contains saved historical metrics, not live order predictions.

## Manual Railway settings

After reviewing and manually publishing the prepared files:

1. Use GitHub repository `nikhilasds25-bit/SupplyBeacon`, repository root `/`,
   root `Dockerfile`, Linux amd64. Connecting a source may immediately build
   or deploy; do that only when you intend to deploy. Review Railway's
   auto-deploy setting separately.
2. Attach a persistent volume at `/app/data/persistent` **before first start**.
   Do not mount over `/app/data` or `/app/deploy`, which would hide packaged
   assets. Railway mounts volumes at runtime, not in the build/pre-deploy step.
3. Use one replica and the Docker CMD (no start-command override):
   `python deploy/start.py`. It execs Uvicorn on `0.0.0.0`, uses Railway's
   `PORT` (8000 locally if absent), validates the port, and uses one worker.
4. Configure healthcheck path `/health`, timeout 120 seconds, restart on
   failure with a bounded retry count (for example 3). `/health` is liveness,
   not a guarantee of valid business evidence; check the read-only routes too.
5. Confirm the environment below. No admin, provider, or LLM key is needed
   for the initial read-only snapshot service.

| Variable | Public deployment value |
| --- | --- |
| `SUPPLYBEACON_ALLOWED_ORIGINS` | `https://supply-beacon.vercel.app` |
| `SUPPLYBEACON_MASTER_WRITE_ENABLED` | `false` |
| `SUPPLYBEACON_LLM_ENABLED` | `false` |
| `LANGSMITH_TRACING` | `false` |
| `LANGCHAIN_TRACING_V2` | `false` |
| `SUPPLYBEACON_CURRENT_DATA_DIR` | `/app/data/persistent/current` |
| `SUPPLYBEACON_MASTER_DB_PATH` | `/app/data/persistent/master/supplybeacon_master.db` |
| `SUPPLYBEACON_MASTER_GENERATED_DIR` | `/app/data/persistent/master/generated` |
| `PORT` | Supplied by Railway; do not hard-code it |

These non-secret defaults are already in the Docker image. The entry point
refuses to start if writes or optional LLM use is enabled; it also disables
inherited tracing flags. On Railway it requires the volume environment
variable and checks that all three runtime paths are inside the mounted
volume. This validates configuration, not the storage provider's durability.

The image retains the base image's root user because Railway volumes are
root-owned. This is a deliberate minimal compatibility choice, not non-root
container hardening. A later non-root deployment needs runtime volume
ownership preparation. Public mutations remain disabled regardless of UID.

Startup uses the existing application lifespan: seed only an **empty** current
directory; preserve any nonempty directory; create SQLite tables with
`IF NOT EXISTS` and metadata with `INSERT OR IGNORE`. The immutable packaged
seed is outside the volume. Restarts do not recopy snapshots or reset an
existing DB. Partial/nonempty current directories are intentionally not
repaired automatically; restore a reviewed backup rather than deleting data.
Enable Railway volume backups before storing valuable runtime state. One
volume/replica and SQLite are the supported arrangement; multi-replica writes
and zero-downtime volume handover have not been verified.

Keep `SUPPLYBEACON_ADMIN_API_KEY` absent for the initial public demo: refresh
then returns 401. If you later intentionally configure protected refresh,
create a secret in Railway only and use `X-SupplyBeacon-Admin-Key` from a
trusted backend/admin client. Provider errors, coverage, and location checks
still apply; refresh is not exercised in this preparation. Never set backend
keys as `VITE_*` values or copy a populated developer `.env` into the image.

The frontend origin is allowed by CORS; other origins are not. CORS is a
browser policy, not authentication. Public read/query endpoints remain
accessible to non-browser clients. Demo login, production identity, rate
limiting, and resource/abuse controls remain outside this change. For a later
frontend deployment, set `VITE_API_BASE_URL` to the actual Railway HTTPS
origin; no frontend source was changed here.

## Local review and verification

Use Python 3.13 and a clean environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
python -m pip check
python deploy/audit_seed.py
python -m pytest tests/test_deployment_runtime.py tests/test_railway_configuration.py tests/test_current_data_storage.py tests/test_api_security.py -q
```

The runtime regression copies only Docker COPY inputs to a temporary folder,
imports the API in a fresh process with no `.env`, blocks external provider
requests, seeds an isolated persistent directory, and checks reads, query
responses, historical metrics, catalog-only scenarios, CORS, blocked master
mutations, and denied refresh. The restart test also checks existing SQLite
metadata and current snapshot bytes survive a second startup. These checks
do not assert that all question-answering semantics are correct.

With a local Docker engine, the remaining Linux build/runtime check is:

```powershell
docker build --platform linux/amd64 -t supplybeacon-api:review .
docker run --rm -p 8780:8000 -v supplybeacon-review:/app/data/persistent supplybeacon-api:review
```

No Docker engine is installed in the preparation environment, so no Docker
build, image-size measurement, vulnerability scan, Railway rollout, or actual
Railway volume permission test is claimed. Linux wheel resolution and hashes
can be verified independently; native execution here uses Windows Python
3.13.7. The image is smaller in scope because heavyweight research packages,
model weights, and frontend files are excluded, but its final size must be
measured after a build. Digest/hash pinning is reproducibility, not a security
audit; review and refresh pins for future security updates.

Existing entity-scoped risk retrieval, delivery-response mapping, warehouse
ranking, and network-summary defects remain documented in README. This
deployment preparation does not fix them or change graph semantics.

## Validation recorded on 18 October 2026

| Check | Result and scope |
| --- | --- |
| Clean dependency installation | Passed in a new Windows Python 3.13.7 venv with system site packages disabled; `pip check` passed. |
| Linux dependency lock | All 50 pinned packages resolved to CPython 3.13/Linux amd64-compatible wheels; a second dry run with `--require-hashes` passed. This is not Linux execution. |
| Seed audit | All 9 existing seed/metrics files passed exact-byte checksums, parsing, file membership, and the targeted sensitive-data checks. Seed content was not changed. |
| Packaged runtime, startup configuration, storage/restart, and API security tests | **26 passed**, one Starlette/AnyIO deprecation warning, using an isolated copy of the publishing clone and clean venv. Includes CORS allow/deny, disabled mutation/upload/publish, denied refresh, and preservation of existing SQLite metadata/current-state bytes. |
| Curated graph and updated startup configuration rerun | **18 passed**. |
| Supervised native Uvicorn HTTP smoke | `/health` plus 10 read-only endpoints returned 200 using a selected `PORT`; the server was stopped afterwards. Uses only the Docker COPY inputs, no `.env`, and isolated persistent paths. |
| Linux Docker image and Railway | Not run: Docker Engine is unavailable here, and no cloud deployment was authorized. |

The first broad regression run had **309 passes and 7 fixture errors** because
the isolated test copy omitted the already-published
`data/external/events/events.csv`. Restoring that file in the test workspace
cleared all seven errors in the 18-test rerun. It was a test-workspace omission,
not a missing read-only container input or an application correction. No
single-pass full-suite success after that restoration is claimed. Initial
sandboxed attempts also hit Windows loopback restrictions; final checks used
local socket access. No live provider refresh or optional LLM call was tested.

The reviewed public-clone COPY payload is about **16.54 MB** of source and
artifacts, excluding dependencies/base layers; this is not the final image
size. Required read-only runtime assets were present in the isolated package.
Raw DataCo inputs, reproducible GraphSAGE training code, live credentials, and
operational company inventory are not supplied by this preparation. Historical
ML retraining and live external intelligence therefore remain separate work.

Official Railway references: [volumes and permissions](https://docs.railway.com/volumes),
[Dockerfiles](https://docs.railway.com/builds/dockerfiles),
[healthchecks](https://docs.railway.com/deployments/healthchecks).
