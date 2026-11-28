# Stage 4 Handover — G185 / MP059 (Cinema Show Occupancy)

- **Student:** Kumari Ishani (`kumariIshani`)
- **Branch:** `ishani-stage4-k8s` (branched from `main` at `04ac6f4`)
- **Course:** SCSE3040 — MLOps
- **Next Owner:** Tanisha (`tanisha17058`)

---

## 1. Repository State Found on `main`

Before starting, `main` was pulled up to commit `04ac6f4`. The following work was already present in the repository:

- **Stage 1 & 2:** Data (`data/raw/train.csv`, `later.csv`, `holdout_features.csv`), DVC tracking, `src/cinema_occupancy/data_analysis.py`, `src/cinema_occupancy/cleaning.py`, `src/cinema_occupancy/train.py`, `tests/test_cleaning.py`, `params.yaml`, and `mlflow.db`.
- **Stage 3:** FastAPI app (`app/main.py`) with `/health`, `/predict`, and JSON stdout logging, `tests/test_api.py`, GitHub Actions CI (`.github/workflows/ci.yml`), `Dockerfile`, `.dockerignore`, and initial Kubernetes manifests (`k8s/deployment.yaml`, `k8s/service.yaml`).
- **Stage 4 (partial, by Sahil):** `src/cinema_occupancy/retrain.py`, `models/model_v1.joblib`, `models/model_v2.joblib`, `models/model.joblib` (updated to v2), and retraining metrics in `reports/findings.yaml` (`champion_version: 2`, `mae_v1_on_check: 8.2222`, `mae_v2_on_check: 7.1388`, `decision: promote`).

Note: `STUDENT_HANDOUT.md` and `PROBLEM.md` were not present in the repository, so validation was aligned against `README.md`, `Dockerfile`, `app/main.py`, `k8s/`, and `reports/findings.yaml`.

---

## 2. Work Completed by Me

1. **Audited Kubernetes, Docker, and FastAPI alignment:**
   - Verified `Dockerfile` builds from `python:3.12-slim`, sets `PYTHONPATH=/app/src`, copies `app/`, `models/model.joblib`, and `src/`, exposes port `8000`, and starts `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
   - Verified `app/main.py` loads `models/model.joblib` relative to repository root (`/app/models/model.joblib` in container) and exposes `GET /health` and `POST /predict`.
   - Verified `k8s/service.yaml` defines `mp059-svc` (`ClusterIP`) selecting `app: mp059-api` and forwarding port `80` to `targetPort: 8000`.
2. **Completed `k8s/deployment.yaml` configuration:**
   - Kept existing `mp059-api` Deployment (`replicas: 3`, `image: mp059-api:1.0`, `containerPort: 8000`, `readinessProbe` and `livenessProbe` on `/health:8000`).
   - Added `metadata.labels.app: mp059-api`.
   - Added `imagePullPolicy: IfNotPresent` so local clusters use the locally built `mp059-api:1.0` image instead of trying to pull from Docker Hub.
   - Added container `resources` (`requests`: `100m` CPU / `256Mi` memory; `limits`: `500m` CPU / `512Mi` memory).
3. **Added automated Kubernetes manifest & alignment tests (`tests/test_k8s.py`):**
   - Validates YAML parsing, labels/selectors, replicas, image and pull policy, container port, readiness/liveness probes, resource requests/limits, `Service` port mapping, `Dockerfile` consistency, and FastAPI route/model presence.
4. **Verified existing API, model, and automated test suite:**
   - Ran full `pytest` suite (14 tests total: API, cleaning, and Kubernetes manifest validation).
   - Verified `app/main.py` `/health` and `/predict` against the promoted `models/model.joblib` using FastAPI `TestClient`.

---

## 3. Files Created or Modified

| File | Action | Purpose |
|---|---|---|
| `k8s/deployment.yaml` | Modified | Added `metadata.labels`, `imagePullPolicy: IfNotPresent`, and container `resources` (`requests`/`limits`). |
| `tests/test_k8s.py` | Created | Automated pytest validation of `k8s/deployment.yaml`, `k8s/service.yaml`, `Dockerfile`, and `app/main.py` alignment. |
| `STAGE4_HANDOVER.md` | Created | Stage 4 progress, test evidence, known issues, and handover notes for Tanisha. |

Unchanged existing files kept intact: `k8s/service.yaml`, `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`, `app/main.py`, `src/cinema_occupancy/retrain.py`, `reports/findings.yaml`.

---

## 4. Commands Executed

```powershell
# 1. Sync main branch with remote
git fetch --all --prune
git checkout main
git merge --ff-only origin/main

# 2. Create working branch for Stage 4 Kubernetes work
git checkout -b ishani-stage4-k8s

# 3. Check Docker and Kubernetes environment availability
docker info --format "{{.ServerVersion}}"
kubectl version --client
kubectl config get-contexts
kubectl cluster-info
kubectl apply --dry-run=client --validate=false -f k8s/ -o yaml

# 4. Validate K8s YAML parsing with Python
python -c "import yaml, pathlib; [print(p.name, yaml.safe_load(p.read_text())) for p in sorted(pathlib.Path('k8s').glob('*.yaml'))]"

# 5. Verify live FastAPI app + promoted v2 model
python -c "from fastapi.testclient import TestClient; from app.main import app; c = TestClient(app); print('HEALTH:', c.get('/health').status_code, c.get('/health').json()); p = {'ticket_price_rs': 321.0, 'days_since_release': 33, 'star_rating': 3.1, 'show_hour': 15, 'screen_seats': 517, 'distance_to_metro_km': 4.74, 'language': 'hindi', 'location': 'Mall'}; r = c.post('/predict', json=p); print('PREDICT:', r.status_code, r.json())"

# 6. Run full test suite (after all edits)
$env:PYTHONPATH="src"; python -m pytest -v
python -m pytest -q
```

---

## 5. Test Results

### Passed Checks
- **`$env:PYTHONPATH="src"; python -m pytest -v`**: **14 passed in 2.34s**
  - `tests/test_api.py` (6/6 passed): `/health`, `/predict`, `422` validation for `screen_seats=701` and `days_since_release=-1`, stdout JSON log format, no log on `422`.
  - `tests/test_cleaning.py` (5/5 passed): leakage removal, sentinel price replacement, Mall distance correction, occupancy typo fix, duplicate row removal.
  - `tests/test_k8s.py` (3/3 passed): `k8s/deployment.yaml` schema/probes/resources, `k8s/service.yaml` selector/ports match, `Dockerfile` and `app/main.py` alignment.
- **`python -m pytest -q`**: **14 passed in 2.44s**.
- **End-to-end API + `models/model.joblib` check via `TestClient`**:
  - `GET /health` -> `200 {'status': 'ok'}`
  - `POST /predict` -> `200 {'prediction': 6.923005361022476}`
  - Stdout JSON log -> `{"event": "prediction", "prediction": 6.923005361022476, "duration_ms": 18049.841}`

### Could Not Be Tested (No Running Docker Daemon / Kubernetes Cluster)
- `docker info`: Failed — Docker Desktop daemon is not running (`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`). Therefore, `docker build -t mp059-api:1.0 .` and container runtime tests were not executed.
- `kubectl config get-contexts` / `kubectl cluster-info`: No kubeconfig (`~/.kube/config` missing) and no Kubernetes API server listening (`localhost:8080` refused connection). `minikube` and `kind` are not installed.
- `kubectl apply --dry-run=client --validate=false -f k8s/`: Failed with `unable to recognize` because `kubectl` (`v1.36.1`) still requires API server discovery when no cluster is reachable.
- Live cluster rollout (`kubectl apply -f k8s/`, `kubectl get pods`, `kubectl port-forward`, and hitting `/health` and `/predict` through `mp059-svc`) could not be executed in this environment.

---

## 6. Remaining Tasks for Tanisha

1. **Docker & Live Kubernetes Runtime Validation (when Docker Desktop Kubernetes / Minikube / kind is running):**
   - Build image: `docker build -t mp059-api:1.0 .`
   - Deploy manifests: `kubectl apply -f k8s/`
   - Verify 3/3 ready pods (`kubectl get pods -l app=mp059-api`) and service (`kubectl get svc mp059-svc`).
   - Port-forward `kubectl port-forward svc/mp059-svc 8000:80` and verify live `GET /health`, `POST /predict`, and pod JSON stdout logs (`kubectl logs -l app=mp059-api`).
2. **Stage 4 Drift Detection & Monitoring (`reports/findings.yaml`):**
   - Implement PSI drift calculation comparing `train.csv` and `later.csv` across the 6 numeric features (`ticket_price_rs`, `days_since_release`, `star_rating`, `show_hour`, `screen_seats`, `distance_to_metro_km`) and fill `monitoring.psi` and `monitoring.most_drifted_feature` in `reports/findings.yaml`.
   - Evaluate model MAE by location group (`City Centre`, `Mall`, `Highway`) and fill `monitoring.mae_by_group` and `monitoring.worst_served_group` in `reports/findings.yaml`.
3. **Final integration & CI check:**
   - Merge PRs to `main` and confirm GitHub Actions CI passes on `main`.

---

## 7. Known Issues

1. **No active Docker/Kubernetes runtime in current environment:** Docker Desktop is installed (`docker.exe`, `kubectl.exe`), but the Docker Engine is stopped and Kubernetes is not enabled/configured locally, preventing live pod/service verification here.
2. **Scikit-learn version warning locally:** Local Python (`3.14.7`) has `scikit-learn 1.9.1`, whereas `models/model.joblib` was saved with `scikit-learn 1.8.0` (matching `requirements.txt`). Unpickling prints `InconsistentVersionWarning` locally; inside the `python:3.12-slim` Docker image and GitHub Actions CI (`requirements.txt` pins `scikit-learn==1.8.0`), versions match cleanly.
3. **`requirements.txt` pins added in `807fa4a`:** `requirements.txt` currently includes `fastapi==0.165.0`, `starlette==1.10.0`, `uvicorn==0.54.0`, `httpx==0.29.0`, `httpx2==2.16.2`. Keep an eye on GitHub Actions CI for `ishani-stage4-k8s` to confirm those pinned versions resolve cleanly on Ubuntu + Python 3.12.

---

## 8. Instructions for Continuing From My Work

```powershell
# Fetch latest branches and check out Ishani's Stage 4 branch (or pull after PR merge into main)
git fetch --all --prune
git checkout ishani-stage4-k8s
git pull origin ishani-stage4-k8s

# Run the test suite
$env:PYTHONPATH="src"; python -m pytest -v

# Once Docker Desktop (with Kubernetes enabled) is started:
docker build -t mp059-api:1.0 .
kubectl apply -f k8s/
kubectl rollout status deployment/mp059-api
kubectl get pods,svc
```
