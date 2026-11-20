# MP059 — Cinema Show Occupancy (Indore)
 
SCSE3040 Machine Learning Operations — Mini Project
**Group ID:** G185  **Problem ID:** MP059
**Faculty:** Richa Sharma  **Batch:** 86
 
## Client
Indore Single-Screen Owners' Forum. Owners fix show timings a week ahead and
want to predict `occupancy_pct` (share of seats filled) per show.
Regression problem, evaluated with MAE.
 
## Members
| Enrolment No. | Name | GitHub |
|---|---|---|
| S24CSEU2281 | Sahil | strange-vibes |
| S24CSEU0051 | Kumari Ishani | kumariIshani |
| S24CSEU0705 | Ashmit Mittal |      -       |
| S24CSEU2257 | Tanisha | tanisha17058 |
 
## Pipeline
Versioned data (DVC) -> cleaning -> MLflow tracking + registry (`champion`)
-> FastAPI `/predict`, `/health` -> Docker (`mp059-api:1.0`) -> GitHub Actions
CI -> Kubernetes (`mp059-api`, `mp059-svc`, 3 replicas) -> JSON logging ->
drift (PSI) + retraining.
 
## Layout
```
src/cinema_occupancy/   package (data_analysis.py = data-problem detection)
data/raw/               train.csv.dvc, later.csv.dvc, holdout_features.csv
reports/findings.yaml   data problems, tracking, monitoring
app/  models/  k8s/  tests/  predictions/
```
 
## Data problems found in train.csv
Detected by `src/cinema_occupancy/data_analysis.py`; recorded in `reports/findings.yaml`.
 
| Problem | Finding |
|---|---|
| Leaking column | `tickets_scanned_pct` |
| Unit mix-up | `distance_to_metro_km`, location `Mall`, factor 1000 (metres) |
| Sentinel value | `ticket_price_rs` = 999, 65 rows |
| Duplicate rows | none |
| Stuck sensor | none |
| Target typos | 15 rows (`occupancy_pct` x10) |
| Spelling variants | none |
 
## Run
```
pip install -r requirements.txt
dvc pull
python -m src.cinema_occupancy.data_analysis
```
 
