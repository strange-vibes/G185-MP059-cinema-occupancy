from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from cinema_occupancy.cleaning import clean_data

config = yaml.safe_load(Path("params.yaml").read_text())

train = clean_data(pd.read_csv(config["data"]["train_path"]))
later = pd.read_csv(config["data"]["later_path"])

later["date"] = pd.to_datetime(later["date"])
later = later.sort_values("date", kind="stable").reset_index(drop=True)

split = int(len(later) * 0.7)

retrain_data = later.iloc[:split]
check_data = later.iloc[split:]

combined = pd.concat([train, retrain_data], ignore_index=True)

print("Combined training records:", len(combined))
print("Evaluation records:", len(check_data))

v1 = joblib.load("models/model_v1.joblib")

features = list(v1.feature_names_in_)
target = config["data"]["target"]

X_train = combined[features]
y_train = combined[target]

X_check = check_data[features]
y_check = check_data[target]

v2 = clone(v1)
v2.fit(X_train, y_train)

mae_v1 = mean_absolute_error(y_check, v1.predict(X_check))
mae_v2 = mean_absolute_error(y_check, v2.predict(X_check))

improvement = ((mae_v1 - mae_v2) / mae_v1) * 100

decision = "promote" if mae_v2 <= 0.95 * mae_v1 else "keep"

print(f"V1 MAE: {mae_v1:.4f}")
print(f"V2 MAE: {mae_v2:.4f}")
print(f"Improvement: {improvement:.2f}%")
print(f"Decision: {decision}")

import mlflow
import mlflow.sklearn
from hashlib import sha256

mlflow_config = config["mlflow"]

mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
mlflow.set_experiment(mlflow_config["experiment_name"])

with mlflow.start_run(run_name="stage4-retraining-v2") as run:
    mlflow.log_params({
        "data_sha256": sha256(
            Path(config["data"]["train_path"]).read_bytes()
        ).hexdigest(),
        "training_records": len(combined),
        "evaluation_records": len(check_data),
        "retrain_fraction": 0.7,
        "model_type": type(v2.steps[-1][1]).__name__,
        "promotion_threshold": 5,
        "decision": decision,
    })

    mlflow.log_metrics({
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "improvement_pct": improvement,
    })

    model_info = mlflow.sklearn.log_model(
        sk_model=v2,
        name="model",
        serialization_format="cloudpickle",
    )

    print("V2 Run ID:", run.info.run_id)
    print("V2 Model URI:", model_info.model_uri)

joblib.dump(v2, "models/model_v2.joblib")