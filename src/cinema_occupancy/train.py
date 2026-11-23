from hashlib import sha256
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cinema_occupancy.cleaning import clean_data


def main():
    with Path("params.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    data_config = config["data"]
    training_config = config["training"]
    mlflow_config = config["mlflow"]

    train_path = Path(data_config["train_path"])
    data_hash = sha256(train_path.read_bytes()).hexdigest()
    cleaned = clean_data(pd.read_csv(train_path))

    numeric = [
        "ticket_price_rs",
        "days_since_release",
        "star_rating",
        "show_hour",
        "screen_seats",
        "distance_to_metro_km",
    ]
    categorical = ["language", "location"]

    X_train, X_val, y_train, y_val = train_test_split(
        cleaned[numeric + categorical],
        cleaned[data_config["target"]],
        **training_config,
    )

    preprocessor = ColumnTransformer([
        (
            "numeric",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            numeric,
        ),
        (
            "categorical",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
            categorical,
        ),
    ])

    model_types = {
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
    }

    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    best_mae = float("inf")
    best_model = None
    best_run_id = None

    for candidate in config["models"]:
        settings = candidate.copy()
        name = settings.pop("name")
        model_type = settings.pop("type")

        if model_type in {"RandomForestRegressor", "GradientBoostingRegressor"}:
            settings["random_state"] = training_config["random_state"]

        estimator = model_types[model_type](**settings)
        model = make_pipeline(clone(preprocessor), estimator)

        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_val, model.predict(X_val))

            mlflow.log_params({
                "model_type": model_type,
                "data_sha256": data_hash,
                "validation_fraction": training_config["test_size"],
                "split_seed": training_config["random_state"],
                **settings,
            })
            mlflow.log_metric("val_mae", mae)
            print(f"{name} | MAE: {mae:.3f} | Run ID: {run.info.run_id}")

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_run_id = run.info.run_id

    if best_model is None:
        raise ValueError("No model was selected. Check the models in params.yaml.")

    with mlflow.start_run(run_id=best_run_id):
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            serialization_format="cloudpickle",
        )

    print(f"Winning validation MAE: {best_mae:.3f}")
    print(f"Winning run ID: {best_run_id}")
    print(f"Saved model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()