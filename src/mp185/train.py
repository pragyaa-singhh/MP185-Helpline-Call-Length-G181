
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "train_clean.csv"

NUMERIC_FEATURES = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age"
]

CATEGORICAL_FEATURES = ["language", "team"]
TARGET = "handle_min"

EXPERIMENT_NAME = "MP185-Helpline-Call-Length"
MODEL_NAME = "MP185-Helpline-Call-Length"

CONFIGURATIONS = [
    {"n_estimators": 50, "max_depth": 4},
    {"n_estimators": 100, "max_depth": 6},
    {"n_estimators": 200, "max_depth": 10}
]


def create_pipeline(n_estimators, max_depth):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        ))
    ])


def train_model():

    df = pd.read_csv(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    for config in CONFIGURATIONS:

        model = create_pipeline(
            config["n_estimators"],
            config["max_depth"]
        )

        with mlflow.start_run(
            run_name=f"RandomForest-{config['n_estimators']}-{config['max_depth']}"
        ) as run:

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            mlflow.log_params({
                "model_type": "RandomForestRegressor",
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "test_size": 0.2,
                "random_state": 42
            })

            mlflow.log_metric("MAE", mae)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle"
            )

            results.append({
                "run_id": run.info.run_id,
                "mae": mae,
                "model": model,
                "config": config
            })

            print(
                f"Configuration: {config}, "
                f"MAE: {mae:.4f}, "
                f"Run ID: {run.info.run_id}"
            )

    best_result = min(results, key=lambda item: item["mae"])

    print("\nTRAINING COMPARISON")
    print("-------------------")

    for result in results:
        print(
            result["config"],
            "MAE:",
            round(result["mae"], 4)
        )

    print("\nBEST MODEL")
    print("Configuration:", best_result["config"])
    print("Best MAE:", round(best_result["mae"], 4))
    print("Best Run ID:", best_result["run_id"])

    model_uri = f"runs:/{best_result['run_id']}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("\nMODEL REGISTERED")
    print("Model name:", MODEL_NAME)
    print("Model version:", registered_model.version)


if __name__ == "__main__":
    train_model()
