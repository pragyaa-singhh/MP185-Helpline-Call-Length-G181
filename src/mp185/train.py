
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

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ))
    ])

    mlflow.set_experiment("MP185-Helpline-Call-Length")

    with mlflow.start_run(run_name="RandomForest-v1"):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth": 6,
            "test_size": 0.2,
            "random_state": 42
        })

        mlflow.log_metric("MAE", mae)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )

        print("\nMODEL TRAINING COMPLETED")
        print("------------------------")
        print("MAE:", round(mae, 4))
        print("Training rows:", len(X_train))
        print("Testing rows:", len(X_test))
        print("MLflow Run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    train_model()
