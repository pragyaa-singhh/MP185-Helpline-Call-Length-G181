import logging
import json
from datetime import datetime, timezone

import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# JSON Logging Setup
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_data)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())

logger = logging.getLogger("mp185-api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


# MLflow Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "MP185-Helpline-Call-Length"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="MP185 Helpline API")


class CallData(BaseModel):
    queue_wait_sec: float = Field(ge=0)
    agent_exp_months: float
    previous_calls: int = Field(le=20)
    issue_count: int
    hour: int
    caller_age: int
    language: str
    team: str


@app.get("/health")
def health():
    logger.info("Health check successful")
    return {"status": "ok"}


@app.post("/predict")
def predict(data: CallData):
    logger.info("Prediction request received")

    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]

    logger.info("Prediction generated successfully")

    return {"prediction": float(prediction)}