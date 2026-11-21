
import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME = "MP185-Helpline-Call-Length"
MODEL_NAME = "MP185-Helpline-Call-Length"

# Latest successful training run
RUN_ID = "79d271022bca42fcab3799d57da68802"


def register_model():

    client = MlflowClient()

    model_uri = f"runs:/{RUN_ID}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("\nMODEL REGISTRATION COMPLETED")
    print("----------------------------")
    print("Model Name:", registered_model.name)
    print("Model Version:", registered_model.version)
    print("Run ID:", RUN_ID)


if __name__ == "__main__":
    register_model()
