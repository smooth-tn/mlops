import os
import json
import mlflow
import pandas as pd

def init():
    global model, threshold

    model_name = os.getenv("MODEL_NAME", "fraud-detection-champion")
    threshold = 0.1 if model_name == "fraud-detection-challenger" else 0.5

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model = mlflow.sklearn.load_model(model_dir)


def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data["input"])

        proba = model.predict_proba(df)[:, 1]
        prediction = (proba >= threshold).astype(int)

        return {
            "predictions": prediction.tolist(),
            "probabilities": proba.tolist()
        }

    except Exception as e:
        return {"error": str(e)}