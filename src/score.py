import os
import json
import mlflow
import joblib
import pandas as pd
import logging
from mlops_model_prototype.repo.src.preprocess import preprocess

logger = logging.getLogger(__name__)
feature_columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig','oldbalanceDest', 'newbalanceDest']
EXAMPLE_INPUT = {
    "input": [{
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0
    }]
}


def init():
    global model, threshold,encoder,scaler

   
    model_name = os.getenv("MODEL_NAME", "fraud-detection-champion")
    threshold = 0.1 if model_name == "fraud-detection-challenger" else 0.5

    model_dir=os.getenv("AZUREML_MODEL_DIR")
    encoder=joblib.load(os.path.join(model_dir, "encoder.pkl"))
    scaler=joblib.load(os.path.join(model_dir, "scaler.pkl"))
    model=mlflow.sklearn.load_model(os.path.join(model_dir))

def run(raw_data):
    try:
        data = json.loads(raw_data)
        if data.get("request") == "schema":
            return {
                "required_columns": feature_columns,
                "types": {
                    "step":           "int   — transaction step (1 step = 1 hour)",
                    "type":           "str   — CASH_IN | CASH_OUT | DEBIT | PAYMENT | TRANSFER",
                    "amount":         "float — transaction amount",
                    "oldbalanceOrg":  "float — sender balance before",
                    "newbalanceOrig": "float — sender balance after",
                    "oldbalanceDest": "float — receiver balance before",
                    "newbalanceDest": "float — receiver balance after"
                },
                "example": EXAMPLE_INPUT
            }

        df = pd.DataFrame(data["input"])
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            return {
                "error": "Missing required columns",
                "missing": missing,
                "required_columns": feature_columns,
                "example": EXAMPLE_INPUT
            }
        processed_df=preprocess(df,encoder,scaler)
        proba = model.predict_proba(processed_df)[:,1]
        prediction = (proba >= threshold).astype(int)

        return {
            "predictions": prediction.tolist(),
            "probabilities": proba.tolist()
        }
    except Exception as e:
        logger.error(f"Scoring error: {e}", exc_info=True)
        return {"error": str(e)}
    
