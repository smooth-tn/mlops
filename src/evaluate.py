import mlflow
import argparse
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score
from mlflow.tracking import MlflowClient


def evaluate_model(model, x_test, y_test,threshold=0.5):
    proba=model.predict_proba(x_test)[:, 1]
    prediction=(proba>=threshold).astype(int)
    
    report=classification_report(y_test, prediction, output_dict=True)
    roc=roc_auc_score(y_test, proba)
    f2=fbeta_score(y_test, prediction, beta=2)
    return {
        'precision':      report['1']['precision'],
        'recall':         report['1']['recall'],
        'f1':             report['1']['f1-score'],
        'roc_auc_score':  roc,
        'fbeta_score':       f2
    }

def passes_threshold(metrics: dict, min_recall=0.8, min_precision=0.25)->bool:
    return metrics['recall']>=min_recall and metrics['precision']>=min_precision

def _parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str)
    return parser.parse_args()

if __name__=="__main__":
    EXPERIMENT_NAME = "fraud_detection"
    client=MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    args= _parse_args()
    x_test=pd.read_csv(f"{args.processed_data}/x_test.csv")
    y_test=pd.read_csv(f"{args.processed_data}/y_test.csv").squeeze()


    model_xgb = mlflow.sklearn.load_model("models:/fraud-detection-xgboost/latest")
    model_forest = mlflow.sklearn.load_model("models:/fraud-detection-randomForest/latest")

    metrics_xgb=evaluate_model(model_xgb, x_test, y_test)
    metrics_forest=evaluate_model(model_forest, x_test, y_test,threshold=0.1)

    passed_xgb=passes_threshold(metrics_xgb)
    passed_forest=passes_threshold(metrics_forest)

    if not passed_xgb and not passed_forest:
        raise RuntimeError("Both models failed threshold — retrain with tuned parameters.")

    candidates={}
    if passed_xgb:
        candidates['xgboost']=(metrics_xgb, model_xgb, "model_xgboost")
    if passed_forest:
        candidates['randomForest']=(metrics_forest, model_forest, "model_randomForest")

    champion_name=max(candidates, key=lambda k: candidates[k][0]['recall'])
    champion_metrics, champion_model, champion_artifact = candidates[champion_name]


    challenger_name=[k for k in candidates if k != champion_name]
    
   
    active = mlflow.active_run()
    if active is None:
        active = mlflow.start_run()
    evaluate_run_id = active.info.run_id
    mlflow.log_metrics({f"champion_metrics_{k}": v for k, v in champion_metrics.items()})
    mlflow.log_param("champion", champion_name)
    mlflow.sklearn.log_model(champion_model,artifact_path=champion_artifact)

    result_champ=mlflow.register_model(
        model_uri=f"runs:/{evaluate_run_id}/{champion_artifact}",
        name="fraud-detection-champion"
    )
    version_champ=result_champ.version
    client.transition_model_version_stage(
        name="fraud-detection-champion",
        version=version_champ,
        stage="Production"
    )

    mlflow.log_artifact(f"{args.processed_data}/encoder.pkl")
    mlflow.log_artifact(f"{args.processed_data}/scaler.pkl")

    if challenger_name:
        challenger_metrics, challenger_model,challenger_artifact=candidates[challenger_name[0]]
        mlflow.log_param("challenger", challenger_name[0])
        mlflow.log_metrics({f"challenger_metrics_{k}": v for k, v in challenger_metrics.items()})
        mlflow.sklearn.log_model(challenger_model,artifact_path=challenger_artifact)
        
        result_challenger=mlflow.register_model(
            model_uri=f"runs:/{evaluate_run_id}/{challenger_artifact}",
            name="fraud-detection-challenger"
        )
        version_chall=result_challenger.version
        client.transition_model_version_stage(
            name="fraud-detection-challenger",
            version=version_chall,
            stage="Staging"
        )          

        