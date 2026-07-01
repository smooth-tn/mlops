import mlflow
import argparse
import pandas as pd
import os
import json
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
    parser.add_argument("--run_info_input", type=str)
    return parser.parse_args()

if __name__=="__main__":
    EXPERIMENT_NAME = "fraud_detection"
    client=MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    args= _parse_args()
    x_test=pd.read_csv(f"{args.processed_data}/x_test.csv")
    y_test=pd.read_csv(f"{args.processed_data}/y_test.csv").squeeze()

    with open(os.path.join(args.run_info_input, "run_info.json"), "r") as f:
        run_info = json.load(f)
    xgb_run_id = run_info["xgb_run_id"]
    rf_run_id  = run_info["rf_run_id"]

    model_xgb = mlflow.sklearn.load_model(f"runs:/{xgb_run_id}/model")
    model_forest  = mlflow.sklearn.load_model(f"runs:/{rf_run_id}/model")

    metrics_xgb=evaluate_model(model_xgb, x_test, y_test)
    metrics_forest=evaluate_model(model_forest, x_test, y_test,threshold=0.1)

    passed_xgb=passes_threshold(metrics_xgb)
    passed_forest=passes_threshold(metrics_forest)

    if not passed_xgb and not passed_forest:
        raise RuntimeError("Both models failed threshold — retrain with tuned parameters.")

    candidates={}
    if passed_xgb:
        candidates['xgboost']=(metrics_xgb, model_xgb, "model")
    if passed_forest:
        candidates['randomForest']=(metrics_forest, model_forest, "model")

    champion_name=max(candidates, key=lambda k: candidates[k][0]['recall'])
    champion_metrics, champion_model, champion_artifact = candidates[champion_name]


    challenger_name=[k for k in candidates if k != champion_name]
    
    run_id_map={"xgboost": xgb_run_id, "randomForest": rf_run_id}
    champ_run_id=run_id_map[champion_name]
    
    mlflow.log_metrics({f"champion_metrics_{k}": v for k, v in champion_metrics.items()})
    mlflow.log_param("champion", champion_name)

    result_champ=mlflow.register_model(
        model_uri=f"runs:/{champ_run_id}/{champion_artifact}",
        name="fraud-detection-champion"
    )
    version_champ=2
    client.transition_model_version_stage(
        name="fraud-detection-champion",
        version=version_champ,
        stage="Production"
    )

    if challenger_name:
        challenger_metrics, challenger_model,challenger_artifact=candidates[challenger_name[0]]
        mlflow.log_param("challenger", challenger_name[0])
        mlflow.log_metrics({f"challenger_metrics_{k}": v for k, v in challenger_metrics.items()})
        chall_run_id = run_id_map[challenger_name[0]]
        result_challenger=mlflow.register_model(
            model_uri=f"runs:/{chall_run_id}/{challenger_artifact}",
            name="fraud-detection-challenger"
        )
        version_chall=result_challenger.version
        client.transition_model_version_stage(
            name="fraud-detection-challenger",
            version=version_chall,
            stage="Staging"
        )          

        