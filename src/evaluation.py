import mlflow
import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient


def get_latest_train_run(client: MlflowClient, experiment_name: str):
    """fetch the latest parent train run and return its two child runs"""
    experiment=client.get_experiment_by_name(experiment_name)
    runs=client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'train'",
        order_by=["start_time DESC"],
        max_results=1
    )
    parent_run=runs[0]
    child_runs=client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
    )
    return {run.data.tags['mlflow.runName']: run for run in child_runs}

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
        'f2_score':       f2
    }

def passes_threshold(metrics: dict, min_recall=0.8, min_precision=0.25)->bool:
    return metrics['recall']>=min_recall and metrics['precision']>=min_precision

def _parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str)
    return parser.parse_args()

if __name__=="__main__":
    EXPERIMENT_NAME = "fraud-detection v1"
    client=MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)

    args= _parse_args()
    x_test=pd.read_csv(f"{args.processed_data}/x_test.csv")
    y_test=pd.read_csv(f"{args.processed_data}/y_test.csv").squeeze()
    encoder=joblib.load(f"{args.processed_data}/encoder.pkl")
    scaler=joblib.load(f"{args.processed_data}/scaler.pkl")

    child_runs=get_latest_train_run(client, EXPERIMENT_NAME)
    xgb_run_id=child_runs['xgboost'].info.run_id
    rf_run_id=child_runs['randomForest'].info.run_id
    model_xgb=mlflow.sklearn.load_model(f"runs:/{xgb_run_id}/model_xgb")
    model_forest=mlflow.sklearn.load_model(f"runs:/{rf_run_id}/model_forest")
  

    metrics_xgb=evaluate_model(model_xgb, x_test, y_test)
    metrics_forest=evaluate_model(model_forest, x_test, y_test,threshold=0.1)

    passed_xgb=passes_threshold(metrics_xgb)
    passed_forest=passes_threshold(metrics_forest)

    if not passed_xgb and not passed_forest:
        raise RuntimeError("Both models failed threshold — retrain with tuned parameters.")

    candidates={}
    if passed_xgb:
        candidates['xgboost']=(metrics_xgb, model_xgb, xgb_run_id, "champion_pipeline")
    if passed_forest:
        candidates['randomForest']=(metrics_forest, model_forest, rf_run_id, "challenger_pipeline")

    champion_name=max(candidates, key=lambda k: candidates[k][0]['f2_score'])
    champion_metrics, champion_model, champion_run_id, champion_artifact = candidates[champion_name]
    champion_pipeline=Pipeline([
        ('encoder', encoder),
        ('scaler', scaler),
        ('champion', champion_model)])

    challenger_name=[k for k in candidates if k != champion_name]
    
    with mlflow.start_run(run_name='evaluate') as evaluate_run :
        evaluate_run_id = evaluate_run.info.run_id

        mlflow.log_metrics({f"test_champion_{k}": v for k, v in champion_metrics.items()})
        mlflow.log_param("champion", champion_name)
        mlflow.sklearn.log_model(champion_pipeline, artifact_path="champion_pipeline")
        mlflow.register_model(
            model_uri=f"runs:/{evaluate_run_id}/{champion_artifact}",
            name="fraud-detection-champion"
        )
        if challenger_name:
            challenger_metrics, challenger_model, challenger_run_id, challenger_artifact= candidates[challenger_name[0]]
            challenger_pipeline=Pipeline([
                ('encoder', encoder),
                ('scaler', scaler),
                ('challenger_model', challenger_model)])
            mlflow.log_param("challenger", challenger_name[0])
            mlflow.log_metrics({f"challenger_{k}": v for k, v in challenger_metrics.items()})
            mlflow.sklearn.log_model(challenger_pipeline, artifact_path="challenger_pipeline")
            mlflow.register_model(
                model_uri=f"runs:/{evaluate_run_id}/{challenger_artifact}",
                name="fraud-detection-challenger"
            )


    