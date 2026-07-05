import mlflow
import argparse
import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,classification_report,fbeta_score


def train_xgb(x_train, y_train, max_depth=6, learning_rate=0.1,n_estimators=100, scale_pos_weight=75) -> XGBClassifier:
    """
    train_xgb trains a XGBClassifier model 
    params:
        x_train=training dataFrame
        y_train=the target series
    outputs:
        model_xgb=a XGBClassifier model
    """

    model_xgb=XGBClassifier(max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=42)
    model_xgb.fit(x_train,y_train)

    return model_xgb

def train_forest(x_train, y_train, n_estimators=50,
                pos_weight=10) -> RandomForestClassifier:
    """
    train_xgb trains a XGBClassifier model 
    params:
        x_train=training dataFrame
        y_train=the target series
    output:
        model_forest=a RandomForestClassifier model
    """

    model_forest= RandomForestClassifier(
    n_estimators=n_estimators,
    class_weight={0:1,1:pos_weight},
    random_state=42,
    n_jobs=-1,
    )
    model_forest.fit(x_train,y_train)

    return model_forest

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str)
    parser.add_argument("--run_info_output", type=str)
    parser.add_argument("--model_type", type=str, default="both",
                        choices=["xgb", "forest", "both"])
    parser.add_argument("--xgb_max_depth", type=int, default=9)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.40829877019115457)
    parser.add_argument("--xgb_n_estimators", type=int, default=150)
    parser.add_argument("--xgb_scale_pos_weight", type=float, default=75)
    parser.add_argument("--xgb_threshold", type=float, default=0.36146844823902946)

    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_pos_weight", type=float, default=25)
    parser.add_argument("--rf_threshold", type=float, default=0.15878201460012448)
    return parser.parse_args()

if __name__=="__main__":

    
    args =_parse_args()
    X_train = pd.read_csv(f"{args.processed_data}/x_train.csv")
    y_train = pd.read_csv(f"{args.processed_data}/y_train.csv").squeeze()
    x_cv=pd.read_csv(f"{args.processed_data}/x_cv.csv")
    y_cv = pd.read_csv(f"{args.processed_data}/y_cv.csv").squeeze()

    nested=args.model_type=="both"
    if nested:
        mlflow.set_experiment("fraud_detection")

    run_ids = {}
    if args.model_type in ("xgb", "both"):
        model_xgb=train_xgb(x_train=X_train,
            y_train=y_train,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            n_estimators=args.xgb_n_estimators,
            scale_pos_weight=args.xgb_scale_pos_weight
        )
        
        xgb_threshold=args.xgb_threshold
        proba_xgb=model_xgb.predict_proba(x_cv)[:,1]
        prediction_xgb=(proba_xgb>=xgb_threshold).astype(int)
        report_xgb=classification_report(y_cv,prediction_xgb,output_dict=True)
        roc_xgb=roc_auc_score(y_cv,model_xgb.predict_proba(x_cv)[:,1])
        fbeta_score_xgb=fbeta_score(y_cv, prediction_xgb, beta=2)

        metrics_xgb={
            'precision':report_xgb['1']['precision'],
            'recall':report_xgb['1']['recall'],
            'f1':report_xgb['1']['f1-score'],
            'roc_auc_score':roc_xgb,
            'fbeta_score':fbeta_score_xgb
        }
        
        params_xgb={
            "max_depth": args.xgb_max_depth,
            "learning_rate": args.xgb_learning_rate,
            "n_estimators": args.xgb_n_estimators,
            "scale_pos_weight": args.xgb_scale_pos_weight,
            "threshold": xgb_threshold
        }
        if nested:
            with mlflow.start_run(run_name='xgboost',nested=nested) as xgb_run:
                mlflow.log_metrics(metrics_xgb)
                mlflow.log_params(params_xgb)
                mlflow.sklearn.log_model(
                    model_xgb,
                    artifact_path='model')
                run_ids["xgb_run_id"]=xgb_run.info.run_id
        else:
            mlflow.log_metrics(metrics_xgb)
            mlflow.log_params(params_xgb)
            mlflow.sklearn.log_model(model_xgb, artifact_path='model')
            active = mlflow.active_run()
            run_ids["xgb_run_id"] = active.info.run_id if active else None

    if args.model_type in ("forest", "both"):
        model_forest=train_forest(x_train=X_train,
            y_train=y_train,
            n_estimators=args.rf_n_estimators,
            pos_weight=args.rf_pos_weight
        )

        threshold=args.rf_threshold
        proba=model_forest.predict_proba(x_cv)[:,1]
        prediction=(proba>=threshold).astype(int)
        report_forest=classification_report(y_cv,prediction,output_dict=True)
        roc_forest=roc_auc_score(y_cv,proba)
        fbeta_score_forest=fbeta_score(y_cv, prediction, beta=2)

        metrics_rf={
            'precision':report_forest['1']['precision'],
            'recall':report_forest['1']['recall'],
            'f1':report_forest['1']['f1-score'],
            'roc_auc_score':roc_forest,
            'fbeta_score':fbeta_score_forest
        }

        params_rf={
            "n_estimators": args.rf_n_estimators,
            "pos_weight": args.rf_pos_weight,
            "threshold": args.rf_threshold,
        }
        if nested:
            with mlflow.start_run(run_name='randomForest',nested=nested) as rf_run:
                mlflow.log_metrics(metrics_rf)
                mlflow.log_params(params_rf)
                mlflow.sklearn.log_model(model_forest,artifact_path='model')
                run_ids["run_id_rf"] = rf_run.info.run_id
        else:
            mlflow.log_metrics(metrics_rf)
            mlflow.log_params(params_rf)
            mlflow.sklearn.log_model(model_forest,artifact_path='model')
            active = mlflow.active_run()
            run_ids["run_id_rf"] = active.info.run_id if active else None

    if args.model_type == "both":
        os.makedirs(args.run_info_output, exist_ok=True)
        with open(os.path.join(args.run_info_output, "run_info.json"), "w") as f:
            json.dump({"xgb_run_id": run_ids["xgb_run_id"], "rf_run_id": run_ids["run_id_rf"]}, f)    