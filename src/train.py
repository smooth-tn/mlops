import mlflow
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,classification_report,fbeta_score


def train_xgb(x_train,y_train)->XGBClassifier:
    """
    train_xgb trains a XGBClassifier model 
    params:
        x_train=training dataFrame
        y_train=the target series
    outputs:
        model_xgb=a XGBClassifier model
    """

    model_xgb=XGBClassifier(scale_pos_weight=75,random_state=42)
    model_xgb.fit(x_train,y_train)

    return model_xgb

def train_forest(x_train,y_train)->RandomForestClassifier:
    """
    train_xgb trains a XGBClassifier model 
    params:
        x_train=training dataFrame
        y_train=the target series
    output:
        model_forest=a RandomForestClassifier model
    """

    model_forest=model_forest = RandomForestClassifier(
    n_estimators=50,
    class_weight={0:1,1:10},
    random_state=42,
    n_jobs=-1  
    )
    model_forest.fit(x_train,y_train)

    return model_forest

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str)
    return parser.parse_args()

if __name__=="__main__":

    mlflow.set_experiment("fraud-detection v1")
    args =_parse_args()
    X_train = pd.read_csv(f"{args.processed_data}/x_train.csv")
    y_train = pd.read_csv(f"{args.processed_data}/y_train.csv").squeeze()
    x_cv=pd.read_csv(f"{args.processed_data}/x_cv.csv")
    y_cv = pd.read_csv(f"{args.processed_data}/y_cv.csv").squeeze()
    model_xgb=train_xgb(X_train,y_train)
    model_forest=train_forest(X_train,y_train)

    report_xgb=classification_report(y_cv,model_xgb.predict(x_cv),output_dict=True)
    roc_xgb=roc_auc_score(y_cv,model_xgb.predict_proba(x_cv)[:,1])
    fbeta_score_xgb=fbeta_score(y_cv, model_xgb.predict(x_cv), beta=2)

    threshold=0.1
    proba=model_forest.predict_proba(x_cv)[:,1]
    prediction=(proba>=threshold).astype(int)
    report_forest=classification_report(y_cv,prediction,output_dict=True)
    roc_forest=roc_auc_score(y_cv,proba)
    fbeta_score_forest=fbeta_score(y_cv, prediction, beta=2)

    with mlflow.start_run(run_name='train'):
        with mlflow.start_run(run_name='xgboost',nested=True):
            mlflow.log_metrics({
                'precision':report_xgb['1']['precision'],
                'recall':report_xgb['1']['recall'],
                'f-1':report_xgb['1']['f1-score'],
                'roc_auc_score':roc_xgb,
                'fbeta_score':fbeta_score_xgb
            })
            mlflow.sklearn.log_model(model_xgb,'model_xgb')

        with mlflow.start_run(run_name='randomForest',nested=True):
            mlflow.log_metrics({
                'precision':report_forest['1']['precision'],
                'recall':report_forest['1']['recall'],
                'f-1':report_forest['1']['f1-score'],
                'roc_auc_score':roc_forest,
                'fbeta_score':fbeta_score_forest
            })

            mlflow.sklearn.log_model(model_forest,'model_forest')  