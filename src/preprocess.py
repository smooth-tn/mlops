import pandas as pd
from sklearn.preprocessing import TargetEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from utils import load_train_data
import os
import joblib

COLS_TO_DROP = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']

def feature_engineering(x_train:pd.DataFrame,y_train:pd.Series):
    """
    x_train.columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest']
    y_train is a pandas series that is used for target encoding the 'type' column
    feature_engineering is function used in trainning 
    df must be split using train_test_spilt to avoid data leakage
    outputs: return a dataset with engineered features and an encoder (target encoder) foe the feature 'type'
    """

    df = x_train.copy()
    df=_compute_features(df)
    encoder =TargetEncoder(smooth='auto')
    df['type_encoded']=encoder.fit_transform(df[['type']],y_train)
    df_engineered=df.drop(columns=COLS_TO_DROP)
    df_engineered,scaler=_scale_data(df_engineered)
    df_engineered = pd.DataFrame(df_engineered, columns=df_engineered.columns)
    return df_engineered,encoder,scaler


def preprocess(df:pd.DataFrame,encoder : TargetEncoder,scaler:StandardScaler)->pd.DataFrame:
    """
    preprocess get a pd.DataFrame and transforme it to digestible data by the model at infrence time
    """

    df=df.copy()
    df=_compute_features(df)
    df['type_encoded']=encoder.transform(df[['type']])
    df_preprocessed=df.drop(columns=COLS_TO_DROP)
    df_preprocessed=scaler.transform(df_preprocessed)
    return df_preprocessed

def _scale_data(df:pd.DataFrame)->tuple:
    scaler=StandardScaler()
    scaled_df=scaler.fit_transform(df)
    return scaled_df,scaler

def _compute_features(df):
    """compute features"""

    df['transferred_amount_orig'] = df['newbalanceOrig'] - (df['oldbalanceOrg'] - df['amount'])
    df['transferred_amount_dest'] = df['newbalanceDest'] - (df['oldbalanceDest'] + df['amount'])
    return df



def split_data(df:pd.DataFrame,train_size=0.6,cv_size=0.5,random_state=42):
    """
    splits the data into smaller datasets for train,cross validation and test
    Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe including the target column 'isFraud'.
        train_size : float, optional
            Proportion of the dataset to use for training. Default is 0.6.
        cv_size : float, optional
            Proportion of the remainder (after train split) to use for CV.
            Default is 0.5, which gives equal CV and test sizes.
        random_state : int, optional
            Random seed for reproducibility. Default is 42.
    Returns
    -------
        x_train : pd.DataFrame
        y_train : pd.Series
        x_cv : pd.DataFrame
        y_cv : pd.Series
        x_test : pd.DataFrame
        y_test : pd.Series
    """

    df=df.copy()
    target=df.isFraud
    df.drop(columns=['isFraud'],inplace=True)
    x_train,x_temp,y_train,y_temp=train_test_split(df,target,train_size=train_size,random_state=random_state)
    x_cv,x_test,y_cv,y_test=train_test_split(x_temp,y_temp,train_size=cv_size,random_state=random_state)
    
    return x_train,y_train,x_cv,y_cv,x_test,y_test

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str)
    parser.add_argument("--processed_data", type=str)
    return parser.parse_args()


if __name__=="__main__":
    args=_parse_args()

    df=load_train_data(args.raw_data)
    x_train,y_train,x_cv,y_cv,x_test,y_test=split_data(df)
    x_train, encoder,scaler = feature_engineering(x_train, y_train)
    x_cv=pd.DataFrame(preprocess(x_cv, encoder,scaler))
    x_test=pd.DataFrame(preprocess(x_test, encoder,scaler))
    

    os.makedirs(args.processed_data, exist_ok=True)
    x_train.to_csv(f"{args.processed_data}/x_train.csv", index=False)
    y_train.to_frame().to_csv(f"{args.processed_data}/y_train.csv", index=False)
    x_cv.to_csv(f"{args.processed_data}/x_cv.csv",       index=False)
    y_cv.to_frame().to_csv(f"{args.processed_data}/y_cv.csv",       index=False)
    x_test.to_csv(f"{args.processed_data}/x_test.csv",   index=False)
    y_test.to_frame().to_csv(f"{args.processed_data}/y_test.csv",   index=False)
    joblib.dump(encoder, f"{args.processed_data}/encoder.pkl")
    joblib.dump(scaler, f"{args.processed_data}/scaler.pkl")
    