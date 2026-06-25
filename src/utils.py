from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import logging



def az_connect(config)->MLClient:
    """function to connect to azure workspaces 
            config:json file
    """

    ml_client=MLClient(
                        DefaultAzureCredential(),
                        subscription_id=config['subscription_id'],
                        resource_group_name=config['resource_group'],
                        workspace_name=config['workspace_name'])

    try:
        wsn=ml_client.workspaces.get(config["workspace_name"])
        logging.info(f"connect to workspace{wsn}")
        print(f"connection succ +{config['workspace_name']}")
    except Exception as e:
        print(f"connection failed + {e}")
        raise

    return ml_client

def load_train_data(path:str)->pd.DataFrame:
    """function to load trainning (tain,test)data from a csv in a azure boble storage
        df.columns=['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
       'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
       'isFlaggedFraud']
       [isFlaggedFraud,nameOrig,nameDest] will be dropped in the load data function cuz it will be absent at infrence time
    """

    df=pd.read_csv(path)
    df.drop(columns=['isFlaggedFraud','nameOrig','nameDest'],inplace=True)
    return df



