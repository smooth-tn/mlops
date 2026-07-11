from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
import pandas as pd
import logging




def az_connect(config)->MLClient:
    ml_client=MLClient(
                        InteractiveBrowserCredential(tenant_id="6d19977e-97cf-425f-b759-dffa37424bd1"),
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

#test 4


