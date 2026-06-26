from src.utils import az_connect
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml import load_job,load_component

if __name__=="__main__":

    login={
        "subscription_id": "3c3b6d94-78ba-4830-b73c-a8e2b2835dae",
        "resource_group": "mlopsmouhib",
        "workspace_name": "amlmouhib"
    }
    az_client=az_connect(login)

    component=load_job("pipeline.yml") 
    az_client.jobs.create_or_update(component )