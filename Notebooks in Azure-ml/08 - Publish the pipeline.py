#
# Publish the pipeline from the run
published_pipeline = pipeline_run.publish_pipeline(
    name="diabetes-training-pipeline", description="Trains diabetes model", version="1.0")

published_pipeline
#

#
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)
#

#
from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print("Authentication header ready.")
#

#
import requests

experiment_name = 'mslearn-diabetes-pipeline'

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": experiment_name})
run_id = response.json()["Id"]
run_id
#

#
from azureml.pipeline.core.run import PipelineRun

published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)
pipeline_run.wait_for_completion(show_output=True)
#

