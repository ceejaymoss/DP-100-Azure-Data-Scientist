#
from azureml.pipeline.core import ScheduleRecurrence, Schedule

# Submit the Pipeline every Monday at 00:00 UTC
recurrence = ScheduleRecurrence(frequency="Week", interval=1, week_days=["Monday"], time_of_day="00:00")
weekly_schedule = Schedule.create(ws, name="weekly-diabetes-training", 
                                  description="Based on time",
                                  pipeline_id=published_pipeline.id, 
                                  experiment_name='mslearn-diabetes-pipeline', 
                                  recurrence=recurrence)
print('Pipeline scheduled.')
#

#
schedules = Schedule.list(ws)
schedules
#

#
pipeline_experiment = ws.experiments.get('mslearn-diabetes-pipeline')
latest_run = list(pipeline_experiment.get_runs())[0]

latest_run.get_details()
#
