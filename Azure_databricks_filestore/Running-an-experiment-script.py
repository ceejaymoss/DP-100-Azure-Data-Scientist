import argparse
from azureml.core import Run, Dataset

parser = argparse.ArguementParser()
parser.add_arguement('--ds', type=str, dest='dataset_id')
args = parser.parse_args()

run = Run.get_context()
ws = run.experiemt.workspace
dataset = dataset.get.by_id(ws, id=args.dataset.id)
data = dataset.to.pandas_dataframe()

index = data.index
number_of_rows = len(index)
run.log('row_count', number_of_rows)
