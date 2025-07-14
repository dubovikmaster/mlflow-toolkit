# mlflow-toolkit


## Getting started

### Installation

```python
pip install git+https://github.com/dubovikmaster/mlflow-toolkit.git
```

### Usage

```python
import pandas as pd
import numpy as np

import mlflow

from mlflow_toolkit import MLflowWorker

# set the tracking_uri and experiment name
mlflow.set_tracking_uri('http://localhost:5000')  # or your MLflow server URI
mlflow.set_experiment('my-awesome-project')

# init mlflow worker
mlflow_worker = MLflowWorker()

# create some artifacts like yaml, txt, csv, parquet files
features = ['a', 'b', 'c', 'd']
params = {'iterations': 100, 'depth': 5, 'cat_features': ['a', 'b']}

df = pd.DataFrame(np.random.random((100, 4)), columns=features)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    # log dataframe as csv file
    mlflow_worker.log_dataframe(df, 'data/train_data.csv', run_id=run_id, output_file_type='csv')
    # log dataframe as parquet file
    mlflow_worker.log_dataframe(df, 'data/data.parq', run_id=run_id)
    # log features names as text file
    mlflow_worker.log_text(run.info.run_id, '\n'.join(features), 'features.txt')
    # log model serialized model params
    mlflow_worker.log_as_pickle(params, 'params.pkl', run_id=run_id)
    # log model params as yaml file 
    mlflow_worker.log_dict(params, 'params.yml', run_id=run_id)

df_loaded = mlflow_worker.load_dataframe('data/train_data.parq', run_id=run_id)
# check the equals of dataframes
print(df_loaded.equals(df))

>> > True
```

