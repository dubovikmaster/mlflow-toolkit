# mlflow-toolkit

Symmetric `log_*` / `load_*` helpers for MLflow artifacts.

MLflow lets you log a dict or a text file, but getting artifacts **back** always means
`download_artifacts()` → temp dir → manual deserialization. `mlflow-toolkit` closes that
gap: log in-memory objects (pandas DataFrames, dicts, arbitrary picklable objects) and
load them back into memory with one call. The file format is inferred from the artifact
path suffix.

## Features

- `log_dataframe` / `load_dataframe` — pandas DataFrame or Series as `.parquet` or `.csv`
- `log_dict` / `load_dict` — dict as `.json`, `.yml` / `.yaml`
- `log_as_pickle` / `load_pickle_artifact` — any object via `pickle`, `dill` or `joblib`
  (backend inferred from `.pkl` / `.dill` / `.joblib` suffix)
- `log_file` / `load_file` — single entry point, format inferred from the suffix
- `log_files` / `load_files` — batch logging/loading of whole artifact directories
- `get_run_params` — run params with Python types restored (MLflow stores them as strings)
- `get_latest_model_version` — latest registered model version without deprecated stages

## Installation

```bash
pip install git+https://github.com/dubovikmaster/mlflow-toolkit.git
# with dill/joblib backends:
pip install "mlflow-toolkit[extras] @ git+https://github.com/dubovikmaster/mlflow-toolkit.git"
```

Requires Python >= 3.10.

## Usage

`MLflowWorker` is a drop-in subclass of `mlflow.MlflowClient` — everything the client
does, plus the helpers. All `log_*` methods take `(run_id, data, artifact_path)`,
all `load_*` methods take `(run_id, artifact_path)`.

```python
import pandas as pd
import numpy as np

import mlflow

from mlflow_toolkit import MLflowWorker

mlflow.set_tracking_uri('http://localhost:5000')  # or your MLflow server URI
mlflow.set_experiment('my-awesome-project')

worker = MLflowWorker()

features = ['a', 'b', 'c', 'd']
params = {'iterations': 100, 'depth': 5, 'cat_features': ['a', 'b']}
df = pd.DataFrame(np.random.random((100, 4)), columns=features)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    # log dataframe as parquet / csv (format inferred from the suffix)
    worker.log_dataframe(run_id, df, 'data/train_data.parquet')
    worker.log_dataframe(run_id, df, 'data/train_data.csv')
    # log feature names as a text file
    worker.log_text(run_id, '\n'.join(features), 'features.txt')
    # log params as pickle and as yaml
    worker.log_as_pickle(run_id, params, 'params.pkl')
    worker.log_dict(run_id, params, 'params.yml')

# ...and load everything back into memory
df_loaded = worker.load_dataframe(run_id, 'data/train_data.parquet')
assert df_loaded.equals(df)

params_loaded = worker.load_pickle_artifact(run_id, 'params.pkl')
features_loaded = worker.load_text_artifact(run_id, 'features.txt').splitlines()

# or load a whole directory at once
artifacts = worker.load_files(run_id)
# {'data/train_data.parquet': <DataFrame>, 'params.yml': {...}, ...}
```

### Typed run params

```python
worker.log_param(run_id, 'iterations', 100)
worker.log_param(run_id, 'cat_features', ['a', 'b'])

worker.get_run_params(run_id)
# {'iterations': 100, 'cat_features': ['a', 'b']}  — not strings!
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
