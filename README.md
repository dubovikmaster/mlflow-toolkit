# mlflow-toolkit

[![CI](https://github.com/dubovikmaster/mlflow-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/dubovikmaster/mlflow-toolkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlflow-toolkit)](https://pypi.org/project/mlflow-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/mlflow-toolkit)](https://pypi.org/project/mlflow-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Symmetric `log_*` / `load_*` helpers for MLflow artifacts.**

MLflow lets you log a dict or a text file, but getting artifacts **back** always means
`download_artifacts()` → temp dir → manual deserialization. And logging an in-memory
DataFrame means saving it to disk first. `mlflow-toolkit` closes that gap:

```python
worker.log_dataframe(run_id, df, 'data/train.parquet')          # object in ──┐
df = worker.load_dataframe(run_id, 'data/train.parquet')        # object out ─┘
```

One call in, one call out — the file format, the serialization backend and the temp-file
juggling are handled for you, inferred from the artifact path suffix.

## Highlights

- 📤 **Log objects straight from memory** — DataFrames, dicts, numpy arrays, figures,
  arbitrary picklable objects. No manual temp files.
- 📥 **Load them back into memory** — the missing half of the MLflow artifact API.
- 🐼 **pandas and polars** — polars `DataFrame` / `Series` / `LazyFrame` are first-class
  citizens; choose the output library with `backend='polars'`.
- 🧩 **Extensible format registry** — one `register_format()` call teaches
  `log_file` / `load_file` / `load_files` a new suffix.
- 📦 **Batch operations** — `load_files(run_id)` pulls a whole run's artifacts
  (recursively) into a dict of live objects.
- 🔢 **Typed run params** — `get_run_params` returns `100`, not `"100"`.
- 🔌 **Drop-in** — `MLflowWorker` subclasses `mlflow.MlflowClient`: everything the
  client does, plus the helpers.

## Installation

```bash
pip install mlflow-toolkit
# with dill/joblib pickle backends:
pip install "mlflow-toolkit[extras]"
# with polars support:
pip install "mlflow-toolkit[polars]"
```

Requires Python ≥ 3.10.

## Quickstart

```python
import mlflow
import numpy as np
import pandas as pd

from mlflow_toolkit import MLflowWorker

mlflow.set_tracking_uri('http://localhost:5000')   # or your MLflow server URI
mlflow.set_experiment('my-awesome-project')

worker = MLflowWorker()

df = pd.DataFrame(np.random.random((100, 4)), columns=['a', 'b', 'c', 'd'])
params = {'iterations': 100, 'depth': 5, 'cat_features': ['a', 'b']}

with mlflow.start_run() as run:
    run_id = run.info.run_id
    worker.log_dataframe(run_id, df, 'data/train.parquet')   # format from suffix
    worker.log_dict(run_id, params, 'params.yml')
    worker.log_as_pickle(run_id, params, 'params.pkl')
    worker.log_text(run_id, 'first experiment', 'notes.txt')

# ...days later, from anywhere:
df = worker.load_dataframe(run_id, 'data/train.parquet')
params = worker.load_dict(run_id, 'params.yml')
notes = worker.load_text_artifact(run_id, 'notes.txt')
```

All `log_*` methods take `(run_id, data, artifact_path)`; all `load_*` methods take
`(run_id, artifact_path)`.

## One entry point: `log_file` / `load_file`

Don't want to remember method names? `log_file` and `load_file` route any supported
suffix through the format registry:

```python
worker.log_file(run_id, df, 'data/train.parquet')       # dataframe → parquet
worker.log_file(run_id, df, 'data/train.csv', index=False)
worker.log_file(run_id, df, 'data/train.feather')       # arrow ipc
worker.log_file(run_id, params, 'config.json')          # dict → json (indented)
worker.log_file(run_id, model, 'model.joblib')          # object → joblib
worker.log_file(run_id, np.eye(3), 'matrix.npy')        # numpy array
worker.log_file(run_id, {'x': xs, 'y': ys}, 'arrays.npz')
worker.log_file(run_id, fig, 'plots/loss.png')          # matplotlib or plotly figure

train = worker.load_file(run_id, 'data/train.parquet')
config = worker.load_file(run_id, 'config.json')
model = worker.load_file(run_id, 'model.joblib')
```

Built-in formats:

| Suffixes | Data | Backed by |
|---|---|---|
| `.parquet`, `.parq` | DataFrame / Series | pandas · polars · pyarrow |
| `.csv` | DataFrame / Series | pandas · polars |
| `.feather` | DataFrame / Series | pandas · polars (Arrow IPC) |
| `.json`, `.yml`, `.yaml` | dict | json · PyYAML |
| `.pkl`, `.pickle`, `.dill`, `.joblib` | any object | pickle · dill · joblib |
| `.txt`, `.md`, `.html` | str | — |
| `.npy`, `.npz` | numpy array / dict of arrays | numpy |
| `.png`, `.jpg`, `.jpeg`, `.bmp`, `.svg` | matplotlib / plotly figure | save-only |

## Register your own format

The registry is public — one call and your suffix behaves like a built-in one
everywhere (`log_file`, `load_file`, `load_files`):

```python
import pandas as pd
from mlflow_toolkit import register_format

register_format(
    'excel', ['.xlsx'],
    save=lambda df, path, **kw: df.to_excel(path, **kw),
    load=lambda path, **kw: pd.read_excel(path, **kw),
)

worker.log_file(run_id, report_df, 'reports/q3.xlsx')
report = worker.load_file(run_id, 'reports/q3.xlsx')
```

Save-only and load-only formats are fine — pass just one of `save` / `load`:

```python
import onnx
from mlflow_toolkit import register_format

register_format(
    'onnx', ['.onnx'],
    save=lambda model, path, **kw: onnx.save(model, str(path)),
    load=lambda path, **kw: onnx.load(str(path)),
)
```

Replacing a built-in handler is explicit, so you can't shadow one by accident:

```python
register_format('csv-semicolon', ['.csv'],
                save=lambda df, path, **kw: df.to_csv(path, sep=';', **kw),
                load=lambda path, **kw: pd.read_csv(path, sep=';', **kw),
                overwrite=True)
```

Introspection helpers: `registered_suffixes()` lists everything the registry knows,
`get_format_handler('some/file.xlsx')` returns the matching handler (or `None`).
Compressed names resolve too: `data.csv.gz` → the `csv` handler.

## Polars

Polars objects are detected automatically on save — including `LazyFrame`, which is
collected for you. Pick the library you want back with `backend`:

```python
import polars as pl

pl_df = pl.DataFrame({'user': ['a', 'b'], 'score': [0.9, 0.7]})

worker.log_dataframe(run_id, pl_df, 'data/scores.parquet')            # polars in
worker.log_dataframe(run_id, pl_df.lazy().filter(pl.col('score') > 0.8),
                     'data/top.parquet')                              # lazy in

df = worker.load_dataframe(run_id, 'data/scores.parquet')                     # pandas out
pl_df = worker.load_dataframe(run_id, 'data/scores.parquet', backend='polars')  # polars out
```

## Whole runs at once

```python
# log several artifacts in one call
worker.log_files(run_id, {
    'data/train.parquet': train_df,
    'data/test.parquet': test_df,
    'params.yml': params,
    'features.txt': '\n'.join(features),
})

# ...and pull every artifact of the run back as a dict (recursive)
artifacts = worker.load_files(run_id)
# {'data/train.parquet': <DataFrame>, 'params.yml': {...}, 'features.txt': '...'}

# or just one directory
data = worker.load_files(run_id, 'data')
```

Files with no registered loader are skipped with a warning instead of failing the
whole batch.

## Typed run params

MLflow stores every param as a string. `get_run_params` gives you Python back:

```python
worker.log_param(run_id, 'iterations', 100)
worker.log_param(run_id, 'lr', 0.05)
worker.log_param(run_id, 'cat_features', ['a', 'b'])

worker.get_run_params(run_id)
# {'iterations': 100, 'lr': 0.05, 'cat_features': ['a', 'b']}   ← not strings
```

## Model registry

```python
latest = worker.get_latest_model_version('churn-model')   # highest version or None
if latest is not None:
    print(latest.version, latest.source)
```

## Development

```bash
git clone https://github.com/dubovikmaster/mlflow-toolkit.git
cd mlflow-toolkit
pip install -e ".[dev]"
pytest
ruff check .
```

Pull requests are welcome — `main` is protected, CI (tests on Python 3.10–3.13 + lint)
must be green to merge.

## License

[MIT](LICENSE)
