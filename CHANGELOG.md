# Changelog

## 0.4.0 (2026-07-14)

### Added

- **Format registry** (`mlflow_toolkit.formats`): file formats are now resolved through
  a suffix → handler registry instead of hard-coded `if/elif` chains. New public API:
  `register_format`, `get_format_handler`, `registered_suffixes`, `FormatHandler` —
  register a custom suffix once and `log_file` / `load_file` / `load_files` handle it
  like a built-in format.
- **Polars support**: `log_dataframe` / `log_file` accept polars DataFrame, Series and
  LazyFrame (LazyFrame is collected on save); loaders accept `backend='polars'` to
  return polars instead of pandas. Install via the `polars` extra.
- **New built-in formats**: `.feather` (pandas/polars), `.npy` (numpy array),
  `.npz` (dict of numpy arrays), `.svg` for figures.
- Figures: `log_file` now saves both matplotlib and plotly figures to image suffixes.

### Changed

- `worker.log_file` / `load_file` are now generic over the registry; error messages for
  unsupported suffixes list all registered suffixes.
- `load_dataframe` gained an explicit `backend` parameter ('pandas' | 'polars').
- `numpy` is declared as an explicit dependency.

## 0.3.0 (2026-07-14)

### Breaking changes

- Unified argument order across the whole API: all `log_*` methods take
  `(run_id, data, artifact_path)`, all `load_*` methods take `(run_id, artifact_path)`.
  Previously `load_parquet_artifact`, `load_csv_artifact`, `load_text_artifact`,
  `load_pickle_artifact` and `load_dataframe` took `(artifact_path, run_id)`.
- `get_latest_model_version` now returns `None` instead of `[]` when the model has no versions.
- `requires-python` raised to `>=3.10` (the package never actually worked on older
  versions: `typing.TypeAlias` was used, which appeared in 3.10).

### Fixed

- `joblib` and `dill` were imported in a single `try` block: if only `dill` was missing,
  the `joblib` backend was silently disabled too. Now imported independently.
- `pandas` and `pyarrow` are declared as dependencies (previously they worked only
  because the full `mlflow` distribution pulls them in).
- Artifact logging no longer relies on the private `MlflowClient._log_artifact_helper`,
  which could break on any MLflow update. The toolkit now has its own context manager.
- `bytes` paths no longer crash path-inspection helpers (`os.fsdecode` is used).
- `load_files` now actually skips unsupported file types with a warning, as documented,
  instead of raising.
- `load_files` is now recursive: files in nested artifact directories (e.g. `data/train.csv`)
  were previously silently invisible.
- Text files are written with explicit UTF-8 encoding (they were already read as UTF-8).
- README examples now match the real method signatures.

### Changed

- JSON dicts are saved indented by default (`indent=2`).
- `pickle.dump/load` and `dill.dump/load` now receive `**kwargs`.
- Added `py.typed` marker, test suite, ruff config and GitHub Actions CI.
