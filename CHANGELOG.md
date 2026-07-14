# Changelog

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
